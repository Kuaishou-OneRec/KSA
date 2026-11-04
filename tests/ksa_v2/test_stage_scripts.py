"""The three public shells directly launch the training recipe."""
import importlib.util
import os
from pathlib import Path
import shlex
import subprocess

import pytest

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "examples/ksa_v2"
SCRIPTS = ROOT / "scripts"
STAGES = [
    ("compress_warmup", 1600, 1e-4, 200, None, 6_710_886_400),
    ("full_cpt", 800, 1e-4, 400, 1600, 3_355_443_200),
    ("indexer_warmup", 4400, 1e-3, 200, 2400, 18_454_937_600),
]


def environment(tmp_path, **overrides):
    env = os.environ.copy()
    for name in ("MODEL_DIR", "MODEL_CONFIG", "DATASET_CONFIG", "CHECKPOINT_DIR", "CHECKPOINT_ID",
                 "GBS", "LR", "WARMUP_STEPS", "TOTAL_STEPS", "SAVE_EVERY", "SAVE_STEPS",
                 "SOURCE_GBS", "RESUME", "RESUME_DATALOADER"):
        env.pop(name, None)
    env.update(NNODES="1", GPUS_PER_NODE="8", NODE_RANK="0",
               MASTER_ADDR="127.0.0.1", MASTER_PORT="29500",
               OUTPUT_DIR=str(tmp_path / "new output"), PYTHONDONTWRITEBYTECODE="1")
    env.update(overrides)
    return env


def shell_command(stage, tmp_path, *extra, **overrides):
    result = subprocess.run(
        ["bash", str(SCRIPTS / (stage + ".sh")), "--dry-run", *extra],
        cwd=tmp_path, env=environment(tmp_path, **overrides),
        capture_output=True, text=True, check=True, timeout=30,
    )
    command = shlex.split(result.stdout)
    assert command[:3] == ["python", "-m", "torch.distributed.run"]
    assert not (tmp_path / "new output").exists()
    return command


def parse_recipe(command):
    spec = importlib.util.spec_from_file_location("public_recipe", REPO / "recipes/ksa_v2.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    start = command.index("recipes/ksa_v2.py") + 1
    return module.get_argument_parser().parse_args(command[start:])


@pytest.mark.parametrize("stage,steps,lr,warmup,parent,tokens", STAGES)
def test_shell_defaults_match_paper(stage, steps, lr, warmup, parent, tokens, tmp_path):
    args = parse_recipe(shell_command(stage, tmp_path))
    assert args.num_training_steps == steps
    assert args.learning_rate == args.min_lr == lr
    assert args.warmup_start_lr == 1e-6
    assert args.num_warmup_steps == warmup
    assert args.num_decay_steps == 0
    assert args.gradient_accumulation_steps == 4
    assert args.context_parallel_size == 1
    assert args.save_checkpoint_per_step == 200
    assert args.save_checkpoint_steps == f"10,{steps}"
    if parent is None:
        assert args.model_dir == "models/KSA-v1-1.9B"
        assert args.checkpoint_id is None
        assert not args.resume_weights and not args.resume_dataloader
    else:
        assert args.model_dir is None
        assert args.checkpoint_id == f"global_step{parent}"
        assert args.resume_weights and args.resume_dataloader
    assert not args.resume_optimizer
    assert args.train_meta_pool_only == (stage == "compress_warmup")
    assert args.enable_meta_pool_attn_distill == (stage == "compress_warmup")
    assert args.train_sksa_indexer_only == (stage == "indexer_warmup")
    assert args.sksa_dense_warmup == (stage == "indexer_warmup")
    assert args.enable_sksa_indexer_distill == (stage == "indexer_warmup")
    assert bool(args.allow_random_init_params) == (stage != "full_cpt")
    assert args.resume_dataloader_source_global_batch_size is None
    assert 32 * 131072 * steps == tokens


@pytest.mark.parametrize("stage", [row[0] for row in STAGES])
def test_shell_full_resume_and_new_data_are_distinct(stage, tmp_path):
    resumed = parse_recipe(shell_command(
        stage, tmp_path, RESUME="1", CHECKPOINT_DIR="output/previous", CHECKPOINT_ID="global_step123",
    ))
    assert resumed.resume_weights and resumed.resume_optimizer and resumed.resume_dataloader
    assert not resumed.allow_random_init_params
    assert resumed.resume_dataloader_source_global_batch_size is None
    assert resumed.checkpoint_dir == "output/previous"
    assert resumed.checkpoint_id == "global_step123"
    fresh = parse_recipe(shell_command(stage, tmp_path, RESUME_DATALOADER="0"))
    assert fresh.resume_weights == (stage != "compress_warmup")
    assert not fresh.resume_dataloader and not fresh.resume_optimizer
    assert fresh.resume_dataloader_source_global_batch_size is None


@pytest.mark.parametrize("stage", [row[0] for row in STAGES])
def test_shell_multinode_overrides_and_short_test(stage, tmp_path):
    command = shell_command(
        stage, tmp_path, "--chunked-loss-minibatch-size", "64",
        NNODES="4", GPUS_PER_NODE="8", NODE_RANK="2", MASTER_ADDR="192.0.2.1",
        MODEL_CONFIG="/model path/model.json", DATASET_CONFIG="/data path/data.json",
        TOTAL_STEPS="3", WARMUP_STEPS="0", SAVE_EVERY="3", SAVE_STEPS="3",
        SOURCE_GBS="32",
    )
    args = parse_recipe(command)
    assert command[command.index("--nnodes") + 1] == "4"
    assert command[command.index("--node-rank") + 1] == "2"
    assert args.gradient_accumulation_steps == 1
    assert args.num_training_steps == 3 and args.num_warmup_steps == 0
    assert args.model_config == "/model path/model.json"
    assert args.dataset_config == "/data path/data.json"
    assert args.chunked_loss_minibatch_size == 64


def test_invalid_gbs_is_not_silently_truncated(tmp_path):
    with pytest.raises(subprocess.CalledProcessError):
        shell_command("full_cpt", tmp_path, GBS="31")


def test_hf_initialization_and_legacy_dcp_are_explicit(tmp_path):
    args = parse_recipe(shell_command("compress_warmup", tmp_path, MODEL_DIR="/local/HF model"))
    assert args.model_dir == "/local/HF model" and args.checkpoint_dir is None
    for flag in ("RESUME", "RESUME_DATALOADER"):
        with pytest.raises(subprocess.CalledProcessError):
            shell_command("compress_warmup", tmp_path, **{flag: "1"})
    args = parse_recipe(shell_command(
        "compress_warmup", tmp_path, CHECKPOINT_DIR="/training/dcp",
        CHECKPOINT_ID="global_step100", SOURCE_GBS="240",
    ))
    assert args.model_dir is None
    assert args.checkpoint_id == "global_step100"
    assert args.resume_weights and args.resume_dataloader
    assert args.resume_dataloader_source_global_batch_size == 240


def test_public_stage_scripts_are_flat_and_only_cover_paper():
    assert {p.relative_to(SCRIPTS).as_posix() for p in SCRIPTS.rglob("*.sh")} == {
        "install.sh", "compress_warmup.sh", "full_cpt.sh", "indexer_warmup.sh",
    }
