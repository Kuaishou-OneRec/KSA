#!/usr/bin/env python3
"""One-Hopper-GPU end-to-end test of the public shells; not a training launcher."""

import argparse
import csv
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = REPO_ROOT / "examples/ksa_v2"
KERNEL_DIR = REPO_ROOT / "summary_attention_kernel/ksa_v2"


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def prepare(output):
    output.mkdir(parents=True, exist_ok=False)
    fixtures = output / "fixtures"
    fixtures.mkdir()
    prefix = fixtures / "train"
    (3 + np.arange(64 * 2048 + 1, dtype=np.uint32) % 253).astype("<u4").tofile(str(prefix) + ".bin")
    Path(str(prefix) + ".bin.meta").write_text("uint32")
    data = fixtures / "dataset.json"
    write_json(data, {
        "name": "KaiMMapDataset", "data_prefix": [1.0, str(prefix)],
        "seq_length": 2048, "seed": 1024, "seeded_sampling": True,
        "global_batch_size": 1, "skip_warmup": True, "num_workers": 1,
    })
    for name, source in (
        ("ksa", "cksa_g128_1b9_128k.json"),
        ("cksa", "cksa_g128_1b9_128k.json"),
        ("sksa", "cksa_g128_sksa_block4_1b9_128k.json"),
    ):
        model = json.loads((ROOT / "configs" / source).read_text())
        model.update(
            vocab_size=256, summary_token_begin=256, embed_dim=256,
            num_layers=4, num_heads=2, num_kv_heads=2, intermediate_dim=768,
            max_seq_len=2048, rope_base=10000, summary_layer_freq=[1, 1, 1, 1],
            summary_sliding_chunk_num=[4, 4, 4, 256],
        )
        if name == "ksa":
            model.update(summary_meta_pool_mode="none", summary_meta_pool_group_size=0,
                         summary_meta_pool_mass_correction="none")
        else:
            model["summary_meta_pool_group_size"] = [8, 8, 8, 0]
        if name == "sksa":
            model.update(sksa_sparse_full_layers=[3], sksa_indexer_token_budget=64,
                         sksa_local_window_tokens=32, sksa_indexer_summary_sink_tokens=8)
        write_json(fixtures / (name + ".json"), model)

    bootstrap = [
        sys.executable, "-m", "torch.distributed.run",
        "--nnodes", "1", "--nproc-per-node", "1", "--node-rank", "0",
        "--master-addr", "127.0.0.1", "--master-port", "29597",
        str(REPO_ROOT / "recipes/ksa_v2.py"),
        "--model-config", str(fixtures / "ksa.json"),
        "--dataset-config", str(data),
        "--context-parallel-size", "1", "--gradient-accumulation-steps", "1",
        "--chunked-loss-minibatch-size", "256", "--learning-rate", "1e-4",
        "--min-lr", "1e-4", "--warmup-start-lr", "1e-6", "--lr-scheduler-type", "wsd",
        "--num-warmup-steps", "0", "--num-decay-steps", "0",
        "--save-checkpoint-per-step", "1", "--fp32-weight",
        "--use-flash-attention-2", "--enable-gradient-checkpointing",
        "--logging-per-step", "1",
        "--seed", "42", "--comment", "test_fixture", "--commit-id", "test",
    ]
    commands = [
        {"segment": "pretrain", "command": bootstrap + [
            "--output-dir", str(output / "pretrain"), "--num-training-steps", "2"],
         "checkpoint": str(output / "pretrain/global_step2"), "expected_phase_step": 2},
        {"segment": "resume", "command": bootstrap + [
            "--output-dir", str(output / "resume"), "--num-training-steps", "4",
            "--checkpoint-dir", str(output / "pretrain"), "--checkpoint-id", "global_step2",
            "--resume-weights", "--resume-optimizer", "--resume-dataloader"],
         "checkpoint": str(output / "resume/global_step4"), "expected_phase_step": 4},
    ]
    checkpoint = Path(commands[-1]["checkpoint"])
    for stage, segment, model, cap, step in (
        ("compress_warmup", "compress", "cksa", 2, 2),
        ("full_cpt", "full_cpt", "cksa", 2, 4),
        ("indexer_warmup", "indexer", "sksa", 2, 6),
        ("indexer_warmup", "indexer_resume", "sksa", 4, 8),
    ):
        env = {
            "NNODES": "1", "GPUS_PER_NODE": "1", "NODE_RANK": "0",
            "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "29597",
            "GBS": "1", "TOTAL_STEPS": str(cap), "WARMUP_STEPS": "0",
            "SAVE_EVERY": "1", "SAVE_STEPS": "1", "SOURCE_GBS": "1",
            "MODEL_CONFIG": str(fixtures / (model + ".json")), "DATASET_CONFIG": str(data),
            "OUTPUT_DIR": str(output / segment),
            "CHECKPOINT_DIR": str(checkpoint.parent), "CHECKPOINT_ID": checkpoint.name,
            "RESUME": "1" if segment == "indexer_resume" else "0", "RESUME_DATALOADER": "1",
        }
        if stage == "compress_warmup":
            env.update(MODEL_DIR=str(output / "hf_ksa"), CHECKPOINT_DIR="", RESUME_DATALOADER="0")
        command = ["bash", str(ROOT / "scripts" / (stage + ".sh")),
                   "--chunked-loss-minibatch-size", "256"]
        if stage == "indexer_warmup":
            command += ["--sksa-indexer-monitor-topk-blocks", "16"]
        checkpoint = output / segment / f"global_step{step}"
        commands.append({"segment": segment, "command": command, "env": env,
                         "checkpoint": str(checkpoint), "expected_phase_step": cap})
    plan = {"status": "prepared_not_run", "commands": commands}
    write_json(output / "smoke_plan.json", plan)
    return plan


def inspect_checkpoint(checkpoint, phase_step):
    import torch
    from torch.distributed.checkpoint import FileSystemReader
    from torch.distributed.checkpoint.default_planner import _EmptyStateDictLoadPlanner
    from torch.distributed.checkpoint.state_dict_loader import _load_state_dict

    reader = FileSystemReader(checkpoint)
    metadata = reader.read_metadata()
    for location in metadata.storage_data.values():
        shard = checkpoint / location.relative_path
        if not shard.is_file() or shard.stat().st_size < location.offset + location.length:
            raise AssertionError(f"Incomplete checkpoint shard: {shard}")
    state = {}
    _load_state_dict(state, storage_reader=reader, planner=_EmptyStateDictLoadPlanner(), no_dist=True)
    app = state["app"]
    assert app["optim"]["state"], "Adam state was not saved"
    assert any(
        value.get("exp_avg") is not None and value["exp_avg"].abs().sum() > 0
        for value in app["optim"]["state"].values()
    ), "No nonzero Adam first moment; trainable weights did not receive a gradient"
    head_state = app["optim"]["state"].get("model.output.weight")
    if head_state is not None:
        assert int(head_state["step"]) == phase_step, "LM head missed an optimizer update"
    assert app["training"]["step_scheduler"]["global_step"] == phase_step
    for name, value in app["model"].items():
        assert torch.isfinite(value).all(), f"Nonfinite model tensor: {name}"
    data = torch.load(checkpoint / "dataloader_state.pt", map_location="cpu", weights_only=False)
    assert data["state_kind"] == "consumer_boundary_v1"
    metrics_path = checkpoint.parent / "metrics.csv"
    with metrics_path.open() as stream:
        rows = list(csv.DictReader(stream))
    losses = [float(row["training/loss"]) for row in rows if row.get("training/loss")]
    assert losses and all(math.isfinite(x) for x in losses), "Missing or nonfinite loss"
    return app["model"], {"loss": losses[-1], "tensors": len(app["model"]), "data_state": data["state_kind"]}


def export_and_infer(output, checkpoint, stage, model_config):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    template = output / "fixtures" / ("hf_" + stage)
    template.mkdir()
    for source in (ROOT / "hf_template").glob("*.py"):
        shutil.copy2(source, template / source.name)
    config = json.loads((ROOT / "hf_template/config.json").read_text())
    config.update(bos_token_id=0, eos_token_id=1, pad_token_id=0)
    write_json(template / "config.json", config)
    write_json(template / "generation_config.json", {"bos_token_id": 0, "eos_token_id": 1, "pad_token_id": 0})
    vocabulary = {"<pad>": 0, "<eos>": 1, "<unk>": 2}
    vocabulary.update({f"t{i}": i for i in range(3, 256)})
    tokenizer = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, eos_token="<eos>", pad_token="<pad>", unk_token="<unk>")
    fast.save_pretrained(template)
    write_json(template / "vocab.json", vocabulary)
    (template / "merges.txt").write_text("#version: 0.2\n")
    export_dir = output / ("hf_" + stage)
    subprocess.run([
        sys.executable, str(ROOT / "tools/export_hf.py"),
        "--checkpoint", str(checkpoint),
        "--model-config", str(output / "fixtures" / (model_config + ".json")),
        "--runtime-dir", str(template),
        "--output-dir", str(export_dir),
    ], check=True, cwd=ROOT)
    # A fresh process cannot reuse training modules or the already imported kernel.
    check = r'''
from pathlib import Path
import sys
target = Path(sys.argv[1])
sys.path[:0] = [str(target / "kernel"), str(target / "kernel/deps")]
import torch
import ksa_attn
from transformers import AutoModelForCausalLM, AutoTokenizer
assert Path(ksa_attn.__file__).is_relative_to(target)
tokenizer = AutoTokenizer.from_pretrained(target, trust_remote_code=True, local_files_only=True)
assert tokenizer.encode("t3 t4", add_special_tokens=False) == [3, 4]
model = AutoModelForCausalLM.from_pretrained(
    target, trust_remote_code=True, local_files_only=True,
    dtype=torch.bfloat16, device_map="cuda:0",
).eval()
with torch.inference_mode():
    for length in (7, 63, 2031):
        ids = (torch.arange(length, device="cuda") % 253 + 3).unsqueeze(0)
        result = model(ids, use_cache=True)
        for _ in range(16):
            assert torch.isfinite(result.logits).all()
            next_id = result.logits[:, -1].argmax(-1, keepdim=True)
            ids = torch.cat((ids, next_id), dim=1)
            inputs = model.prepare_inputs_for_generation(
                ids, past_key_values=result.past_key_values, use_cache=True,
            )
            result = model(**inputs)
        assert torch.isfinite(result.logits).all()
assert not any(name == "muse" or name.startswith("muse.") for name in sys.modules)
'''
    subprocess.run([sys.executable, "-B", "-c", check, str(export_dir)],
                   check=True, cwd=output, env=os.environ | {"PYTHONPATH": ""})
    return {"prefill_lengths": [7, 63, 2031], "decode_tokens": 16, "finite_logits": True}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--run-prepared", action="store_true", help="Execute a previously prepared, unstarted smoke directory")
    args = p.parse_args()
    if args.prepare_only and args.run_prepared:
        p.error("--prepare-only and --run-prepared are mutually exclusive")
    output = args.output.resolve()
    plan = json.loads((output / "smoke_plan.json").read_text()) if args.run_prepared else prepare(output)
    if args.prepare_only:
        print(f"Prepared CPU fixtures and GPU commands in {output}; no training executed.")
        return
    temp = Path(os.environ.get("TMPDIR", output / "tmp"))
    temp.mkdir(parents=True, exist_ok=True)
    cache = output / ".cache"
    os.environ.update(
        PYTHONPATH=os.pathsep.join([str(KERNEL_DIR / "v0.1"), str(REPO_ROOT)]),
        PATH=str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", ""),
        PYTHONDONTWRITEBYTECODE="1", TMPDIR=str(temp), TMP=str(temp), TEMP=str(temp),
        XDG_CACHE_HOME=str(cache), HF_HOME=str(cache / "hf"),
        TORCH_HOME=str(cache / "torch"), TORCH_EXTENSIONS_DIR=str(cache / "extensions"),
        TRITON_CACHE_DIR=str(cache / "triton"), CUDA_CACHE_PATH=str(cache / "cuda"),
        HF_MODULES_CACHE=str(cache / "modules"), TORCHINDUCTOR_CACHE_DIR=str(cache / "inductor"),
        CUTE_DSL_CACHE_DIR=str(cache / "cute"), NUMBA_CACHE_DIR=str(cache / "numba"),
        MPLCONFIGDIR=str(cache / "matplotlib"), SKSA_LIGHTNING_BACKEND="cute",
        SKSA_SUMMARY_SPARSE_CORE="cute_mask", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
    )
    for key in ("OMPI_COMM_WORLD_RANK", "OMPI_COMM_WORLD_LOCAL_RANK", "OMPI_COMM_WORLD_SIZE"):
        os.environ.pop(key, None)
    sys.path.insert(0, str(KERNEL_DIR / "v0.1"))
    import torch
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        raise RuntimeError("Select an idle H100/H800 (SM90) via CUDA_VISIBLE_DEVICES")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one reserved GPU via CUDA_VISIBLE_DEVICES")
    report = {"status": "running", "stages": [], "inference": {}}
    write_json(output / "smoke_result.json", report)
    previous = None
    for item in plan["commands"]:
        print(f"Running {item['segment']}", flush=True)
        log = output / (item["segment"] + ".log")
        with log.open("x") as stream:
            subprocess.run(item["command"], env=os.environ | item.get("env", {}),
                           check=True, cwd=REPO_ROOT, stdout=stream, stderr=subprocess.STDOUT)
        weights, result = inspect_checkpoint(Path(item["checkpoint"]), item["expected_phase_step"])
        if item["segment"] in {"compress", "indexer", "indexer_resume"}:
            for name in previous:
                if item["segment"].startswith("indexer") and "sksa_indexer" in name:
                    continue
                torch.testing.assert_close(weights[name], previous[name], rtol=0, atol=0)
            result["frozen_backbone_unchanged"] = True
        result["segment"] = item["segment"]
        report["stages"].append(result)
        if item["segment"] == "resume":
            report["inference"]["ksa"] = export_and_infer(
                output, Path(item["checkpoint"]), "ksa", "ksa",
            )
        write_json(output / "smoke_result.json", report)
        previous = weights
    for item in plan["commands"]:
        stage = item["segment"]
        if stage in {"compress", "full_cpt", "indexer_resume"}:
            model_config = "sksa" if stage == "indexer_resume" else "cksa"
            report["inference"][stage] = export_and_infer(
                output, Path(item["checkpoint"]), stage, model_config,
            )
            write_json(output / "smoke_result.json", report)
    report["status"] = "passed"
    write_json(output / "smoke_result.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
