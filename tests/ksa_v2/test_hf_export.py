"""Offline checks for the single infra HF runtime and self-contained exports."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
import torch.distributed.checkpoint as dcp
from safetensors.torch import load_file

from export_hf import (
    canonical_checkpoint_keys,
    export,
    load_and_map_dcp,
    load_training_config,
    runtime_files_for,
    validate_runtime_config,
)
from muse.models.qwen3_sa.modeling import Qwen3SummaryModel
from test_model_cpu import configuration

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "examples/ksa_v2"
KERNEL = REPO / "summary_attention_kernel/ksa_v2/inference"


def test_runtime_layer_patterns_do_not_execute_python():
    path = ROOT / "hf_template/runtime_config.py"
    spec = importlib.util.spec_from_file_location("runtime_config_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    parse = module._parse_config_pattern
    assert parse("([128]*3+[0])*6") == [128, 128, 128, 0] * 6
    assert parse("([1])*36") == [1] * 36
    assert parse([1, 2]) == [1, 2]
    assert parse(8) == 8
    for value in ("__import__('os')", "[x for x in range(4)]", "True", "obj.attr"):
        with pytest.raises(ValueError):
            parse(value)


def test_runtime_profile_accepts_all_public_stages():
    for path in sorted((ROOT / "configs").glob("*.json")):
        validate_runtime_config(load_training_config(path))
    parent = load_training_config(ROOT / "configs/cksa_g128_1b9_128k.json")
    parent.update(summary_meta_pool_mode="none", summary_meta_pool_group_size=0,
                  summary_meta_pool_mass_correction="none")
    validate_runtime_config(parent)


@pytest.mark.parametrize("field,value", [
    ("summary_chunk_size", 4),
    ("sksa_indexer_block_size", 8),
    ("sksa_indexer_qk_norm", True),
    ("num_kv_heads", 8),
    ("summary_partial_chunk_policy", "summarize"),
    ("sksa_indexer_summary_sink_tokens", 4),
    ("summary_independent_qkv", True),
    ("summary_meta_pool_mode", "scalar"),
])
def test_runtime_profile_rejects_unsupported_configurations(field, value):
    config = load_training_config(ROOT / "configs/cksa_g128_sksa_block4_1b9_128k.json")
    config[field] = value
    with pytest.raises(ValueError, match=field):
        validate_runtime_config(config)


def test_runtime_dependencies_are_checked(tmp_path):
    (tmp_path / "modeling_qwen3.py").write_text("from .missing_helper import forward\n")
    with pytest.raises(ValueError, match="unbundled relative import"):
        runtime_files_for(tmp_path)
    assert len(runtime_files_for(ROOT / "hf_template")) == 13
    assert [path.name for path in ROOT.glob("hf_*") if path.is_dir()] == ["hf_template"]


def test_export_canonicalizes_legacy_keys_and_rejects_collisions(tmp_path):
    old = "sc" + "sa_"
    tensor = torch.ones(2, 3)
    mapped = canonical_checkpoint_keys({f"model.layers.3.attn.{old}indexer.key_proj.weight": tensor})
    assert mapped["model.layers.3.attn.sksa_indexer.key_proj.weight"] is tensor
    with pytest.raises(ValueError, match="conflicting checkpoint aliases"):
        canonical_checkpoint_keys({old + "enabled": True, "sksa_enabled": True})
    path = ROOT / "configs/cksa_g128_sksa_block4_1b9_128k.json"
    expected = load_training_config(path)
    legacy = {name.replace("sksa_", old): value for name, value in json.loads(path.read_text()).items()}
    source = tmp_path / "legacy.json"
    source.write_text(json.dumps(legacy))
    assert load_training_config(source) == expected


@pytest.mark.parametrize("sparse", [False, True])
def test_export_preserves_weights_and_loads_without_training_checkout(tmp_path, sparse):
    config = configuration(sparse=sparse)
    config.sksa_indexer_distill_attention = False
    config.sksa_apply_sparse_attention = True
    model = Qwen3SummaryModel(config).eval()
    checkpoint = tmp_path / "global_step1"
    dcp.save({"app": {"model": model.state_dict()}}, checkpoint_id=checkpoint)
    model_json = tmp_path / "model.json"
    model_json.write_text(config.to_json())
    target = tmp_path / "hf"
    # Exercise the documented CLI defaults as well as the Python entrypoint.
    if sparse:
        result = subprocess.run([
            sys.executable, "-B", str(ROOT / "tools/export_hf.py"),
            "--checkpoint", str(checkpoint),
            "--model-config", str(model_json),
            "--output-dir", str(target),
        ], cwd=tmp_path, capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr
    else:
        export(checkpoint, model_json, ROOT / "hf_template", KERNEL,
               ROOT / "tools/generate.py", target, 1.0)
    kernel = target / "kernel"
    assert (kernel / "ksa_attn/__init__.py").is_file()
    assert not (kernel / "v0.1").exists()
    assert not (kernel / "README.md").exists()
    manifest = json.loads((target / "export_manifest.json").read_text())
    hashes = manifest["kernel_files_sha256"]
    assert set(hashes) == {
        str(path.relative_to(kernel)) for path in kernel.rglob("*") if path.is_file()
    }
    for name, digest in hashes.items():
        assert hashlib.sha256((kernel / name).read_bytes()).hexdigest() == digest
    assert "kernel/v0.1" not in (target / "generate.py").read_text()
    actual = {}
    for path in target.glob("*.safetensors"):
        actual.update(load_file(path))
    expected = load_and_map_dcp(checkpoint, load_training_config(model_json))
    assert actual.keys() == expected.keys()
    for name in actual:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
    assert manifest["runtime"] == "infra"
    assert manifest["model_family"] == ("C-KSA + S-KSA" if sparse else "C-KSA")
    assert str(tmp_path) not in json.dumps(manifest)
    assert len(manifest["runtime_files_sha256"]) == 13
    assert (target / "kernel/deps/flash_attn/LICENSE").is_file()
    assert (target / "LICENSE").is_file()

    # A separate process must use the export's kernels, never Muse or FlashAttn2.
    check = r'''
import importlib.abc
from pathlib import Path
import sys

target = Path(sys.argv[1])
sys.path[:0] = [str(target / "kernel"), str(target / "kernel/deps")]
class NoFlash2(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {"flash_attn_2_cuda", "flash_attn.flash_attn_interface"}:
            raise ImportError("Export must not require FlashAttention 2")
sys.meta_path.insert(0, NoFlash2())
import torch
import pytest
import ksa_attn
import flash_attn.cute.interface
from transformers import AutoModelForCausalLM
torch.set_num_threads(1)
assert not torch.cuda.is_available()
for module in (ksa_attn, flash_attn.cute.interface):
    assert Path(module.__file__).is_relative_to(target)
model, info = AutoModelForCausalLM.from_pretrained(
    target, trust_remote_code=True, local_files_only=True,
    dtype=torch.float32, device_map="cpu", attn_implementation="sdpa",
    output_loading_info=True,
)
for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
    assert not info.get(key), (key, info.get(key))
model.eval()
with torch.inference_mode():
    with pytest.raises(ValueError, match="one nonempty request"):
        model(torch.ones((2, 8), dtype=torch.long))
    with pytest.raises(ValueError, match="inputs_embeds"):
        model(inputs_embeds=torch.zeros((1, 8, model.config.hidden_size)))
    with pytest.raises(ValueError, match="Padded inputs"):
        model(torch.ones((1, 8), dtype=torch.long), attention_mask=torch.zeros((1, 8)))
    for length in (3, 31, 63):
        ids = (torch.arange(length).unsqueeze(0) % 63) + 1
        actual = model(ids, use_cache=False).logits
        assert torch.isfinite(actual).all()
    ids = (torch.arange(31).unsqueeze(0) % 63) + 1
    output = model.generate(ids, max_new_tokens=4, do_sample=False, eos_token_id=None, pad_token_id=0)
    expected = ids
    for _ in range(4):
        next_id = model(expected, use_cache=False).logits[:, -1].argmax(-1, keepdim=True)
        expected = torch.cat((expected, next_id), dim=1)
    assert torch.equal(output, expected)
    cached = model(ids, use_cache=True)
    with pytest.raises(ValueError, match="one new text token"):
        model.prepare_inputs_for_generation(
            torch.cat((ids, ids[:, :2]), dim=1), past_key_values=cached.past_key_values,
        )
assert output.shape == (1, 35)
assert not any(name == "muse" or name.startswith("muse.") for name in sys.modules)
'''
    result = subprocess.run([sys.executable, "-B", "-c", check, str(target)],
                            cwd=tmp_path, capture_output=True, text=True,
                            env=os.environ | {"CUDA_VISIBLE_DEVICES": ""})
    assert result.returncode == 0, result.stdout + result.stderr
