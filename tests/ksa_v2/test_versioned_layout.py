"""Versioned entry points, V1 export configuration and shell safety."""

import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import pytest
import torch
import torch.distributed.checkpoint as dcp
from safetensors.torch import load_file

from muse.models.qwen3_sa.modeling import Qwen3SummaryModel
from test_v1_compatibility import tiny_config

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT / "examples/ksa_v1/pretrain"


@pytest.mark.parametrize("context", [8, 32, 64, 128])
def test_versioned_shells_have_expected_dry_run(context, tmp_path):
    env = os.environ | {"CHECKPOINT_DIR": str(tmp_path / "parent"), "OUTPUT_DIR": str(tmp_path / "run")}
    canonical = V1 / f"run_pretrain_{context}k.sh"
    plan = subprocess.check_output(
        ["bash", str(canonical), "--dry-run"], env=env, cwd=tmp_path, text=True,
    )
    args = shlex.split(plan)
    assert args[:3] == ["python", "-m", "torch.distributed.run"]
    assert "recipes/ksa_v1.py" in args
    assert "--resume-optimizer" not in args
    assert not (tmp_path / "run").exists()


def test_legacy_pretrain_directory_removed():
    assert not (ROOT / "examples/pretrain").exists()


def test_set_env_only_exports_without_printing_or_installing(tmp_path):
    script = tmp_path / "set_env.sh"
    script.write_text((ROOT / "set_env.sh").read_text())
    (tmp_path / ".env").write_text("KSA_TEST_SECRET=not_for_logs\n")
    checked = subprocess.run(
        ["bash", "-c", 'source "$1"; test "$KSA_TEST_SECRET" = not_for_logs', "test", str(script)],
        capture_output=True, text=True, check=True,
    )
    assert not checked.stdout and not checked.stderr
    assert not (ROOT / "killall.sh").exists()
    assert not (ROOT / "killall.py").exists()


def test_shell_scripts_parse():
    for path in [ROOT / "set_env.sh", *ROOT.glob("examples/**/*.sh")]:
        if "output" not in path.relative_to(ROOT).parts:
            subprocess.run(["bash", "-n", str(path)], check=True)


def test_v1_converter_uses_training_config_and_shared_tokenizer(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(V1 / "convert"))
    spec = importlib.util.spec_from_file_location("v1_convert", V1 / "convert/convert.py")
    converter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(converter)
    template = tmp_path / "template"
    shutil.copytree(V1 / "hf_template", template)
    chat_template = "{% for message in messages %}{{ message['content'] }}{% endfor %}"
    (template / "chat_template.jinja").write_text(chat_template)
    config = tiny_config(tie_word_embeddings=False)
    source = Qwen3SummaryModel(config).eval()
    checkpoint = tmp_path / "global_step1"
    dcp.save({"app": {"model": source.state_dict()}}, checkpoint_id=checkpoint)
    (tmp_path / "model_config.json").write_text(config.to_json())
    output = Path(converter.convert_dcp_to_hf(tmp_path, "global_step1", template, remap="muse2hf"))
    assert (output / "chat_template.jinja").read_text() == chat_template
    exported = json.loads((output / "config.json").read_text())
    assert exported["hidden_size"] == config.embed_dim
    assert exported["num_hidden_layers"] == config.num_layers
    assert exported["max_position_embeddings"] == config.max_seq_len
    assert exported["summary_sliding_chunk_num"] == config.summary_sliding_chunk_num
    assert exported["tie_word_embeddings"] is False
    assert exported["vocab_size"] == config.vocab_size + config.summary_token_num
    for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
        assert (output / name).read_bytes() == (ROOT / "assets/tokenizer" / name).read_bytes()
    tensors = {}
    for path in output.glob("*.safetensors"):
        tensors.update(load_file(path))
    torch.testing.assert_close(tensors["lm_head.weight"][:config.vocab_size],
                               source.model.output.weight.to(torch.bfloat16), rtol=0, atol=0)
    with pytest.raises(FileExistsError):
        converter.convert_dcp_to_hf(tmp_path, "global_step1", V1 / "hf_template", remap="muse2hf")


def test_documentation_local_links_exist():
    import re
    files = [ROOT / "README.md", ROOT / "README_zh.md"]
    files += list((ROOT / "docs").glob("*.md"))
    files += list((ROOT / "examples/ksa_v1").rglob("*.md"))
    files += [ROOT / "examples/ksa_v2/README.md"]
    for path in files:
        content = path.read_text()
        links = re.findall(r"\]\(([^)]+)\)", content)
        links += re.findall(r'''(?:src|href)=["']([^"']+)["']''', content)
        for match in links:
            target = match.split("#", 1)[0]
            if not target or "://" in target:
                continue
            assert (path.parent / target).exists(), (path.relative_to(ROOT), target)
