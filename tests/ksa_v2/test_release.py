import ast
import importlib.util
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = REPO_ROOT / "examples/ksa_v2"
KERNEL_DIR = REPO_ROOT / "summary_attention_kernel/ksa_v2"


@pytest.fixture
def dataset(tmp_path):
    prefix = tmp_path / "train"
    np.arange(129, dtype="<u4").tofile(str(prefix) + ".bin")
    Path(str(prefix) + ".bin.meta").write_text("uint32")
    config = tmp_path / "dataset.json"
    config.write_text(json.dumps({
        "name": "KaiMMapDataset", "data_prefix": [1.0, str(prefix)],
        "seq_length": 16, "seed": 1024, "seeded_sampling": True,
        "global_batch_size": 1, "skip_warmup": True, "num_workers": 1,
    }))
    return config


def test_public_configs_are_posttraining_only():
    assert {path.name for path in (ROOT / "configs").glob("*.json")} == {
        "cksa_g128_1b9_128k.json",
        "cksa_g128_sksa_block4_1b9_128k.json",
    }
    assert not (ROOT / "tools/smoke_test.py").exists()


@pytest.mark.parametrize("name", sorted(p.name for p in (ROOT / "configs").glob("*.json")))
def test_config_schema_and_hf_export(name):
    from muse.config import load_config
    from export_hf import build_export_config, load_training_config
    source = ROOT / "configs" / name
    model = load_config(str(source))
    training = load_training_config(source)
    exported = build_export_config(ROOT / "hf_template", source)
    assert exported["num_hidden_layers"] == model.num_layers == 24
    assert exported["hidden_size"] == model.embed_dim == 2048
    assert exported["vocab_size"] == model.vocab_size + 1
    assert exported["summary_chunk_size"] == 8
    assert exported["max_position_embeddings"] == model.max_seq_len
    assert exported["sksa_enabled"] == model.sksa_enabled
    assert exported["max_window_layers"] == model.num_layers
    if not model.sksa_enabled:
        assert not any(key.startswith("sksa_") and key != "sksa_enabled" for key in exported)
    if "sksa" in name:
        assert model.sksa_indexer_teacher_pooling == "max"
        assert model.sksa_indexer_block_size == 4
        assert model.sksa_indexer_summary_sink_tokens == 8
        assert not model.sksa_indexer_chunk_attention_include_summary
        assert exported["sksa_indexer_teacher_pooling"] == training["sksa_indexer_teacher_pooling"]


def test_data_state_real_mmap_roundtrip_and_gbs_change(dataset):
    from muse.data.datasets.kai_mmap import KaiMMapDataset
    config = json.loads(dataset.read_text())
    config.pop("name")
    config.update(global_batch_size=2, world_size=1, rank=0)
    original = KaiMMapDataset(**config)
    saved = original.state_dict(consumer_batches=5)
    resumed = KaiMMapDataset(**config)
    resumed.load_state_dict(saved)
    roundtrip = resumed.state_dict(consumer_batches=0)
    assert saved["state_kind"] == "consumer_boundary_v1"
    assert saved["resume_fingerprint"] == roundtrip["resume_fingerprint"]
    np.testing.assert_equal(saved["dataset_state"], roundtrip["dataset_state"])
    larger = dict(config, global_batch_size=4)
    with pytest.raises(ValueError, match="fingerprint"):
        KaiMMapDataset(**larger).load_state_dict(saved)
    changed = KaiMMapDataset(**larger, resume_source_global_batch_size=2)
    changed.load_state_dict(saved)
    np.testing.assert_equal(saved["dataset_state"], changed.state_dict(consumer_batches=0)["dataset_state"])
    with pytest.raises(ValueError, match="fingerprint"):
        KaiMMapDataset(**dict(config, seq_length=8)).load_state_dict(saved)


def test_public_sources_have_valid_syntax_and_kernel_license():
    sources = [REPO_ROOT / "muse", KERNEL_DIR]
    sources += [ROOT / name for name in ("tools", "configs", "hf_template", "scripts")]
    files = [REPO_ROOT / "recipes/ksa_v2.py", ROOT / "README.md", ROOT / "requirements.txt"]
    for directory in sources:
        files.extend(path for path in directory.rglob("*")
                     if path.suffix in {".py", ".json", ".jsonl", ".md", ".sh", ".toml", ".txt"})
    for path in files:
        assert not path.is_symlink()
        raw = path.read_bytes()
        for forbidden in (b"/mmu_nlp_hdd/", b"git.corp.", b"oversea-squid", b"/home/hadoop/"):
            assert forbidden not in raw, path
        if path.suffix == ".py":
            ast.parse(raw, filename=str(path))
    wheels = list((KERNEL_DIR / "v0.1/wheels").glob("*.whl"))
    assert len(wheels) == 1
    with zipfile.ZipFile(wheels[0]) as wheel:
        assert any(name.endswith("/LICENSE") for name in wheel.namelist())
        assert not any(name.endswith((".so", ".dll")) for name in wheel.namelist())


def test_hf_template_defaults_match_the_final_model():
    from export_hf import build_export_config
    template = json.loads((ROOT / "hf_template/config.json").read_text())
    expected = build_export_config(ROOT / "hf_template", ROOT / "configs/cksa_g128_sksa_block4_1b9_128k.json")
    assert template == expected
    assert template["num_hidden_layers"] == 24
    assert template["hidden_size"] == 2048
    assert template["max_position_embeddings"] == 131072
    assert template["sksa_indexer_teacher_pooling"] == "max"


def test_smoke_cpu_preparation(tmp_path):
    from muse.config import load_config
    spec = importlib.util.spec_from_file_location("ksa_v2_smoke", Path(__file__).with_name("smoke_test.py"))
    smoke = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(smoke)
    directory = tmp_path / "smoke"
    plan = smoke.prepare(directory)
    assert plan["status"] == "prepared_not_run"
    assert [item["segment"] for item in plan["commands"]] == [
        "pretrain", "resume", "compress", "full_cpt", "indexer", "indexer_resume",
    ]
    assert plan["commands"][-1]["checkpoint"] == str(directory / "indexer_resume/global_step8")
    for name in ("ksa", "cksa", "sksa"):
        config = load_config(str(directory / "fixtures" / (name + ".json")))
        assert config.max_seq_len == 2048
        assert config.num_layers == 4
        if name == "ksa":
            assert config.summary_meta_pool_mode == "none"
            assert config.summary_meta_pool_group_size == 0
            assert config.summary_meta_pool_mass_correction == "none"
    assert not (directory / "pretrain").exists()
