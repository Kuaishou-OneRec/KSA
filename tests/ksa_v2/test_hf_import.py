"""HF initialization loads every backbone tensor without remote model imports."""

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from export_hf import build_export_config, muse_to_hf_name
from muse.models.qwen3_sa.modeling import Qwen3SummaryModel
from muse.training.hf_checkpoint import hf_to_muse, load_ksa_hf_checkpoint, validate_hf_parent
from test_model_cpu import configuration

ROOT = Path(__file__).resolve().parents[2] / "examples/ksa_v2"


def fixture(tmp_path, tied=False, independent=False):
    config = configuration()
    config.summary_meta_pool_mode = "none"
    config.summary_meta_pool_group_size = [0] * 4
    config.tie_word_embeddings = tied
    config.summary_independent_qkv = independent
    model = Qwen3SummaryModel(config)
    state = model.state_dict()
    hf = {muse_to_hf_name(name): value.clone() for name, value in state.items()
          if name not in {"summary_embedding.weight", "model.tok_embeddings.weight"}}
    hf["model.embed_tokens.weight"] = torch.cat((state["model.tok_embeddings.weight"], state["summary_embedding.weight"]))
    if not tied:
        hf["lm_head.weight"] = torch.cat((state["model.output.weight"], torch.zeros(config.summary_token_num, config.embed_dim)))
    config_path = tmp_path / "training.json"
    config_path.write_text(config.to_json())
    metadata = build_export_config(ROOT / "hf_template", config_path)
    (tmp_path / "config.json").write_text(json.dumps(metadata))
    save_file(hf, tmp_path / "model.safetensors")
    return config, state, hf, metadata


@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("independent", [False, True])
def test_hf_roundtrip_is_tensor_exact(tmp_path, tied, independent):
    config, expected, hf, metadata = fixture(tmp_path, tied, independent)
    mapped = hf_to_muse(hf, config)
    assert set(expected) == set(mapped)
    for name in expected:
        torch.testing.assert_close(mapped[name], expected[name], rtol=0, atol=0)
    with torch.device("meta"):
        target = Qwen3SummaryModel(config)
    loaded = load_ksa_hf_checkpoint(tmp_path, target)
    for name in expected:
        torch.testing.assert_close(loaded[name], expected[name], rtol=0, atol=0)


def test_only_new_pooling_parameters_may_be_missing(tmp_path):
    config, expected, _, _ = fixture(tmp_path)
    config.summary_meta_pool_mode = "dimension_wise"
    config.summary_meta_pool_group_size = [2, 2, 2, 0]
    with torch.device("meta"):
        target = Qwen3SummaryModel(config)
    with pytest.raises(ValueError, match="missing="):
        load_ksa_hf_checkpoint(tmp_path, target)
    loaded = load_ksa_hf_checkpoint(tmp_path, target, "meta_pool_scorer.weight,meta_pool_position_bias")
    assert set(loaded) == set(target.state_dict())
    for name in loaded:
        if name in expected:
            torch.testing.assert_close(loaded[name], expected[name], rtol=0, atol=0)
        else:
            assert torch.count_nonzero(loaded[name]) == 0


@pytest.mark.parametrize("name,value", [
    ("summary_chunk_size", 4), ("mix_coeff", 1), ("rope_theta", 500000),
    ("tie_word_embeddings", True), ("summary_token_begin", 63),
    ("summary_sliding_chunk_num", [2, 2, 2, 8]),
])
def test_rejects_semantically_incompatible_parent(tmp_path, name, value):
    config, _, _, hf = fixture(tmp_path)
    hf[name] = value
    with pytest.raises(ValueError, match="incompatible"):
        validate_hf_parent(hf, config)


def test_missing_or_unexpected_backbone_weights_are_errors(tmp_path):
    config, _, hf, _ = fixture(tmp_path)
    with torch.device("meta"):
        target = Qwen3SummaryModel(config)
    hf.pop("model.layers.0.self_attn.k_proj.weight")
    hf["unrecognized.weight"] = torch.zeros(1)
    save_file(hf, tmp_path / "model.safetensors")
    with pytest.raises(ValueError, match="missing=.*k_proj.*unexpected=.*unrecognized"):
        load_ksa_hf_checkpoint(tmp_path, target, "meta_pool_scorer.weight")


def test_duplicate_hf_shards_are_not_silently_merged(tmp_path):
    config, _, hf, _ = fixture(tmp_path)
    save_file(hf, tmp_path / "duplicate.safetensors")
    with torch.device("meta"):
        target = Qwen3SummaryModel(config)
    with pytest.raises(ValueError, match="Duplicate HF tensor"):
        load_ksa_hf_checkpoint(tmp_path, target)
