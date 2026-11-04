"""CPU compatibility checks for the original public KSA entry points."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import torch

from muse.config import Qwen3Config, Qwen3SummaryAttentionConfig, load_config
from muse.layers.summary_utils import maybe_build_summary_batch
from muse.models.qwen3.modeling import Qwen3Model
from muse.models.qwen3_sa.modeling import Qwen3SummaryModel

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = Path(__file__).with_name("fixtures") / "v1_forward.json"


def tiny_config(**overrides):
    value = json.loads(FIXTURE.read_text())["config"]
    value.update(overrides)
    return Qwen3SummaryAttentionConfig(**value)


def test_v1_model_state_keys_and_forward_match_public_reference():
    # Golden output from public main 9998565, Torch 2.6.0, without training.
    fixture = json.loads(FIXTURE.read_text())
    torch.manual_seed(fixture["seed"])
    model = Qwen3SummaryModel(tiny_config()).eval()
    assert list(model.state_dict()) == fixture["state_keys"]
    ids = torch.arange(16).unsqueeze(0)
    batch, context = maybe_build_summary_batch({
        "input_ids": ids,
        "loss_mask": torch.ones_like(ids),
        "position_ids": torch.arange(16).unsqueeze(0),
    }, model.config)
    with torch.no_grad():
        actual = model(tokens=batch["input_ids"], input_pos=batch["position_ids"], summary_ctx=context)
    torch.testing.assert_close(actual, torch.tensor(fixture["logits"]), rtol=1e-5, atol=1e-6)


def test_v1_config_keeps_new_features_disabled():
    config = load_config(str(ROOT / "examples/ksa_v1/pretrain/model_config/model_config_1b9_hybrid.json"))
    assert config.summary_meta_pool_mode == "none"
    assert config.summary_meta_pool_group_size == 0
    assert config.sksa_enabled is False
    assert config.summary_partial_chunk_policy == "legacy"
    assert config.get_layer_sliding_chunk_num(0) == 128
    assert config.get_layer_sliding_chunk_num(3) * config.summary_chunk_size >= config.max_seq_len


def test_v1_per_layer_sliding_window_and_rope_config_survive():
    config = tiny_config(
        summary_layer_freq=[1, 0],
        layer_sliding_window="[4, 8]",
        rope_scaling={"rope_type": "default"},
    )
    assert config.rope_scaling == {"rope_type": "default"}
    assert [config.get_layer_sliding_window(i) for i in range(2)] == [4, 8]
    model = Qwen3SummaryModel(config)
    assert [layer.attn.sliding_window for layer in model.model.layers] == [4, 8]
    plain = Qwen3Config(
        model_class="Qwen3Model", vocab_size=16, embed_dim=16,
        num_layers=2, num_heads=2, num_kv_heads=2, head_dim=8,
        intermediate_dim=32, max_seq_len=16, layer_sliding_window=[4, 8],
    )
    assert [layer.attn.sliding_window for layer in Qwen3Model(plain).model.layers] == [4, 8]


@pytest.mark.parametrize("policy,expected", [
    ("legacy", [2, 4]),
    ("summarize", [4, 4]),
    ("defer", []),
])
def test_partial_chunk_policy_does_not_change_v1_tail_positions(policy, expected):
    config = tiny_config(summary_token_num=2, summary_partial_chunk_policy=policy)
    ids = torch.arange(5).unsqueeze(0)
    batch, context = maybe_build_summary_batch({
        "input_ids": ids,
        "position_ids": ids.clone(),
        "loss_mask": torch.ones_like(ids),
    }, config)
    tail_positions = context.samples[0].chunks[-1].summary_positions
    assert batch["position_ids"][0, tail_positions].tolist() == expected


def test_v1_sampler_retains_global_rng_and_legacy_resume():
    from muse.data.datasets.kai_mmap import BlendableWeightedSamplingDataset

    sources = [SimpleNamespace(name=name, size=100, num_sample=12, _chunk_size=8) for name in ("a", "b")]
    dataset = BlendableWeightedSamplingDataset(
        sources, [0.25, 0.75], seed=1024, global_batch_size=4,
        num_workers=1, dp_rank=0, dp_world_size=1,
    )
    assert dataset.seeded_sampling is False
    assert dataset._np_rng is np.random
    rng_state = np.random.get_state()
    try:
        np.random.seed(91)
        state = dataset.state_dict()
        expected = np.random.choice(["a", "b"], size=4, p=[0.25, 0.75])
        dataset.load_state_dict(state)
        actual = dataset._np_rng.choice(["a", "b"], size=4, p=[0.25, 0.75])
        np.testing.assert_array_equal(actual, expected)
        with pytest.raises(ValueError, match="seeded_sampling"):
            dataset.state_dict(consumer_batches=1)
    finally:
        np.random.set_state(rng_state)


def test_exact_v2_state_cannot_be_loaded_with_v1_rng(tmp_path):
    from muse.data.datasets.kai_mmap import KaiMMapDataset
    prefix = tmp_path / "tokens"
    np.arange(33, dtype="<u4").tofile(str(prefix) + ".bin")
    Path(str(prefix) + ".bin.meta").write_text("uint32")
    config = dict(data_prefix=[1.0, str(prefix)], seq_length=8,
                  global_batch_size=1, seeded_sampling=True, skip_warmup=True)
    state = KaiMMapDataset(**config).state_dict(consumer_batches=2)
    legacy = KaiMMapDataset(**dict(config, seeded_sampling=False))
    with pytest.raises(ValueError, match="seeded_sampling"):
        legacy.load_state_dict(state)


def test_v1_checkpoint_helper_keeps_boolean_return_and_noarg_state_dict(monkeypatch, tmp_path):
    from muse.training import checkpoint

    folder = tmp_path / "global_step1"
    folder.mkdir()
    dataset = SimpleNamespace(
        supports_dataloader_resume=True,
        state_dict=lambda: {"cursor": 7},
        load_state_dict=lambda state: None,
    )
    monkeypatch.setattr(checkpoint, "get_checkpoint_path", lambda *args: str(folder))
    monkeypatch.setattr(checkpoint.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(checkpoint.dist, "barrier", lambda: None)
    assert checkpoint.load_dataloader_state(dataset, str(tmp_path), "global_step1") is False
    checkpoint.save_dataloader_state(dataset, str(tmp_path), 1)
    assert checkpoint.load_dataloader_state(dataset, str(tmp_path), "global_step1") is True
    assert checkpoint.load_dataloader_state(dataset, str(tmp_path), "global_step1", return_kind=True) == "legacy_soft"


@pytest.mark.parametrize("return_lse", [False, True])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_v1_and_v2_dispatch_to_shared_kernel(monkeypatch, return_lse, batch_size):
    from muse.layers import summary_attention_func as attention

    called = []
    def shared_kernel(q, k, v, *args, **kwargs):
        called.append(kwargs)
        lse = torch.ones(q.shape[0], q.shape[2], q.shape[1]) if kwargs.get("force_return_lse") else None
        return q.clone(), lse

    def legacy_kernel(*args, **kwargs):
        raise AssertionError("An installed legacy kernel must not shadow ksa_attn")

    monkeypatch.setitem(sys.modules, "summary_attn", SimpleNamespace(summary_attn_func=legacy_kernel))
    monkeypatch.setitem(sys.modules, "ksa_attn", SimpleNamespace(summary_attn_func=shared_kernel))
    monkeypatch.setattr(attention, "_summary_attn_func", None)
    q = torch.randn(batch_size, 5, 2, 8)
    mask = torch.tensor([False, False, False, False, True])
    actual = attention.summary_attention_forward(q, q, q, 4, 1, 1, mask, return_lse=return_lse)
    if return_lse:
        actual, lse = actual
        torch.testing.assert_close(lse, torch.ones(batch_size, 2, 5))
    torch.testing.assert_close(actual, q.reshape(batch_size, 5, 16))
    assert len(called) == batch_size
    for kwargs in called:
        assert list(kwargs) == ["summary_pos"] + (["force_return_lse"] if return_lse else [])


def test_v1_recipe_parser_remains_available():
    spec = importlib.util.spec_from_file_location("v1_recipe", ROOT / "recipes/ksa_v1.py")
    recipe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(recipe)
    args = recipe.get_argument_parser().parse_args([])
    assert not hasattr(args, "train_meta_pool_only")
    assert not hasattr(args, "train_sksa_indexer_only")
