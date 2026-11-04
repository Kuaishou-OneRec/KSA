"""Training / HF numerical checks; these do not substitute for CUDA tests."""

import copy
import json
from pathlib import Path

import torch
import pytest

from muse.config.model_config import Qwen3SummaryAttentionConfig
from muse.layers.summary_utils import maybe_build_summary_batch
from muse.models.qwen3_sa.modeling import Qwen3SummaryModel

REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = REPO_ROOT / "examples/ksa_v2"
KERNEL_DIR = REPO_ROOT / "summary_attention_kernel/ksa_v2"
torch.set_num_threads(1)


def configuration(sparse=False):
    name = "cksa_g128_sksa_block4_1b9_128k.json" if sparse else "cksa_g128_1b9_128k.json"
    value = json.loads((ROOT / "configs" / name).read_text())
    value.pop("__class__")
    value.update(
        vocab_size=64,
        summary_token_begin=64,
        embed_dim=32,
        num_layers=4,
        num_heads=2,
        num_kv_heads=2,
        head_dim=16,
        intermediate_dim=64,
        max_seq_len=64,
        rope_base=10000,
        attention_function="eager",
        summary_layer_freq=[1, 1, 1, 1],
        summary_sliding_chunk_num=[1, 1, 1, 8],
        summary_attention_mode="block",
        summary_block_size=32,
        summary_meta_pool_group_size=[2, 2, 2, 0],
    )
    if sparse:
        value.update(
            sksa_sparse_full_layers=[3],
            sksa_indexer_query_rank=16,
            sksa_indexer_heads=2,
            sksa_indexer_head_dim=8,
            sksa_indexer_rope_dim=4,
            sksa_indexer_query_block_size=7,
            sksa_indexer_teacher_query_block_size=3,
            sksa_indexer_scorer_backend="torch",
            sksa_indexer_token_budget=16,
            sksa_local_window_tokens=8,
            sksa_indexer_distill_attention=True,
            sksa_apply_sparse_attention=False,
        )
    return Qwen3SummaryAttentionConfig(**value)


def forward(model, ids=None):
    if ids is None:
        torch.manual_seed(100)
        ids = torch.randint(0, 64, (1, 64))
    length = ids.shape[1]
    expanded, context = maybe_build_summary_batch({
        "input_ids": ids,
        "loss_mask": torch.ones_like(ids),
        "position_ids": torch.arange(length).unsqueeze(0),
        "cu_seqlens": torch.tensor([0, length], dtype=torch.int32),
    }, model.config)
    return model(
        tokens=expanded["input_ids"],
        input_pos=expanded["position_ids"],
        cu_seqlens=expanded["cu_seqlens"],
        summary_ctx=context,
    )


def test_dimension_wise_pool_matches_mean_at_initialization_and_updates_only_pool():
    torch.manual_seed(24)
    config = configuration()
    # Before layer 0 all summary embeddings are identical. Test learned pooling
    # after at least one layer has made their hidden states content-dependent.
    config.summary_meta_pool_group_size = [0, 2, 2, 0]
    mean_config = copy.deepcopy(config)
    mean_config.summary_meta_pool_mode = "mean"
    mean = Qwen3SummaryModel(mean_config).eval()
    learned = Qwen3SummaryModel(config).eval()
    missing, unexpected = learned.load_state_dict(mean.state_dict(), strict=False)
    assert not unexpected and missing and all("meta_pool_" in name for name in missing)
    for name, parameter in learned.named_parameters():
        parameter.requires_grad = "meta_pool_" in name
    expected = forward(mean)
    actual = forward(learned)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    actual.square().mean().backward()
    for name, parameter in learned.named_parameters():
        if "meta_pool_" in name:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
            assert parameter.grad.abs().sum() > 0
        else:
            assert parameter.grad is None


@pytest.mark.parametrize("architecture", ["ksa", "cksa", "sksa"])
def test_real_dcp_export_loads_standalone_hf_model_on_cpu(tmp_path, architecture):
    import torch.distributed.checkpoint as dcp
    from transformers import AutoModelForCausalLM
    from export_hf import export

    config = configuration(sparse=architecture == "sksa")
    if architecture == "ksa":
        config.summary_meta_pool_mode = "none"
        config.summary_meta_pool_group_size = [0, 0, 0, 0]
    config.sksa_indexer_distill_attention = False
    config.sksa_apply_sparse_attention = True
    torch.manual_seed(73)
    source = Qwen3SummaryModel(config).eval()
    with torch.no_grad():
        for name, parameter in source.named_parameters():
            if "meta_pool_" in name:
                parameter.normal_(mean=0, std=0.1)
    checkpoint = tmp_path / "global_step1"
    dcp.save({"app": {"model": source.state_dict()}}, checkpoint_id=checkpoint)
    model_json = tmp_path / "model.json"
    model_json.write_text(config.to_json())
    target = tmp_path / "hf"
    export(
        checkpoint, model_json, ROOT / "hf_template",
        KERNEL_DIR / "inference", ROOT / "tools/generate.py", target, 1.0,
    )
    for name in ("modeling_qwen3.py", "summary_context.py", "hierarchical_summary_attention.py", "sksa_attention.py"):
        assert (target / name).is_file()
    loaded = AutoModelForCausalLM.from_pretrained(
        target, trust_remote_code=True, local_files_only=True,
        dtype=torch.float32, device_map="cpu", attn_implementation="sdpa",
    )
    assert loaded.config.sksa_enabled == (architecture == "sksa")
    if architecture == "sksa":
        assert loaded.config.sksa_indexer_teacher_pooling == "max"
    assert loaded.config.summary_meta_pool_group_size == config.summary_meta_pool_group_size
    weights = loaded.state_dict()
    assert all(torch.isfinite(tensor).all() for tensor in weights.values())
    torch.testing.assert_close(
        loaded.get_input_embeddings().weight[:64],
        source.model.tok_embeddings.weight,
        rtol=0, atol=0,
    )
    assert any("sksa_indexer" in key for key in weights) == (architecture == "sksa")
    with torch.inference_mode():
        # Training S-KSA uses complete chunks; HF also supports partial prompts,
        # covered independently by the cached/full-prefill comparisons below.
        lengths = (8, 16, 32, 48, 64) if architecture == "sksa" else (3, 7, 8, 9, 15, 16, 17, 31, 63)
        for length in lengths:
            ids = (torch.arange(length).unsqueeze(0) % 63) + 1
            expected = forward(source, ids)
            actual = loaded(ids, use_cache=False, logits_to_keep=length).logits
            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
        # Cross Summary, local-window eviction and Meta-group boundaries.
        for start in (3, 15, 31):
            ids = (torch.arange(start).unsqueeze(0) % 63) + 1
            cached = loaded(ids, use_cache=True)
            for offset in range(18):
                next_id = torch.tensor([[(start + offset) % 63 + 1]])
                ids = torch.cat((ids, next_id), dim=1)
                model_inputs = loaded.prepare_inputs_for_generation(
                    ids, past_key_values=cached.past_key_values, use_cache=True,
                )
                cached = loaded(**model_inputs)
                expected = loaded(ids, use_cache=False).logits
                torch.testing.assert_close(cached.logits, expected, rtol=2e-5, atol=2e-6)


def test_sksa_max_warmup_preserves_logits_and_isolates_indexer_gradients():
    torch.manual_seed(25)
    baseline = Qwen3SummaryModel(configuration()).eval()
    indexed = Qwen3SummaryModel(configuration(sparse=True)).eval()
    missing, unexpected = indexed.load_state_dict(baseline.state_dict(), strict=False)
    assert not unexpected
    assert len(missing) == 9 and all(".sksa_indexer." in name for name in missing)
    for name, parameter in indexed.named_parameters():
        parameter.requires_grad = ".sksa_indexer." in name
    logits, kl = forward(indexed)
    torch.testing.assert_close(logits, forward(baseline), rtol=0, atol=0)
    assert torch.isfinite(kl) and kl > 0
    kl.backward()
    for name, parameter in indexed.named_parameters():
        if ".sksa_indexer." in name:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
            assert parameter.grad.abs().sum() > 0
        else:
            assert parameter.grad is None
