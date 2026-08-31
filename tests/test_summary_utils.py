from types import SimpleNamespace

import torch

from muse.layers.summary_utils import maybe_build_summary_batch


def _config(**overrides):
    values = {
        "summary_chunk_size": 2,
        "summary_token_num": 2,
        "summary_token_begin": 64,
        "summary_sliding_chunk_num": 1,
        "summary_independent_parameters": False,
        "summary_chunk_position_ids_type": "origin",
        "summary_token_position_ids_type": "last_chunk_slice_right",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_vectorized_summary_batch_preserves_layout_and_metadata():
    input_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    loss_mask = torch.ones_like(input_ids)
    batch = {
        "input_ids": input_ids,
        "loss_mask": loss_mask,
        "position_ids": torch.arange(4, dtype=torch.int32).expand(2, -1).clone(),
        "cu_seqlens": torch.tensor([0, 3, 4], dtype=torch.int32),
    }

    expanded, context = maybe_build_summary_batch(batch, _config())

    assert expanded["input_ids"].tolist() == [
        [1, 2, 64, 65, 3, 4, 64, 65],
        [5, 6, 64, 65, 7, 8, 64, 65],
    ]
    assert expanded["position_ids"].tolist() == [
        [0, 1, 0, 1, 2, 3, 2, 3],
        [0, 1, 0, 1, 2, 3, 2, 3],
    ]
    assert context.summary_mask.tolist() == [
        [False, False, True, True, False, False, True, True],
        [False, False, True, True, False, False, True, True],
    ]
    assert expanded["cu_seqlens"].tolist() == [0, 5, 8]
    assert expanded["loss_mask"] is loss_mask

    for sample in context.samples:
        assert len(sample.chunks) == 2
        assert sample.chunks[0].text_positions.tolist() == [0, 1]
        assert sample.chunks[0].summary_positions.tolist() == [2, 3]
        assert sample.chunks[0].prefix_summary_positions.tolist() == []
        assert sample.chunks[1].text_positions.tolist() == [4, 5]
        assert sample.chunks[1].summary_positions.tolist() == [6, 7]
        assert sample.chunks[1].prefix_summary_positions.tolist() == [2, 3]


def test_summary_prefix_metadata_uses_linear_shared_storage():
    seq_len = 8192
    input_ids = torch.arange(seq_len).unsqueeze(0)
    batch = {
        "input_ids": input_ids,
        "loss_mask": torch.ones_like(input_ids),
        "position_ids": torch.arange(seq_len).unsqueeze(0),
        "cu_seqlens": torch.tensor([0, seq_len], dtype=torch.int32),
    }

    _, context = maybe_build_summary_batch(
        batch,
        _config(
            summary_chunk_size=8,
            summary_token_num=1,
            summary_independent_parameters=True,
        ),
    )

    chunks = context.samples[0].chunks
    assert len(chunks) == 1024
    prefixes = [chunk.prefix_summary_positions for chunk in chunks[1:]]
    assert len({prefix.untyped_storage().data_ptr() for prefix in prefixes}) == 1
    assert prefixes[-1].untyped_storage().nbytes() == 1024 * 8
    assert prefixes[-1].tolist() == [chunk_idx * 9 + 8 for chunk_idx in range(1023)]
