"""Read V1's demo directly, including EOF boundaries and exact cursor resume."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from muse.data.datasets.kai_mmap import KaiMMapDataset, MMapDataset

REPO = Path(__file__).resolve().parents[2]


def test_demo_is_read_without_copy_or_padding(monkeypatch):
    monkeypatch.chdir(REPO)
    config = json.loads((REPO / "examples/ksa_v2/dataset_config/demo_128k.json").read_text())
    source = REPO / (config["data_prefix"][1] + ".bin")
    before = hashlib.sha256(source.read_bytes()).digest()
    data = MMapDataset(config["data_prefix"][1], 131072, 1, 0, 1, skip_warmup=True)
    assert data.size == 1048576
    assert data.num_sample == 7
    raw = np.fromfile(source, dtype="<u4")
    for i in range(data.num_sample):
        np.testing.assert_array_equal(data.get(i), raw[i * 131072:(i + 1) * 131072 + 1])
    with pytest.raises(IndexError):
        data.get(7)
    assert hashlib.sha256(source.read_bytes()).digest() == before


@pytest.mark.parametrize("tokens,count", [(16, 0), (17, 1), (32, 1), (33, 2), (35, 2)])
def test_real_lookahead_determines_number_of_samples(tmp_path, tokens, count):
    prefix = tmp_path / "source"
    np.arange(tokens, dtype="<u4").tofile(str(prefix) + ".bin")
    Path(str(prefix) + ".bin.meta").write_text("uint32")
    if not count:
        with pytest.raises(ValueError, match="no complete training sample"):
            MMapDataset(str(prefix), 16, 1, 0, 1, skip_warmup=True)
        return
    dataset = MMapDataset(str(prefix), 16, 1, 0, 1, skip_warmup=True)
    assert dataset.num_sample == count
    for i in range(count):
        assert dataset.get(i).tolist() == list(range(i * 16, i * 16 + 17))
    with pytest.raises(IndexError):
        dataset.get(count)


def test_qa_mask_samples_do_not_require_lookahead(tmp_path):
    prefix = tmp_path / "qa"
    np.arange(32, dtype="<u4").tofile(str(prefix) + ".bin")
    np.ones(32, dtype="<u4").tofile(str(prefix) + "_loss_mask.bin")
    Path(str(prefix) + ".bin.meta").write_text("uint32")
    dataset = MMapDataset(str(prefix), 16, 1, 0, 1, skip_warmup=True)
    assert dataset.num_sample == 2
    text, mask = dataset.get(1)
    assert text.tolist() == list(range(16, 32))
    assert mask.tolist() == [1] * 16


@pytest.mark.parametrize("rank", range(4))
def test_demo_cursor_resume_across_rollover(monkeypatch, rank):
    monkeypatch.chdir(REPO)
    config = json.loads((REPO / "examples/ksa_v2/dataset_config/demo_128k.json").read_text())
    config.pop("name")
    config.update(global_batch_size=8, world_size=4, rank=rank)
    source = KaiMMapDataset(**config)
    stream = iter(source)
    for _ in range(6):
        next(stream)
    saved = source.state_dict(consumer_batches=3)
    expected = [next(stream)["input_ids"].numpy() for _ in range(6)]
    restored = KaiMMapDataset(**config)
    restored.load_state_dict(saved)
    actual = iter(restored)
    for value in expected:
        np.testing.assert_array_equal(next(actual)["input_ids"].numpy(), value)
