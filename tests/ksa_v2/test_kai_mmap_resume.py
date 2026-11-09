import importlib.util
from types import SimpleNamespace
from pathlib import Path

import pytest


def _load_kai_mmap_dataset_class():
    module_path = Path(__file__).resolve().parents[2] / "muse/data/datasets/kai_mmap.py"
    spec = importlib.util.spec_from_file_location("_kai_mmap_test_module", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.KaiMMapDataset


def test_kai_mmap_dataset_delegates_stateful_resume_to_underlying_dataset():
    KaiMMapDataset = _load_kai_mmap_dataset_class()

    class DummyStatefulDataset:
        def __init__(self):
            self.loaded = None

        def state_dict(self, consumer_batches=None):
            assert consumer_batches is None
            return {"cursor": 7}

        def load_state_dict(self, state_dict, old_seq_length=None, seq_length=None):
            self.loaded = {
                "state_dict": state_dict,
                "old_seq_length": old_seq_length,
                "seq_length": seq_length,
            }

    dataset = KaiMMapDataset.__new__(KaiMMapDataset)
    dataset.dataset = DummyStatefulDataset()
    dataset.seq_length = 16

    assert dataset.supports_dataloader_resume is True
    assert dataset.state_dict() == {
        "dataset_state": {"cursor": 7},
        "seq_length": 16,
        "state_kind": "legacy_live_state",
        "state_type": "kai_mmap",
    }

    dataset.load_state_dict({
        "dataset_state": {"cursor": 9},
        "seq_length": 8,
        "state_type": "kai_mmap",
    })

    assert dataset.dataset.loaded == {
        "state_dict": {"cursor": 9},
        "old_seq_length": 8,
        "seq_length": 16,
    }


def test_kai_mmap_dataset_rejects_resume_for_stateless_source():
    KaiMMapDataset = _load_kai_mmap_dataset_class()

    dataset = KaiMMapDataset.__new__(KaiMMapDataset)
    dataset.dataset = SimpleNamespace(name="single-source")

    assert dataset.supports_dataloader_resume is False

    with pytest.raises(RuntimeError, match="resume_dataloader"):
        dataset.state_dict()

    with pytest.raises(RuntimeError, match="resume_dataloader"):
        dataset.load_state_dict({"cursor": 3})


def test_kai_mmap_dataset_loads_legacy_state_without_wrapper_metadata():
    KaiMMapDataset = _load_kai_mmap_dataset_class()

    class DummyLegacyDataset:
        def __init__(self):
            self.loaded = None

        def state_dict(self, consumer_batches=None):
            assert consumer_batches is None
            return {"cursor": 1}

        def load_state_dict(self, state_dict, old_seq_length=None, seq_length=None):
            self.loaded = {
                "state_dict": state_dict,
                "old_seq_length": old_seq_length,
                "seq_length": seq_length,
            }

    dataset = KaiMMapDataset.__new__(KaiMMapDataset)
    dataset.dataset = DummyLegacyDataset()
    dataset.seq_length = 16

    dataset.load_state_dict({"cursor": 5})

    assert dataset.dataset.loaded == {
        "state_dict": {"cursor": 5},
        "old_seq_length": None,
        "seq_length": 16,
    }


def test_kai_mmap_dataset_serializes_exact_consumer_boundary():
    KaiMMapDataset = _load_kai_mmap_dataset_class()

    class DummyExactDataset:
        seeded_sampling = True
        def __init__(self):
            self.loaded = None

        def state_dict(self, consumer_batches=None):
            return {"cursor": consumer_batches}

        def load_state_dict(self, state_dict, old_seq_length=None, seq_length=None):
            self.loaded = state_dict

        def _resume_fingerprint(self):
            return "same-config"

    dataset = KaiMMapDataset.__new__(KaiMMapDataset)
    dataset.dataset = DummyExactDataset()
    dataset.seq_length = 16

    state = dataset.state_dict(consumer_batches=9)

    assert state == {
        "state_type": "kai_mmap",
        "state_kind": "consumer_boundary_v1",
        "seq_length": 16,
        "dataset_state": {"cursor": 9},
        "consumer_batches": 9,
        "resume_fingerprint": "same-config",
    }
    dataset.load_state_dict(state)
    assert dataset.dataset.loaded == {"cursor": 9}
    assert dataset.last_resume_kind == "consumer_boundary_v1"


def test_kai_mmap_dataset_rejects_exact_resume_config_mismatch():
    KaiMMapDataset = _load_kai_mmap_dataset_class()

    class DummyChangedDataset:
        seeded_sampling = True
        def state_dict(self, consumer_batches=None):
            return {"cursor": 0}

        def load_state_dict(self, state_dict, old_seq_length=None, seq_length=None):
            raise AssertionError("mismatched state must not load")

        def _resume_fingerprint(self):
            return "changed-config"

    dataset = KaiMMapDataset.__new__(KaiMMapDataset)
    dataset.dataset = DummyChangedDataset()
    dataset.seq_length = 16

    with pytest.raises(ValueError, match="resume fingerprint mismatch"):
        dataset.load_state_dict(
            {
                "state_type": "kai_mmap",
                "state_kind": "consumer_boundary_v1",
                "seq_length": 16,
                "dataset_state": {"cursor": 9},
                "consumer_batches": 9,
                "resume_fingerprint": "original-config",
            }
        )


def test_kai_mmap_dataset_allows_an_explicit_validated_gbs_transition():
    KaiMMapDataset = _load_kai_mmap_dataset_class()

    class DummyChangedBatchDataset:
        seeded_sampling = True
        global_bs = 64

        def __init__(self):
            self.loaded = None

        def state_dict(self, consumer_batches=None):
            return {"cursor": consumer_batches}

        def load_state_dict(self, state_dict, old_seq_length=None, seq_length=None):
            self.loaded = state_dict

        def _resume_fingerprint(self, *, global_batch_size=None):
            batch_size = (
                self.global_bs
                if global_batch_size is None
                else global_batch_size
            )
            return f"same-except-gbs-{batch_size}"

    dataset = KaiMMapDataset.__new__(KaiMMapDataset)
    dataset.dataset = DummyChangedBatchDataset()
    dataset.seq_length = 16
    dataset._resume_source_global_batch_size = 256
    dataset.load_state_dict(
        {
            "state_type": "kai_mmap",
            "state_kind": "consumer_boundary_v1",
            "seq_length": 16,
            "dataset_state": {"cursor": 9},
            "consumer_batches": 9,
            "resume_fingerprint": "same-except-gbs-256",
        }
    )

    assert dataset.dataset.loaded == {"cursor": 9}
    assert dataset.last_resume_kind == "consumer_boundary_v1"
