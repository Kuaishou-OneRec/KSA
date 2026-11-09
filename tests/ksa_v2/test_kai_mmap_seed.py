from dataclasses import dataclass

import numpy as np
from torch.utils.data import DataLoader

from muse.data.datasets.kai_mmap import BlendableWeightedSamplingDataset


@dataclass
class _FakeDataset:
    name: str
    size: int = 10_000
    num_sample: int = 1_000
    _chunk_size: int = 8

    def __getitem__(self, index):
        return {"source": self.name, "index": index}


def _make_dataset(seed):
    sources = [_FakeDataset("a"), _FakeDataset("b"), _FakeDataset("c")]
    return BlendableWeightedSamplingDataset(
        sources,
        [0.2, 0.3, 0.5],
        seed=seed,
        seeded_sampling=True,
        global_batch_size=64,
        num_workers=1,
        dp_rank=0,
        dp_world_size=1,
    )


def _make_worker_dataset(seed, weights=(0.2, 0.3, 0.5)):
    sources = [_FakeDataset("a"), _FakeDataset("b"), _FakeDataset("c")]
    return BlendableWeightedSamplingDataset(
        sources,
        list(weights),
        seed=seed,
        seeded_sampling=True,
        global_batch_size=4,
        num_workers=1,
        dp_rank=0,
        dp_world_size=4,
    )


def _identity(value):
    return value


def _local_state(dataset):
    return {
        0: {
            "consume_sample_dict": dict(dataset.consume_sample_dict),
            "consume_token_dict": dict(dataset.consume_token_dict),
            "past_epoch_dict": dict(dataset.past_epoch_dict),
        }
    }


def test_kai_blend_seed_is_reproducible_and_effective():
    first = _make_dataset(1024)
    same = _make_dataset(1024)
    different = _make_dataset(2048)

    first_names, _ = first._sampling_index(_local_state(first), 0)
    same_names, _ = same._sampling_index(_local_state(same), 0)
    different_names, _ = different._sampling_index(_local_state(different), 0)

    np.testing.assert_array_equal(first_names, same_names)
    assert not np.array_equal(first_names, different_names)


def test_kai_blend_rng_state_round_trip():
    source = _make_dataset(7)
    source_local = _local_state(source)
    source._sampling_index(source_local, 0)
    saved = source.state_dict()

    restored = _make_dataset(999)
    restored.load_state_dict(saved)
    expected_names, _ = source._sampling_index(source_local, 0)
    restored_names, _ = restored._sampling_index(_local_state(restored), 0)
    np.testing.assert_array_equal(restored_names, expected_names)


def test_consumer_boundary_excludes_prefetch_frontier():
    dataset = _make_worker_dataset(1024)
    state = dataset.state_dict(consumer_batches=7)
    sources = [value for key, value in state.items() if key != "np_rng_state"]

    assert sum(value["c_sample"] for value in sources) == 7 * 4
    assert not np.array_equal(
        state["np_rng_state"][1],
        np.random.RandomState(1024).get_state()[1],
    )


def test_consumer_boundary_advances_from_loaded_resume_cursor():
    initial = _make_worker_dataset(1024)
    resume_state = initial.state_dict(consumer_batches=7)

    resumed = _make_worker_dataset(1024)
    resumed.load_state_dict(resume_state)
    continued_state = resumed.state_dict(consumer_batches=5)

    uninterrupted_state = initial.state_dict(consumer_batches=12)
    for name in ("a", "b", "c"):
        assert continued_state[name] == uninterrupted_state[name]
    expected_rng = uninterrupted_state["np_rng_state"]
    actual_rng = continued_state["np_rng_state"]
    assert expected_rng[0] == actual_rng[0]
    np.testing.assert_array_equal(expected_rng[1], actual_rng[1])
    assert expected_rng[2:] == actual_rng[2:]


def test_worker_prefetch_resume_matches_uninterrupted_stream():
    consumed_batches = 7
    compare_batches = 8
    source = _make_worker_dataset(1024)
    loader = DataLoader(
        source,
        batch_size=None,
        num_workers=1,
        prefetch_factor=2,
        collate_fn=_identity,
    )
    iterator = iter(loader)
    try:
        for _ in range(consumed_batches):
            next(iterator)
        state = source.state_dict(consumer_batches=consumed_batches)
        expected = [next(iterator) for _ in range(compare_batches)]
    finally:
        iterator._shutdown_workers()

    restored = _make_worker_dataset(1024)
    restored.load_state_dict(state)
    restored_loader = DataLoader(
        restored,
        batch_size=None,
        num_workers=1,
        prefetch_factor=2,
        collate_fn=_identity,
    )
    restored_iterator = iter(restored_loader)
    try:
        actual = [next(restored_iterator) for _ in range(compare_batches)]
    finally:
        restored_iterator._shutdown_workers()

    assert actual == expected


def test_resume_fingerprint_changes_with_sampling_config():
    original = _make_worker_dataset(1024)
    changed_seed = _make_worker_dataset(2048)
    changed_weights = _make_worker_dataset(1024, weights=(0.1, 0.3, 0.6))

    assert original._resume_fingerprint() != changed_seed._resume_fingerprint()
    assert original._resume_fingerprint() != changed_weights._resume_fingerprint()
