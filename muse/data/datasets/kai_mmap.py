"""
Kai MMap Dataset for Muse.

Ported from kai-megatron's dataset pipeline with minimal modifications.
Only Megatron-specific imports (get_args, mpu, print_rank_0) are replaced;
all dataset logic is preserved as-is for correctness.

Original sources:
  - megatron/core/datasets/iterable_dataset.py  -> MMapDataset, make_dataset
  - megatron/core/datasets/blendable_dataset.py -> BlendableWeightedSamplingDataset
  - megatron/core/datasets/klear_dataset.py     -> build / parse functions
"""

import os
import time
import hashlib
import json
from functools import lru_cache
from itertools import accumulate
from collections import OrderedDict
import multiprocessing
import itertools

import numpy as np
import torch

from muse.utils.common import print_rank_0


# ============================================================================
# Ported from: megatron/core/datasets/iterable_dataset.py
# Changes: removed get_args(), print_rank_0 source changed, removed mpu import
# ============================================================================

def get_available_dataset_impl():
    return ['lazy', 'cached', 'mmap']


def data_file_path(prefix_path):
    return prefix_path + '.bin'


def loss_mask_file_path(prefix_path):
    return prefix_path + '_loss_mask.bin'


def meta_file_path(prefix_path):
    return prefix_path + '.bin.meta'


def _warmup_mmap_file(path):
    with open(path, 'rb') as stream:
        while stream.read(100 * 1024 * 1024):
            pass


class MMapDataset(torch.utils.data.Dataset):
    """Memory-mapped dataset that reads pre-tokenized binary files.

    Ported from kai-megatron iterable_dataset.py with Megatron imports removed.
    """

    def __init__(self, path, chunk_size, batch_size_per_iter,
                 data_parallel_rank, data_parallel_size, skip_warmup=False):
        super().__init__()

        with open(meta_file_path(path)) as r:
            dtype = r.read()
        self._dtype = getattr(np, dtype)
        self._path = path
        self.name = path

        self._chunk_size = chunk_size
        self._v_chunk_size = chunk_size + 1

        self._itemsize = self._dtype().itemsize
        self._chunk_bytes = self._chunk_size * self._itemsize
        self._v_chunk_bytes = self._v_chunk_size * self._itemsize
        self._total_bytes = os.path.getsize(data_file_path(path))
        assert self._total_bytes % self._itemsize == 0
        if self.num_sample == 0:
            raise ValueError(f"Dataset {path} has no complete training sample")

        self._bin_buffer = None
        self._do_init(skip_warmup)

    def __getstate__(self):
        return {
            'path': self._path,
            'chunk_size': self._chunk_size,
            'dtype_name': self._dtype.__name__,
            'total_bytes': self._total_bytes,
        }

    def __setstate__(self, state):
        self._path = state['path']
        self.name = self._path
        self._chunk_size = state['chunk_size']
        self._v_chunk_size = self._chunk_size + 1
        self._dtype = getattr(np, state['dtype_name'])
        self._itemsize = self._dtype().itemsize
        self._chunk_bytes = self._chunk_size * self._itemsize
        self._v_chunk_bytes = self._v_chunk_size * self._itemsize
        self._total_bytes = state['total_bytes']
        self._bin_buffer = None
        self._do_init(skip_warmup=True)

    def __del__(self):
        if hasattr(self, '_bin_buffer_mmap') and self._bin_buffer_mmap is not None:
            try:
                self._bin_buffer_mmap._mmap.close()
            except Exception:
                pass
            del self._bin_buffer_mmap

    def _do_init(self, skip_warmup):
        if not skip_warmup:
            print_rank_0("    warming up data mmap file...")
            _warmup_mmap_file(data_file_path(self._path))
        print_rank_0("    creating numpy buffer of mmap...")
        self._bin_buffer_mmap = np.memmap(data_file_path(self._path), mode='r', order='C')
        print_rank_0("    creating memory view of numpy buffer...")
        self._bin_buffer = memoryview(self._bin_buffer_mmap)
        if self.is_qa_dataset:
            self._loss_mask_bin_buffer_mmap = np.memmap(loss_mask_file_path(self._path), mode='r', order='C')
            self._loss_mask_bin_buffer = memoryview(self._loss_mask_bin_buffer_mmap)

    @property
    def size(self):
        return self._total_bytes // self._itemsize

    @property
    def num_sample(self):
        # Causal samples need one real look-ahead token. Do not read past EOF
        # or invent a target when the file ends exactly on a chunk boundary.
        lookahead = 0 if self.is_qa_dataset else 1
        return max(0, (self.size - lookahead) // self._chunk_size)

    def get(self, s_idx):
        if not 0 <= s_idx < self.num_sample:
            raise IndexError(f"Sample {s_idx} outside [0, {self.num_sample})")
        if self.is_qa_dataset:
            np_array = np.frombuffer(self._bin_buffer, dtype=self._dtype,
                                     count=self._chunk_size, offset=s_idx * self._chunk_bytes)
            loss_mask = np.frombuffer(self._loss_mask_bin_buffer, dtype=self._dtype,
                                      count=self._chunk_size, offset=s_idx * self._chunk_bytes)
            return np_array, loss_mask
        else:
            np_array = np.frombuffer(self._bin_buffer, dtype=self._dtype,
                                     count=self._v_chunk_size, offset=s_idx * self._chunk_bytes)
            return np_array

    def __getitem__(self, s_idx):
        sample = self.get(s_idx)
        if self.is_qa_dataset:
            sample = np.concatenate(sample, axis=-1, dtype=np.int64)
            return {'text': sample}
        else:
            sample = np.array(sample, dtype=np.int64)
            return {'text': sample}

    @property
    def is_qa_dataset(self):
        return os.path.exists(loss_mask_file_path(self._path))

    @staticmethod
    def exists(path):
        return (
                os.path.exists(meta_file_path(path)) and os.path.exists(data_file_path(path))
        )


def make_dataset(path, impl, chunk_size, batch_size_per_iter, data_parallel_rank, data_parallel_size,
                 skip_warmup=False):
    dataset = None
    if not MMapDataset.exists(path):
        print(f"Dataset does not exist: {path}")
        print("Path should be a basename that both .meta and .bin can be appended to get full filenames.")
    if impl == 'mmap':
        dataset = MMapDataset(path, chunk_size, batch_size_per_iter, data_parallel_rank, data_parallel_size, skip_warmup)

    return dataset


# ============================================================================
# Ported from: megatron/core/datasets/blendable_dataset.py
# Changes: get_args() -> constructor params, mpu.* -> constructor params
# ============================================================================

class BlendableWeightedSamplingDataset(torch.utils.data.IterableDataset):
    """Weighted sampling across multiple MMapDatasets.

    Ported from kai-megatron blendable_dataset.py with Megatron imports removed.
    All constructor params that were previously read from get_args() or mpu are now
    passed explicitly.
    """

    def __init__(self, datasets, weights, *,
                 seed, global_batch_size, num_workers,
                 dp_rank, dp_world_size, seeded_sampling=False):
        self.seed = seed
        self.seeded_sampling = seeded_sampling
        # V1 uses NumPy's worker-global RNG; V2 opts into an isolated stream.
        self._sampling_rng = np.random.RandomState(seed) if seeded_sampling else None
        self.global_bs = global_batch_size

        num_datasets = len(datasets)
        assert num_datasets == len(weights)
        self._num_datasets = num_datasets
        self._datasets = {dataset.name: dataset for dataset in datasets}
        self.sizes = [dataset.size for dataset in self._datasets.values()]
        self.dataset_name_to_idx = {name: idx for idx, name in enumerate(self._datasets)}

        self._dp_rank = dp_rank
        self._dp_world_size = dp_world_size
        self._num_workers = num_workers if num_workers >= 1 else 1
        assert self.global_bs % (self._num_workers * self._dp_world_size) == 0

        manager = multiprocessing.Manager()
        self.consume_sample_dict = manager.dict()
        self.consume_token_dict = manager.dict()
        self.past_epoch_dict = manager.dict()
        for dataset in datasets:
            self.consume_sample_dict[dataset.name] = 0
            self.past_epoch_dict[dataset.name] = 0
            self.consume_token_dict[dataset.name] = 0
        # Exact checkpoint boundaries are reconstructed relative to this
        # cursor. It is None for a fresh run and replaced with the loaded
        # cursor when continuing from a dataloader checkpoint.
        self._consumer_boundary_base_state = None
        self._initial_source_state_sha256 = None

        weights = np.array(weights, dtype=np.float64)
        sum_weights = np.sum(weights)
        assert sum_weights > 0.0
        weights /= sum_weights
        self.weights = weights

        print_rank_0('> total number of tokens of blendable dataset: '
                     '{} tokens'.format(sum(self.sizes)))

    @property
    def size(self):
        return sum(self.sizes)

    @property
    def _np_rng(self):
        # Resolve the worker's global RNG lazily; storing the module itself
        # would make legacy datasets unpicklable with spawned workers.
        return self._sampling_rng if self.seeded_sampling else np.random

    def _resume_fingerprint(self, *, global_batch_size=None):
        fingerprint_global_bs = (
            self.global_bs
            if global_batch_size is None
            else int(global_batch_size)
        )
        payload = {
            "seed": self.seed,
            "global_batch_size": fingerprint_global_bs,
            "num_workers": self._num_workers,
            "dp_world_size": self._dp_world_size,
            "sources": [
                {
                    "name": name,
                    "weight": float(weight),
                    "size": int(dataset.size),
                    "num_sample": int(dataset.num_sample),
                    "chunk_size": int(dataset._chunk_size),
                }
                for (name, dataset), weight in zip(
                    self._datasets.items(), self.weights
                )
            ],
        }
        if self._initial_source_state_sha256 is not None:
            payload["initial_source_state_sha256"] = self._initial_source_state_sha256
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(encoded).hexdigest()

    def _state_at_consumer_boundary(self, consumer_batches):
        """Reconstruct sampling state before worker prefetch for a batch count."""
        if not self.seeded_sampling:
            raise ValueError("Exact consumer boundaries require seeded_sampling=True")
        if isinstance(consumer_batches, bool) or not isinstance(
            consumer_batches, (int, np.integer)
        ):
            raise TypeError("consumer_batches must be an integer")
        consumer_batches = int(consumer_batches)
        if consumer_batches < 0:
            raise ValueError("consumer_batches must be >= 0")

        base_state = self._consumer_boundary_base_state
        rng = np.random.RandomState(self.seed)
        if base_state is not None:
            rng.set_state(base_state["np_rng_state"])
        names = list(self._datasets)
        selection_counts = np.zeros(len(names), dtype=np.int64)
        remaining = consumer_batches * self.global_bs
        # Bound temporary memory while preserving RandomState.choice order.
        while remaining:
            draw_count = min(remaining, 1_000_000)
            selected = rng.choice(len(names), size=draw_count, p=self.weights)
            selection_counts += np.bincount(selected, minlength=len(names))
            remaining -= draw_count

        saved_state_dict = OrderedDict()
        for (name, dataset), count in zip(self._datasets.items(), selection_counts):
            selections = int(count)
            # Preserve the ported sampler's exact rollover semantics. Its
            # cursor cycles through 1..num_sample, then 0 on the rollover draw.
            cycle = dataset.num_sample + 1
            if base_state is None or name not in base_state:
                base_past_epoch = 0
                base_c_sample = 0
                base_c_token = 0
            else:
                base_past_epoch = int(base_state[name]["past_epoch"])
                base_c_sample = int(base_state[name]["c_sample"])
                base_c_token = int(base_state[name]["c_token"])
            advanced_cursor = base_c_sample + selections
            # Context extension can leave a zero-weight short source's cursor
            # beyond its new capacity. The iterator only rolls over on equality.
            if base_c_sample > dataset.num_sample:
                next_sample, next_epoch = advanced_cursor, base_past_epoch
            else:
                next_sample = advanced_cursor % cycle
                next_epoch = base_past_epoch + advanced_cursor // cycle
            saved_state_dict[name] = {
                "past_epoch": next_epoch,
                "c_sample": next_sample,
                "c_token": base_c_token + selections * dataset._chunk_size,
                "num_tokens": dataset.size,
            }
        saved_state_dict["np_rng_state"] = rng.get_state()
        return saved_state_dict

    def state_dict(self, consumer_batches=None):
        if consumer_batches is not None:
            return self._state_at_consumer_boundary(consumer_batches)
        saved_state_dict = OrderedDict()
        for name, dataset in self._datasets.items():
            saved_state_dict[name] = {}
            saved_state_dict[name]['past_epoch'] = self.past_epoch_dict[name]
            saved_state_dict[name]['c_sample'] = self.consume_sample_dict[name]
            saved_state_dict[name]['c_token'] = self.consume_token_dict[name]
            saved_state_dict[name]['num_tokens'] = dataset.size
            saved_state_dict['np_rng_state'] = self._np_rng.get_state()
        return saved_state_dict

    def load_state_dict(self, state_dict, old_seq_length=None, seq_length=None):
        state = state_dict['np_rng_state']
        self._np_rng.set_state(state)
        for name, dataset in self._datasets.items():
            if name in state_dict:
                old_c_sample = state_dict[name]['c_sample']
                if old_seq_length is not None and seq_length is not None:
                    new_c_sample = (old_c_sample * old_seq_length + seq_length - 1) // seq_length
                else:
                    new_c_sample = old_c_sample
                self.past_epoch_dict[name] = state_dict[name]['past_epoch']
                self.consume_sample_dict[name] = new_c_sample
                self.consume_token_dict[name] = state_dict[name]['c_token']
            else:
                print_rank_0(f"Appending new Dataset {name} with {dataset.size} tokens")
        # Future exact boundaries advance from the cursor that was actually
        # loaded, instead of restarting the sampling stream from seed.
        self._consumer_boundary_base_state = self.state_dict()

    def _sampling_index(self, local_dicts, worker_id):

        dataset_name = self._np_rng.choice(
            list(self._datasets.keys()), size=self.global_bs, p=self.weights
        )
        sample_index = [0] * self.global_bs
        local_consume_sample_dict = local_dicts[worker_id]['consume_sample_dict']
        local_consume_token_dict = local_dicts[worker_id]['consume_token_dict']
        local_past_epoch_dict = local_dicts[worker_id]['past_epoch_dict']

        for i, d_name in enumerate(dataset_name):
            if local_consume_sample_dict[d_name] == self._datasets[d_name].num_sample:
                sample_index[i] = 0
                local_consume_sample_dict[d_name] = 0
                local_past_epoch_dict[d_name] += 1
            else:
                sample_index[i] = local_consume_sample_dict[d_name]
                local_consume_sample_dict[d_name] += 1
            local_consume_token_dict[d_name] += self._datasets[d_name]._chunk_size

        if worker_id == (self._num_workers - 1):
            self.consume_sample_dict.update(local_consume_sample_dict)
            self.consume_token_dict.update(local_consume_token_dict)
            self.past_epoch_dict.update(local_past_epoch_dict)

        return dataset_name, sample_index

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id
        local_dicts = {}
        local_dicts[worker_id] = {
            'consume_sample_dict': dict(self.consume_sample_dict),
            'consume_token_dict': dict(self.consume_token_dict),
            'past_epoch_dict': dict(self.past_epoch_dict)
        }

        for _ in itertools.count():
            dataset_name, sample_index = self._sampling_index(local_dicts, worker_id)
            dp_offset = self._dp_rank
            dw_offset = dp_offset + worker_id * self._dp_world_size
            while True:
                try:
                    name, s_idx = dataset_name[dw_offset], sample_index[dw_offset]
                    yield self._datasets[name][s_idx], name
                    dw_offset += self._num_workers * self._dp_world_size
                except Exception:
                    break


# ============================================================================
# Ported from: megatron/core/datasets/klear_dataset.py
# Changes: mpu.* -> params, print_rank_0 source changed, local imports
# ============================================================================

def get_datasets_weights(data_prefix):
    """Parse weight-prefix pairs from a flat list.

    The data prefix should be in the format of:
      weight-1, data-prefix-1, weight-2, data-prefix-2, ..
    """
    assert len(data_prefix) % 2 == 0
    num_datasets = len(data_prefix) // 2
    weights = [0] * num_datasets
    prefixes = [0] * num_datasets
    for i in range(num_datasets):
        weights[i] = float(data_prefix[2 * i])
        prefixes[i] = (data_prefix[2 * i + 1]).strip()
    weight_sum = 0.0
    for weight in weights:
        weight_sum += weight
    assert weight_sum > 0.0
    weights = [weight / weight_sum for weight in weights]

    return prefixes, weights


def build_train_valid_test_datasets(data_impl,
                                    seq_length,
                                    micro_batch_size,
                                    skip_warmup,
                                    train_data_prefix=None,
                                    valid_data_prefix=None,
                                    test_data_prefix=None,
                                    dp_rank=0,
                                    dp_world_size=1,
                                    seed=1024,
                                    global_batch_size=1,
                                    num_workers=1,
                                    seeded_sampling=False):
    """Build train, valid, and test datasets."""

    print_rank_0("Separate data paths provided for train, valid & test. Split string will be ignored.")

    train_dataset, valid_dataset, test_dataset = None, None, None
    if train_data_prefix is not None:
        train_dataset = build_dataset("train", train_data_prefix, data_impl,
                                      seq_length, micro_batch_size, skip_warmup,
                                      dp_rank=dp_rank, dp_world_size=dp_world_size,
                                      seed=seed, global_batch_size=global_batch_size,
                                      num_workers=num_workers,
                                      seeded_sampling=seeded_sampling)

    if valid_data_prefix is not None:
        valid_dataset = build_dataset("valid", valid_data_prefix, data_impl,
                                      seq_length, micro_batch_size, False,
                                      dp_rank=dp_rank, dp_world_size=dp_world_size,
                                      seed=seed, global_batch_size=global_batch_size,
                                      num_workers=num_workers,
                                      seeded_sampling=seeded_sampling)
    if test_data_prefix is not None:
        test_dataset = build_dataset("test", test_data_prefix, data_impl,
                                     seq_length, micro_batch_size, False,
                                     dp_rank=dp_rank, dp_world_size=dp_world_size,
                                     seed=seed, global_batch_size=global_batch_size,
                                     num_workers=num_workers,
                                     seeded_sampling=seeded_sampling)

    return (train_dataset, valid_dataset, test_dataset)


def build_dataset(dataset_name, data_prefix, data_impl,
                  seq_length, micro_batch_size, skip_warmup,
                  dp_rank=0, dp_world_size=1,
                  seed=1024, global_batch_size=1, num_workers=1,
                  seeded_sampling=False):
    dataset = None
    if len(data_prefix) == 1:
        batch_size_per_iter = micro_batch_size
        dataset = _build_dataset(dataset_name,
                                 data_prefix[0], data_impl,
                                 seq_length,
                                 batch_size_per_iter,
                                 skip_warmup,
                                 dp_rank=dp_rank,
                                 dp_world_size=dp_world_size)
    else:
        batch_size_per_iter = 1
        output = get_datasets_weights(data_prefix)
        prefixes, weights = output

        datasets = []
        for i in range(len(prefixes)):
            ds = _build_dataset(dataset_name,
                                prefixes[i], data_impl,
                                seq_length,
                                batch_size_per_iter,
                                skip_warmup,
                                dp_rank=dp_rank,
                                dp_world_size=dp_world_size)
            if ds:
                datasets.append(ds)
        if datasets:
            dataset = BlendableWeightedSamplingDataset(
                datasets, weights,
                seed=seed,
                global_batch_size=global_batch_size,
                num_workers=num_workers,
                dp_rank=dp_rank,
                dp_world_size=dp_world_size,
                seeded_sampling=seeded_sampling,
            )
    return dataset


def _build_dataset(dataset_name, data_prefix, data_impl,
                   seq_length, batch_size_per_iter, skip_warmup,
                   dp_rank=0, dp_world_size=1):
    """Build dataset for individual train, valid, test splits."""

    dataset = get_noindexed_dataset_(data_prefix,
                                     data_impl,
                                     seq_length,
                                     batch_size_per_iter,
                                     skip_warmup,
                                     dp_rank=dp_rank,
                                     dp_world_size=dp_world_size)

    print_rank_0('    {}/{}:'.format(dataset_name, data_prefix))
    if dataset is not None:
        print_rank_0('    number of tokens: {}'.format(dataset.size))

    return dataset


def get_noindexed_dataset_(data_prefix, data_impl, seq_length, batch_size_per_iter, skip_warmup,
                           dp_rank=0, dp_world_size=1):
    """Build noindexed dataset."""

    start_time = time.time()

    dataset = make_dataset(data_prefix,
                           data_impl,
                           seq_length,
                           batch_size_per_iter,
                           dp_rank,
                           dp_world_size,
                           skip_warmup)
    print_rank_0(' > finished creating no-indexed dataset in {:4f} '
                 'seconds'.format(time.time() - start_time))
    if dataset is not None:
        print_rank_0('    number of tokens: {}'.format(dataset.size))
    else:
        print_rank_0('Dataset is None')
    return dataset


# ============================================================================
# New: recipe file parser + thin Muse adapter
# ============================================================================

def parse_data_prefix(data_prefix):
    """Parse data_prefix into a flat list suitable for get_datasets_weights.

    Supports input formats:
    1. str single prefix: "/path/prefix" -> (["/path/prefix"], None)
    2. str recipe file (2-col): "weight<tab>path" per line -> (flat_list, None)
    3. str recipe file (3-col): "domain<tab>weight<tab>path" per line
       -> (flat_list, {path: domain})
    4. List[str] flat list: ["w1", "p1", "w2", "p2"] -> (list, None)

    Shell wrapper lines (containing '=' or starting with '"') are skipped.

    Returns:
        (flat_list, domain_map): flat_list is [w1, p1, w2, p2, ...],
        domain_map is {path: domain} when 3-col format is detected, else None.
    """
    if isinstance(data_prefix, str):
        if os.path.isfile(data_prefix) and not data_prefix.endswith(('.bin', '.meta')):
            flat_list = []
            domain_map = {}
            with open(data_prefix) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#') or line.startswith('"'):
                        continue
                    if '=' in line:
                        continue
                    # Try 3-column: domain weight path
                    parts = line.split(None, 2)
                    if len(parts) == 3:
                        try:
                            w = float(parts[1])
                        except ValueError:
                            pass
                        else:
                            domain, weight, path = parts[0], parts[1], parts[2].strip()
                            if w > 0:
                                flat_list.append(weight)
                                flat_list.append(path)
                                domain_map[path] = domain
                            continue
                    # Fallback: 2-column: weight path
                    parts = line.split(None, 1)
                    if len(parts) == 2:
                        try:
                            w = float(parts[0])
                        except ValueError:
                            continue
                        weight, path = parts[0], parts[1].strip()
                        if w > 0:
                            flat_list.append(weight)
                            flat_list.append(path)
            if not flat_list:
                raise ValueError(f"Recipe file {data_prefix} contains no valid weight-path pairs")
            return flat_list, domain_map or None
        else:
            return [data_prefix], None
    return list(data_prefix), None


class KaiMMapDataset(torch.utils.data.IterableDataset):
    """Thin adapter that wraps Kai's MMap dataset pipeline for Muse consumption.

    Internally uses the ported MMapDataset / BlendableWeightedSamplingDataset
    and converts their output to Muse's batch format.
    """

    def __init__(self, data_prefix, seq_length,
                 seed=1024, rank=0, world_size=1,
                 num_workers=1, global_batch_size=1,
                 skip_warmup=False,
                 resume_source_global_batch_size=None,
                 initial_source_state_file=None,
                 seeded_sampling=False,
                 **kwargs):
        super().__init__()
        data_prefix, self._domain_map = parse_data_prefix(data_prefix)
        self.dataset, _, _ = build_train_valid_test_datasets(
            data_impl='mmap',
            seq_length=seq_length,
            micro_batch_size=1,
            skip_warmup=skip_warmup,
            train_data_prefix=data_prefix,
            dp_rank=rank,
            dp_world_size=world_size,
            seed=seed,
            global_batch_size=global_batch_size,
            num_workers=num_workers,
            seeded_sampling=seeded_sampling,
        )
        self.seq_length = seq_length
        self._resume_source_global_batch_size = (
            int(resume_source_global_batch_size)
            if resume_source_global_batch_size is not None
            else None
        )
        if self.dataset is None:
            raise RuntimeError(
                f"Failed to build KaiMMapDataset from data_prefix={data_prefix}. "
                "Check that .bin and .bin.meta files exist."
            )
        if initial_source_state_file is not None:
            self._load_initial_source_state(initial_source_state_file)

    def _load_initial_source_state(self, path):
        """Start a stage from frozen source cursors and the configured RNG seed.

        ``seed`` accepts either an integer or NumPy seed words. A subsequent
        checkpoint resume replaces both these cursors and the RNG state.
        """
        if not isinstance(self.dataset, BlendableWeightedSamplingDataset):
            raise ValueError("initial_source_state_file requires a blendable dataset")
        with open(path, "rb") as stream:
            raw = stream.read()
        initial = json.loads(raw)
        if initial.get("state_type") != "kai_mmap_source_cursors_v1":
            raise ValueError("Unsupported initial source state format")
        old_length = initial["seq_length"]
        if not 0 < old_length <= self.seq_length:
            raise ValueError("Initial source state requires a positive, non-decreasing sequence length")
        state = initial["dataset_state"]
        for name, dataset in self.dataset._datasets.items():
            if name in state and state[name]["num_tokens"] != dataset.size:
                raise ValueError(f"Initial source size changed: {name}")
        state["np_rng_state"] = self.dataset._np_rng.get_state()
        self.dataset.load_state_dict(
            state, old_seq_length=old_length, seq_length=self.seq_length
        )
        self.dataset._initial_source_state_sha256 = hashlib.sha256(raw).hexdigest()
        print_rank_0(f"Initialized data source cursors from {path}; sampling uses the configured seed")

    @property
    def supports_dataloader_resume(self):
        return (
            hasattr(self.dataset, "state_dict")
            and callable(self.dataset.state_dict)
            and hasattr(self.dataset, "load_state_dict")
            and callable(self.dataset.load_state_dict)
        )

    def state_dict(self, consumer_batches=None):
        if not self.supports_dataloader_resume:
            raise RuntimeError(
                "resume_dataloader is not supported for stateless KaiMMapDataset "
                "sources. Use a blendable dataset backend or disable dataloader resume."
            )
        if consumer_batches is None:
            dataset_state = self.dataset.state_dict()
            state_kind = "legacy_live_state"
        else:
            dataset_state = self.dataset.state_dict(
                consumer_batches=consumer_batches
            )
            state_kind = "consumer_boundary_v1"
        state = {
            "state_type": "kai_mmap",
            "state_kind": state_kind,
            "seq_length": self.seq_length,
            "dataset_state": dataset_state,
        }
        if consumer_batches is not None:
            state["consumer_batches"] = int(consumer_batches)
        if consumer_batches is not None and hasattr(self.dataset, "_resume_fingerprint"):
            state["resume_fingerprint"] = self.dataset._resume_fingerprint()
        return state

    def load_state_dict(self, state_dict):
        if not self.supports_dataloader_resume:
            raise RuntimeError(
                "resume_dataloader is not supported for stateless KaiMMapDataset "
                "sources. Use a blendable dataset backend or disable dataloader resume."
            )
        if (
            isinstance(state_dict, dict)
            and state_dict.get("state_type") == "kai_mmap"
            and "dataset_state" in state_dict
        ):
            expected_fingerprint = state_dict.get("resume_fingerprint")
            if expected_fingerprint is not None:
                if not self.dataset.seeded_sampling:
                    raise ValueError("An exact V2 state requires seeded_sampling=True")
                current_fingerprint = self.dataset._resume_fingerprint()
                if current_fingerprint != expected_fingerprint:
                    source_global_bs = getattr(
                        self, "_resume_source_global_batch_size", None
                    )
                    source_fingerprint = None
                    if source_global_bs is not None:
                        source_fingerprint = self.dataset._resume_fingerprint(
                            global_batch_size=source_global_bs
                        )
                    if source_fingerprint != expected_fingerprint:
                        raise ValueError(
                            "KaiMMapDataset resume fingerprint mismatch: dataset "
                            "mixture, seed, batch size, workers, or DP size changed"
                        )
                    print_rank_0(
                        "KaiMMapDataset validated an explicit global-batch-size "
                        f"transition: {source_global_bs} -> "
                        f"{self.dataset.global_bs}"
                    )
            self.dataset.load_state_dict(
                state_dict["dataset_state"],
                old_seq_length=state_dict.get("seq_length"),
                seq_length=self.seq_length,
            )
            self.last_resume_kind = state_dict.get(
                "state_kind", "legacy_soft"
            )
            return
        self.dataset.load_state_dict(state_dict, seq_length=self.seq_length)
        self.last_resume_kind = "legacy_soft"

    def __iter__(self):
        if isinstance(self.dataset, BlendableWeightedSamplingDataset):
            for sample, source_name in self.dataset:
                yield self._to_muse_batch(sample, source_name)
        else:
            source_name = self.dataset.name
            while True:
                for idx in range(self.dataset.num_sample):
                    yield self._to_muse_batch(self.dataset[idx], source_name)

    def _to_muse_batch(self, sample, source_name):
        """Convert Kai's {'text': ndarray} to Muse's batch dict.

        Muse convention (same as TextDataset):
          - input_ids has length L (fits within max_seq_len)
          - loss_mask has length L
          - CrossEntropyLoss(shift_labels=True) shifts internally:
            logits[:,:-1] predicts labels[:,1:]

        Non-QA: Kai provides seq_length+1 tokens. We take first seq_length
        as input_ids. The last token (text[seq_length]) becomes the prediction
        target for position seq_length-1 via Muse's internal shift.
        loss_mask is all-1 except the last position (which has no target after
        shift, so it doesn't matter, but we keep it 1 for simplicity since
        shift_labels discards it anyway).

        QA: Kai provides 2*seq_length (tokens || loss_mask). Used as-is.
        """
        text = torch.tensor(sample['text'], dtype=torch.long)

        if text.shape[0] == 2 * self.seq_length:
            input_ids = text[:self.seq_length]
            loss_mask = text[self.seq_length:]
        else:
            # input_ids = text
            # loss_mask = torch.ones_like(input_ids) # 数据侧 shift
            input_ids = text[:self.seq_length]
            loss_mask = torch.ones(self.seq_length, dtype=torch.long) # pretrain 入口内  shift

        domain = self._domain_map.get(source_name) if self._domain_map else None
        L = input_ids.shape[0]
        batch = {
            "input_ids": input_ids.unsqueeze(0),
            "loss_mask": loss_mask.unsqueeze(0),
            "position_ids": torch.arange(L, dtype=torch.int32).unsqueeze(0),
            "cu_seqlens": torch.tensor([0, L], dtype=torch.int32),
            "data_source": [source_name],
        }
        if domain is not None:
            batch["data_domain"] = [domain]
        return batch
