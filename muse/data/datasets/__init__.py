from .kai_mmap import KaiMMapDataset

__all__ = [
    'DistributedDataset',
    'KaiMMapDataset',
]


def __getattr__(name):
    # Mmap-only training does not require the optional parquet/image stack.
    if name == 'DistributedDataset':
        from .base import DistributedDataset
        return DistributedDataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
