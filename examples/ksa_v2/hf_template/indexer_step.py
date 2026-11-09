from contextvars import ContextVar
from functools import wraps

_INDEXER_STEP = ContextVar("indexer_step", default=None)


def shared_indexer_step(forward):
    @wraps(forward)
    def scoped(*args, **kwargs):
        token = _INDEXER_STEP.set({"positions": {}, "bounds": {}})
        try:
            return forward(*args, **kwargs)
        finally:
            _INDEXER_STEP.reset(token)
    return scoped


def shared_indexer_positions(source, kind, factory, *layout):
    scope = _INDEXER_STEP.get()
    if scope is None:
        return factory()
    key = (id(source), kind, *layout)
    if key not in scope["positions"]:
        # Retain the source, so tensor identity cannot be recycled in this step.
        scope["positions"][key] = (source, factory())
    return scope["positions"][key][1]


def _validated_position_bounds(position_ids):
    scope = _INDEXER_STEP.get()
    key = id(position_ids)
    if scope is not None and key in scope["bounds"]:
        return scope["bounds"][key][1]
    low = int(position_ids.min())
    if low < 0:
        raise ValueError("indexer position IDs must be non-negative")
    bounds = (low, int(position_ids.max()))
    if scope is not None:
        scope["bounds"][key] = (position_ids, bounds)
    return bounds
