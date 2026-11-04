"""Summary-query block routing for S-KSA prefill and cached decoding."""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from .sparse_decode_gather import fused_summary_decode_gather


# Metadata is owned by one model forward, never by a model layer or KV cache.
# Inputs/derived positions are immutable during this forward. Nested forwards
# receive a separate scope; reset in finally also releases exception paths.
from .indexer_step import shared_indexer_step, shared_indexer_positions, _validated_position_bounds


_INDEXER_SCORER_BACKENDS = {"torch", "fused", "auto"}
_FUSED_INDEXER_SCORER = None
_FUSED_INDEXER_SCORER_LOAD_ATTEMPTED = False
_FUSED_INDEXER_SCORER_IMPORT_ERROR = None


class _IndexerRMSNorm(nn.Module):
    """RMSNorm with the same ``weight`` state-dict contract as HF Qwen3."""

    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        hidden_states = hidden_states * torch.rsqrt(
            hidden_states.pow(2).mean(-1, keepdim=True) + self.eps
        )
        return self.weight * hidden_states.to(input_dtype)


def aligned_summary_block_local_start(
    query_position: int,
    local_window_tokens: int,
    raw_block_size: int,
) -> int:
    """Return the block-aligned inclusive left edge of the local window.

    Routing only admits raw blocks that end before this edge.  Aligning the
    edge down keeps those remote blocks disjoint from the local window while
    ensuring that the one raw block straddling the nominal window boundary is
    never dropped for one to ``raw_block_size - 1`` decode positions.
    """
    if local_window_tokens <= 0 or raw_block_size <= 0:
        raise ValueError("local window and raw block size must be positive")
    nominal_start = max(query_position - local_window_tokens + 1, 0)
    return nominal_start // raw_block_size * raw_block_size


def _resolve_indexer_scorer_backend(config_backend: str) -> str:
    """Resolve the scorer backend from the unchanged exported configuration."""
    backend = config_backend
    if not isinstance(backend, str):
        raise ValueError("S-KSA indexer scorer backend must be a string")
    backend = backend.strip().lower()
    if backend not in _INDEXER_SCORER_BACKENDS:
        raise ValueError(
            "sksa_indexer_scorer_backend must be 'torch', 'fused', or 'auto'"
        )
    return backend


def _load_fused_indexer_scorer():
    """Load the shared KSA Indexer scorer."""
    global _FUSED_INDEXER_SCORER
    global _FUSED_INDEXER_SCORER_LOAD_ATTEMPTED
    global _FUSED_INDEXER_SCORER_IMPORT_ERROR
    if not _FUSED_INDEXER_SCORER_LOAD_ATTEMPTED:
        _FUSED_INDEXER_SCORER_LOAD_ATTEMPTED = True
        try:
            from ksa_attn import fused_indexer_candidate_scores
        except ImportError as error:
            _FUSED_INDEXER_SCORER_IMPORT_ERROR = error
        else:
            _FUSED_INDEXER_SCORER = fused_indexer_candidate_scores
    return _FUSED_INDEXER_SCORER


def _fused_indexer_scorer_unavailable_reason(
    query: torch.Tensor,
    key: torch.Tensor,
    head_weight: torch.Tensor,
) -> str | None:
    if torch.is_grad_enabled():
        return "gradient mode is enabled"
    if query.ndim != 4 or key.ndim != 3 or head_weight.ndim != 3:
        return "expected query[B,Q,H,D], key[B,M,D], and weight[B,Q,H]"
    if not (query.is_cuda and key.is_cuda and head_weight.is_cuda):
        return "all scorer tensors must be CUDA tensors"
    if not (query.device == key.device == head_weight.device):
        return "all scorer tensors must share one CUDA device"
    if query.dtype not in (torch.float16, torch.bfloat16):
        return "query must use fp16 or bf16"
    if key.dtype != query.dtype:
        return "query and key dtypes must match"
    if head_weight.dtype != torch.float32:
        return "head weights must use fp32"
    if query.shape[-1] not in (64, 128):
        return "indexer head_dim must be 64 or 128"
    if not 1 <= query.shape[2] <= 8:
        return "indexer head count must be between 1 and 8"
    if query.stride(-1) != 1 or key.stride(-1) != 1:
        return "query/key head dimensions must be contiguous"
    if head_weight.stride(-1) != 1:
        return "head weights must be contiguous over heads"
    if _load_fused_indexer_scorer() is None:
        detail = (
            f": {_FUSED_INDEXER_SCORER_IMPORT_ERROR}"
            if _FUSED_INDEXER_SCORER_IMPORT_ERROR is not None
            else ""
        )
        return f"ksa_attn fused scorer is unavailable{detail}"
    return None


def score_indexer_candidates(
    query: torch.Tensor,
    key: torch.Tensor,
    head_weight: torch.Tensor,
    score_scale: float,
    *,
    scorer_backend: str = "torch",
) -> torch.Tensor:
    """Score index keys using the persisted train/runtime backend contract."""
    backend = _resolve_indexer_scorer_backend(scorer_backend)
    if backend != "torch":
        unavailable = _fused_indexer_scorer_unavailable_reason(
            query, key, head_weight
        )
        if unavailable is None:
            # Never hide a fused execution or compilation failure behind a
            # numerically different PyTorch route.
            return _load_fused_indexer_scorer()(
                query, key, head_weight, score_scale
            )
        if backend == "fused":
            raise RuntimeError(
                f"strict fused S-KSA indexer scorer is unavailable: {unavailable}"
            )
    dot = torch.einsum("bqhd,bmd->bqhm", query, key)
    activated = dot.float()
    activated.mul_(score_scale).relu_()
    activated.mul_(head_weight[..., None])
    return activated.sum(dim=2)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    left, right = x.chunk(2, dim=-1)
    return torch.cat((-right, left), dim=-1)


def _apply_partial_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rope_dim: int,
) -> torch.Tensor:
    """Apply the indexer's RoPE frequencies to its trailing dimensions.

    ``x`` is ``[B,S,H,D]`` (or ``[B,S,1,D]`` for shared index keys).  Following
    the official Lightning Indexer layout, RoPE occupies the trailing
    ``rope_dim`` coordinates and the leading coordinates remain unrotated.
    """
    if rope_dim == 0:
        return x
    if rope_dim < 0 or rope_dim > x.shape[-1] or rope_dim % 2:
        raise ValueError("S-KSA indexer rope_dim must be even and within head_dim")
    if cos.shape[:2] != x.shape[:2] or sin.shape != cos.shape:
        raise ValueError("indexer RoPE tensors do not match projected states")
    if cos.shape[-1] < rope_dim:
        raise ValueError("model RoPE width is smaller than indexer rope_dim")
    rotated = x[..., -rope_dim:]
    rope_cos = cos[..., :rope_dim].unsqueeze(2).to(rotated.dtype)
    rope_sin = sin[..., :rope_dim].unsqueeze(2).to(rotated.dtype)
    rotated = rotated * rope_cos + _rotate_half(rotated) * rope_sin
    return torch.cat((x[..., :-rope_dim], rotated), dim=-1)


class SummaryBlockLightningIndexer(nn.Module):
    """Per-Full-layer Lightning Indexer over shared routing keys.

    The module contains only the newly trained indexer parameters.  It accepts
    the already-normalized pre-attention hidden states so its query signal is
    specific to the current Full layer.  Keys are shared across indexer heads,
    matching the inexpensive Lightning Indexer geometry.
    """

    def __init__(
        self,
        hidden_size: int,
        *,
        query_rank: int,
        num_heads: int,
        head_dim: int,
        rope_dim: int,
        rope_theta: float,
        max_seq_len: int,
        query_block_size: int,
        scorer_backend: str = "torch",
        chunk_attention_qk_source: str = "backbone",
        chunk_num_heads: int | None = None,
        chunk_num_kv_heads: int | None = None,
        chunk_head_dim: int | None = None,
        chunk_norm_eps: float = 1e-6,
        chunk_q_bias: bool = False,
        chunk_k_bias: bool = False,
        chunk_attention_v_source: str = "backbone",
        chunk_v_bias: bool = False,
        chunk_attention_include_summary: bool = True,
    ) -> None:
        super().__init__()
        if min(query_rank, num_heads, head_dim, query_block_size) <= 0:
            raise ValueError("S-KSA indexer dimensions must be positive")
        if rope_dim < 0 or rope_dim > head_dim or rope_dim % 2:
            raise ValueError("S-KSA indexer rope_dim must be even and <= head_dim")
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.rope_dim = rope_dim
        self.rope_theta = float(rope_theta)
        self.query_block_size = query_block_size
        if scorer_backend not in _INDEXER_SCORER_BACKENDS:
            raise ValueError(
                "sksa_indexer_scorer_backend must be 'torch', 'fused', or 'auto'"
            )
        self.scorer_backend = scorer_backend
        if chunk_attention_qk_source not in {"backbone", "independent"}:
            raise ValueError(
                "chunk_attention_qk_source must be 'backbone' or 'independent'"
            )
        self.chunk_attention_qk_source = chunk_attention_qk_source
        if chunk_attention_v_source not in {"backbone", "independent"}:
            raise ValueError("invalid chunk_attention_v_source")
        if chunk_attention_v_source == "independent" and chunk_attention_qk_source != "independent":
            raise ValueError("independent chunk V requires independent chunk Q/K")
        self.chunk_attention_v_source = chunk_attention_v_source
        self.chunk_attention_include_summary = chunk_attention_include_summary
        if not chunk_attention_include_summary and chunk_attention_qk_source != "independent":
            raise ValueError("raw-only chunk attention requires independent chunk Q/K")
        self.chunk_v_proj = None
        self.chunk_num_heads = None
        self.chunk_num_kv_heads = None
        self.chunk_head_dim = None
        self.chunk_q_proj = None
        self.chunk_k_proj = None
        self.chunk_q_norm = None
        self.chunk_k_norm = None
        if chunk_attention_qk_source == "independent":
            if (
                chunk_num_heads is None
                or chunk_num_kv_heads is None
                or chunk_head_dim is None
                or min(chunk_num_heads, chunk_num_kv_heads, chunk_head_dim) <= 0
            ):
                raise ValueError(
                    "independent chunk Q/K requires positive head geometry"
                )
            if chunk_num_heads % chunk_num_kv_heads:
                raise ValueError(
                    "chunk query heads must be divisible by chunk KV heads"
                )
            self.chunk_num_heads = int(chunk_num_heads)
            self.chunk_num_kv_heads = int(chunk_num_kv_heads)
            self.chunk_head_dim = int(chunk_head_dim)
            self.chunk_q_proj = nn.Linear(
                hidden_size,
                self.chunk_num_heads * self.chunk_head_dim,
                bias=chunk_q_bias,
            )
            self.chunk_k_proj = nn.Linear(
                hidden_size,
                self.chunk_num_kv_heads * self.chunk_head_dim,
                bias=chunk_k_bias,
            )
            self.chunk_q_norm = _IndexerRMSNorm(
                self.chunk_head_dim, chunk_norm_eps
            )
            self.chunk_k_norm = _IndexerRMSNorm(
                self.chunk_head_dim, chunk_norm_eps
            )
        if chunk_attention_v_source == "independent":
            self.chunk_v_proj = nn.Linear(
                hidden_size, self.chunk_num_kv_heads * self.chunk_head_dim,
                bias=chunk_v_bias,
            )
        inv_freq = 1.0 / (
            float(rope_theta)
            ** (
                torch.arange(0, rope_dim, 2, dtype=torch.float32)
                / max(rope_dim, 1)
            )
        )
        self.register_buffer("rope_inv_freq", inv_freq, persistent=False)
        self._rope_cache_device = inv_freq.device
        sequence = torch.arange(max_seq_len, dtype=inv_freq.dtype)
        angle = torch.einsum("i,j->ij", sequence, inv_freq).float()
        embedding = torch.cat((angle, angle), dim=-1)
        self.register_buffer(
            "rope_cache",
            torch.stack((embedding.cos(), embedding.sin()), dim=-1),
            persistent=False,
        )
        self.query_down_proj = nn.Linear(hidden_size, query_rank, bias=False)
        self.query_up_proj = nn.Linear(
            query_rank, num_heads * head_dim, bias=False
        )
        self.key_proj = nn.Linear(hidden_size, head_dim, bias=False)
        self.head_weight_proj = nn.Linear(hidden_size, num_heads, bias=False)

    def project_chunk_qk(
        self,
        hidden_states: torch.Tensor,
        *,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project and normalize the independent chunk-attention Q/K stream."""
        return (
            self.project_chunk_queries(hidden_states, cos=cos, sin=sin),
            self.project_chunk_keys(hidden_states, cos=cos, sin=sin),
        )

    def _validate_chunk_rope(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor | None,
        sin: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if (cos is None) != (sin is None):
            raise ValueError("chunk Q/K RoPE requires both cosine and sine")
        if cos is not None:
            if cos.shape != sin.shape or cos.ndim != 3:
                raise ValueError("chunk Q/K RoPE must use [B,S,D] tensors")
            if tuple(cos.shape[:2]) != tuple(hidden_states.shape[:2]):
                raise ValueError("chunk Q/K RoPE geometry does not match hidden states")
            if cos.shape[-1] != self.chunk_head_dim:
                raise ValueError("chunk Q/K RoPE width must equal chunk head_dim")
            return cos.to(hidden_states.dtype), sin.to(hidden_states.dtype)
        return None, None

    def project_chunk_queries(
        self,
        hidden_states: torch.Tensor,
        *,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Project only the independent chunk-attention queries."""
        if self.chunk_attention_qk_source != "independent":
            raise RuntimeError("independent chunk Q/K is not enabled")
        if hidden_states.ndim != 3:
            raise ValueError("chunk Q/K projection expects hidden[B,S,E]")
        if (
            self.chunk_q_proj is None
            or self.chunk_q_norm is None
        ):
            raise RuntimeError("independent chunk Q/K modules are incomplete")
        batch, sequence, _ = hidden_states.shape
        query = self.chunk_q_proj(hidden_states).view(
            batch,
            sequence,
            self.chunk_num_heads,
            self.chunk_head_dim,
        )
        query = self.chunk_q_norm(query)
        rope_cos, rope_sin = self._validate_chunk_rope(
            hidden_states, cos, sin
        )
        if rope_cos is not None:
            query = query * rope_cos.unsqueeze(2) + _rotate_half(query) * rope_sin.unsqueeze(2)
        return query

    def project_chunk_keys(
        self,
        hidden_states: torch.Tensor,
        *,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Project only the independent chunk-attention keys."""
        if self.chunk_attention_qk_source != "independent":
            raise RuntimeError("independent chunk Q/K is not enabled")
        if hidden_states.ndim != 3:
            raise ValueError("chunk Q/K projection expects hidden[B,S,E]")
        if (
            self.chunk_k_proj is None
            or self.chunk_k_norm is None
            or self.chunk_head_dim is None
        ):
            raise RuntimeError("independent chunk K modules are incomplete")
        batch, sequence, _ = hidden_states.shape
        key = self.chunk_k_proj(hidden_states).view(
            batch,
            sequence,
            self.chunk_num_kv_heads,
            self.chunk_head_dim,
        )
        key = self.chunk_k_norm(key)
        rope_cos, rope_sin = self._validate_chunk_rope(
            hidden_states, cos, sin
        )
        if rope_cos is not None:
            key = key * rope_cos.unsqueeze(2) + _rotate_half(key) * rope_sin.unsqueeze(2)
        return key

    def project_chunk_values(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project auxiliary V without position encoding or normalization."""
        if self.chunk_v_proj is None:
            raise RuntimeError("independent chunk V is not enabled")
        batch, sequence, _ = hidden_states.shape
        return self.chunk_v_proj(hidden_states).view(
            batch, sequence, self.chunk_num_kv_heads, self.chunk_head_dim,
        )

    def expand_chunk_values(self, values: torch.Tensor) -> torch.Tensor:
        return values.repeat_interleave(
            self.chunk_num_heads // self.chunk_num_kv_heads, dim=-2,
        )

    def aggregate_chunk_attention_summary(
        self,
        query_chunks: torch.Tensor,
        key_chunks: torch.Tensor,
        value_chunks: torch.Tensor,
        *,
        summary_chunk_size: int,
        summary_token_num: int,
        indexer_block_size: int | None = None,
    ) -> torch.Tensor:
        """Run causal local attention for the summary positions only.

        ``query_chunks`` is ``[B,N,W,H,D]`` and ``key_chunks`` is
        ``[B,N,W,KV,D]``. Values must already be expanded to query heads.
        """
        if (
            query_chunks.ndim != 5
            or key_chunks.ndim != 5
            or value_chunks.ndim != 5
        ):
            raise ValueError("chunk attention tensors must be rank-5")
        batch, chunks, query_width, query_heads, head_dim = query_chunks.shape
        width = key_chunks.shape[2]
        if key_chunks.shape[0] != batch or key_chunks.shape[1] != chunks:
            raise ValueError("chunk Q/K geometry mismatch")
        if value_chunks.shape[:3] != (batch, chunks, width):
            raise ValueError("chunk K/V geometry mismatch")
        if summary_chunk_size <= 0 or summary_token_num <= 0:
            raise ValueError("chunk attention sizes must be positive")
        if summary_chunk_size + summary_token_num != width:
            raise ValueError("chunk attention width is inconsistent")
        if query_width not in (summary_token_num, width):
            raise ValueError("chunk query width does not match summary geometry")
        if key_chunks.shape[-1] != head_dim:
            raise ValueError("chunk Q/K head dimensions do not match")
        if value_chunks.shape[-2:] != (query_heads, head_dim):
            raise ValueError("chunk values do not match query geometry")
        if key_chunks.shape[-2] != query_heads:
            if query_heads % key_chunks.shape[-2]:
                raise ValueError("chunk query heads must divide KV heads")
            groups = query_heads // key_chunks.shape[-2]
            key_chunks = (
                key_chunks.unsqueeze(4)
                .expand(batch, chunks, width, key_chunks.shape[-2], groups, head_dim)
                .reshape(batch, chunks, width, query_heads, head_dim)
            )
        query_summary = (
            query_chunks
            if query_width == summary_token_num
            else query_chunks[:, :, summary_chunk_size:, :, :]
        )
        block_size = indexer_block_size or summary_chunk_size
        if block_size != summary_chunk_size or not self.chunk_attention_include_summary:
            if (
                summary_token_num != 1
                or block_size not in (2, 4, 8)
                or summary_chunk_size % block_size
            ):
                raise ValueError("raw-block attention requires one summary and blocks of 2, 4 or 8")
            # Summary supplies Q only; normalize over each raw-text block.
            groups = summary_chunk_size // block_size
            raw_k = key_chunks[:, :, :summary_chunk_size].reshape(
                batch, chunks, groups, block_size, query_heads, head_dim
            )
            raw_v = value_chunks[:, :, :summary_chunk_size].reshape_as(raw_k)
            logits = torch.einsum(
                "bnhd,bngshd->bnghs", query_summary[:, :, 0], raw_k
            ).float()
            probability = torch.softmax(logits * (head_dim ** -0.5), dim=-1)
            return torch.einsum(
                "bnghs,bngshd->bnghd", probability.to(raw_v.dtype), raw_v
            )
        logits = torch.einsum(
            "bnthd,bnshd->bnhts", query_summary, key_chunks
        ).float()
        logits.mul_(head_dim ** -0.5)
        query_positions = summary_chunk_size + torch.arange(
            summary_token_num, device=query_chunks.device
        )
        key_positions = torch.arange(width, device=query_chunks.device)
        causal = key_positions[None, :] <= query_positions[:, None]
        probability = torch.softmax(
            logits.masked_fill(~causal[None, None, None], float("-inf")),
            dim=-1,
        )
        return torch.einsum(
            "bnhts,bnshd->bnthd",
            probability.to(value_chunks.dtype),
            value_chunks,
        )

    def build_chunk_attention_summary_from_projected(
        self,
        query_chunks: torch.Tensor,
        key_chunks: torch.Tensor,
        value_chunks: torch.Tensor,
        *,
        summary_chunk_size: int,
        summary_token_num: int,
        indexer_block_size: int | None = None,
    ) -> torch.Tensor:
        summary = self.aggregate_chunk_attention_summary(
            query_chunks,
            key_chunks,
            value_chunks,
            summary_chunk_size=summary_chunk_size,
            summary_token_num=summary_token_num,
            indexer_block_size=indexer_block_size,
        )
        return summary.reshape(summary.shape[0], -1, summary.shape[-2] * summary.shape[-1])

    @torch.no_grad()
    def build_chunk_attention_summary(
        self,
        hidden_chunks: torch.Tensor,
        value_chunks: torch.Tensor,
        *,
        summary_chunk_size: int,
        summary_token_num: int,
        chunk_cos: torch.Tensor | None = None,
        chunk_sin: torch.Tensor | None = None,
        indexer_block_size: int | None = None,
    ) -> torch.Tensor:
        """Build independent-Q/K summary states from a complete chunk batch."""
        if hidden_chunks.ndim != 4 or value_chunks.ndim != 5:
            raise ValueError(
                "chunk attention expects hidden[B,N,W,E] and value[B,N,W,H,D]"
            )
        batch, chunks, width, hidden_size = hidden_chunks.shape
        if value_chunks.shape[:3] != (batch, chunks, width):
            raise ValueError("chunk hidden/value geometry mismatch")
        flat_hidden = hidden_chunks.reshape(batch, chunks * width, hidden_size)
        flat_summary_hidden = hidden_chunks[:, :, summary_chunk_size:, :].reshape(
            batch, chunks * summary_token_num, hidden_size
        )
        flat_cos = (
            None
            if chunk_cos is None
            else chunk_cos.reshape(batch, chunks * width, chunk_cos.shape[-1])
        )
        flat_sin = (
            None
            if chunk_sin is None
            else chunk_sin.reshape(batch, chunks * width, chunk_sin.shape[-1])
        )
        summary_cos = (
            None
            if flat_cos is None
            else flat_cos.reshape(batch, chunks, width, flat_cos.shape[-1])[
                :, :, summary_chunk_size:, :
            ].reshape(batch, chunks * summary_token_num, flat_cos.shape[-1])
        )
        summary_sin = (
            None
            if flat_sin is None
            else flat_sin.reshape(batch, chunks, width, flat_sin.shape[-1])[
                :, :, summary_chunk_size:, :
            ].reshape(batch, chunks * summary_token_num, flat_sin.shape[-1])
        )
        if self.chunk_v_proj is not None:
            value_chunks = self.expand_chunk_values(
                self.project_chunk_values(flat_hidden.detach())
            ).view(batch, chunks, width, self.chunk_num_heads, self.chunk_head_dim)
        query = self.project_chunk_queries(
            flat_summary_hidden.detach(), cos=summary_cos, sin=summary_sin
        )
        key = self.project_chunk_keys(
            flat_hidden.detach(), cos=flat_cos, sin=flat_sin
        )
        query = query.view(
            batch,
            chunks,
            summary_token_num,
            self.chunk_num_heads,
            self.chunk_head_dim,
        )
        key = key.view(
            batch,
            chunks,
            width,
            self.chunk_num_kv_heads,
            self.chunk_head_dim,
        )
        return self.build_chunk_attention_summary_from_projected(
            query,
            key,
            value_chunks.detach(),
            summary_chunk_size=summary_chunk_size,
            summary_token_num=summary_token_num,
            indexer_block_size=indexer_block_size,
        )

    def _rope_cos_sin(
        self, position_ids: torch.Tensor, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if position_ids.ndim != 2:
            raise ValueError("indexer position IDs must use [B,S] layout")
        if self.rope_dim == 0:
            empty = torch.empty(
                (*position_ids.shape, 0),
                dtype=dtype,
                device=position_ids.device,
            )
            return empty, empty
        self._ensure_rope_cache_device(position_ids.device)
        if position_ids.numel():
            _, max_position = _validated_position_bounds(position_ids)
            # Device/table identity and capacity remain layer-local. Even a
            # reused validated range must perform this layer's growth check.
            self._grow_rope_cache(max_position + 1)
        cache = self.rope_cache[position_ids]
        return cache[..., 0].to(dtype), cache[..., 1].to(dtype)

    @torch.no_grad()
    def _ensure_rope_cache_device(self, device: torch.device) -> None:
        if (
            self._rope_cache_device == device
            and self.rope_inv_freq.dtype == torch.float32
            and self.rope_cache.dtype == torch.float32
        ):
            return
        inv_freq = 1.0 / (
            self.rope_theta
            ** (torch.arange(0, self.rope_dim, 2, dtype=torch.float32, device=device)
                / max(self.rope_dim, 1))
        )
        positions = torch.arange(
            self.rope_cache.shape[0], dtype=torch.float32, device=device
        )
        angle = torch.einsum("i,j->ij", positions, inv_freq).float()
        embedding = torch.cat((angle, angle), dim=-1)
        self.rope_inv_freq = inv_freq
        self.rope_cache = torch.stack((embedding.cos(), embedding.sin()), dim=-1)
        self._rope_cache_device = device

    @torch.no_grad()
    def _grow_rope_cache(self, required_length: int) -> None:
        """Extend the non-persistent indexer RoPE cache on demand.

        Exported models may prefill exactly ``max_position_embeddings`` tokens
        and then decode beyond that boundary.  Preserve every existing cache
        row byte-for-byte and append new rows geometrically.  The extension is
        built with FP32 frequencies on the execution device, matching training.
        Existing rows remain unchanged when generation crosses the boundary.
        """
        current_length = self.rope_cache.shape[0]
        if required_length <= current_length:
            return
        new_length = max(required_length, max(1, current_length * 2))
        positions = torch.arange(
            current_length, new_length, dtype=torch.float32, device=self.rope_cache.device
        )
        inv_freq = self.rope_inv_freq.detach().float()
        angle = torch.einsum("i,j->ij", positions, inv_freq).float()
        embedding = torch.cat((angle, angle), dim=-1)
        extension = torch.stack(
            (embedding.cos(), embedding.sin()), dim=-1
        ).to(
            device=self.rope_cache.device,
            dtype=self.rope_cache.dtype,
        )
        self.rope_cache = torch.cat((self.rope_cache, extension), dim=0)

    def project_keys(
        self,
        summary_hidden: torch.Tensor,
        summary_position_ids: torch.Tensor,
    ) -> torch.Tensor:
        key = self.key_proj(summary_hidden).unsqueeze(2)
        summary_cos, summary_sin = self._rope_cos_sin(
            summary_position_ids, key.dtype
        )
        return _apply_partial_rope(
            key, summary_cos, summary_sin, self.rope_dim
        ).squeeze(2)

    def project_queries(
        self,
        text_hidden: torch.Tensor,
        text_position_ids: torch.Tensor,
    ) -> torch.Tensor:
        query = self.query_up_proj(
            self.query_down_proj(text_hidden)
        ).view(
            text_hidden.shape[0],
            text_hidden.shape[1],
            self.num_heads,
            self.head_dim,
        )
        text_cos, text_sin = self._rope_cos_sin(
            text_position_ids, query.dtype
        )
        return _apply_partial_rope(
            query, text_cos, text_sin, self.rope_dim
        )

    @torch.no_grad()
    def select_blocks(
        self,
        text_hidden: torch.Tensor,
        index_keys: torch.Tensor,
        text_position_ids: torch.Tensor,
        *,
        raw_block_size: int,
        local_window_tokens: int,
        selected_block_capacity: int,
        force_sink: bool,
        sink_block_count: int = 1,
        sink_token_count: int | None = None,
        local_alignment_tokens: int | None = None,
    ) -> torch.Tensor:
        """Return padded per-query route IDs in ``[B,text,K]``.

        ``raw_block_size`` is the raw-token span represented by one route ID.
        ``local_alignment_tokens`` controls only the left edge of the local
        window.  They match the block span for summary-block routing; raw-token
        routing uses span one while retaining the experiment's raw-8 local
        alignment. ``-1`` pads rows with fewer than K remote candidates.
        """
        if text_hidden.ndim != 3 or index_keys.ndim != 3:
            raise ValueError("indexer hidden/key tensors must be rank-3")
        batch, text_count, _ = text_hidden.shape
        if index_keys.shape[0] != batch or selected_block_capacity < 0:
            raise ValueError("indexer key batch/capacity is invalid")
        if sink_token_count is not None:
            sink_block_count = sink_token_count
        if sink_block_count <= 0:
            raise ValueError("indexer sink token count must be positive")
        if force_sink and selected_block_capacity < sink_block_count:
            raise ValueError(
                "indexer route capacity must cover every forced sink token"
            )
        if local_alignment_tokens is None:
            local_alignment_tokens = raw_block_size
        if (
            raw_block_size <= 0
            or local_alignment_tokens <= 0
            or local_alignment_tokens % raw_block_size
        ):
            raise ValueError("invalid indexer route span/local alignment")
        selected = torch.full(
            (batch, text_count, selected_block_capacity),
            -1,
            dtype=torch.int32,
            device=text_hidden.device,
        )
        meta_count = index_keys.shape[1]
        if not text_count or not meta_count or not selected_block_capacity:
            return selected

        # A block is remote iff its final raw token lies left of the raw-block
        # aligned local start.  This is numerically the same candidate prefix
        # used by training, while the aligned local window also covers the
        # nominal boundary block until it becomes fully remote.
        query_ids = torch.arange(text_count, device=text_hidden.device)
        nominal_local_start = (query_ids - local_window_tokens + 1).clamp_min(0)
        local_start = torch.div(
            nominal_local_start,
            local_alignment_tokens,
            rounding_mode="floor",
        ) * local_alignment_tokens
        eligible_count = torch.div(
            local_start, raw_block_size, rounding_mode="floor"
        ).clamp_max(meta_count)
        query_all = self.project_queries(text_hidden, text_position_ids)
        head_weight_all = self.head_weight_proj(text_hidden).float()
        head_weight_all.mul_(self.num_heads**-0.5)
        scale = self.head_dim**-0.5
        for start in range(0, text_count, self.query_block_size):
            end = min(text_count, start + self.query_block_size)
            # Match training's causal candidate prefix exactly.  Scoring the
            # masked future suffix wastes work and can make tied TopK behavior
            # depend on a different input extent between train and runtime.
            tile_candidate_count = min(
                meta_count,
                (
                    max(0, end - local_window_tokens)
                    // local_alignment_tokens
                    * local_alignment_tokens
                    // raw_block_size
                ),
            )
            if tile_candidate_count == 0:
                continue
            max_take = min(tile_candidate_count, selected_block_capacity)
            query = query_all[:, start:end]
            score = score_indexer_candidates(
                query,
                index_keys[:, :tile_candidate_count],
                head_weight_all[:, start:end],
                scale,
                scorer_backend=self.scorer_backend,
            )
            eligible = eligible_count[start:end]
            key_ids = torch.arange(
                tile_candidate_count, device=text_hidden.device
            )
            valid = key_ids[None, None, :] < eligible[None, :, None]
            score.masked_fill_(~valid, float("-inf"))
            if force_sink:
                sink_count = min(sink_block_count, tile_candidate_count, selected_block_capacity)
                sink_valid = valid[..., :sink_count]
                score[..., :sink_count] = torch.where(
                    sink_valid,
                    score.new_full((), float("inf")),
                    score[..., :sink_count],
                )
            values, indices = torch.topk(
                score, max_take, dim=-1, sorted=False
            )
            # With ``sorted=False`` valid scores are not guaranteed to occupy
            # the first ``eligible`` output slots.  As in training, identify
            # padding solely by the masked -inf value before sorting IDs.
            keep = values != float("-inf")
            tile_selected = torch.where(
                keep, indices.to(torch.int32), indices.new_full((), -1).to(torch.int32)
            )
            if max_take < selected_block_capacity:
                tile_selected = F.pad(
                    tile_selected,
                    (0, selected_block_capacity - max_take),
                    value=-1,
                )
            sort_key = torch.where(
                tile_selected >= 0,
                tile_selected,
                tile_selected.new_full((), tile_candidate_count + 1),
            )
            selected[:, start:end] = torch.gather(
                tile_selected,
                -1,
                torch.argsort(sort_key, dim=-1),
            )
        return selected


def _summary_block_sparse_text_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    selected_block_indices: torch.Tensor,
    *,
    raw_block_size: int,
    local_window_tokens: int,
    local_alignment_tokens: int | None = None,
    query_tile_size: int = 4,
) -> torch.Tensor:
    """Exact, bounded-memory reference for per-query summary-block routing.

    Q/K/V use ``[B,H,S,D]``.  The implementation gathers raw tokens, never
    summary K/V, and tiles query rows so variable prefill lengths cannot fall
    back to a dense ``S x S`` attention allocation.
    """
    if q.ndim != 4 or k.ndim != 4 or v.shape != k.shape:
        raise ValueError("summary-block attention expects rank-4 Q/K/V")
    batch, query_heads, text_count, head_dim = q.shape
    if k.shape[0] != batch or k.shape[2:] != (text_count, head_dim):
        raise ValueError("summary-block attention Q/K/V geometry mismatch")
    if query_heads % k.shape[1]:
        raise ValueError("query heads must be divisible by K/V heads")
    if selected_block_indices.shape[:2] != (batch, text_count):
        raise ValueError("one selected-block row is required per text query")
    if local_alignment_tokens is None:
        local_alignment_tokens = raw_block_size
    if min(
        raw_block_size,
        local_window_tokens,
        local_alignment_tokens,
        query_tile_size,
    ) <= 0:
        raise ValueError("summary-block attention sizes must be positive")
    if local_alignment_tokens % raw_block_size:
        raise ValueError("local alignment must be divisible by route span")

    k_seq = k.transpose(1, 2)
    v_seq = v.transpose(1, 2)
    kv_heads = k.shape[1]
    groups = query_heads // kv_heads
    selected = selected_block_indices.long()
    block_offsets = torch.arange(raw_block_size, device=q.device)
    # Aligning the local edge down can grow the physical window by at most one
    # raw block minus one token.  Keep a fixed bounded gather width and mask the
    # unused suffix independently for every query.
    local_offsets = torch.arange(
        local_window_tokens + local_alignment_tokens - 1, device=q.device
    )
    batch_ids = torch.arange(batch, device=q.device)[:, None, None]
    output = torch.empty_like(q)
    scale = head_dim**-0.5

    for start in range(0, text_count, query_tile_size):
        end = min(text_count, start + query_tile_size)
        query_ids = torch.arange(start, end, device=q.device)
        selected_tile = selected[:, start:end]
        remote = (
            selected_tile[..., None] * raw_block_size
            + block_offsets
        ).flatten(2)
        remote_valid = (
            selected_tile[..., None] >= 0
        ).expand_as(
            selected_tile[..., None] * raw_block_size + block_offsets
        ).flatten(2)
        nominal_local_start = (
            query_ids - local_window_tokens + 1
        ).clamp_min(0)
        local_start = torch.div(
            nominal_local_start,
            local_alignment_tokens,
            rounding_mode="floor",
        ) * local_alignment_tokens
        local = (local_start[:, None] + local_offsets)[None].expand(
            batch, -1, -1
        )
        local_valid = local <= query_ids[None, :, None]
        candidates = torch.cat((remote, local), dim=-1)
        valid = torch.cat((remote_valid, local_valid), dim=-1)
        valid &= (candidates >= 0) & (candidates <= query_ids[None, :, None])
        safe = candidates.clamp(min=0, max=max(text_count - 1, 0))
        gathered_k = k_seq[batch_ids, safe]
        gathered_v = v_seq[batch_ids, safe]
        if groups != 1:
            gathered_k = gathered_k.repeat_interleave(groups, dim=3)
            gathered_v = gathered_v.repeat_interleave(groups, dim=3)
        logits = torch.einsum(
            "bhqd,bqkhd->bhqk", q[:, :, start:end].float(), gathered_k.float()
        ) * scale
        probability = torch.softmax(
            logits.masked_fill(~valid[:, None], float("-inf")), dim=-1
        ).to(v.dtype)
        output[:, :, start:end] = torch.einsum(
            "bhqk,bqkhd->bhqd", probability, gathered_v
        )
    return output


def summary_block_sparse_prefill_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    summary_mask: torch.Tensor,
    selected_block_indices: torch.Tensor,
    *,
    raw_block_size: int,
    summary_chunk_size: int,
    summary_token_num: int,
    local_window_tokens: int,
    text_positions: torch.Tensor | None = None,
    summary_positions: torch.Tensor | None = None,
) -> torch.Tensor:
    """Full-layer prefill for summary-key routing; returns ``[B,H,S,D]``."""
    if summary_token_num != 1 or raw_block_size not in (2, 4, 8) or summary_chunk_size % raw_block_size:
        raise ValueError("summary index routing requires one summary and raw block size 2/4/8 dividing the chunk")
    if summary_mask.ndim == 1:
        summary_mask = summary_mask[None].expand(q.shape[0], -1)
    if summary_mask.shape != (q.shape[0], q.shape[2]):
        raise ValueError("summary mask does not match expanded Q/K/V")
    shared = summary_mask[0]
    if summary_mask.shape[0] > 1 and not torch.equal(
        summary_mask, shared[None].expand_as(summary_mask)
    ):
        raise ValueError("summary-block runtime requires a shared batch layout")
    if text_positions is None:
        text_positions = torch.nonzero(~shared, as_tuple=False).flatten()
    if summary_positions is None:
        summary_positions = torch.nonzero(shared, as_tuple=False).flatten()
    text_count = text_positions.numel()
    if selected_block_indices.shape[:2] != (q.shape[0], text_count):
        raise ValueError("selected blocks do not match text query count")
    q_text = q.index_select(2, text_positions).contiguous()
    k_text = k.index_select(2, text_positions).contiguous()
    v_text = v.index_select(2, text_positions).contiguous()

    if q.is_cuda and text_count > local_window_tokens:
        try:
            from ksa_attn import summary_block_sparse_attn_func
        except ImportError:
            summary_block_sparse_attn_func = None
        can_run_cute = torch.cuda.get_device_capability(q.device)[0] >= 9
        if summary_block_sparse_attn_func is not None and can_run_cute:
            text_output = summary_block_sparse_attn_func(
                q_text.transpose(1, 2).contiguous(),
                k_text.transpose(1, 2).contiguous(),
                v_text.transpose(1, 2).contiguous(),
                selected_block_indices.to(torch.int32).contiguous(),
                raw_block_size,
                local_window_tokens,
                softmax_scale=q.shape[-1] ** -0.5,
                deterministic=False,
            ).transpose(1, 2).contiguous()
        else:
            text_output = _summary_block_sparse_text_reference(
                q_text,
                k_text,
                v_text,
                selected_block_indices,
                raw_block_size=raw_block_size,
                local_window_tokens=local_window_tokens,
            )
    elif text_count <= local_window_tokens:
        text_output = F.scaled_dot_product_attention(
            q_text, k_text, v_text, is_causal=True
        )
    else:
        text_output = _summary_block_sparse_text_reference(
            q_text,
            k_text,
            v_text,
            selected_block_indices,
            raw_block_size=raw_block_size,
            local_window_tokens=local_window_tokens,
        )

    chunk_block = summary_chunk_size + summary_token_num
    complete_chunks = summary_positions.numel() // summary_token_num
    complete_prefix = complete_chunks * chunk_block
    expected_prefix = torch.cat((
        torch.zeros(summary_chunk_size, dtype=torch.bool, device=q.device),
        torch.ones(summary_token_num, dtype=torch.bool, device=q.device),
    )).repeat(complete_chunks)
    if (
        complete_prefix > q.shape[2]
        or not torch.equal(shared[:complete_prefix], expected_prefix)
        or torch.any(shared[complete_prefix:])
        or q.shape[2] - complete_prefix >= summary_chunk_size
    ):
        raise ValueError("summary-block attention requires the regular summary layout")
    summary_output = (
        _summary_chunk_local_attention(
            q[:, :, :complete_prefix],
            k[:, :, :complete_prefix],
            v[:, :, :complete_prefix],
            summary_chunk_size=summary_chunk_size,
            summary_token_num=summary_token_num,
        ).reshape(q.shape[0], q.shape[1], -1, q.shape[-1])
        if summary_positions.numel()
        else q.new_empty(q.shape[0], q.shape[1], 0, q.shape[-1])
    )
    output = torch.empty_like(q)
    output.index_copy_(2, text_positions, text_output)
    if summary_positions.numel():
        output.index_copy_(2, summary_positions, summary_output)
    return output




def gather_summary_block_decode_kv(
    key: torch.Tensor,
    value: torch.Tensor,
    selected_block_indices: torch.Tensor,
    *,
    raw_block_size: int,
    local_window_tokens: int,
    local_alignment_tokens: int | None = None,
    workspace: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Expand summary block IDs into raw K/V for one decode query."""
    if key.shape != value.shape or key.ndim != 4:
        raise ValueError("decode K/V must share [B,H,L,D]")
    if selected_block_indices.ndim != 3 or selected_block_indices.shape[1] != 1:
        raise ValueError("decode selected blocks must have [B,1,K] layout")
    if workspace is not None:
        return fused_summary_decode_gather(
            key, value, selected_block_indices, raw_block_size=raw_block_size,
            local_window_tokens=local_window_tokens,
            local_alignment_tokens=local_alignment_tokens, workspace=workspace,
        )
    batch, heads, text_length, head_dim = key.shape
    if local_alignment_tokens is None:
        local_alignment_tokens = raw_block_size
    if local_alignment_tokens % raw_block_size:
        raise ValueError("local alignment must be divisible by route span")
    block_ids = selected_block_indices[:, 0].long()
    offsets = torch.arange(raw_block_size, device=key.device)
    remote = block_ids[..., None] * raw_block_size + offsets
    remote_valid = (block_ids[..., None] >= 0) & (remote < text_length)
    local_start = aligned_summary_block_local_start(
        text_length - 1, local_window_tokens, local_alignment_tokens
    )
    max_local_tokens = local_window_tokens + local_alignment_tokens - 1
    local = (
        local_start
        + torch.arange(max_local_tokens, device=key.device)
    )[None].expand(batch, -1)
    local_valid = (local >= 0) & (local < text_length)
    candidates = torch.cat((remote.flatten(1), local), dim=1)
    valid = torch.cat((remote_valid.flatten(1), local_valid), dim=1)
    # Router candidates end before the aligned local window, so no semantic
    # duplicate exists. Sentinel padding retains a fixed SDPA width across
    # decode steps.
    sentinel = text_length
    candidates = candidates.masked_fill(~valid, sentinel)
    safe = candidates.clamp_max(max(text_length - 1, 0))
    gather_index = safe[:, None, :, None].expand(
        batch, heads, safe.shape[1], head_dim
    )
    return (
        torch.gather(key, 2, gather_index),
        torch.gather(value, 2, gather_index),
        (candidates < sentinel)[:, None, None],
    )








_COMPILED_FLEX_ATTENTION = None






def _summary_chunk_local_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    summary_chunk_size: int,
    summary_token_num: int,
) -> torch.Tensor:
    """Exact causal Full-layer output for inserted summary queries only."""
    batch, heads, expanded_sequence, head_dim = q.shape
    chunk_block = summary_chunk_size + summary_token_num
    if expanded_sequence % chunk_block:
        raise ValueError("block-routed S-KSA requires a complete summary layout")
    chunk_count = expanded_sequence // chunk_block
    q_chunks = q.view(batch, heads, chunk_count, chunk_block, head_dim)
    k_chunks = k.view(batch, heads, chunk_count, chunk_block, head_dim)
    v_chunks = v.view(batch, heads, chunk_count, chunk_block, head_dim)
    q_summary = q_chunks[:, :, :, summary_chunk_size:]
    logits = torch.einsum(
        "bhntd,bhnsd->bhnts", q_summary, k_chunks
    ).float()
    logits.mul_(head_dim ** -0.5)
    query_positions = summary_chunk_size + torch.arange(
        summary_token_num, device=q.device
    )
    key_positions = torch.arange(chunk_block, device=q.device)
    causal = key_positions[None, :] <= query_positions[:, None]
    probability = torch.softmax(
        logits.masked_fill(~causal[None, None, None], float("-inf")), dim=-1
    )
    return torch.einsum(
        "bhnts,bhnsd->bhntd", probability.to(v.dtype), v_chunks
    )


def summary_chunk_attention_indexer_key_input(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    summary_chunk_size: int,
    summary_token_num: int,
) -> torch.Tensor:
    """Build Full-layer summary index entries from current-layer chunk context.

    Runtime Q/K/V use ``[B,H,S,D]``.  The returned tensor uses the Indexer's
    hidden-state contract ``[B,num_summary,H*D]`` and intentionally remains
    before the model output projection: the newly trained ``key_proj`` learns
    the inexpensive composition needed for routing.
    """
    summary_output = _summary_chunk_local_attention(
        q,
        k,
        v,
        summary_chunk_size=summary_chunk_size,
        summary_token_num=summary_token_num,
    )
    batch, heads, chunks, summary_count, head_dim = summary_output.shape
    return summary_output.permute(0, 2, 3, 1, 4).reshape(
        batch, chunks * summary_count, heads * head_dim
    )








__all__ = ['SummaryBlockLightningIndexer', 'aligned_summary_block_local_start', 'gather_summary_block_decode_kv', 'score_indexer_candidates', 'summary_block_sparse_prefill_attention', 'summary_chunk_attention_indexer_key_input']
