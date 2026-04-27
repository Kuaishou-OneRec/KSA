# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Optional, Literal

import torch
from torch import nn
from muse.layers.attention_utils import get_attention_function
from muse.layers.feed_forward import FeedForward
from muse.layers.kv_cache import KVCache
from muse.utils.common import print_rank_0

from muse.training.parallel import get_context_parallel_world_size, \
    get_context_parallel_group, SeqAllToAll4D

logger = logging.getLogger(__name__)


def _format_window_label(window_size: Optional[int]) -> str:
    if window_size is None or window_size <= 0:
        return "full"
    return str(int(window_size))


def _log_runtime_attention_once(
    module: nn.Module,
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor],
    is_causal: bool,
    runtime_window_size: Optional[int],
) -> None:
    if getattr(module, "_runtime_window_log_emitted", False):
        return

    module._runtime_window_log_emitted = True
    layer_index = getattr(module, "layer_index", -1)
    layer_label = f"L{layer_index:02d}" if layer_index >= 0 else "L??"
    backend_name = getattr(module, "attention_function_name", type(module._attention_function).__name__)
    configured_window = getattr(module, "sliding_window", -1)
    cp_size = get_context_parallel_world_size()
    mask_shape = tuple(mask.shape) if isinstance(mask, torch.Tensor) else None

    print_rank_0(
        "[SWA runtime] "
        f"layer={layer_label} "
        f"backend={backend_name} "
        f"configured_window={_format_window_label(configured_window)} "
        f"runtime_window={_format_window_label(runtime_window_size)} "
        f"is_causal={is_causal} "
        f"cp_size={cp_size} "
        f"q={tuple(q.shape)} "
        f"k={tuple(k.shape)} "
        f"v={tuple(v.shape)} "
        f"mask={mask_shape}"
    )

def qwen3_mlp(dim: int, hidden_dim: int) -> FeedForward:
    """
    Build the MLP layer associated with the Qwen2 model.
    """
    gate_proj = nn.Linear(dim, hidden_dim, bias=False)
    down_proj = nn.Linear(hidden_dim, dim, bias=False)
    up_proj = nn.Linear(dim, hidden_dim, bias=False)
    return FeedForward(gate_proj=gate_proj, down_proj=down_proj, up_proj=up_proj)

class Qwen3Attention(nn.Module):
    """
    Basically, it is standard multihead attention, but with QK-norm applied before
    the RoPE. It is unusual for most of the models, but Qwen3 became an exception to the rule.

    Args:
        embed_dim (int): embedding dimension for the model
        num_heads (int): number of query heads. For MHA this is also the
            number of heads for key and value
        num_kv_heads (int): number of key and value heads. User should ensure
            ``num_heads % num_kv_heads == 0``. For standard MHA set ``num_kv_heads == num_heads``,
            for GQA ``num_kv_heads < num_heads``, and for MQA set ``num_kv_heads == 1``.
        head_dim (int): dimension of each head, calculated by ``embed_dim // num_heads``.
        q_proj (nn.Module): projection layer for query.
        k_proj (nn.Module): projection layer for key.
        v_proj (nn.Module): projection layer for value.
        output_proj (nn.Module): projection layer for output.
        pos_embeddings (Optional[nn.Module]): positional embeddings layer, e.g. RotaryPositionalEmbeddings or LlamaRotaryPositionalEmbeddings.
        q_norm (Optional[nn.Module]): normalization layer for query, e.g. RMSNorm. For decoding, this is applied
            before updating from kv_cache. This means it will only support token wide normalization and not
            batch or sequence wide normalization.
        k_norm (Optional[nn.Module]): normalization layer for key, must be set if q_norm is.
        kv_cache (Optional[KVCache]): KVCache object used to cache key and value
        max_seq_len (int): maximum sequence length supported by the model.
            This is needed to compute the RoPE Cache. Default: 4096.
        is_causal (bool): sets the default mask to causal when no mask is provided
        attn_dropout (float): dropout value passed onto the scaled_dot_product_attention function.
            Default value is 0.0.
    Raises:
        ValueError:
            If ``num_heads % num_kv_heads != 0``, **or**
            if ``embed_dim % num_heads != 0``, **or**
            if ``attn_dropout < 0`` or ``attn_dropout > 1``, **or**
            if q_norm is defined without k_norm or vice versa
    """

    def __init__(
        self,
        *,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        q_proj: nn.Module,
        k_proj: nn.Module,
        v_proj: nn.Module,
        output_proj: nn.Module,
        pos_embeddings: Optional[nn.Module] = None,
        q_norm: Optional[nn.Module] = None,
        k_norm: Optional[nn.Module] = None,
        kv_cache: Optional[KVCache] = None,
        max_seq_len: int = 4096,
        is_causal: bool = True,
        attn_dropout: float = 0.0,
        attention_function: Literal["eager", "flash_attention_2"] = "eager",
        sliding_window: int = -1,
        layer_index: int = -1,
    ) -> None:
        super().__init__()
        if num_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_heads ({num_heads}) must be divisible by "
                f"num_kv_heads ({num_kv_heads})"
            )

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )

        if attn_dropout < 0 or attn_dropout > 1:
            raise ValueError(f"attn_dropout ({embed_dim}) must be between 0.0 and 1.0")

        if bool(q_norm) ^ bool(k_norm):
            raise ValueError("q and k norm must be set together")

        # Set attributes
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.embed_dim = embed_dim
        self.attn_dropout = attn_dropout
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.is_causal = is_causal
        self.sliding_window = sliding_window if sliding_window is not None and sliding_window > 0 else -1
        self.layer_index = layer_index

        # Set layers
        self.kv_cache = kv_cache
        self.q_proj = q_proj
        self.k_proj = k_proj
        self.v_proj = v_proj
        self.output_proj = output_proj
        self.q_norm = q_norm
        self.k_norm = k_norm
        self.pos_embeddings = pos_embeddings

        self._attention_function = get_attention_function(attention_function)
        self.attention_function_name = attention_function
        self._runtime_window_log_emitted = False

        # this flag indicates whether to update the kv-cache during forward
        # passes. when disabled, we can have the cache setup but still
        # perform normal forward passes
        self.cache_enabled = False

    def setup_cache(
        self, batch_size: int, dtype: torch.dtype, max_seq_len: int
    ) -> None:
        """Setup key value caches for attention calculation. If called
        after kv_cache is already setup, this will be skipped.
        Args:
            batch_size (int): batch size for the caches.
            dtype (torch.dtype): dtype for the caches.
            max_seq_len (int): maximum sequence length model will be run with.
        """
        # Don't overwrite user defined kv_cache from init
        if self.kv_cache is not None:
            logger.warning(
                "Key value caches are already setup. You cannot call ``setup_caches()`` twice. Skipping."
            )
        else:
            self.kv_cache = KVCache(
                batch_size=batch_size,
                max_seq_len=max_seq_len,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                dtype=dtype,
            )
            self.cache_enabled = True

    def reset_cache(self):
        """Reset the key value caches."""
        if self.kv_cache is None:
            raise RuntimeError(
                "Key value caches are not setup. Call ``setup_caches()`` first."
            )
        self.kv_cache.reset()

    def forward(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        *,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor with shape [b x s_x x d] for the query
            y (Optional[torch.Tensor]): second input tensor with shape [b x s_y x d], is the input
                for k and v. For self attention, x=y. Optional only with kv_cache enabled.
            mask (Optional[_MaskType]): Used to mask the scores after the query-key multiplication
                and before the softmax. Either:
                A boolean tensor with shape ``[b x s x s]``, ``[b x s x self.encoder_max_cache_seq_len]``,
                or ``[b x s x self.decoder_max_cache_seq_len]`` if using KV-cacheing with encoder/decoder layers.
                A value of True in row ``i`` and column ``j`` means token ``i`` attends to token ``j``. A value of False means
                token ``i`` does not attend to token ``j``. If no mask is specified, a causal mask
                is used by default.
                A :class:`~torch.nn.attention.flex_attention.BlockMask` for document masking in a packed sequence
                created via `create_block_mask <https://pytorch.org/blog/flexattention/#mask-mods>`_. We  use
                :func:`~torch.nn.attention.flex_attention.flex_attention` when computing attention with block masks.
                Default is None.
            input_pos (Optional[torch.Tensor]): Optional tensor which contains the position ids
                of each token. During training, this is used to indicate the positions
                of each token relative to its sample when packed, shape [b x s].
                During inference, this indicates the position of the current token.
                If none, assume the index of the token is its position id. Default is None.
        Raises:
            ValueError: If no ``y`` input and ``kv_cache`` is not enabled.
        Returns:
            torch.Tensor: output tensor with attention applied
        Notation used for tensor shapes:
            - b: batch size
            - s_x: sequence length for x
            - s_y: sequence length for y
            - n_h: num heads
            - n_kv: num kv heads
            - d: embed dim
            - h_d: head dim
        """
        # x has shape [b, s_x, d]
        # y has shape [b, s_y, d]
        b, s_x, _ = x.shape
        s_y = y.shape[1] if y is not None else 0

        # q has shape [b, s_x, num_heads * head_dim]
        q = self.q_proj(x)

        # number of queries per key/value
        q_per_kv = self.num_heads // self.num_kv_heads
        q = q.view(b, s_x, self.num_kv_heads * q_per_kv, self.head_dim)

        # Qwen3 applies QK-norm before the RoPE, which is different from most of the models.
        # Normalize q
        if self.q_norm is not None:
            q = self.q_norm(q)

        # Apply positional embeddings after q-norm
        if self.pos_embeddings is not None:
            q = self.pos_embeddings(q, input_pos=input_pos)

        # print(f"self.cache_enabled={self.cache_enabled}", f"attention={self._attention_function}")
        if y is None:
            if self.kv_cache is None or not self.cache_enabled:
                raise ValueError(
                    "Must provide y input or use kv_cache to enable streaming decoding"
                )
            k = self.kv_cache.k_cache
            v = self.kv_cache.v_cache
        else:
            # Update k and v shape, positional embeddings, and normalization
            # k,v shape [b, s_y, num_kv_heads * head_dim]
            k = self.k_proj(y)
            v = self.v_proj(y)

            # Apply positional embeddings
            # k,v shape: [b, s_y, n_kv, h_d]
            k = k.view(b, s_y, -1, self.head_dim)
            v = v.view(b, s_y, -1, self.head_dim)

            # Normalize k
            if self.k_norm is not None:
                k = self.k_norm(k)

            # Apply positional embeddings after k-norm
            if self.pos_embeddings is not None:
                k = self.pos_embeddings(k, input_pos=input_pos)

            # Update key-value cache
            if self.kv_cache is not None and self.cache_enabled:
                
                k, v = self.kv_cache.update(k, v)


        if get_context_parallel_world_size() > 1:
            cpg = get_context_parallel_group()
            # q: [b, s_x, n_h, h_d], k/v: [b, s_x, n_kv, h_d]
            # -> q: [b, s_x * P, n_h // P, h_d], k/v: [b, s_x * P, n_kv // P, h_d]
            q = SeqAllToAll4D.apply(cpg, q, 2, 1)
            k = SeqAllToAll4D.apply(cpg, k, 2, 1)  # n_kv heads, saves 4x vs expanded
            v = SeqAllToAll4D.apply(cpg, v, 2, 1)  # n_kv heads, saves 4x vs expanded

        # GQA expand AFTER all-to-all (local operation, zero communication)
        # k,v shape: [b, s, n_kv(/P), h_d] -> [b, s, n_h(/P), h_d]
        if self.num_heads != self.num_kv_heads:
            q_per_kv = self.num_heads // self.num_kv_heads
            expand_shape = (b, -1, k.shape[2], q_per_kv, self.head_dim)
            k = k.unsqueeze(3).expand(expand_shape).flatten(2, 3)
            v = v.unsqueeze(3).expand(expand_shape).flatten(2, 3)

        # print(f"q={q.shape}, k={k.shape}, v={v.shape}, self._attention_function={self._attention_function}, kwargs={kwargs}")
        # print(f"self.kv cache is None={self.kv_cache is None}, mask is None={mask is None}, self.is_causal={self.is_causal}")
        is_causal = (self.kv_cache is not None) or (mask is None and self.is_causal)
        # is_causal = self.kv_cache is None and mask is None and self.is_causal
        # 1. self.kv_cache is not None的时候必须是is_causal
        # 2. self.is_causal=True的时候也必须是is_causal
        attn_kwargs = dict(kwargs)
        if "window_size" not in attn_kwargs and self.sliding_window > 0:
            attn_kwargs["window_size"] = self.sliding_window
        runtime_window_size = attn_kwargs.get("window_size", -1)
        _log_runtime_attention_once(
            self,
            q=q,
            k=k,
            v=v,
            mask=mask,
            is_causal=is_causal,
            runtime_window_size=runtime_window_size,
        )

        output = self._attention_function(
            q=q,
            k=k,
            v=v,
            mask=mask,
            attention_dropout=self.attn_dropout,
            is_causal=is_causal,
            training=self.training,
            **attn_kwargs,
        )
        if get_context_parallel_world_size() > 1:
            cpg = get_context_parallel_group()
            # output: [b, s_x * P, n_h // P, h_d] -> [b, s_x, n_h, h_d]
            output = SeqAllToAll4D.apply(cpg, output, 1, 2)
        # reshape the output to be the same shape as the input
        output = output.contiguous().view(b, s_x, -1)
        return self.output_proj(output)
