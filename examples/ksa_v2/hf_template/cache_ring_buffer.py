# coding=utf-8
# Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import Any, Callable, Optional, Union
import torch
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from .runtime_config import _get_shared_batch_mask, _parse_config_pattern
from .layer_norm_rope import rotate_half


class Qwen3RingBufferCache:
    """
    Ring buffer KV cache with summary support.

    Two strategies based on per-layer sliding_chunk_num:
    - Large window layers (is_large_window=True): append-only buffer storing only text KV.
      Summary KV is NOT stored since text tokens attend to all text KV directly.
    - Small window layers (is_large_window=False):
      Three buffers:
        1. key_cache: [ring(ws) | old_summaries(growing) | chunk_mirror(≤C)]
           → attention input, steady state is a single contiguous slice
        2. new_summary_buf: ring buffer of size scn, stores summaries whose text
           is still in the window (not needed for attention)
        3. chunk_buf: size C, holds current chunk's text KV

    RoPE position information is baked into KV, so physical order doesn't matter.
    """

    is_compileable = False
    _SUMMARY_INIT_CAP = 512
    _APPEND_HEADROOM = 1024

    def __init__(self, config: Qwen3Config, sliding_chunk_nums: list[int]):
        super().__init__()
        self.summary_chunk_size = getattr(config, "summary_chunk_size", 0)
        self.summary_token_num = getattr(config, "summary_token_num", 0)
        self.num_hidden_layers = config.num_hidden_layers

        self.sliding_chunk_nums = sliding_chunk_nums
        large_window_threshold = min(sliding_chunk_nums) * self.summary_chunk_size
        self.is_large_window = [sv * self.summary_chunk_size > large_window_threshold for sv in sliding_chunk_nums]
        self.window_sizes = [sv * self.summary_chunk_size for sv in sliding_chunk_nums]
        pool_mode = getattr(config, "summary_meta_pool_mode", "none") or "none"
        pool_groups = _parse_config_pattern(
            getattr(config, "summary_meta_pool_group_size", 0) or 0
        )
        if not isinstance(pool_groups, list):
            pool_groups = [int(pool_groups)] * config.num_hidden_layers
        self.meta_pool_group_sizes = [int(v) for v in pool_groups]
        self.meta_pool_enabled = [
            pool_mode in ("mean", "scalar", "dimension_wise")
            and group_size > 1
            for group_size in self.meta_pool_group_sizes
        ]
        self.meta_pool_mode = pool_mode

        self.key_cache = [None for _ in range(config.num_hidden_layers)]
        self.value_cache = [None for _ in range(config.num_hidden_layers)]
        # Full-layer Lightning Indexer keys are not main-attention K/V.  The
        # summary-indexer variant stores one key per raw chunk; token-indexer
        # mode stores one key per raw text token.
        self._sksa_index_keys = [None for _ in range(config.num_hidden_layers)]
        self._sparse_gather_storage = {}
        # Independent chunk-Q/K mode needs the auxiliary K states for the
        # currently accumulating raw chunk.  They are separate from the
        # backbone attention cache and are discarded once its summary token is
        # emitted.
        self._sksa_chunk_k = [None for _ in range(config.num_hidden_layers)]
        self._sksa_chunk_v = [None for _ in range(config.num_hidden_layers)]

        # Large window: append-only
        self._text_len = [0] * config.num_hidden_layers
        self._capacity = [0] * config.num_hidden_layers

        # Small window: ring buffer + summary
        self._window_write_ptr = [0] * config.num_hidden_layers
        self._n_valid_window = [0] * config.num_hidden_layers
        self._old_summary_len = [0] * config.num_hidden_layers   # old summaries in key_cache
        self._old_summary_cap = [0] * config.num_hidden_layers
        self._old_summary_total = [0] * config.num_hidden_layers

        # Hierarchical old-summary state. Complete groups are stored as post-RoPE
        # meta KV; fewer than group_size evicted summaries remain residual KV.
        self._meta_key_cache = [None for _ in range(config.num_hidden_layers)]
        self._meta_value_cache = [None for _ in range(config.num_hidden_layers)]
        self._meta_len = [0] * config.num_hidden_layers
        self._meta_cap = [0] * config.num_hidden_layers
        self._residual_pre_key_buf = [None for _ in range(config.num_hidden_layers)]
        self._residual_score_buf = [None for _ in range(config.num_hidden_layers)]
        self._residual_cos_buf = [None for _ in range(config.num_hidden_layers)]
        self._residual_sin_buf = [None for _ in range(config.num_hidden_layers)]
        self._residual_position_buf = [None for _ in range(config.num_hidden_layers)]

        # New summary ring buffer (small window only): summaries whose text is still in window
        self._new_sum_key_buf = [None for _ in range(config.num_hidden_layers)]
        self._new_sum_value_buf = [None for _ in range(config.num_hidden_layers)]
        self._new_sum_pre_key_buf = [None for _ in range(config.num_hidden_layers)]
        self._new_sum_score_buf = [None for _ in range(config.num_hidden_layers)]
        self._new_sum_cos_buf = [None for _ in range(config.num_hidden_layers)]
        self._new_sum_sin_buf = [None for _ in range(config.num_hidden_layers)]
        self._new_sum_position_buf = [None for _ in range(config.num_hidden_layers)]
        self._new_sum_len = [0] * config.num_hidden_layers       # how many filled (≤ scn)
        self._new_sum_write_ptr = [0] * config.num_hidden_layers  # ring write pointer

        # Mirror region length (small window only): partial chunk text KV count in key_cache
        self._chunk_buf_len = [0] * config.num_hidden_layers

        # Common
        self.cur_chunk_sizes = [0] * config.num_hidden_layers
        self.true_tokens = [0] * config.num_hidden_layers
        self._total_chunks = [0] * config.num_hidden_layers  # completed chunks count
        self._reorganized = [False] * config.num_hidden_layers
        self._prefill_text_positions = None
        self._prefill_summary_positions = None




    def __len__(self):
        return self.num_hidden_layers

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns nonzero only when this layer's cache is in post-reorganize state."""
        if layer_idx >= self.num_hidden_layers:
            return 0
        if not self._reorganized[layer_idx]:
            return 0
        if self.is_large_window[layer_idx]:
            return self._text_len[layer_idx]
        else:
            return (self._n_valid_window[layer_idx] + self._chunk_buf_len[layer_idx]
                    + self._old_summary_len[layer_idx] + self._new_sum_len[layer_idx])

    def get_cur_chunk_size(self, layer_idx: Optional[int] = None) -> int:
        if layer_idx is None:
            layer_idx = self.num_hidden_layers - 1
        return self.cur_chunk_sizes[layer_idx]

    def get_true_token_num(self, layer_idx: Optional[int] = None) -> int:
        if layer_idx is None:
            layer_idx = self.num_hidden_layers - 1
        return self.true_tokens[layer_idx]

    def set_sksa_index_keys(
        self, layer_idx: int, index_keys: torch.Tensor
    ) -> None:
        if index_keys.ndim != 3:
            raise ValueError("S-KSA index keys must use [B,M,D] layout")
        self._sksa_index_keys[layer_idx] = index_keys.contiguous()

    def append_sksa_index_keys(
        self, layer_idx: int, index_keys: torch.Tensor
    ) -> None:
        if index_keys.ndim != 3:
            raise ValueError("S-KSA index keys must use [B,M,D] layout")
        current = self._sksa_index_keys[layer_idx]
        if current is None:
            self._sksa_index_keys[layer_idx] = index_keys.contiguous()
        else:
            if current.shape[0] != index_keys.shape[0] or current.shape[2] != index_keys.shape[2]:
                raise ValueError("S-KSA index key cache geometry changed during decode")
            self._sksa_index_keys[layer_idx] = torch.cat(
                (current, index_keys), dim=1
            )

    def get_sksa_index_keys(self, layer_idx: int) -> torch.Tensor:
        index_keys = self._sksa_index_keys[layer_idx]
        if index_keys is None:
            raise RuntimeError(f"S-KSA index key cache for layer {layer_idx} is uninitialized")
        return index_keys

    def set_sksa_chunk_k_from_prefill(
        self,
        layer_idx: int,
        chunk_key_states: Optional[torch.Tensor],
        summary_mask: torch.Tensor,
    ) -> None:
        """Retain auxiliary K for the incomplete raw chunk after prefill."""
        if chunk_key_states is None:
            self._sksa_chunk_k[layer_idx] = None
            return
        if chunk_key_states.ndim != 4:
            raise ValueError("S-KSA chunk K must use [B,H,S,D] layout")
        shared = _get_shared_batch_mask(summary_mask, "summary_mask")
        positions = self._prefill_text_positions
        text_key = (chunk_key_states.index_select(2, positions) if positions is not None
                    else chunk_key_states[:, :, ~shared, :])
        partial = text_key.shape[2] % self.summary_chunk_size
        if partial:
            self._sksa_chunk_k[layer_idx] = text_key[:, :, -partial:, :].contiguous()
        else:
            self._sksa_chunk_k[layer_idx] = None

    def append_sksa_chunk_k(
        self, layer_idx: int, chunk_key_states: Optional[torch.Tensor]
    ) -> None:
        """Append one decoded raw token's independent auxiliary K state."""
        if chunk_key_states is None:
            return
        if chunk_key_states.ndim != 4 or chunk_key_states.shape[2] != 1:
            raise ValueError("decoded S-KSA chunk K must use [B,H,1,D] layout")
        current = self._sksa_chunk_k[layer_idx]
        self._sksa_chunk_k[layer_idx] = (
            chunk_key_states.contiguous()
            if current is None
            else torch.cat((current, chunk_key_states), dim=2).contiguous()
        )

    def get_current_sksa_chunk_k(self, layer_idx: int) -> Optional[torch.Tensor]:
        return self._sksa_chunk_k[layer_idx]

    def clear_sksa_chunk_k(self, layer_idx: int) -> None:
        self._sksa_chunk_k[layer_idx] = None

    def set_sksa_chunk_v_from_prefill(
        self,
        layer_idx: int,
        chunk_value_states: Optional[torch.Tensor],
        summary_mask: torch.Tensor,
    ) -> None:
        """Retain auxiliary V for the incomplete raw chunk after prefill."""
        if chunk_value_states is None:
            self._sksa_chunk_v[layer_idx] = None
            return
        if chunk_value_states.ndim != 4:
            raise ValueError("S-KSA chunk V must use [B,H,S,D] layout")
        shared = _get_shared_batch_mask(summary_mask, "summary_mask")
        positions = self._prefill_text_positions
        text_value = (chunk_value_states.index_select(2, positions) if positions is not None
                      else chunk_value_states[:, :, ~shared, :])
        partial = text_value.shape[2] % self.summary_chunk_size
        if partial:
            self._sksa_chunk_v[layer_idx] = text_value[:, :, -partial:, :].contiguous()
        else:
            self._sksa_chunk_v[layer_idx] = None

    def append_sksa_chunk_v(
        self, layer_idx: int, chunk_value_states: Optional[torch.Tensor]
    ) -> None:
        """Append one decoded raw token's independent auxiliary V state."""
        if chunk_value_states is None:
            return
        if chunk_value_states.ndim != 4 or chunk_value_states.shape[2] != 1:
            raise ValueError("decoded S-KSA chunk V must use [B,H,1,D] layout")
        current = self._sksa_chunk_v[layer_idx]
        self._sksa_chunk_v[layer_idx] = (
            chunk_value_states.contiguous()
            if current is None
            else torch.cat((current, chunk_value_states), dim=2).contiguous()
        )

    def get_current_sksa_chunk_v(self, layer_idx: int) -> Optional[torch.Tensor]:
        return self._sksa_chunk_v[layer_idx]

    def clear_sksa_chunk_v(self, layer_idx: int) -> None:
        self._sksa_chunk_v[layer_idx] = None

    # ── Prefill: per-layer one-shot init into ring buffer layout ──

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Deprecated for prefill — use init_layer_from_prefill instead.

        Kept only to surface accidental calls; the prefill path now skips this
        intermediate full-length buffer entirely.
        """
        raise NotImplementedError(
            "Qwen3RingBufferCache.update() is no longer used; prefill goes through "
            "init_layer_from_prefill, decode goes through update_text/update_summary."
        )

    def prepare_prefill(self, text_positions, summary_positions):
        self._prefill_text_positions = text_positions
        self._prefill_summary_positions = summary_positions

    def init_layer_from_prefill(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        summary_mask: torch.Tensor,
        layer_idx: int,
        *,
        pre_rope_key_states: Optional[torch.Tensor] = None,
        summary_scores: Optional[torch.Tensor] = None,
        summary_cos: Optional[torch.Tensor] = None,
        summary_sin: Optional[torch.Tensor] = None,
        summary_position_ids: Optional[torch.Tensor] = None,
        meta_key_states: Optional[torch.Tensor] = None,
        meta_value_states: Optional[torch.Tensor] = None,
        sksa_chunk_key_states: Optional[torch.Tensor] = None,
    ):
        """Allocate this layer's cache directly into the post-reorganize ring buffer
        layout from prefill K/V (already projected + RoPE applied), with NO intermediate
        full-length buffer. Replaces the old update() + reorganize_after_prefill() pair
        for the prefill path.

        key_states / value_states: [bsz, kv_heads, prefill_len_expanded, head_dim]
        summary_mask:               [bsz, prefill_len_expanded] (bool, batch-uniform)
        """
        assert not self._reorganized[layer_idx], f"layer {layer_idx} already reorganized"

        prefill_k = key_states
        prefill_v = value_states
        bsz, heads, prefill_len, head_dim = prefill_k.shape
        device, dtype = prefill_k.device, prefill_k.dtype

        sm = _get_shared_batch_mask(summary_mask, "summary_mask")
        text_mask = ~sm
        text_positions = self._prefill_text_positions
        summary_positions = self._prefill_summary_positions
        if text_positions is None:
            text_positions = torch.nonzero(text_mask, as_tuple=False).flatten()
            summary_positions = torch.nonzero(sm, as_tuple=False).flatten()
        n_text = text_positions.numel()

        block = self.summary_chunk_size + self.summary_token_num
        # Match what update() + reorganize_after_prefill() would have set:
        # update(): cur_chunk_sizes += prefill_len; true_tokens += n_text
        # reorganize: cur_chunk_sizes %= block
        self.cur_chunk_sizes[layer_idx] = prefill_len % block
        self.true_tokens[layer_idx] = n_text

        if self.is_large_window[layer_idx]:
            text_k = prefill_k.index_select(2, text_positions)
            text_v = prefill_v.index_select(2, text_positions)
            cap = n_text + self._APPEND_HEADROOM
            new_k = torch.empty(bsz, heads, cap, head_dim, dtype=dtype, device=device)
            new_v = torch.empty(bsz, heads, cap, head_dim, dtype=dtype, device=device)
            if n_text > 0:
                new_k[:, :, :n_text, :].copy_(text_k)
                new_v[:, :, :n_text, :].copy_(text_v)
            self.key_cache[layer_idx] = new_k
            self.value_cache[layer_idx] = new_v
            self._text_len[layer_idx] = n_text
            self._capacity[layer_idx] = cap
            self._total_chunks[layer_idx] = n_text // self.summary_chunk_size
        else:
            all_summary_k = prefill_k.index_select(2, summary_positions)
            all_summary_v = prefill_v.index_select(2, summary_positions)
            n_summary = all_summary_k.shape[2]

            C = self.summary_chunk_size
            ws = self.window_sizes[layer_idx]
            scn = self.sliding_chunk_nums[layer_idx]

            n_complete_chunks = n_text // C
            n_partial = n_text % C
            n_complete_text = n_complete_chunks * C

            n_window_chunks = min(scn, n_complete_chunks)
            n_window_text = n_window_chunks * C
            window_start = n_complete_text - n_window_text
            # Only retain the live window and partial chunk, avoiding a
            # full-context text KV temporary on compressed layers.
            live_positions = text_positions[window_start:]
            text_k = prefill_k.index_select(2, live_positions)
            text_v = prefill_v.index_select(2, live_positions)

            n_old_total = max(0, n_summary - n_window_chunks)
            n_new = n_summary - n_old_total
            hierarchical = self.meta_pool_enabled[layer_idx]
            group_size = self.meta_pool_group_sizes[layer_idx]
            n_meta = n_old_total // group_size if hierarchical else 0
            n_old = n_old_total % group_size if hierarchical else n_old_total
            old_begin = n_meta * group_size if hierarchical else 0

            old_s_cap = max(self._SUMMARY_INIT_CAP, (n_old + 1) * 2)
            total_cap = ws + old_s_cap + C
            new_k = torch.empty(bsz, heads, total_cap, head_dim, dtype=dtype, device=device)
            new_v = torch.empty(bsz, heads, total_cap, head_dim, dtype=dtype, device=device)

            if n_window_text > 0:
                new_k[:, :, :n_window_text, :].copy_(text_k[:, :, :n_window_text, :])
                new_v[:, :, :n_window_text, :].copy_(text_v[:, :, :n_window_text, :])
            self._n_valid_window[layer_idx] = n_window_text
            self._window_write_ptr[layer_idx] = n_window_text % ws

            if n_old > 0:
                new_k[:, :, ws:ws + n_old, :].copy_(
                    all_summary_k[:, :, old_begin:n_old_total, :]
                )
                new_v[:, :, ws:ws + n_old, :].copy_(
                    all_summary_v[:, :, old_begin:n_old_total, :]
                )
            self._old_summary_len[layer_idx] = n_old
            self._old_summary_total[layer_idx] = n_old_total
            self._old_summary_cap[layer_idx] = old_s_cap

            if hierarchical:
                required = (
                    pre_rope_key_states,
                    summary_scores,
                    summary_cos,
                    summary_sin,
                    summary_position_ids,
                    meta_key_states,
                    meta_value_states,
                )
                if any(item is None for item in required):
                    raise ValueError("hierarchical prefill cache requires pre-RoPE and meta state")
                all_pre_k = pre_rope_key_states.index_select(2, summary_positions)
                meta_cap = max(self._SUMMARY_INIT_CAP, (n_meta + 1) * 2)
                meta_k_buf = torch.empty(
                    bsz, heads, meta_cap, head_dim, dtype=dtype, device=device
                )
                meta_v_buf = torch.empty_like(meta_k_buf)
                if n_meta:
                    meta_k_buf[:, :, :n_meta, :].copy_(meta_key_states[:, :, :n_meta, :])
                    meta_v_buf[:, :, :n_meta, :].copy_(meta_value_states[:, :, :n_meta, :])
                self._meta_key_cache[layer_idx] = meta_k_buf
                self._meta_value_cache[layer_idx] = meta_v_buf
                self._meta_len[layer_idx] = n_meta
                self._meta_cap[layer_idx] = meta_cap

                residual_pre = torch.empty(
                    bsz, heads, group_size, head_dim, dtype=dtype, device=device
                )
                score_width = (
                    heads * head_dim
                    if self.meta_pool_mode == "dimension_wise"
                    else 1
                )
                residual_score = torch.empty(
                    bsz,
                    group_size,
                    score_width,
                    dtype=summary_scores.dtype,
                    device=device,
                )
                residual_cos = torch.empty(
                    bsz, group_size, head_dim, dtype=summary_cos.dtype, device=device
                )
                residual_sin = torch.empty_like(residual_cos)
                residual_position = torch.empty(
                    bsz, group_size, dtype=torch.long, device=device
                )
                if n_old:
                    residual_pre[:, :, :n_old, :].copy_(
                        all_pre_k[:, :, old_begin:n_old_total, :]
                    )
                    residual_score[:, :n_old, :].copy_(
                        summary_scores[:, old_begin:n_old_total]
                    )
                    residual_cos[:, :n_old, :].copy_(
                        summary_cos[:, old_begin:n_old_total, :]
                    )
                    residual_sin[:, :n_old, :].copy_(
                        summary_sin[:, old_begin:n_old_total, :]
                    )
                    residual_position[:, :n_old].copy_(
                        summary_position_ids[:, old_begin:n_old_total]
                    )
                self._residual_pre_key_buf[layer_idx] = residual_pre
                self._residual_score_buf[layer_idx] = residual_score
                self._residual_cos_buf[layer_idx] = residual_cos
                self._residual_sin_buf[layer_idx] = residual_sin
                self._residual_position_buf[layer_idx] = residual_position

            if n_partial > 0:
                mirror_start = ws + n_old
                new_k[:, :, mirror_start:mirror_start + n_partial, :].copy_(
                    text_k[:, :, n_window_text:, :])
                new_v[:, :, mirror_start:mirror_start + n_partial, :].copy_(
                    text_v[:, :, n_window_text:, :])

            self.key_cache[layer_idx] = new_k
            self.value_cache[layer_idx] = new_v
            self._capacity[layer_idx] = total_cap
            self._text_len[layer_idx] = 0

            ns_buf_k = torch.empty(bsz, heads, scn, head_dim, dtype=dtype, device=device)
            ns_buf_v = torch.empty(bsz, heads, scn, head_dim, dtype=dtype, device=device)
            if n_new > 0:
                ns_buf_k[:, :, :n_new, :].copy_(all_summary_k[:, :, n_old_total:, :])
                ns_buf_v[:, :, :n_new, :].copy_(all_summary_v[:, :, n_old_total:, :])
            self._new_sum_key_buf[layer_idx] = ns_buf_k
            self._new_sum_value_buf[layer_idx] = ns_buf_v
            if hierarchical:
                ns_pre = torch.empty(bsz, heads, scn, head_dim, dtype=dtype, device=device)
                score_width = (
                    heads * head_dim
                    if self.meta_pool_mode == "dimension_wise"
                    else 1
                )
                ns_score = torch.empty(
                    bsz,
                    scn,
                    score_width,
                    dtype=summary_scores.dtype,
                    device=device,
                )
                ns_cos = torch.empty(bsz, scn, head_dim, dtype=summary_cos.dtype, device=device)
                ns_sin = torch.empty_like(ns_cos)
                ns_position = torch.empty(
                    bsz, scn, dtype=torch.long, device=device
                )
                if n_new:
                    ns_pre[:, :, :n_new, :].copy_(all_pre_k[:, :, n_old_total:, :])
                    ns_score[:, :n_new, :].copy_(summary_scores[:, n_old_total:])
                    ns_cos[:, :n_new, :].copy_(summary_cos[:, n_old_total:, :])
                    ns_sin[:, :n_new, :].copy_(summary_sin[:, n_old_total:, :])
                    ns_position[:, :n_new].copy_(
                        summary_position_ids[:, n_old_total:]
                    )
                self._new_sum_pre_key_buf[layer_idx] = ns_pre
                self._new_sum_score_buf[layer_idx] = ns_score
                self._new_sum_cos_buf[layer_idx] = ns_cos
                self._new_sum_sin_buf[layer_idx] = ns_sin
                self._new_sum_position_buf[layer_idx] = ns_position
            self._new_sum_len[layer_idx] = n_new
            self._new_sum_write_ptr[layer_idx] = n_new % scn

            self._chunk_buf_len[layer_idx] = n_partial
            self._total_chunks[layer_idx] = n_old_total + n_new

        self._reorganized[layer_idx] = True
        self.set_sksa_chunk_k_from_prefill(
            layer_idx, sksa_chunk_key_states, summary_mask
        )

    # ── Reorganize after prefill ──

    def reorganize_after_prefill(self, summary_mask: torch.Tensor):
        """No-op sanity check.

        Prefill path now writes each layer directly into the ring buffer layout via
        init_layer_from_prefill, so by the time Qwen3Model.forward calls this,
        all layers must already be reorganized.
        """
        for layer_idx in range(self.num_hidden_layers):
            assert self._reorganized[layer_idx], (
                f"layer {layer_idx} was not initialized via init_layer_from_prefill"
            )

        self._prefill_text_positions = None
        self._prefill_summary_positions = None

    # ── Decode: text token update ──

    def update_text(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        *,
        sksa_chunk_key_states: Optional[torch.Tensor] = None,
    ):
        """Write a single text token KV during decode."""
        if self.is_large_window[layer_idx]:
            cur = self._text_len[layer_idx]
            new_len = cur + 1
            if new_len > self._capacity[layer_idx]:
                cap = max(new_len + self._APPEND_HEADROOM, self._capacity[layer_idx] * 2)
                old_k, old_v = self.key_cache[layer_idx], self.value_cache[layer_idx]
                bsz, heads, _, head_dim = old_k.shape
                new_k = torch.empty(bsz, heads, cap, head_dim, dtype=old_k.dtype, device=old_k.device)
                new_v = torch.empty(bsz, heads, cap, head_dim, dtype=old_v.dtype, device=old_v.device)
                new_k[:, :, :cur, :].copy_(old_k[:, :, :cur, :])
                new_v[:, :, :cur, :].copy_(old_v[:, :, :cur, :])
                self.key_cache[layer_idx] = new_k
                self.value_cache[layer_idx] = new_v
                self._capacity[layer_idx] = cap
            self.key_cache[layer_idx][:, :, cur:new_len, :].copy_(key_states)
            self.value_cache[layer_idx][:, :, cur:new_len, :].copy_(value_states)
            self._text_len[layer_idx] = new_len
        else:
            # Write only to key_cache mirror region (chunk_buf eliminated)
            ws = self.window_sizes[layer_idx]
            n_old = self._old_summary_len[layer_idx]
            pos = self._chunk_buf_len[layer_idx]
            mirror_pos = ws + n_old + pos
            self.key_cache[layer_idx][:, :, mirror_pos:mirror_pos+1, :].copy_(key_states)
            self.value_cache[layer_idx][:, :, mirror_pos:mirror_pos+1, :].copy_(value_states)
            self._chunk_buf_len[layer_idx] = pos + 1

        self.cur_chunk_sizes[layer_idx] += 1
        self.true_tokens[layer_idx] += 1
        self.append_sksa_chunk_k(layer_idx, sksa_chunk_key_states)

    # ── Decode: summary token update ──

    def update_summary(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        *,
        pre_rope_key_states: Optional[torch.Tensor] = None,
        summary_scores: Optional[torch.Tensor] = None,
        summary_cos: Optional[torch.Tensor] = None,
        summary_sin: Optional[torch.Tensor] = None,
        summary_position_ids: Optional[torch.Tensor] = None,
        position_bias: Optional[torch.Tensor] = None,
    ):
        """Write summary token KV during decode (chunk boundary).

        Large window: skip.
        Small window (order matters — flush mirror before evict to avoid clobbering):
          1. Flush mirror region → ring
          2. Evict oldest new_summary → old_summary in key_cache (if full)
          3. Write new summary → new_summary_buf
        """
        n_summary = key_states.shape[2]

        if self.is_large_window[layer_idx]:
            self.cur_chunk_sizes[layer_idx] += n_summary
            self._total_chunks[layer_idx] += n_summary
            return

        C = self.summary_chunk_size
        ws = self.window_sizes[layer_idx]
        scn = self.sliding_chunk_nums[layer_idx]
        hierarchical = self.meta_pool_enabled[layer_idx]
        group_size = self.meta_pool_group_sizes[layer_idx]
        cbl = self._chunk_buf_len[layer_idx]
        ptr = self._window_write_ptr[layer_idx]
        n_old = self._old_summary_len[layer_idx]

        # Step 1: Flush mirror region → ring (must happen before evict touches mirror[0])
        mirror_start = ws + n_old
        if ptr + cbl <= ws:
            self.key_cache[layer_idx][:, :, ptr:ptr + cbl, :].copy_(
                self.key_cache[layer_idx][:, :, mirror_start:mirror_start + cbl, :])
            self.value_cache[layer_idx][:, :, ptr:ptr + cbl, :].copy_(
                self.value_cache[layer_idx][:, :, mirror_start:mirror_start + cbl, :])
        else:
            first = ws - ptr
            self.key_cache[layer_idx][:, :, ptr:ws, :].copy_(
                self.key_cache[layer_idx][:, :, mirror_start:mirror_start + first, :])
            self.value_cache[layer_idx][:, :, ptr:ws, :].copy_(
                self.value_cache[layer_idx][:, :, mirror_start:mirror_start + first, :])
            rest = cbl - first
            self.key_cache[layer_idx][:, :, :rest, :].copy_(
                self.key_cache[layer_idx][:, :, mirror_start + first:mirror_start + cbl, :])
            self.value_cache[layer_idx][:, :, :rest, :].copy_(
                self.value_cache[layer_idx][:, :, mirror_start + first:mirror_start + cbl, :])

        self._window_write_ptr[layer_idx] = (ptr + cbl) % ws
        if self._n_valid_window[layer_idx] < ws:
            self._n_valid_window[layer_idx] = min(ws, self._n_valid_window[layer_idx] + cbl)
        self._chunk_buf_len[layer_idx] = 0

        # Step 2: Evict oldest new_summary → old_summary (now safe — mirror already flushed)
        if self._new_sum_len[layer_idx] >= scn:
            read_ptr = self._new_sum_write_ptr[layer_idx]
            old_dst = ws + n_old  # == mirror_start, but mirror data is already in ring

            # Check capacity for old_summary growth
            needed = old_dst + 1 + C  # +1 for new old_sum, +C for future chunk mirror
            if needed > self._capacity[layer_idx]:
                new_s_cap = max(self._old_summary_cap[layer_idx] * 2, n_old + self._SUMMARY_INIT_CAP)
                new_total = ws + new_s_cap + C
                old_k, old_v = self.key_cache[layer_idx], self.value_cache[layer_idx]
                bsz, heads, _, head_dim = old_k.shape
                nk = torch.empty(bsz, heads, new_total, head_dim, dtype=old_k.dtype, device=old_k.device)
                nv = torch.empty(bsz, heads, new_total, head_dim, dtype=old_v.dtype, device=old_v.device)
                copy_len = ws + n_old
                nk[:, :, :copy_len, :].copy_(old_k[:, :, :copy_len, :])
                nv[:, :, :copy_len, :].copy_(old_v[:, :, :copy_len, :])
                self.key_cache[layer_idx] = nk
                self.value_cache[layer_idx] = nv
                self._old_summary_cap[layer_idx] = new_s_cap
                self._capacity[layer_idx] = new_total

            self.key_cache[layer_idx][:, :, old_dst:old_dst+1, :].copy_(
                self._new_sum_key_buf[layer_idx][:, :, read_ptr:read_ptr+1, :])
            self.value_cache[layer_idx][:, :, old_dst:old_dst+1, :].copy_(
                self._new_sum_value_buf[layer_idx][:, :, read_ptr:read_ptr+1, :])
            n_old += 1
            self._old_summary_total[layer_idx] += 1

            if hierarchical:
                self._residual_pre_key_buf[layer_idx][:, :, n_old-1:n_old, :].copy_(
                    self._new_sum_pre_key_buf[layer_idx][:, :, read_ptr:read_ptr+1, :]
                )
                self._residual_score_buf[layer_idx][:, n_old-1:n_old, :].copy_(
                    self._new_sum_score_buf[layer_idx][:, read_ptr:read_ptr+1, :]
                )
                self._residual_cos_buf[layer_idx][:, n_old-1:n_old, :].copy_(
                    self._new_sum_cos_buf[layer_idx][:, read_ptr:read_ptr+1, :]
                )
                self._residual_sin_buf[layer_idx][:, n_old-1:n_old, :].copy_(
                    self._new_sum_sin_buf[layer_idx][:, read_ptr:read_ptr+1, :]
                )
                self._residual_position_buf[layer_idx][:, n_old-1:n_old].copy_(
                    self._new_sum_position_buf[layer_idx][
                        :, read_ptr:read_ptr+1
                    ]
                )

                if n_old == group_size:
                    scores = self._residual_score_buf[layer_idx][
                        :, :group_size, :
                    ]
                    if self.meta_pool_mode == "dimension_wise":
                        if position_bias is None:
                            weights = torch.full_like(scores, 1.0 / group_size)
                        else:
                            weights = torch.softmax(
                                (
                                    scores
                                    + position_bias.view(
                                        1, group_size, scores.shape[-1]
                                    )
                                ).float(),
                                dim=1,
                            ).to(key_states.dtype)
                        bsz, heads, _, head_dim = self._residual_pre_key_buf[
                            layer_idx
                        ].shape
                        weights = weights.view(
                            bsz, group_size, heads, head_dim
                        ).permute(0, 2, 1, 3)
                    else:
                        scalar_scores = scores.squeeze(-1)
                        if position_bias is None:
                            scalar_weights = torch.full_like(
                                scalar_scores, 1.0 / group_size
                            )
                        else:
                            scalar_weights = torch.softmax(
                                (
                                    scalar_scores
                                    + position_bias.view(1, group_size)
                                ).float(),
                                dim=1,
                            ).to(key_states.dtype)
                        weights = scalar_weights[:, None, :, None]
                    meta_pre_k = (
                        self._residual_pre_key_buf[layer_idx][:, :, :group_size, :]
                        * weights
                    ).sum(dim=2)
                    meta_v = (
                        self.value_cache[layer_idx][:, :, ws:ws + group_size, :]
                        * weights
                    ).sum(dim=2)
                    last_cos = self._residual_cos_buf[layer_idx][:, group_size - 1, :]
                    last_sin = self._residual_sin_buf[layer_idx][:, group_size - 1, :]
                    meta_k = (
                        meta_pre_k * last_cos[:, None, :]
                        + rotate_half(meta_pre_k) * last_sin[:, None, :]
                    )

                    meta_dst = self._meta_len[layer_idx]
                    if meta_dst >= self._meta_cap[layer_idx]:
                        old_meta_k = self._meta_key_cache[layer_idx]
                        old_meta_v = self._meta_value_cache[layer_idx]
                        new_cap = max(meta_dst + self._SUMMARY_INIT_CAP, self._meta_cap[layer_idx] * 2)
                        bsz, heads, _, head_dim = old_meta_k.shape
                        new_meta_k = torch.empty(
                            bsz, heads, new_cap, head_dim,
                            dtype=old_meta_k.dtype, device=old_meta_k.device,
                        )
                        new_meta_v = torch.empty_like(new_meta_k)
                        new_meta_k[:, :, :meta_dst, :].copy_(old_meta_k[:, :, :meta_dst, :])
                        new_meta_v[:, :, :meta_dst, :].copy_(old_meta_v[:, :, :meta_dst, :])
                        self._meta_key_cache[layer_idx] = new_meta_k
                        self._meta_value_cache[layer_idx] = new_meta_v
                        self._meta_cap[layer_idx] = new_cap
                    self._meta_key_cache[layer_idx][:, :, meta_dst:meta_dst+1, :].copy_(
                        meta_k.unsqueeze(2)
                    )
                    self._meta_value_cache[layer_idx][:, :, meta_dst:meta_dst+1, :].copy_(
                        meta_v.unsqueeze(2)
                    )
                    self._meta_len[layer_idx] += 1
                    n_old = 0

            self._old_summary_len[layer_idx] = n_old

        # Step 3: Write new summary to new_summary_buf (overwrite oldest slot)
        w_ptr = self._new_sum_write_ptr[layer_idx]
        self._new_sum_key_buf[layer_idx][:, :, w_ptr:w_ptr+1, :].copy_(key_states)
        self._new_sum_value_buf[layer_idx][:, :, w_ptr:w_ptr+1, :].copy_(value_states)
        if hierarchical:
            if any(
                item is None
                for item in (
                    pre_rope_key_states,
                    summary_scores,
                    summary_cos,
                    summary_sin,
                    summary_position_ids,
                )
            ):
                raise ValueError("hierarchical decode requires pre-RoPE summary state")
            self._new_sum_pre_key_buf[layer_idx][:, :, w_ptr:w_ptr+1, :].copy_(
                pre_rope_key_states
            )
            self._new_sum_score_buf[layer_idx][:, w_ptr:w_ptr+1, :].copy_(
                summary_scores
            )
            self._new_sum_cos_buf[layer_idx][:, w_ptr:w_ptr+1, :].copy_(summary_cos)
            self._new_sum_sin_buf[layer_idx][:, w_ptr:w_ptr+1, :].copy_(summary_sin)
            self._new_sum_position_buf[layer_idx][
                :, w_ptr:w_ptr+1
            ].copy_(summary_position_ids)
        self._new_sum_write_ptr[layer_idx] = (w_ptr + 1) % scn
        if self._new_sum_len[layer_idx] < scn:
            self._new_sum_len[layer_idx] += 1

        self.cur_chunk_sizes[layer_idx] += n_summary
        self._total_chunks[layer_idx] += n_summary

    # ── Decode: get KV for attention ──

    def get_attention_kv(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Get full KV for text token attention.

        Large window: buffer[:text_len]
        Small window (steady state): key_cache[:ws + n_old + cbl] — single slice, zero cat
        """
        if self.is_large_window[layer_idx]:
            tl = self._text_len[layer_idx]
            return (self.key_cache[layer_idx][:, :, :tl, :],
                    self.value_cache[layer_idx][:, :, :tl, :], 0)

        ws = self.window_sizes[layer_idx]
        nv = self._n_valid_window[layer_idx]
        cbl = self._chunk_buf_len[layer_idx]
        n_old = self._old_summary_len[layer_idx]

        # Steady state: ring full → [ring(ws) | old_sums(n_old) | chunk_mirror(cbl)] contiguous
        if nv == ws:
            end = ws + n_old + cbl
            base_k = self.key_cache[layer_idx][:, :, :end, :]
            base_v = self.value_cache[layer_idx][:, :, :end, :]
        else:
            # Warmup: ring not full, [nv:ws] is gap → cat
            parts_k, parts_v = [], []
            if nv > 0:
                parts_k.append(self.key_cache[layer_idx][:, :, :nv, :])
                parts_v.append(self.value_cache[layer_idx][:, :, :nv, :])
            if cbl > 0:
                mirror_start = ws + n_old
                parts_k.append(self.key_cache[layer_idx][:, :, mirror_start:mirror_start + cbl, :])
                parts_v.append(self.value_cache[layer_idx][:, :, mirror_start:mirror_start + cbl, :])
            if n_old > 0:
                parts_k.append(self.key_cache[layer_idx][:, :, ws:ws + n_old, :])
                parts_v.append(self.value_cache[layer_idx][:, :, ws:ws + n_old, :])
            base_k = parts_k[0] if len(parts_k) == 1 else torch.cat(parts_k, dim=2)
            base_v = parts_v[0] if len(parts_v) == 1 else torch.cat(parts_v, dim=2)

        meta_len = self._meta_len[layer_idx]
        if meta_len:
            base_k = torch.cat(
                (base_k, self._meta_key_cache[layer_idx][:, :, :meta_len, :]), dim=2
            )
            base_v = torch.cat(
                (base_v, self._meta_value_cache[layer_idx][:, :, :meta_len, :]), dim=2
            )
        return base_k, base_v, meta_len

    def get_current_chunk_kv(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get KV of the current chunk's C text tokens for summary token attention."""
        C = self.summary_chunk_size
        if self.is_large_window[layer_idx]:
            tl = self._text_len[layer_idx]
            return (self.key_cache[layer_idx][:, :, tl - C:tl, :],
                    self.value_cache[layer_idx][:, :, tl - C:tl, :])
        else:
            ws = self.window_sizes[layer_idx]
            n_old = self._old_summary_len[layer_idx]
            cbl = self._chunk_buf_len[layer_idx]
            mirror_start = ws + n_old
            return (self.key_cache[layer_idx][:, :, mirror_start:mirror_start + cbl, :],
                    self.value_cache[layer_idx][:, :, mirror_start:mirror_start + cbl, :])

    def reset_chunk_counter(self):
        """Reset chunk counters after a chunk boundary step completes."""
        block = self.summary_chunk_size + self.summary_token_num
        for layer_idx in range(self.num_hidden_layers):
            if self.cur_chunk_sizes[layer_idx] >= block:
                self.cur_chunk_sizes[layer_idx] %= block
