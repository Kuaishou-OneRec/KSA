"""
Summary Attention layers for Qwen3@ccl.

SummaryQwen3Attention extends Qwen3Attention with:
- Summary attention computation via CUDA kernel or native PyTorch (configurable)
- Independent QKV projections for summary tokens
- Independent layer norm for summary tokens (optional)
- Attention distillation output
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from torch import Tensor

from muse.models.qwen3._layers import Qwen3Attention
from muse.layers.rms_norm import RMSNorm
from muse.layers.summary_attention_func import (
    summary_attention_forward,
    summary_attention_forward_native,
)
from muse.layers.summary_context import SummaryBatchContext
from muse.training.parallel import (
    get_context_parallel_world_size,
    get_context_parallel_group,
    SeqAllToAll4D,
)


class SummaryQwen3Attention(Qwen3Attention):
    """Qwen3Attention variant with summary attention support.

    When summary_ctx is provided, uses the CUDA kernel summary_attn_func
    instead of standard flash/eager attention. When summary_ctx is None,
    falls back to standard Qwen3Attention behavior.

    Must fully rewrite forward() because the base class's forward is a
    tightly-coupled block (QKV→Norm→RoPE→CP→GQA→Attention→CP→Proj) and
    summary attention needs to intervene at multiple points.
    """

    def __init__(
        self,
        *,
        summary_config,
        layer_index: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.summary_config = summary_config
        self.layer_index = layer_index
        self.sliding_chunk_num = summary_config.get_layer_sliding_chunk_num(layer_index)

        # Independent summary QKV projections
        if summary_config.use_independent_qkv:
            self.q_proj_summary = nn.Linear(
                self.embed_dim, self.num_heads * self.head_dim,
                bias=self.q_proj.bias is not None,
            )
            self.k_proj_summary = nn.Linear(
                self.embed_dim, self.num_kv_heads * self.head_dim,
                bias=self.k_proj.bias is not None,
            )
            self.v_proj_summary = nn.Linear(
                self.embed_dim, self.num_kv_heads * self.head_dim,
                bias=self.v_proj.bias is not None,
            )
            # Independent QK norms for summary tokens (on head_dim)
            # When summary_independent_qk_norm=False (default), summary Q/K reuse
            # the shared q_norm/k_norm, aligning with Megatron behavior.
            if self.q_norm is not None and summary_config.summary_independent_qk_norm:
                self.q_norm_summary = RMSNorm(self.head_dim, eps=1e-6)
                self.k_norm_summary = RMSNorm(self.head_dim, eps=1e-6)

            # NOTE: sa_norm_summary (pre-attention layernorm on embed_dim) is NOT here.
            # It belongs to SummaryTransformerSelfAttentionLayer (the layer, not attention).
            # See modeling.py for where it's created and used.

        # Distillation config
        self.enable_distill = summary_config.enable_summary_distill_attention
        self.distill_norm = summary_config.summary_distill_attention_norm

    def _get_summary_mix_coeff(self, curr_iteration: int) -> float:
        """Summary qkv weight in [0, 1], linearly annealed by training iteration."""
        if not self.summary_config.summary_mix_qkv:
            return 1.0
        start = self.summary_config.summary_mix_start_iter
        end = max(start + 1, self.summary_config.summary_mix_end_iter)
        if curr_iteration <= start:
            return 1.0
        if curr_iteration >= end:
            return 0.0
        return 1.0 - (curr_iteration - start) / (end - start)

    def forward(
        self,
        x: Tensor,
        y: Optional[Tensor] = None,
        *,
        mask: Optional[Tensor] = None,
        input_pos: Optional[Tensor] = None,
        summary_ctx: Optional[SummaryBatchContext] = None,
        **kwargs,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Forward pass with optional summary attention.

        Returns:
            (output, distill_out): output tensor [b, s, d] and optional
            distillation output.
        """
        # ① Fallback to standard attention when no summary context
        if summary_ctx is None or not summary_ctx.enabled:
            output = super().forward(x, y, mask=mask, input_pos=input_pos, **kwargs)
            distill_out = None
            if self.enable_distill:
                distill_out = output.clone()
                if self.distill_norm:
                    rms = distill_out.pow(2).mean(-1, keepdim=True).sqrt() + 1e-6
                    distill_out = distill_out / rms
            return output, distill_out

        b, s_x, _ = x.shape
        y = y if y is not None else x

        # ② Standard QKV projection + reshape
        q = self.q_proj(x).view(b, s_x, self.num_heads, self.head_dim)
        k = self.k_proj(y).view(b, s_x, self.num_kv_heads, self.head_dim)
        v = self.v_proj(y).view(b, s_x, self.num_kv_heads, self.head_dim)

        # ③ QK Norm (Qwen3: norm before RoPE)
        if self.q_norm is not None:
            q = self.q_norm(q)
        if self.k_norm is not None:
            k = self.k_norm(k)

        # ④ Independent summary parameter replacement
        if self.summary_config.use_independent_qkv:
            summary_mask = summary_ctx.summary_mask  # [b, s] bool
            q = q.clone()
            k = k.clone()
            v = v.clone()

            # Extract summary hidden states
            # x is already normed (by sa_norm or sa_norm_summary in the layer)
            x_summary = x[summary_mask]  # [num_summary, d]

            q_s = self.q_proj_summary(x_summary).view(-1, self.num_heads, self.head_dim)
            k_s = self.k_proj_summary(x_summary).view(-1, self.num_kv_heads, self.head_dim)
            v_s = self.v_proj_summary(x_summary).view(-1, self.num_kv_heads, self.head_dim)

            if hasattr(self, 'q_norm_summary'):
                q_s = self.q_norm_summary(q_s)
                k_s = self.k_norm_summary(k_s)
            elif self.q_norm is not None:
                q_s = self.q_norm(q_s)
                k_s = self.k_norm(k_s)

            # ④b Mixoff blend: linearly anneal from summary QKV to base QKV
            if self.summary_config.summary_mix_qkv:
                mix = self._get_summary_mix_coeff(summary_ctx.curr_iteration)
                # q/k/v[summary_mask] still hold base QKV (after clone, before assign)
                q_base_s = q[summary_mask]
                k_base_s = k[summary_mask]
                v_base_s = v[summary_mask]
                q_s = mix * q_s + (1.0 - mix) * q_base_s
                k_s = mix * k_s + (1.0 - mix) * k_base_s
                v_s = mix * v_s + (1.0 - mix) * v_base_s

            q[summary_mask] = q_s
            k[summary_mask] = k_s
            v[summary_mask] = v_s

        # ⑤ RoPE — use summary_ctx.position_ids
        summary_pos_ids = summary_ctx.position_ids  # [b, s]
        if self.pos_embeddings is not None:
            q = self.pos_embeddings(q, input_pos=summary_pos_ids)
            k = self.pos_embeddings(k, input_pos=summary_pos_ids)

        # ⑥ CP all-to-all: seq partition → heads partition
        if get_context_parallel_world_size() > 1:
            cpg = get_context_parallel_group()
            q = SeqAllToAll4D.apply(cpg, q, 2, 1)
            k = SeqAllToAll4D.apply(cpg, k, 2, 1)
            v = SeqAllToAll4D.apply(cpg, v, 2, 1)

        # ⑦ GQA expansion
        if self.num_heads != self.num_kv_heads:
            q_per_kv = self.num_heads // self.num_kv_heads
            expand_shape = (b, -1, k.shape[2], q_per_kv, self.head_dim)
            k = k.unsqueeze(3).expand(expand_shape).flatten(2, 3)
            v = v.unsqueeze(3).expand(expand_shape).flatten(2, 3)

        # ⑧ Summary attention — dispatch by mode
        # After CP all-to-all: q/k/v are [b, full_seq, heads/P, head_dim]
        # Use full_summary_mask (never CP-split) since tensors are full-sequence here
        if self.summary_config.summary_attention_mode == "kernel":
            kernel_summary_mask = summary_ctx.full_summary_mask if summary_ctx.full_summary_mask is not None else summary_ctx.summary_mask
            attn_out = summary_attention_forward(
                q, k, v,
                summary_chunk_size=self.summary_config.summary_chunk_size,
                summary_token_num=self.summary_config.summary_token_num,
                summary_sliding_chunk_num=self.sliding_chunk_num,
                summary_pos=kernel_summary_mask.squeeze(),
            )
        else:
            attn_out = summary_attention_forward_native(
                q, k, v,
                summary_chunk_size=self.summary_config.summary_chunk_size,
                summary_token_num=self.summary_config.summary_token_num,
                summary_sliding_chunk_num=self.sliding_chunk_num,
                attention_mode=self.summary_config.summary_attention_mode,
            )
        # attn_out: [b, full_seq, heads/P * head_dim]

        # ⑨ CP inverse all-to-all: heads partition → seq partition
        if get_context_parallel_world_size() > 1:
            cpg = get_context_parallel_group()
            # Reshape back to [b, s, h/P, d] for all-to-all
            h_local = q.shape[2]  # heads/P after the first all-to-all
            attn_out = attn_out.view(b, -1, h_local, self.head_dim)
            attn_out = SeqAllToAll4D.apply(cpg, attn_out, 1, 2)
            attn_out = attn_out.contiguous().view(b, s_x, -1)
        else:
            attn_out = attn_out.contiguous().view(b, s_x, -1)

        # ⑩ Output projection
        output = self.output_proj(attn_out)

        # ⑪ Distillation output
        distill_out = None
        if self.enable_distill:
            distill_out = output.clone()
            if self.distill_norm:
                rms = distill_out.pow(2).mean(-1, keepdim=True).sqrt() + 1e-6
                distill_out = distill_out / rms

        return output, distill_out
