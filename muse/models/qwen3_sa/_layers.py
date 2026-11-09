"""
Summary Attention layers for Qwen3.

SummaryQwen3Attention extends Qwen3Attention with:
- Summary attention computation via CUDA kernel or native PyTorch (configurable)
- Independent QKV projections for summary tokens
- Independent layer norm for summary tokens (optional)
- Attention distillation output
"""

import math

import torch
import torch.nn as nn
from typing import Optional, Tuple
from torch import Tensor

from muse.models.qwen3._layers import Qwen3Attention
from muse.layers.rms_norm import RMSNorm
from muse.layers.summary_attention_func import (
    hierarchical_summary_attention_forward,
    summary_attention_forward,
    summary_attention_forward_native,
    summary_sliding_attention_pooled_block_mode,
)
from muse.layers.summary_context import SummaryBatchContext
from muse.layers.sksa_summary_indexer import (
    SKSASummaryLightningIndexer,
    _can_reuse_full_attention_lse_for_dense_teacher,
    _extract_regular_summary_text_lse,
    _summary_chunk_local_attention,
    _split_regular_layout,
    sksa_summary_indexer_sparse_attention,
    split_summary_indexer_inputs,
)
from muse.training.parallel import (
    get_context_parallel_world_size,
    get_context_parallel_group,
    SeqAllToAll4D,
)
from muse.training.activations import (
    is_activation_checkpoint_active,
    is_activation_checkpoint_recomputing,
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
        self.meta_pool_mode = summary_config.summary_meta_pool_mode
        self.meta_pool_group_size = summary_config.get_layer_meta_pool_group_size(
            layer_index
        )
        self.meta_pool_enabled = (
            self.meta_pool_mode != "none" and self.meta_pool_group_size > 1
        )
        self.meta_pool_distill_enabled = (
            self.meta_pool_enabled
            and summary_config.summary_meta_pool_distill_attention
        )
        self.sksa_target_enabled = (
            summary_config.sksa_enabled
            and layer_index in summary_config.sksa_sparse_full_layers
        )
        self.sksa_routing_mode = summary_config.sksa_routing_mode

        if (
            (self.sksa_target_enabled)
        ):
            # Preserve initialization of all parameters shared with the parent.
            with torch.random.fork_rng(devices=[]):
                indexer_cls = (
                    (SKSASummaryLightningIndexer)
                )
                self.sksa_indexer = indexer_cls(
                    embed_dim=self.embed_dim,
                    query_rank=summary_config.sksa_indexer_query_rank,
                    indexer_heads=summary_config.sksa_indexer_heads,
                    indexer_head_dim=summary_config.sksa_indexer_head_dim,
                    rope_dim=summary_config.sksa_indexer_rope_dim,
                    max_seq_len=summary_config.max_seq_len,
                    rope_base=summary_config.rope_base,
                    query_block_size=(
                        summary_config.sksa_indexer_query_block_size
                    ),
                    scorer_backend=(
                        summary_config.sksa_indexer_scorer_backend
                    ),
                    **((dict(
                        chunk_attention_qk_source=getattr(
                            summary_config,
                            "sksa_indexer_chunk_attention_qk_source",
                            "backbone",
                        ),
                        chunk_num_heads=self.num_heads,
                        chunk_num_kv_heads=self.num_kv_heads,
                        chunk_head_dim=self.head_dim,
                        chunk_norm_eps=summary_config.norm_eps,
                        chunk_q_bias=self.q_proj.bias is not None,
                        chunk_k_bias=self.k_proj.bias is not None,
                        chunk_attention_v_source=(
                            summary_config.sksa_indexer_chunk_attention_v_source
                        ),
                        chunk_attention_include_summary=(
                            summary_config.sksa_indexer_chunk_attention_include_summary
                        ),
                        chunk_v_bias=self.v_proj.bias is not None,
                    ))),
                )

        if self.meta_pool_enabled and self.meta_pool_mode in {
            "scalar", "dimension_wise"
        }:
            scorer_width = (
                1
                if self.meta_pool_mode == "scalar"
                else self.num_kv_heads * self.head_dim
            )
            # Preserve the caller's RNG stream so adding the scorer does not
            # perturb initialization of any shared baseline parameters.
            with torch.random.fork_rng(devices=[]):
                self.meta_pool_scorer = nn.Linear(
                    self.embed_dim, scorer_width, bias=False
                )
                nn.init.zeros_(self.meta_pool_scorer.weight)
            position_bias_shape = (
                (self.meta_pool_group_size,)
                if self.meta_pool_mode == "scalar"
                else (self.meta_pool_group_size, scorer_width)
            )
            self.meta_pool_position_bias = nn.Parameter(torch.zeros(
                position_bias_shape
            ))

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

    @torch.no_grad()
    def initialize_sksa_indexer_chunk_qk_from_backbone(self) -> bool:
        """Copy this layer's backbone Q/K into its Indexer chunk Q/K.

        The independent chunk-attention path is constructed before the source
        checkpoint is loaded, so its parameters normally keep their model
        initialization.  This hook is called after the checkpoint load for
        the initialization ablation: it copies the corresponding backbone
        projections and Q/K RMSNorm scales without changing any backbone
        parameter or sharing storage with it.

        Returns:
            ``True`` when an independent S-KSA chunk path was initialized,
            otherwise ``False`` for layers without an Indexer or when the
            Indexer is configured to reuse backbone Q/K.
        """
        indexer = getattr(self, "sksa_indexer", None)
        if indexer is None or indexer.chunk_attention_qk_source != "independent":
            return False

        required = (
            indexer.chunk_q_proj,
            indexer.chunk_k_proj,
            indexer.chunk_q_norm,
            indexer.chunk_k_norm,
            self.q_proj,
            self.k_proj,
            self.q_norm,
            self.k_norm,
        )
        if any(module is None for module in required):
            raise RuntimeError(
                "Backbone chunk-Q/K initialization requires backbone and "
                "independent Indexer Q/K projections and norms"
            )

        pairs = (
            (indexer.chunk_q_proj.weight, self.q_proj.weight, "chunk_q_proj"),
            (indexer.chunk_k_proj.weight, self.k_proj.weight, "chunk_k_proj"),
            (indexer.chunk_q_norm.scale, self.q_norm.scale, "chunk_q_norm"),
            (indexer.chunk_k_norm.scale, self.k_norm.scale, "chunk_k_norm"),
        )
        if indexer.chunk_q_proj.bias is not None:
            if self.q_proj.bias is None:
                raise RuntimeError("chunk_q_proj has bias but backbone q_proj does not")
            pairs += ((indexer.chunk_q_proj.bias, self.q_proj.bias, "chunk_q_proj.bias"),)
        if indexer.chunk_k_proj.bias is not None:
            if self.k_proj.bias is None:
                raise RuntimeError("chunk_k_proj has bias but backbone k_proj does not")
            pairs += ((indexer.chunk_k_proj.bias, self.k_proj.bias, "chunk_k_proj.bias"),)

        for target, source, name in pairs:
            if target.shape != source.shape:
                raise RuntimeError(
                    f"Cannot initialize {name} from backbone: "
                    f"target shape {tuple(target.shape)} != "
                    f"source shape {tuple(source.shape)}"
                )
            target.copy_(source)
        return True

    def _build_meta_kv(
        self,
        x: Tensor,
        k_pre_rope: Tensor,
        v: Tensor,
        summary_ctx: SummaryBatchContext,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Pool fixed groups of summary K/V before applying RoPE to the keys."""
        if get_context_parallel_world_size() != 1:
            raise RuntimeError(
                "Hierarchical summary pooling currently supports context parallel size 1"
            )
        if self.summary_config.summary_token_num != 1:
            raise RuntimeError(
                "Hierarchical summary pooling currently requires summary_token_num=1"
            )

        b, _, num_kv_heads, head_dim = k_pre_rope.shape
        summary_mask = summary_ctx.summary_mask
        if (
            summary_ctx.summary_layout_validated
            and summary_ctx.summary_layout_signature is not None
        ):
            summary_count = summary_ctx.summary_layout_signature[0]
        else:
            summary_count = int(summary_mask[0].sum().item())
        if (
            not summary_ctx.summary_layout_validated
            and not torch.all(summary_mask.sum(dim=1) == summary_count)
        ):
            raise ValueError("All samples must contain the same number of summaries")

        group_size = self.meta_pool_group_size
        group_count = summary_count // group_size
        complete_end = group_count * group_size
        x_summary = x[summary_mask].view(b, summary_count, self.embed_dim)
        k_summary = k_pre_rope[summary_mask].view(
            b, summary_count, num_kv_heads, head_dim
        )
        v_summary = v[summary_mask].view(
            b, summary_count, num_kv_heads, head_dim
        )

        x_grouped = x_summary[:, :complete_end].view(
            b, group_count, group_size, self.embed_dim
        )
        k_grouped = k_summary[:, :complete_end].view(
            b, group_count, group_size, num_kv_heads, head_dim
        )
        v_grouped = v_summary[:, :complete_end].view(
            b, group_count, group_size, num_kv_heads, head_dim
        )

        if self.meta_pool_mode == "mean":
            weights = torch.full(
                (b, group_count, group_size),
                1.0 / group_size,
                dtype=k_pre_rope.dtype,
                device=k_pre_rope.device,
            )
            meta_k = (weights[..., None, None] * k_grouped).sum(dim=2)
            meta_v = (weights[..., None, None] * v_grouped).sum(dim=2)
        elif self.meta_pool_mode == "scalar":
            logits = self.meta_pool_scorer(x_grouped).squeeze(-1)
            logits = logits + self.meta_pool_position_bias.view(1, 1, group_size)
            weights = torch.softmax(logits.float(), dim=-1).to(k_pre_rope.dtype)
            meta_k = (weights[..., None, None] * k_grouped).sum(dim=2)
            meta_v = (weights[..., None, None] * v_grouped).sum(dim=2)
        elif self.meta_pool_mode == "dimension_wise":
            kv_width = num_kv_heads * head_dim
            logits = self.meta_pool_scorer(x_grouped)
            logits = logits + self.meta_pool_position_bias.view(
                1, 1, group_size, kv_width
            )
            weights = torch.softmax(logits.float(), dim=2).to(k_pre_rope.dtype)
            weights = weights.view(
                b, group_count, group_size, num_kv_heads, head_dim
            )
            meta_k = (weights * k_grouped).sum(dim=2)
            meta_v = (weights * v_grouped).sum(dim=2)
        else:
            raise RuntimeError(f"Unsupported meta-pool mode: {self.meta_pool_mode}")
        summary_position_ids = summary_ctx.position_ids[summary_mask].view(
            b, summary_count
        )
        meta_position_ids = summary_position_ids[
            :, group_size - 1 : complete_end : group_size
        ]
        return meta_k, meta_v, meta_position_ids

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

    def _get_meta_pool_summary_coeff(self, curr_iteration: int) -> float:
        """Original-summary attention weight during summary-to-meta annealing."""
        if not self.summary_config.summary_meta_pool_anneal:
            return 0.0
        start = self.summary_config.summary_meta_pool_anneal_start_iter
        end = self.summary_config.summary_meta_pool_anneal_end_iter
        if curr_iteration <= start:
            return 1.0
        if curr_iteration >= end:
            return 0.0
        return 1.0 - (curr_iteration - start) / (end - start)

    def _run_original_summary_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        summary_ctx: SummaryBatchContext,
        *,
        return_lse: bool = False,
    ) -> Tensor | tuple[Tensor, Optional[Tensor]]:
        """Run the uncompressed KSA path for this layer."""
        if self.summary_config.summary_attention_mode == "kernel":
            kernel_summary_mask = (
                summary_ctx.full_summary_mask
                if summary_ctx.full_summary_mask is not None
                else summary_ctx.summary_mask
            )
            return summary_attention_forward(
                q,
                k,
                v,
                summary_chunk_size=self.summary_config.summary_chunk_size,
                summary_token_num=self.summary_config.summary_token_num,
                summary_sliding_chunk_num=self.sliding_chunk_num,
                summary_pos=kernel_summary_mask.squeeze(),
                return_lse=return_lse,
            )

        use_static_layout = get_context_parallel_world_size() == 1
        output = summary_attention_forward_native(
            q,
            k,
            v,
            summary_chunk_size=self.summary_config.summary_chunk_size,
            summary_token_num=self.summary_config.summary_token_num,
            summary_sliding_chunk_num=self.sliding_chunk_num,
            attention_mode=self.summary_config.summary_attention_mode,
            block_size=self.summary_config.summary_block_size,
            summary_mask=summary_ctx.summary_mask,
            summary_layout_signature=(
                summary_ctx.summary_layout_signature
                if use_static_layout
                else None
            ),
            summary_layout_validated=(
                summary_ctx.summary_layout_validated and use_static_layout
            ),
        )
        return (output, None) if return_lse else output

    def forward(
        self,
        x: Tensor,
        y: Optional[Tensor] = None,
        *,
        mask: Optional[Tensor] = None,
        input_pos: Optional[Tensor] = None,
        summary_ctx: Optional[SummaryBatchContext] = None,
        **kwargs,
    ) -> tuple:
        """Forward pass with optional summary attention.

        Returns:
            (output, distill_out): output tensor [b, s, d] and optional
            distillation output.
        """
        distill_override = (
            summary_ctx.sksa_indexer_distill_active
            if summary_ctx is not None
            else None
        )
        sksa_indexer_distill_active = (
            self.summary_config.sksa_indexer_distill_attention
            and distill_override is not False
        )
        # ① Fallback to standard attention when no summary context
        if summary_ctx is None or not summary_ctx.enabled:
            output = super().forward(x, y, mask=mask, input_pos=input_pos, **kwargs)
            if self.meta_pool_distill_enabled:
                return output, output.new_zeros(())
            if (
                self.sksa_target_enabled
                and sksa_indexer_distill_active
            ):
                return output, output.new_zeros(())
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

        # Reuse normalized, unrotated main K only for the backbone-key Indexer.

        # ⑤ Build meta K/V before RoPE, then rotate base and meta keys.
        meta_k = None
        meta_v = None
        meta_position_ids = None
        if self.meta_pool_enabled:
            meta_k, meta_v, meta_position_ids = self._build_meta_kv(
                x, k, v, summary_ctx
            )
        summary_pos_ids = summary_ctx.position_ids  # [b, s]
        if self.pos_embeddings is not None:
            q = self.pos_embeddings(q, input_pos=summary_pos_ids)
            k = self.pos_embeddings(k, input_pos=summary_pos_ids)
            if meta_k is not None:
                meta_k = self.pos_embeddings(
                    meta_k, input_pos=meta_position_ids
                )

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
            if meta_k is not None:
                meta_expand_shape = (
                    b, -1, meta_k.shape[2], q_per_kv, self.head_dim
                )
                meta_k = meta_k.unsqueeze(3).expand(meta_expand_shape).flatten(2, 3)
                meta_v = meta_v.unsqueeze(3).expand(meta_expand_shape).flatten(2, 3)

        # S-KSA routing is consumed only on explicitly selected global/full KSA
        # layers.  Each target consumes the detached per-query meta selections
        # emitted by its preceding compressed layers; main attention still
        # reads the current full layer's original text K/V.
        sksa_indexer_loss = None
        sksa_selected_blocks = None
        sksa_dense_warmup_attn_out = None
        independent_chunk_qk = False
        if self.sksa_target_enabled:
            if get_context_parallel_world_size() != 1:
                raise RuntimeError("S-KSA currently requires context parallel size 1")
            cu_seqlens = kwargs.get("cu_seqlens")
            if cu_seqlens is not None and cu_seqlens.numel() != 2:
                raise RuntimeError(
                    "S-KSA currently requires one unpacked document per sequence"
                )
            (
                text_hidden,
                summary_hidden,
                text_position_ids,
                summary_position_ids,
            ) = split_summary_indexer_inputs(
                x,
                summary_ctx.position_ids,
                summary_ctx.summary_mask,
                summary_chunk_size=(
                    self.summary_config.summary_chunk_size
                ),
                summary_token_num=self.summary_config.summary_token_num,
                summary_layout_signature=(
                    summary_ctx.summary_layout_signature
                ),
                summary_layout_validated=(
                    summary_ctx.summary_layout_validated
                ),
            )
            trusted_layout_signature = (
                summary_ctx.summary_layout_signature
                if summary_ctx.summary_layout_validated
                else (
                    s_x
                    // (
                        self.summary_config.summary_chunk_size
                        + self.summary_config.summary_token_num
                    )
                    * self.summary_config.summary_token_num,
                    True,
                )
            )
            text_count = text_hidden.shape[1]
            q_text, _, q_chunks = _split_regular_layout(
                q,
                summary_ctx.summary_mask,
                summary_chunk_size=(
                    self.summary_config.summary_chunk_size
                ),
                summary_token_num=self.summary_config.summary_token_num,
                summary_layout_signature=trusted_layout_signature,
                summary_layout_validated=True,
            )
            k_text, _, k_chunks = _split_regular_layout(
                k,
                summary_ctx.summary_mask,
                summary_chunk_size=(
                    self.summary_config.summary_chunk_size
                ),
                summary_token_num=self.summary_config.summary_token_num,
                summary_layout_signature=trusted_layout_signature,
                summary_layout_validated=True,
            )
            if (
                (self.summary_config.sksa_indexer_summary_key_source == 'chunk_attention')
            ):
                independent_chunk_qk = (
                    getattr(
                        self.summary_config,
                        "sksa_indexer_chunk_attention_qk_source",
                        "backbone",
                    )
                    == "independent"
                )
                # Auxiliary Q/K pool independent V, backbone V, or main K
                # (after QK norm, before RoPE). All backbone inputs detach;
                # this side path never changes the residual stream.
                _, _, v_chunks = _split_regular_layout(
                    (v),
                    summary_ctx.summary_mask,
                    summary_chunk_size=(
                        self.summary_config.summary_chunk_size
                    ),
                    summary_token_num=(
                        self.summary_config.summary_token_num
                    ),
                    summary_layout_signature=trusted_layout_signature,
                    summary_layout_validated=True,
                )
                if independent_chunk_qk:
                    _, _, hidden_chunks = _split_regular_layout(
                        x,
                        summary_ctx.summary_mask,
                        summary_chunk_size=(
                            self.summary_config.summary_chunk_size
                        ),
                        summary_token_num=(
                            self.summary_config.summary_token_num
                        ),
                        summary_layout_signature=trusted_layout_signature,
                        summary_layout_validated=True,
                    )
                    _, _, position_chunks = _split_regular_layout(
                        summary_ctx.position_ids,
                        summary_ctx.summary_mask,
                        summary_chunk_size=(
                            self.summary_config.summary_chunk_size
                        ),
                        summary_token_num=(
                            self.summary_config.summary_token_num
                        ),
                        summary_layout_signature=trusted_layout_signature,
                        summary_layout_validated=True,
                    )
                    summary_hidden = (
                        self.sksa_indexer.build_chunk_attention_summary(
                            hidden_chunks,
                            v_chunks,
                            position_chunks,
                            summary_chunk_size=(
                                self.summary_config.summary_chunk_size
                            ),
                            summary_token_num=(
                                self.summary_config.summary_token_num
                            ),
                            pos_embeddings=self.pos_embeddings,
                            indexer_block_size=self.summary_config.sksa_indexer_block_size,
                        )
                    )
                    block_size = self.summary_config.sksa_indexer_block_size
                    if block_size < self.summary_config.summary_chunk_size:
                        summary_position_ids = position_chunks[
                            :, :, block_size - 1 : self.summary_config.summary_chunk_size : block_size
                        ].reshape(position_chunks.shape[0], -1)
                else:
                    with torch.no_grad():
                        summary_hidden = _summary_chunk_local_attention(
                            q_chunks.detach(),
                            k_chunks.detach(),
                            v_chunks.detach(),
                            summary_chunk_size=(
                                self.summary_config.summary_chunk_size
                            ),
                        ).reshape(
                            summary_hidden.shape[0],
                            summary_hidden.shape[1],
                            -1,
                        )
                if summary_hidden.shape[-1] != self.embed_dim:
                    raise RuntimeError(
                        "chunk-attention summary key width does not match "
                        "the Full-layer hidden width"
                    )
            recomputing = is_activation_checkpoint_recomputing()
            checkpoint_active = is_activation_checkpoint_active()
            checkpoint_cache = None
            cache_stack = None
            dense_warmup = (
                not self.summary_config.sksa_apply_sparse_attention
            )
            reuse_full_attention_lse = (
                sksa_indexer_distill_active
                and ((dense_warmup))
                and _can_reuse_full_attention_lse_for_dense_teacher(
                    summary_attention_mode=(
                        self.summary_config.summary_attention_mode
                    ),
                    summary_layout_validated=(
                        summary_ctx.summary_layout_validated
                    ),
                    summary_layout_signature=(
                        summary_ctx.summary_layout_signature
                    ),
                    expanded_sequence_length=s_x,
                    text_count=text_count,
                    summary_chunk_size=(
                        self.summary_config.summary_chunk_size
                    ),
                    summary_token_num=(
                        self.summary_config.summary_token_num
                    ),
                    sliding_chunk_num=self.sliding_chunk_num,
                )
            )
            precomputed_dense_lse = None
            cache_dense_lse_after_routing = False
            if reuse_full_attention_lse:
                if checkpoint_active and recomputing:
                    dense_lse_stack = (
                        summary_ctx
                        ._sksa_dense_teacher_lse_checkpoint_cache
                        .get(self.layer_index)
                    )
                    if not dense_lse_stack:
                        raise RuntimeError(
                            "S-KSA all-token teacher checkpoint recomputation is "
                            f"missing the layer-{self.layer_index} Full LSE cache"
                        )
                    # ``None`` is a deliberate sentinel: a non-CuTe kernel
                    # could not expose LSE in the original forward, so the
                    # indexer must execute its established dense fallback.
                    precomputed_dense_lse = dense_lse_stack.pop()
                    if not dense_lse_stack:
                        summary_ctx._sksa_dense_teacher_lse_checkpoint_cache.pop(
                            self.layer_index, None
                        )
                else:
                    if dense_warmup:
                        (
                            sksa_dense_warmup_attn_out,
                            expanded_full_lse,
                        ) = self._run_original_summary_attention(
                            q,
                            k,
                            v,
                            summary_ctx,
                            return_lse=True,
                        )
                    else:
                        # The sparse main path still needs the dense Full
                        # denominator for all-token Indexer supervision.
                        # It is a detached teacher and never enters the
                        # residual stream or the backbone gradient graph.
                        with torch.no_grad():
                            _, expanded_full_lse = (
                                self._run_original_summary_attention(
                                    q.detach(),
                                    k.detach(),
                                    v.detach(),
                                    summary_ctx,
                                    return_lse=True,
                                )
                            )
                    if expanded_full_lse is not None:
                        precomputed_dense_lse = (
                            _extract_regular_summary_text_lse(
                                expanded_full_lse,
                                summary_chunk_size=(
                                    self.summary_config.summary_chunk_size
                                ),
                                summary_token_num=(
                                    self.summary_config.summary_token_num
                                ),
                                expected_text_count=text_count,
                            )
                        )
                    cache_dense_lse_after_routing = (
                        checkpoint_active and torch.is_grad_enabled()
                    )
            if checkpoint_active:
                cache_stack = (
                    summary_ctx._sksa_indexer_checkpoint_cache.get(
                        self.layer_index
                    )
                )
                if recomputing:
                    if (not cache_stack and self.summary_config.sksa_apply_sparse_attention
                            and not summary_ctx.sksa_indexer_full_candidate_kl):
                        raise RuntimeError(
                            "S-KSA indexer checkpoint recomputation is "
                            f"missing the layer-{self.layer_index} route cache"
                        )
                    if cache_stack:
                        checkpoint_cache = cache_stack.pop()
                        if not cache_stack:
                            summary_ctx._sksa_indexer_checkpoint_cache.pop(
                                self.layer_index, None
                            )
            routing = self.sksa_indexer(
                text_hidden,
                summary_hidden,
                text_position_ids,
                summary_position_ids,
                summary_block_size=(
                    self.summary_config.sksa_indexer_block_size
                ),
                topk_blocks=(
                    self.summary_config.get_sksa_indexer_topk_blocks()
                ),
                local_window_tokens=(
                    self.summary_config.sksa_local_window_tokens
                ),
                force_sink=self.summary_config.sksa_indexer_force_sink,
                sink_block_count=(
                    self.summary_config.sksa_indexer_summary_sink_tokens
                    // self.summary_config.sksa_indexer_block_size
                ),
                distill=sksa_indexer_distill_active,
                dense_warmup=(
                    not self.summary_config.sksa_apply_sparse_attention
                ),
                full_candidate_kl=summary_ctx.sksa_indexer_full_candidate_kl,
                teacher_pooling=(
                    self.summary_config.sksa_indexer_teacher_pooling
                ),
                main_q=q_text,
                main_k=k_text,
                teacher_query_block_size=(
                    self.summary_config.sksa_indexer_teacher_query_block_size
                ),
                precomputed_dense_lse=precomputed_dense_lse,
                preserve_summary_grad=independent_chunk_qk,
                kl_loss_factor=summary_ctx.sksa_indexer_kl_factor,
                monitor_topk_blocks=summary_ctx.sksa_indexer_monitor_topk_blocks,
                recompute_cache=checkpoint_cache,
            )
            sksa_selected_blocks = routing.selected_block_indices
            if cache_dense_lse_after_routing:
                summary_ctx._sksa_dense_teacher_lse_checkpoint_cache.setdefault(
                    self.layer_index, []
                ).append(precomputed_dense_lse)
            if (
                checkpoint_active
                and not recomputing
                and torch.is_grad_enabled()
                and routing.checkpoint_cache is not None
            ):
                summary_ctx._sksa_indexer_checkpoint_cache.setdefault(
                    self.layer_index, []
                ).append(routing.checkpoint_cache)
            sksa_indexer_loss = routing.indexer_loss
            if not is_activation_checkpoint_recomputing():
                if routing.kl_loss is not None:
                    summary_ctx._sksa_indexer_kl_diagnostics.append(
                        routing.kl_loss
                    )
                if routing.oracle_topk_recall is not None:
                    summary_ctx._sksa_indexer_oracle_topk_recall_diagnostics.append(
                        routing.oracle_topk_recall
                    )
                if routing.selected_attention_mass is not None:
                    summary_ctx._sksa_indexer_selected_attention_mass_diagnostics.append(
                        routing.selected_attention_mass
                    )
                if routing.oracle_topk_attention_mass is not None:
                    summary_ctx._sksa_indexer_oracle_topk_attention_mass_diagnostics.append(
                        routing.oracle_topk_attention_mass
                    )
                if routing.selected_oracle_attention_mass_ratio is not None:
                    summary_ctx._sksa_indexer_selected_oracle_attention_mass_ratio_diagnostics.append(
                        routing.selected_oracle_attention_mass_ratio
                    )

        # ⑧ Summary attention — dispatch by mode
        # After CP all-to-all: q/k/v are [b, full_seq, heads/P, head_dim]
        # Use full_summary_mask (never CP-split) since tensors are full-sequence here
        if (
            self.sksa_target_enabled
            and self.summary_config.sksa_apply_sparse_attention
        ):
            attn_out = sksa_summary_indexer_sparse_attention(
                q,
                k,
                v,
                summary_ctx.summary_mask,
                sksa_selected_blocks,
                summary_chunk_size=(
                    self.summary_config.summary_chunk_size
                ),
                summary_token_num=self.summary_config.summary_token_num,
                local_window_tokens=(
                    self.summary_config.sksa_local_window_tokens
                ),
                summary_layout_signature=trusted_layout_signature,
                summary_layout_validated=True,
                precomputed_q_text=q_text,
                precomputed_k_text=k_text,
                indexer_block_size=self.summary_config.sksa_indexer_block_size,
            )
        elif (
            self.meta_pool_enabled
            and self.summary_config.summary_attention_mode == "kernel"
        ):
            compressed_result = hierarchical_summary_attention_forward(
                q,
                k,
                v,
                meta_k,
                meta_v,
                summary_chunk_size=self.summary_config.summary_chunk_size,
                summary_sliding_chunk_num=self.sliding_chunk_num,
                meta_pool_group_size=self.meta_pool_group_size,
                mass_correction=(
                    self.summary_config.summary_meta_pool_mass_correction
                    == "log_group_size"
                ),
                return_lse=False,
            )
            attn_out = compressed_result
        elif self.meta_pool_enabled:
            use_static_layout = get_context_parallel_world_size() == 1
            attn_out = summary_sliding_attention_pooled_block_mode(
                q,
                k,
                v,
                meta_k,
                meta_v,
                summary_chunk_size=self.summary_config.summary_chunk_size,
                summary_token_num=self.summary_config.summary_token_num,
                summary_sliding_chunk_num=self.sliding_chunk_num,
                meta_pool_group_size=self.meta_pool_group_size,
                meta_logit_bias=(
                    math.log(self.meta_pool_group_size)
                    if self.summary_config.summary_meta_pool_mass_correction
                    == "log_group_size"
                    else 0.0
                ),
                block_size=self.summary_config.summary_block_size,
                summary_mask=summary_ctx.summary_mask,
                summary_layout_signature=(
                    summary_ctx.summary_layout_signature
                    if use_static_layout
                    else None
                ),
                summary_layout_validated=(
                    summary_ctx.summary_layout_validated and use_static_layout
                ),
            )

        # Layer-local KSA teacher.  The teacher sees exactly the same layer
        # input and frozen Q/K/V as the DW student, but its output is detached
        # and never enters the residual stream.
        meta_pool_teacher_out = None
        if self.meta_pool_distill_enabled:
            with torch.no_grad():
                meta_pool_teacher_out = self._run_original_summary_attention(
                    q.detach(), k.detach(), v.detach(), summary_ctx
                )

        # During structural warm-up, retain the original KSA path and linearly
        # transfer attention output mass from summary memory to meta memory.
        # At lambda=1 the zero-weight meta term deliberately stays in the graph:
        # a meta-pool-only run still has a valid backward graph on its first step.
        summary_coeff = (
            self._get_meta_pool_summary_coeff(summary_ctx.curr_iteration)
            if self.meta_pool_enabled
            else 0.0
        )
        if self.meta_pool_enabled and summary_coeff > 0.0:
            summary_attn_out = self._run_original_summary_attention(
                q, k, v, summary_ctx
            )
            attn_out = (
                summary_coeff * summary_attn_out
                + (1.0 - summary_coeff) * attn_out
            )
        elif not self.meta_pool_enabled and not (
            self.sksa_target_enabled
            and self.summary_config.sksa_apply_sparse_attention
        ):
            # Dense Summary-Lightning warm-up may have run this exact normal
            # Full-attention call early to reuse its LSE as the indexer teacher
            # denominator.  All other modes retain their original dispatch.
            attn_out = sksa_dense_warmup_attn_out
            if attn_out is None:
                attn_out = self._run_original_summary_attention(
                    q, k, v, summary_ctx
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
            if meta_pool_teacher_out is not None:
                meta_pool_teacher_out = meta_pool_teacher_out.view(
                    b, -1, h_local, self.head_dim
                )
                meta_pool_teacher_out = SeqAllToAll4D.apply(
                    cpg, meta_pool_teacher_out, 1, 2
                )
                meta_pool_teacher_out = meta_pool_teacher_out.contiguous().view(
                    b, s_x, -1
                )
        else:
            attn_out = attn_out.contiguous().view(b, s_x, -1)
            if meta_pool_teacher_out is not None:
                meta_pool_teacher_out = meta_pool_teacher_out.contiguous().view(
                    b, s_x, -1
                )

        meta_pool_attn_mse = None
        if meta_pool_teacher_out is not None:
            text_mask = ~summary_ctx.summary_mask
            squared_error = (
                attn_out[text_mask].float()
                - meta_pool_teacher_out[text_mask].float()
            ).square()
            # Match the established KSA distillation recipe: cap isolated
            # outliers while retaining a dense per-coordinate learning signal.
            meta_pool_attn_mse = squared_error.clamp(max=0.1).mean()

        # ⑩ Output projection
        output = self.output_proj(attn_out)


        # ⑪ Distillation output
        if self.meta_pool_distill_enabled:
            return output, meta_pool_attn_mse

        attention_aux = sksa_indexer_loss

        distill_out = None
        if self.enable_distill:
            distill_out = output.clone()
            if self.distill_norm:
                rms = distill_out.pow(2).mean(-1, keepdim=True).sqrt() + 1e-6
                distill_out = distill_out / rms

            attention_aux = distill_out

        return output, attention_aux
