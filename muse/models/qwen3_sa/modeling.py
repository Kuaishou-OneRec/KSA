"""
Qwen3 Summary Attention Model@ccl.

Contains:
- SummaryTransformerSelfAttentionLayer: handles tuple return from SummaryQwen3Attention
- SummaryTransformerDecoder: passes summary_ctx, collects distill outputs, filters summary tokens
- Qwen3SummaryModel: top-level model with summary embedding and layer construction
"""

from typing import Dict, Callable, Optional, List, Union
import math
import torch
import torch.nn as nn
import torch.nn.init as init
import logging

from muse.models.base import Model
from muse.config.model_config import Qwen3SummaryAttentionConfig
from muse.layers.position_embeddings import RotaryPositionalEmbeddings
from muse.layers.transformer import TransformerDecoder, TransformerSelfAttentionLayer
from muse.layers.rms_norm import RMSNorm
from muse.layers.linear import TiedLinear
from muse.layers.feed_forward import FeedForward
from muse.layers.summary_context import SummaryBatchContext

from muse.models.qwen3._layers import qwen3_mlp, Qwen3Attention
from muse.models.qwen3_sa._layers import SummaryQwen3Attention

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Initialization helpers (self-contained, no dependency on Qwen3Model)
# ---------------------------------------------------------------------------

def _lecun_normal_(tensor: torch.Tensor) -> None:
    if tensor.dim() < 2:
        std = 0.01
    elif tensor.dim() == 2:
        fan_in = tensor.size(1)
        std = math.sqrt(1.0 / fan_in)
    else:
        fan_in = tensor.size(1)
        for s in tensor.size()[2:]:
            fan_in *= s
        std = math.sqrt(1.0 / fan_in)
    init.normal_(tensor, mean=0.0, std=std)


def _default_flax_embed_init(tensor: torch.Tensor) -> None:
    init.normal_(tensor, mean=0.0, std=0.02)


# ---------------------------------------------------------------------------
# SummaryTransformerSelfAttentionLayer
# ---------------------------------------------------------------------------

class SummaryTransformerSelfAttentionLayer(TransformerSelfAttentionLayer):
    """TransformerSelfAttentionLayer that handles SummaryQwen3Attention's tuple return.

    When summary_independent_attention_layernorm is True, applies separate norms
    to text tokens (sa_norm) and summary tokens (sa_norm_summary) before attention.
    """

    def __init__(
        self,
        attn,
        mlp: nn.Module,
        *,
        sa_norm: Optional[nn.Module] = None,
        mlp_norm: Optional[nn.Module] = None,
        sa_norm_summary: Optional[nn.Module] = None,
        **kwargs,
    ) -> None:
        super().__init__(attn=attn, mlp=mlp, sa_norm=sa_norm, mlp_norm=mlp_norm, **kwargs)
        self.sa_norm_summary = sa_norm_summary  # None when not using independent layernorm

    def forward(
        self,
        x: torch.Tensor,
        *,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[torch.Tensor, tuple]:
        summary_ctx = kwargs.get('summary_ctx', None)

        # Norm: split text/summary if independent layernorm is enabled
        if (
            summary_ctx is not None
            and summary_ctx.enabled
            and self.sa_norm_summary is not None
        ):
            # Text tokens → sa_norm, summary tokens → sa_norm_summary
            h = self.sa_norm(x)
            summary_mask = summary_ctx.summary_mask  # [b, s]
            h = h.clone()
            h[summary_mask] = self.sa_norm_summary(x[summary_mask])
        else:
            # All tokens share sa_norm (default)
            h = self.sa_norm(x)

        # Attention
        attn_result = self.attn(h, h, mask=mask, input_pos=input_pos, **kwargs)

        # SummaryQwen3Attention returns (output, distill_out)
        if isinstance(attn_result, tuple):
            attn_out, distill_out = attn_result
        else:
            attn_out, distill_out = attn_result, None

        # Residual + MLP
        h = self.sa_scale(attn_out) + x
        mlp_out = self.mlp(self.mlp_norm(h))
        out = h + self.mlp_scale(mlp_out)

        if distill_out is not None:
            return out, distill_out
        return out


# ---------------------------------------------------------------------------
# FullAttentionLayerWithSummaryBypass
# ---------------------------------------------------------------------------

class FullAttentionLayerWithSummaryBypass(TransformerSelfAttentionLayer):
    """Standard causal attention layer used in hybrid mode (non-SA layers).

    Summary tokens are filtered out before attention+MLP and restored via
    identity mapping afterwards. Text tokens use standard causal attention
    without seeing summary tokens.
    """

    def forward(
        self,
        x: torch.Tensor,
        *,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        summary_ctx = kwargs.get('summary_ctx')
        if summary_ctx is None or not summary_ctx.enabled:
            return super().forward(x, mask=mask, input_pos=input_pos, **kwargs)

        summary_mask = summary_ctx.summary_mask  # [b, s] bool
        text_mask = ~summary_mask
        b, s, d = x.shape
        s_text = text_mask[0].sum().item()

        # ① Filter: extract text-only tokens
        x_text = x[text_mask].view(b, s_text, d)
        text_pos = summary_ctx.position_ids[text_mask].view(b, s_text)
        residual_summary = x[summary_mask]  # [num_summary_total, d]

        # ② Norm → Attention → Residual → MLP (same as TransformerSelfAttentionLayer)
        h = self.sa_norm(x_text)
        attn_out = self.attn(h, h, mask=mask, input_pos=text_pos)
        h = self.sa_scale(attn_out) + x_text
        mlp_out = self.mlp(self.mlp_norm(h))
        out_text = h + self.mlp_scale(mlp_out)

        # ③ Reconstruct full sequence
        output = torch.zeros(b, s, d, dtype=out_text.dtype, device=out_text.device)
        output[text_mask] = out_text.view(-1, d)
        output[summary_mask] = residual_summary

        return output


# ---------------------------------------------------------------------------
# SummaryTransformerDecoder
# ---------------------------------------------------------------------------

class SummaryTransformerDecoder(TransformerDecoder):
    """TransformerDecoder with summary attention support.

    Adds:
    - summary_ctx parameter in forward
    - Distillation output collection
    - Summary token filtering before output layer
    """

    def __init__(self, *, enable_distill: bool = False, **kwargs):
        super().__init__(**kwargs)
        self._enable_distill = enable_distill

    def forward(
        self,
        tokens: Optional[torch.Tensor],
        *,
        mask: Optional[torch.Tensor] = None,
        encoder_input: Optional[torch.Tensor] = None,
        encoder_mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        input_embeds: Optional[torch.Tensor] = None,
        summary_ctx: Optional[SummaryBatchContext] = None,
        **kwargs,
    ) -> Union[torch.Tensor, tuple]:

        self._validate_inputs(
            tokens=tokens, mask=mask,
            encoder_input=encoder_input, encoder_mask=encoder_mask,
            input_pos=input_pos, input_embeds=input_embeds,
        )

        # Embedding
        h = self.tok_embeddings(tokens) if input_embeds is None else input_embeds

        hidden = []
        attn_outputs = [] if self._enable_distill else None

        for i, layer in enumerate(self.layers):
            if i in self.output_hidden_states:
                hidden.append(h)

            result = layer(
                h,
                mask=mask,
                encoder_input=encoder_input,
                encoder_mask=encoder_mask,
                input_pos=input_pos,
                summary_ctx=summary_ctx,
                **kwargs,
            )

            if isinstance(result, tuple):
                h, distill_out = result
                if attn_outputs is not None and distill_out is not None:
                    # Filter out summary positions from distill output
                    if summary_ctx is not None:
                        text_mask = ~summary_ctx.summary_mask
                        distill_out = distill_out[text_mask].view(
                            h.shape[0], -1, h.shape[-1]
                        )
                    attn_outputs.append(distill_out)
            else:
                h = result

        if len(self.layers) in self.output_hidden_states:
            hidden.append(h)

        # Filter summary tokens before output layer
        if summary_ctx is not None:
            text_mask = ~summary_ctx.summary_mask  # [b, s]
            h = h[text_mask].view(h.shape[0], -1, h.shape[-1])

        # Norm + output projection
        output = self.unembed(h)

        # Build return value
        if hidden:
            output = [*hidden, output]
        if attn_outputs:
            return output, attn_outputs
        return output


# ---------------------------------------------------------------------------
# Qwen3SummaryModel
# ---------------------------------------------------------------------------

class Qwen3SummaryModel(Model):
    """Qwen3 model with Summary Attention.

    Supports hybrid mode: some layers use SummaryQwen3Attention, others use
    standard Qwen3Attention with summary token bypass (controlled by
    config.summary_layer_freq). When summary_ctx is not provided during
    forward, all layers fall back to standard Qwen3Attention behavior.
    """

    def __init__(self, config: Qwen3SummaryAttentionConfig):
        super().__init__(config)
        self.config = config

        head_dim = config.head_dim or config.embed_dim // config.num_heads
        num_heads = config.num_heads
        num_kv_heads = config.num_kv_heads if config.num_kv_heads else num_heads

        # RoPE (standard 1D only for now)
        self.rope = RotaryPositionalEmbeddings(
            dim=head_dim, max_seq_len=config.max_seq_len, base=config.rope_base,
        )

        # Build layers — hybrid: SA layers vs. full-attention bypass layers
        layer_is_summary = config.get_layer_is_summary()
        layers = nn.ModuleList()
        for i in range(config.num_layers):
            layer_sliding_window = config.get_layer_sliding_window(i)
            if layer_is_summary[i]:
                # Summary attention layer
                self_attn = SummaryQwen3Attention(
                    summary_config=config,
                    layer_index=i,
                    embed_dim=config.embed_dim,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    q_proj=nn.Linear(config.embed_dim, num_heads * head_dim, bias=config.q_proj_bias),
                    k_proj=nn.Linear(config.embed_dim, num_kv_heads * head_dim, bias=config.k_proj_bias),
                    v_proj=nn.Linear(config.embed_dim, num_kv_heads * head_dim, bias=config.v_proj_bias),
                    output_proj=nn.Linear(num_heads * head_dim, config.embed_dim, bias=False),
                    pos_embeddings=self.rope,
                    q_norm=RMSNorm(dim=head_dim, eps=config.norm_eps) if config.q_norm else None,
                    k_norm=RMSNorm(dim=head_dim, eps=config.norm_eps) if config.k_norm else None,
                    kv_cache=None,
                    max_seq_len=config.max_seq_len,
                    attn_dropout=config.attn_dropout,
                    attention_function=config.attention_function,
                    sliding_window=layer_sliding_window,
                )
                layer = SummaryTransformerSelfAttentionLayer(
                    attn=self_attn,
                    mlp=qwen3_mlp(dim=config.embed_dim, hidden_dim=config.intermediate_dim),
                    sa_norm=RMSNorm(dim=config.embed_dim, eps=config.norm_eps),
                    mlp_norm=RMSNorm(dim=config.embed_dim, eps=config.norm_eps),
                    sa_norm_summary=(
                        RMSNorm(dim=config.embed_dim, eps=config.norm_eps)
                        if config.summary_independent_attention_layernorm
                        else None
                    ),
                )
            else:
                # Standard causal attention layer with summary bypass
                self_attn = Qwen3Attention(
                    embed_dim=config.embed_dim,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    q_proj=nn.Linear(config.embed_dim, num_heads * head_dim, bias=config.q_proj_bias),
                    k_proj=nn.Linear(config.embed_dim, num_kv_heads * head_dim, bias=config.k_proj_bias),
                    v_proj=nn.Linear(config.embed_dim, num_kv_heads * head_dim, bias=config.v_proj_bias),
                    output_proj=nn.Linear(num_heads * head_dim, config.embed_dim, bias=False),
                    pos_embeddings=self.rope,
                    q_norm=RMSNorm(dim=head_dim, eps=config.norm_eps) if config.q_norm else None,
                    k_norm=RMSNorm(dim=head_dim, eps=config.norm_eps) if config.k_norm else None,
                    kv_cache=None,
                    max_seq_len=config.max_seq_len,
                    attn_dropout=config.attn_dropout,
                    attention_function=config.attention_function,
                    sliding_window=layer_sliding_window,
                    layer_index=i,
                )
                layer = FullAttentionLayerWithSummaryBypass(
                    attn=self_attn,
                    mlp=qwen3_mlp(dim=config.embed_dim, hidden_dim=config.intermediate_dim),
                    sa_norm=RMSNorm(dim=config.embed_dim, eps=config.norm_eps),
                    mlp_norm=RMSNorm(dim=config.embed_dim, eps=config.norm_eps),
                )
            layers.append(layer)

        # Embeddings + output
        tok_embeddings = nn.Embedding(config.vocab_size, config.embed_dim)
        if config.tie_word_embeddings:
            output_proj = TiedLinear(tok_embeddings)
        else:
            output_proj = nn.Linear(config.embed_dim, config.vocab_size, bias=False)

        self.model = SummaryTransformerDecoder(
            tok_embeddings=tok_embeddings,
            layers=layers,
            max_seq_len=config.max_seq_len,
            num_heads=num_heads,
            head_dim=head_dim,
            norm=RMSNorm(dim=config.embed_dim, eps=config.norm_eps),
            output=output_proj,
            enable_distill=config.enable_summary_distill_attention,
        )

        # Summary embedding (independent parameters)
        if config.summary_independent_parameters:
            self.summary_embedding = nn.Embedding(config.summary_token_num, config.embed_dim)

    def forward(
        self,
        tokens: Optional[torch.Tensor] = None,
        *,
        summary_ctx: Optional[SummaryBatchContext] = None,
        **kwargs,
    ) -> Union[torch.Tensor, tuple]:
        # Replace summary token embeddings with independent embedding
        if (
            summary_ctx is not None
            and self.config.summary_independent_parameters
            and hasattr(self, 'summary_embedding')
        ):
            h = self.model.tok_embeddings(tokens)
            summary_mask = summary_ctx.summary_mask  # [b, s] bool
            num_summary = summary_mask.sum().item()
            summary_ids = torch.arange(
                self.config.summary_token_num, device=h.device, dtype=torch.long,
            ).repeat(num_summary // self.config.summary_token_num)
            h = h.clone()
            h[summary_mask] = self.summary_embedding(summary_ids)
            return self.model(
                tokens=None, input_embeds=h,
                summary_ctx=summary_ctx, **kwargs,
            )
        return self.model(tokens=tokens, summary_ctx=summary_ctx, **kwargs)

    def get_initializer(self, name: str) -> Callable[[torch.Tensor], None]:
        """Return an initializer function for the given parameter name."""
        module_name = name.rsplit(".", 1)[0]
        param_suffix = name.rsplit(".", 1)[1] if "." in name else ""

        module = None
        parent_module = None
        for mod_name, mod in self.named_modules():
            if mod_name == module_name:
                module = mod
                if "." in mod_name:
                    parent_name = ".".join(mod_name.rsplit(".", 1)[:-1])
                    for p_name, p_mod in self.named_modules():
                        if p_name == parent_name:
                            parent_module = p_mod
                            break
                break

        # Handle TiedLinear
        if module is None:
            parts = module_name.split(".")
            try:
                current = self
                for part in parts:
                    current = getattr(current, part)
                if isinstance(current, TiedLinear):
                    module = current.tied_module
                else:
                    module = current if isinstance(current, nn.Module) else None
            except AttributeError:
                module = None

        if module is None:
            return _lecun_normal_

        # Qwen3Attention covers both standard and SummaryQwen3Attention (subclass)
        is_attention = isinstance(parent_module, Qwen3Attention)
        is_mlp = isinstance(parent_module, FeedForward)

        if isinstance(module, nn.Linear):
            def linear_init(tensor: torch.Tensor):
                if param_suffix == "weight" and tensor is not None:
                    if is_attention or is_mlp:
                        init.xavier_uniform_(tensor)
                    else:
                        _lecun_normal_(tensor)
                elif param_suffix == "bias" and tensor is not None:
                    if is_mlp:
                        init.normal_(tensor, mean=0.0, std=1e-6)
                    else:
                        init.zeros_(tensor)
            return linear_init

        elif isinstance(module, nn.Embedding):
            def embedding_init(tensor: torch.Tensor):
                if param_suffix == "weight" and tensor is not None:
                    _default_flax_embed_init(tensor)
            return embedding_init

        elif (isinstance(module, RMSNorm)
              or "RMSNorm" in module.__class__.__name__
              or "LayerNorm" in module.__class__.__name__):
            def norm_init(tensor: torch.Tensor):
                if param_suffix in ["weight", "scale"] and tensor is not None:
                    init.ones_(tensor)
                elif param_suffix == "bias" and tensor is not None:
                    init.zeros_(tensor)
            return norm_init

        return _lecun_normal_

    def convert_hf_state_dict(
        self,
        hf_state_dict: Dict[str, torch.Tensor],
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Convert HF Qwen3 state dict to muse Qwen3SummaryModel format.

        Handles both standard Qwen3 keys and summary-specific keys.
        In hybrid mode, summary-specific keys for non-SA layers are skipped.
        """
        converted = {}
        skipped = []
        layer_is_summary = self.config.get_layer_is_summary()

        for hf_key, tensor in hf_state_dict.items():
            # Skip lm_head if tie_word_embeddings
            if self.config.tie_word_embeddings and hf_key == "lm_head.weight":
                skipped.append(hf_key)
                continue

            # Embedding
            if hf_key == "model.embed_tokens.weight":
                converted["model.tok_embeddings.weight"] = tensor
                continue

            # Final norm (RMSNorm: .weight → .scale)
            if hf_key == "model.norm.weight":
                converted["model.norm.scale"] = tensor
                continue

            # Output layer
            if hf_key == "lm_head.weight":
                converted["model.output.weight"] = tensor
                continue

            # Summary embedding (model-level)
            if hf_key == "summary_embedding.weight":
                converted["summary_embedding.weight"] = tensor
                continue

            # Transformer layers
            if hf_key.startswith("model.layers."):
                parts = hf_key.split(".", 3)
                if len(parts) < 4:
                    skipped.append(hf_key)
                    continue
                layer_idx = parts[2]
                rest = parts[3]

                # In hybrid mode, skip summary-specific keys for non-SA layers
                layer_i = int(layer_idx)
                is_summary_key = (
                    "summary" in rest
                )
                if is_summary_key and layer_i < len(layer_is_summary) and not layer_is_summary[layer_i]:
                    skipped.append(hf_key)
                    continue

                # Summary layer norm: input_layernorm_summary.weight → sa_norm_summary.scale
                if rest == "input_layernorm_summary.weight":
                    converted[f"model.layers.{layer_idx}.sa_norm_summary.scale"] = tensor
                    continue

                # Attention (covers both standard and summary projections)
                # self_attn.q_proj.weight → attn.q_proj.weight
                # self_attn.q_proj_summary.weight → attn.q_proj_summary.weight
                if rest.startswith("self_attn."):
                    attn_key = rest.replace("self_attn.", "attn.")
                    attn_key = attn_key.replace("o_proj", "output_proj")
                    attn_key = attn_key.replace("q_norm.weight", "q_norm.scale")
                    attn_key = attn_key.replace("k_norm.weight", "k_norm.scale")
                    converted[f"model.layers.{layer_idx}.{attn_key}"] = tensor
                    continue

                # MLP
                if rest.startswith("mlp."):
                    mlp_key = rest.replace("mlp.", "")
                    if mlp_key == "gate_proj.weight":
                        converted[f"model.layers.{layer_idx}.mlp.w1.weight"] = tensor
                    elif mlp_key == "up_proj.weight":
                        converted[f"model.layers.{layer_idx}.mlp.w3.weight"] = tensor
                    elif mlp_key == "down_proj.weight":
                        converted[f"model.layers.{layer_idx}.mlp.w2.weight"] = tensor
                    else:
                        skipped.append(hf_key)
                    continue

                # Layer norms
                if rest == "input_layernorm.weight":
                    converted[f"model.layers.{layer_idx}.sa_norm.scale"] = tensor
                    continue
                if rest == "post_attention_layernorm.weight":
                    converted[f"model.layers.{layer_idx}.mlp_norm.scale"] = tensor
                    continue

            skipped.append(hf_key)

        if skipped:
            logger.warning(f"Skipped {len(skipped)} keys: {skipped[:10]}")
        logger.info(f"Converted {len(converted)} / {len(hf_state_dict)} keys")
        return converted

    def revert_hf_state_dict(
        self,
        muse_state_dict: Dict[str, torch.Tensor],
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Revert muse state dict back to HF Qwen3 format.

        Includes summary-specific keys so the HF summary-aware
        modeling_qwen3.py can load them.
        """
        hf_state_dict = {}
        skipped = []

        for muse_key, tensor in muse_state_dict.items():
            # Embedding
            if muse_key == "model.tok_embeddings.weight":
                hf_state_dict["model.embed_tokens.weight"] = tensor
                continue
            # Final norm
            if muse_key == "model.norm.scale":
                hf_state_dict["model.norm.weight"] = tensor
                continue
            # Output layer
            if muse_key == "model.output.weight":
                hf_state_dict["lm_head.weight"] = tensor
                continue
            # Summary embedding (model-level)
            if muse_key == "summary_embedding.weight":
                hf_state_dict["summary_embedding.weight"] = tensor
                continue

            # Transformer layers
            if muse_key.startswith("model.layers."):
                parts = muse_key.split(".", 3)
                if len(parts) < 4:
                    skipped.append(muse_key)
                    continue
                layer_idx = parts[2]
                rest = parts[3]

                # Summary layer norm: sa_norm_summary.scale → input_layernorm_summary.weight
                if rest == "sa_norm_summary.scale":
                    hf_state_dict[f"model.layers.{layer_idx}.input_layernorm_summary.weight"] = tensor
                    continue

                # Attention (covers both standard and summary projections)
                # attn.q_proj.weight → self_attn.q_proj.weight
                # attn.q_proj_summary.weight → self_attn.q_proj_summary.weight
                if rest.startswith("attn."):
                    hf_rest = rest.replace("attn.", "self_attn.")
                    hf_rest = hf_rest.replace("output_proj", "o_proj")
                    hf_rest = hf_rest.replace("q_norm_summary.scale", "q_norm_summary.weight")
                    hf_rest = hf_rest.replace("k_norm_summary.scale", "k_norm_summary.weight")
                    hf_rest = hf_rest.replace("q_norm.scale", "q_norm.weight")
                    hf_rest = hf_rest.replace("k_norm.scale", "k_norm.weight")
                    hf_state_dict[f"model.layers.{layer_idx}.{hf_rest}"] = tensor
                    continue

                # MLP
                if rest == "mlp.w1.weight":
                    hf_state_dict[f"model.layers.{layer_idx}.mlp.gate_proj.weight"] = tensor
                    continue
                if rest == "mlp.w3.weight":
                    hf_state_dict[f"model.layers.{layer_idx}.mlp.up_proj.weight"] = tensor
                    continue
                if rest == "mlp.w2.weight":
                    hf_state_dict[f"model.layers.{layer_idx}.mlp.down_proj.weight"] = tensor
                    continue

                # Layer norms
                if rest == "sa_norm.scale":
                    hf_state_dict[f"model.layers.{layer_idx}.input_layernorm.weight"] = tensor
                    continue
                if rest == "mlp_norm.scale":
                    hf_state_dict[f"model.layers.{layer_idx}.post_attention_layernorm.weight"] = tensor
                    continue

            skipped.append(muse_key)

        # tie_word_embeddings: ensure lm_head exists
        if self.config.tie_word_embeddings and "lm_head.weight" not in hf_state_dict:
            if "model.embed_tokens.weight" in hf_state_dict:
                hf_state_dict["lm_head.weight"] = hf_state_dict["model.embed_tokens.weight"]

        if skipped:
            logger.warning(f"Skipped {len(skipped)} muse keys: {skipped[:10]}")
        logger.info(f"Reverted {len(hf_state_dict)} / {len(muse_state_dict)} keys")
        return hf_state_dict

    def get_layers_to_shard(self):
        return self.model.layers

    def get_checkpointable_module_classes(self):
        return {SummaryTransformerSelfAttentionLayer, FullAttentionLayerWithSummaryBypass}

    
