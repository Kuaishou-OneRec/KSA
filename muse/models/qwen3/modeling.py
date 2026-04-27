from typing import Dict, Callable, Optional, List
from functools import partial
import math
import torch
import torch.nn as nn
import torch.nn.init as init
import logging
import torch.nn.functional as F

from muse.models.base import Model
from muse.config import Qwen3Config
from muse.layers.position_embeddings import RotaryPositionalEmbeddings
from muse.layers.transformer import TransformerDecoder, TransformerSelfAttentionLayer
from muse.layers.rms_norm import RMSNorm
from muse.layers.linear import TiedLinear
from muse.models.qwen3._layers import qwen3_mlp, Qwen3Attention
from muse.layers.feed_forward import FeedForward

# Import will be done when muse.models is imported, avoiding circular import
# The actual registration happens in __init__.py after import

logger = logging.getLogger(__name__)


def lecun_normal_(tensor: torch.Tensor) -> None:
    """LeCun normal initialization.
    
    LeCun normal initialization: std = sqrt(1 / fan_in)
    This is similar to Kaiming normal but uses fan_in instead of fan_out.
    
    For Linear layers: fan_in = in_features
    For Conv2d layers: fan_in = in_channels * kernel_size[0] * kernel_size[1]
    """
    if tensor.dim() < 2:
        # For 1D tensors (bias, etc.), use a small std
        std = 0.01
    elif tensor.dim() == 2:
        # Linear layer: (out_features, in_features)
        fan_in = tensor.size(1)
        std = math.sqrt(1.0 / fan_in)
    else:
        # Convolutional layer: (out_channels, in_channels, kernel_h, kernel_w, ...)
        # fan_in = in_channels * product of kernel sizes
        fan_in = tensor.size(1)
        for s in tensor.size()[2:]:
            fan_in *= s
        std = math.sqrt(1.0 / fan_in)
    init.normal_(tensor, mean=0.0, std=std)


def default_flax_embed_init(tensor: torch.Tensor) -> None:
    """Default Flax embedding initialization.
    
    Uses normal distribution with std = 1.0
    """
    init.normal_(tensor, mean=0.0, std=1.0)


class Qwen3Model(Model):
    def __init__(self, config: Qwen3Config):
        super().__init__(config)
        self.config = config

        head_dim = config.head_dim or config.embed_dim // config.num_heads
        num_heads = config.num_heads
        num_kv_heads = config.num_kv_heads if config.num_kv_heads else num_heads

        self.rope = RotaryPositionalEmbeddings(
            dim=head_dim, max_seq_len=config.max_seq_len, base=config.rope_base
        )

        layers = nn.ModuleList()
        for layer_idx in range(config.num_layers):
            layer_sliding_window = config.get_layer_sliding_window(layer_idx)
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
                q_norm=RMSNorm(dim=head_dim, eps=config.norm_eps) if config.q_norm else None, # norm on head_dim
                k_norm=RMSNorm(dim=head_dim, eps=config.norm_eps) if config.k_norm else None,
                kv_cache=None,
                max_seq_len=config.max_seq_len,
                attn_dropout=config.attn_dropout,
                attention_function=config.attention_function,
                sliding_window=layer_sliding_window,
                layer_index=layer_idx,
            )
            mlp = qwen3_mlp(dim=config.embed_dim, hidden_dim=config.intermediate_dim)
            layer = TransformerSelfAttentionLayer(
                attn=self_attn,
                mlp=mlp,
                sa_norm=RMSNorm(dim=config.embed_dim, eps=config.norm_eps),
                mlp_norm=RMSNorm(dim=config.embed_dim, eps=config.norm_eps),
            )
            layers.append(layer)

        tok_embeddings = nn.Embedding(config.vocab_size, config.embed_dim)
        if config.tie_word_embeddings:
            output_proj = TiedLinear(tok_embeddings)
        else:
            output_proj = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        
        self.model = TransformerDecoder(
            tok_embeddings=tok_embeddings,
            layers=layers,
            max_seq_len=config.max_seq_len,
            num_heads=num_heads,
            head_dim=head_dim,
            norm=RMSNorm(dim=config.embed_dim, eps=config.norm_eps),
            output=output_proj,
        )

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
    
    def get_initializer(self, name: str) -> Callable[[torch.Tensor], None]:
        """Return an initializer function for the given parameter name.
        
        Initialization scheme:
        - Attention modules (Qwen3Attention): xavier_uniform_ for weights, zeros_ for bias
        - MLP modules (FeedForward): xavier_uniform_ for weights, normal_(std=1e-6) for bias
        - Linear/Conv2d layers: lecun_normal_ for weights, zeros_ for bias
        - Embedding layers: default_flax_embed_init (normal with std=1.0)
        - LayerNorm/RMSNorm: weight=1, bias=0

        Args:
            name: Parameter name (e.g., "model.layers.0.attn.q_proj.weight")
            
        Returns:
            A callable function that takes a tensor and initializes it
        """
        # Find the module corresponding to this parameter name
        # Remove the parameter suffix (e.g., ".weight", ".bias", ".scale")
        module_name = name.rsplit(".", 1)[0]
        param_suffix = name.rsplit(".", 1)[1] if "." in name else ""
        
        # Get the module and its parent
        module = None
        parent_module = None
        for mod_name, mod in self.named_modules():
            if mod_name == module_name:
                module = mod
                # Get parent module name
                if "." in mod_name:
                    parent_name = ".".join(mod_name.rsplit(".", 1)[:-1])
                    for p_name, p_mod in self.named_modules():
                        if p_name == parent_name:
                            parent_module = p_mod
                            break
                break
        
        # Handle TiedLinear: its weight is actually the tied_module's weight
        # TiedLinear is not an nn.Module, so it won't appear in named_modules()
        if module is None:
            # Try to get the module by attribute access
            parts = module_name.split(".")
            try:
                current = self
                for part in parts:
                    current = getattr(current, part)
                # Check if it's a TiedLinear
                if isinstance(current, TiedLinear):
                    # TiedLinear's weight is the tied_module's weight
                    module = current.tied_module
                else:
                    module = current if isinstance(current, nn.Module) else None
            except AttributeError:
                module = None
        
        if module is None:
            # If module not found, return a default lecun_normal initializer
            return lecun_normal_
        
        # Check if this is part of an Attention module
        is_attention = isinstance(parent_module, Qwen3Attention)
        
        # Check if this is part of an MLP/FeedForward module
        is_mlp = isinstance(parent_module, FeedForward)
        
        # Define initializer based on module type
        if isinstance(module, nn.Linear):
            def linear_init(tensor: torch.Tensor):
                if param_suffix == "weight" and tensor is not None:
                    if is_attention:
                        # Attention layers use xavier_uniform_
                        init.xavier_uniform_(tensor)
                    elif is_mlp:
                        # MLP layers use xavier_uniform_ for weights
                        init.xavier_uniform_(tensor)
                    else:
                        # Other Linear layers use lecun_normal_
                        lecun_normal_(tensor)
                elif param_suffix == "bias" and tensor is not None:
                    if is_mlp:
                        # MLP bias uses normal with std=1e-6
                        init.normal_(tensor, mean=0.0, std=1e-6)
                    else:
                        # Other biases use zeros
                        init.zeros_(tensor)
            return linear_init
            
        elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, 
                                  nn.ConvTranspose1d, nn.ConvTranspose2d)):
            def conv_init(tensor: torch.Tensor):
                if param_suffix == "weight" and tensor is not None:
                    lecun_normal_(tensor)
                elif param_suffix == "bias" and tensor is not None:
                    init.zeros_(tensor)
            return conv_init
            
        elif isinstance(module, nn.Embedding):
            # TODO: use better embedding initialization
            def embedding_init(tensor: torch.Tensor):
                if param_suffix == "weight" and tensor is not None:
                    # Use default_flax_embed_init (normal with std=1.0)
                    default_flax_embed_init(tensor)
            return embedding_init
            
        elif isinstance(module, nn.MultiheadAttention):
            def mha_init(tensor: torch.Tensor):
                if param_suffix == "weight" and tensor is not None:
                    init.xavier_uniform_(tensor)
                elif param_suffix == "bias" and tensor is not None:
                    init.zeros_(tensor)
            return mha_init
            
        elif (isinstance(module, (nn.GroupNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))
              or "LayerNorm" in module.__class__.__name__
              or "RMSNorm" in module.__class__.__name__
              or isinstance(module, RMSNorm)):
            def norm_init(tensor: torch.Tensor):
                if param_suffix in ["weight", "scale"] and tensor is not None:
                    init.ones_(tensor)
                elif param_suffix == "bias" and tensor is not None:
                    init.zeros_(tensor)
            return norm_init
        
        # Default: lecun_normal initialization
        return lecun_normal_

    def get_layers_to_shard(self):
        # Return the ModuleList directly - it's iterable and supports reversed()
        # fully_shard will be applied to each individual layer, not the ModuleList itself
        return self.model.layers

    def get_checkpointable_module_classes(self):
        return {TransformerSelfAttentionLayer}

    def convert_hf_state_dict(self,
                              hf_state_dict: Dict[str, torch.Tensor],
                              **kwargs) -> Dict[str, torch.Tensor]:
        """Convert a Hugging Face state dictionary to a model state dictionary
        
        Args:
            hf_state_dict (Dict[str, torch.Tensor]): The Hugging Face state dictionary.
            **kwargs: Additional keyword arguments.
        
        Returns:
            A dictionary of model state with converted key names.
        """
        converted_state_dict = {}
        tie_word_embeddings = self.config.tie_word_embeddings
        skipped_keys = []
        
        for hf_key, tensor in hf_state_dict.items():
            # Skip lm_head if tie_word_embeddings is True
            if tie_word_embeddings and hf_key == "lm_head.weight":
                skipped_keys.append(hf_key)
                continue
            
            # Handle embedding layer
            if hf_key == "model.embed_tokens.weight":
                converted_key = "model.tok_embeddings.weight"
                converted_state_dict[converted_key] = tensor
                continue
            
            # Handle final norm (RMSNorm uses 'scale' not 'weight')
            if hf_key == "model.norm.weight":
                converted_key = "model.norm.scale"
                converted_state_dict[converted_key] = tensor
                continue
            
            # Handle output layer (lm_head)
            if hf_key == "lm_head.weight":
                converted_key = "model.output.weight"
                converted_state_dict[converted_key] = tensor
                continue
            
            # Handle transformer layers
            if hf_key.startswith("model.layers."):
                # Extract layer index and remaining key
                parts = hf_key.split(".", 3)  # ["model", "layers", "{i}", "rest"]
                if len(parts) < 4:
                    # Skip keys that don't match expected format
                    skipped_keys.append(hf_key)
                    continue
                
                layer_idx = parts[2]
                rest_key = parts[3]
                
                # Handle attention weights
                if rest_key.startswith("self_attn."):
                    attn_key = rest_key.replace("self_attn.", "attn.")
                    # Map o_proj to output_proj
                    attn_key = attn_key.replace("o_proj", "output_proj")
                    # Handle q_norm and k_norm (they are RMSNorm, use 'scale' not 'weight')
                    # Hugging Face uses .weight, Muse uses .scale
                    attn_key = attn_key.replace("q_norm.weight", "q_norm.scale")
                    attn_key = attn_key.replace("k_norm.weight", "k_norm.scale")
                    converted_key = f"model.layers.{layer_idx}.{attn_key}"
                    converted_state_dict[converted_key] = tensor
                    continue
                
                # Handle MLP weights
                # Hugging Face uses gate_proj/up_proj/down_proj
                # Muse uses w1/w3/w2 (gate_proj->w1, up_proj->w3, down_proj->w2)
                if rest_key.startswith("mlp."):
                    mlp_key = rest_key.replace("mlp.", "")
                    if mlp_key == "gate_proj.weight":
                        converted_key = f"model.layers.{layer_idx}.mlp.w1.weight"
                    elif mlp_key == "up_proj.weight":
                        converted_key = f"model.layers.{layer_idx}.mlp.w3.weight"
                    elif mlp_key == "down_proj.weight":
                        converted_key = f"model.layers.{layer_idx}.mlp.w2.weight"
                    else:
                        # Skip unknown MLP keys
                        skipped_keys.append(hf_key)
                        continue
                    converted_state_dict[converted_key] = tensor
                    continue
                
                # Handle layer norms (RMSNorm uses 'scale' not 'weight')
                if rest_key == "input_layernorm.weight":
                    converted_key = f"model.layers.{layer_idx}.sa_norm.scale"
                    converted_state_dict[converted_key] = tensor
                    continue
                
                if rest_key == "post_attention_layernorm.weight":
                    converted_key = f"model.layers.{layer_idx}.mlp_norm.scale"
                    converted_state_dict[converted_key] = tensor
                    continue
            
            # If key doesn't match any pattern, skip it
            # This handles keys like "model.config" or other non-weight keys
            skipped_keys.append(hf_key)
        
        if skipped_keys:
            logger.warning(
                f"Skipped {len(skipped_keys)} keys during conversion. "
                f"First few: {skipped_keys[:10]}"
            )
        
        logger.info(
            f"Converted {len(converted_state_dict)} keys from "
            f"{len(hf_state_dict)} Hugging Face keys"
        )
        
        return converted_state_dict

    def revert_hf_state_dict(
        self,
        muse_state_dict: Dict[str, torch.Tensor],
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Revert a Muse(Qwen3Model in this repo) state dict back to the original HF Qwen3 naming.

        这是 `convert_hf_state_dict` 的反函数：
        - 输入：`muse/models/qwen3/modeling.py`（本 repo 的实现）保存出来的 ckpt key
        - 输出：`muse/models/qwen3/modeling_qwen_ori.py`（原始 HF 代码）所期望的 ckpt key

        规则：严格逐条逆转 `convert_hf_state_dict` 的映射。

        注意：
        - 当 `tie_word_embeddings=True` 时，muse 的 `model.output.weight` 与 `model.tok_embeddings.weight` 共享；
          原始 HF 里通常仍然会存在 `lm_head.weight`（是否保存取决于保存方式）。
          为了保证“可逆”，这里如果看到 `model.output.weight` 就会写出 `lm_head.weight`；
          如果 tie_word_embeddings=True 且 `model.output.weight` 不存在，则会用 `model.tok_embeddings.weight` 补出。

        Args:
            muse_state_dict: muse 格式 state_dict。

        Returns:
            hf_state_dict: 原始 HF 格式 state_dict。
        """

        hf_state_dict: Dict[str, torch.Tensor] = {}
        tie_word_embeddings = self.config.tie_word_embeddings
        skipped_keys: List[str] = []

        for muse_key, tensor in muse_state_dict.items():
            # Embedding layer
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

            # Transformer layers
            if muse_key.startswith("model.layers."):
                parts = muse_key.split(".", 3)
                if len(parts) < 4:
                    skipped_keys.append(muse_key)
                    continue

                layer_idx = parts[2]
                rest_key = parts[3]

                # Attention
                if rest_key.startswith("attn."):
                    hf_rest = rest_key.replace("attn.", "self_attn.")
                    # Map output_proj -> o_proj
                    hf_rest = hf_rest.replace("output_proj", "o_proj")
                    # q_norm/k_norm: muse uses .scale, HF uses .weight
                    hf_rest = hf_rest.replace("q_norm.scale", "q_norm.weight")
                    hf_rest = hf_rest.replace("k_norm.scale", "k_norm.weight")
                    hf_state_dict[f"model.layers.{layer_idx}.{hf_rest}"] = tensor
                    continue

                # MLP
                if rest_key.startswith("mlp."):
                    if rest_key == "mlp.w1.weight":
                        hf_state_dict[f"model.layers.{layer_idx}.mlp.gate_proj.weight"] = tensor
                        continue
                    if rest_key == "mlp.w3.weight":
                        hf_state_dict[f"model.layers.{layer_idx}.mlp.up_proj.weight"] = tensor
                        continue
                    if rest_key == "mlp.w2.weight":
                        hf_state_dict[f"model.layers.{layer_idx}.mlp.down_proj.weight"] = tensor
                        continue

                # Layer norms
                if rest_key == "sa_norm.scale":
                    hf_state_dict[f"model.layers.{layer_idx}.input_layernorm.weight"] = tensor
                    continue

                if rest_key == "mlp_norm.scale":
                    hf_state_dict[f"model.layers.{layer_idx}.post_attention_layernorm.weight"] = tensor
                    continue

            skipped_keys.append(muse_key)

        # tie_word_embeddings 时，允许用 embed_tokens 补出 lm_head
        if tie_word_embeddings and "lm_head.weight" not in hf_state_dict:
            # 如果 muse_state_dict 里没存 output.weight，至少保证 hf ckpt 里有 lm_head.weight
            if "model.embed_tokens.weight" in hf_state_dict:
                hf_state_dict["lm_head.weight"] = hf_state_dict["model.embed_tokens.weight"]

        if skipped_keys:
            logger.warning(
                f"Skipped {len(skipped_keys)} muse keys during revert. First few: {skipped_keys[:10]}"
            )

        logger.info(
            f"Reverted {len(hf_state_dict)} keys from {len(muse_state_dict)} muse keys"
        )

        return hf_state_dict

    def generate(
        self,  # muse.models.qwen3.Qwen3Model
        input_ids: torch.Tensor,
        max_length: int = 512,
        max_new_tokens: Optional[int] = None,  # 添加max_new_tokens参数
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        **kwargs
    ) -> List[torch.Tensor]:
        """
        使用Muse模型进行文本生成

        Args:
            model: Muse模型实例 (muse.models.qwen3.Qwen3Model)
            input_ids: 输入token ids，形状为 [batch_size, seq_length]
            max_length: 生成序列的最大长度
            max_new_tokens: 要生成的新token数量
            temperature: 采样温度
            top_k: 仅考虑概率最高的k个token
            top_p: 仅考虑累积概率达到p的token
            eos_token_id: 结束token id
            pad_token_id: 填充token id
            **kwargs: 其他传递给模型的参数

        Returns:
            生成的token序列列表，每个元素形状为 [seq_length]
        """
        device = input_ids.device
        batch_size = input_ids.size(0)
        input_seq_len = input_ids.size(1)

        # 处理max_new_tokens参数
        if max_new_tokens is not None:
            # 如果指定了max_new_tokens，计算生成的总长度
            max_length = input_seq_len + max_new_tokens

        # 设置最大生成长度
        if max_length <= input_seq_len:
            return input_ids.tolist()

        # 初始化生成的序列
        generated = input_ids.clone()

        # 设置KV缓存
        self.model.setup_caches(
            batch_size=batch_size,
            dtype=next(self.parameters()).dtype,
            decoder_max_seq_len=max_length
        )

        # 预热阶段 - 处理输入序列，填充KV缓存
        with torch.no_grad():
            # 创建预热阶段的位置id (从0到input_seq_len-1)
            prefill_pos = torch.arange(input_seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
            # 使用完整的model()调用，并提供正确的input_pos
            self(generated, input_pos=prefill_pos,# is_causal=True
                **kwargs)

        # 自回归生成阶段
        for step in range(input_seq_len, max_length):
            with torch.no_grad():
                # 当前的token是生成序列的最后一个token
                current_token = generated[:, -1:]
                # 计算当前的位置id (应该是step，因为是接着输入序列之后的位置)
                current_pos = torch.tensor([[step]], device=device).expand(batch_size, -1)

                # 前向传播 - 使用完整的model()调用，提供当前位置id
                logits = self(current_token, input_pos=current_pos,#  is_causal=True
                                **kwargs)

                # 采样下一个token
                next_token_logits = logits[:, -1, :]

                # 应用温度缩放
                if temperature > 0:
                    next_token_logits = next_token_logits / temperature
                
                # 应用top-k采样
                if top_k is not None:
                    # 修复：将非top-k的logits设置为负无穷，而不是只保留top-k的值
                    # 这样可以保持原始的token索引信息
                    top_k_logits, _ = next_token_logits.topk(top_k, dim=-1)
                    # 创建掩码：将低于top-k阈值的logits设为负无穷
                    next_token_logits = torch.where(
                        next_token_logits >= top_k_logits[:, -1:], 
                        next_token_logits, 
                        -float("inf")
                    )
                
                # 应用top-p采样 (nucleus sampling)
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    # 移除累积概率超过p的token
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # 确保至少保留一个token
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = 0
                    
                    # 修复：使用scatter将掩码映射回原始token索引空间
                    indices_to_remove = torch.zeros_like(next_token_logits, dtype=torch.bool)
                    indices_to_remove = indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits = next_token_logits.masked_fill(indices_to_remove, -float("inf"))

                # 添加安全检查：确保至少有一个有效token
                if (next_token_logits == -float("inf")).all(): 
                    next_token_logits = torch.zeros_like(next_token_logits)

                # 计算概率分布
                probs = F.softmax(next_token_logits, dim=-1)

                # 采样下一个token
                next_token = torch.multinomial(probs, num_samples=1)

                # 将生成的token添加到序列中
                generated = torch.cat([generated, next_token], dim=1)

                # 检查是否所有序列都已生成结束token
                if eos_token_id is not None:
                    done = (next_token == eos_token_id).any(dim=1).all()
                    if done:
                        break


        self.model.reset_caches()

        # 将生成的序列转换为列表
        generated_list = generated.tolist()

        return generated_list
