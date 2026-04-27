import torch
import torch.nn as nn
from typing import Callable, Optional, List, Dict, Any
from functools import partial
import os
import json
import glob
import logging
import contextlib
from pathlib import Path
from safetensors import torch as safetensors_torch
from muse.config.model_config import ModelConfig
from muse.config import get_config
from muse.training.checkpoint import load_hf_checkpoint
from muse.training.common import set_default_dtype

logger = logging.getLogger(__name__)

class Model(nn.Module):
    """Base class for all models."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
    
    def forward(self, *args, **kwargs):
        pass

    @classmethod
    def from_pretrained(cls,
                        model_dir: str,
                        strict: bool = False,
                        **kwargs) -> "Model":
        """Load weights from a pretrained model.
        Args:
            model_dir (str): The directory to load the weights from.
            strict (bool): If True, all parameters must be present in the checkpoint, otherwise raise error.
                          If False, missing parameters will be randomly initialized and a warning will be printed.
            dtype (Optional[torch.dtype]): The dtype to use when creating the model.
            **kwargs: Additional keyword arguments passed to model constructor.
        """
        model_dir = Path(model_dir)
        config_path = model_dir / "config.json"
        
        if not config_path.exists():
            raise FileNotFoundError(
                f"Config file not found: {config_path}. "
                f"Expected Muse format config.json in {model_dir}"
            )
        
        # Load config (Muse format) using get_config which handles __class__ field
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        # Load config using get_config (handles __class__ field)
        config = get_config(config_dict)

        # Override config values with kwargs if key exists in config
        for key in list(kwargs.keys()):
            if hasattr(config, key):
                setattr(config, key, kwargs.pop(key))
        
        # Get model_class from config
        model_class_name = config.model_class
        if not model_class_name:
            raise ValueError(
                f"Config file {config_path} must contain 'model_class' field"
            )
        
        # Get model class from registry
        # Import here to avoid circular import
        from muse.models import get_model_class
        model_cls = get_model_class(model_class_name)

        model = model_cls(config, **kwargs)
        
        # Always load weights
        # Check for sharded weights (index file)
        index_path = model_dir / "model.safetensors.index.json"
        if not index_path.exists():
            index_path = model_dir / "pytorch_model.bin.index.json"
        
        if index_path.exists():
            # Load sharded weights
            state_dict = load_hf_checkpoint(str(model_dir))
        else:
            # Try single file
            single_safetensors = model_dir / "model.safetensors"
            single_bin = model_dir / "pytorch_model.bin"
            
            if single_safetensors.exists():
                state_dict = load_hf_checkpoint(str(model_dir))
            elif single_bin.exists():
                state_dict = load_hf_checkpoint(str(model_dir))
            else:
                # Try to find any safetensors or bin files
                safetensors_files = glob.glob(str(model_dir / "*.safetensors"))
                bin_files = glob.glob(str(model_dir / "*.bin"))
                
                if safetensors_files or bin_files:
                    state_dict = load_hf_checkpoint(str(model_dir))
                else:
                    raise FileNotFoundError(
                        f"No weight files found in {model_dir}. "
                        f"Expected model.safetensors, pytorch_model.bin, "
                        f"or sharded files with index.json"
                    )
        
        # Load state dict with strict mode
        if strict:
            # Strict mode: all parameters must be present
            model.load_state_dict(state_dict, strict=True)
        else:
            # Non-strict mode: missing parameters will be randomly initialized
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            
            if unexpected_keys:
                logger.warning(
                    f"Unexpected keys in checkpoint (will be ignored): "
                    f"{unexpected_keys[:10]}{'...' if len(unexpected_keys) > 10 else ''}"
                )
            
            if missing_keys:
                # Randomly initialize missing parameters
                for param_name in missing_keys:
                    # Get the actual parameter from the model
                    param = dict(model.named_parameters()).get(param_name)
                    if param is not None:
                        # Initialize randomly
                        if hasattr(model, 'get_initializer'):
                            init_fn = model.get_initializer(param_name)
                            with torch.no_grad():
                                init_fn(param.data)
                        else:
                            # Use better default initialization based on parameter shape
                            with torch.no_grad():
                                if param.ndim >= 2:
                                    # For weight matrices (2D+), use Kaiming normal initialization
                                    # This is better for ReLU and similar activations
                                    nn.init.kaiming_normal_(param.data, a=0, mode='fan_in', nonlinearity='relu')
                                else:
                                    # For biases and 1D parameters, initialize to zero
                                    nn.init.zeros_(param.data)
                
                logger.warning(
                    f"Missing keys in checkpoint (randomly initialized): "
                    f"{missing_keys[:10]}{'...' if len(missing_keys) > 10 else ''}"
                )
        
        return model

    def save_pretrained(self,
                        model_dir: str,
                        save_safetensors: bool = True):
        """Save the model to a directory
        Args:
            model_dir (str): The directory to save the model to.
            save_safetensors (bool): Whether to save the model in safetensors format.
        """
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        config_path = model_dir / "config.json"
        self.config.save(str(config_path))
        
        # Get state dict
        state_dict = self.state_dict()
        
        # Determine if we need to shard (5GB per shard)
        max_shard_size = 5 * 1024 * 1024 * 1024  # 5GB in bytes
        
        if save_safetensors:
            # Calculate total size
            total_size = sum(tensor.numel() * tensor.element_size() 
                           for tensor in state_dict.values())
            
            if total_size > max_shard_size:
                # Shard the weights
                self._save_sharded_safetensors(state_dict, model_dir, max_shard_size)
            else:
                # Save as single file
                output_path = model_dir / "model.safetensors"
                safetensors_torch.save_file(state_dict, str(output_path))
        else:
            # Calculate total size
            total_size = sum(tensor.numel() * tensor.element_size() 
                           for tensor in state_dict.values())
            
            if total_size > max_shard_size:
                # Shard the weights
                self._save_sharded_bin(state_dict, model_dir, max_shard_size)
            else:
                # Save as single file
                output_path = model_dir / "pytorch_model.bin"
                torch.save(state_dict, str(output_path))
    
    def _save_sharded_safetensors(self, state_dict: Dict[str, torch.Tensor], 
                                   model_dir: Path, max_shard_size: int):
        """Save state dict as sharded safetensors files."""
        shards = []
        current_shard = {}
        current_size = 0
        shard_index = 1
        temp_shard_files = []
        
        # Sort keys for deterministic ordering
        sorted_keys = sorted(state_dict.keys())
        
        for key in sorted_keys:
            tensor = state_dict[key]
            tensor_size = tensor.numel() * tensor.element_size()
            
            # If adding this tensor would exceed max size and current shard is not empty
            if current_size + tensor_size > max_shard_size and current_shard:
                # Save current shard to temporary file
                temp_path = model_dir / f"model-{shard_index:05d}-temp.safetensors"
                safetensors_torch.save_file(current_shard, str(temp_path))
                temp_shard_files.append((temp_path, current_shard.copy()))
                current_shard = {}
                current_size = 0
                shard_index += 1
            
            current_shard[key] = tensor
            current_size += tensor_size
        
        # Save last shard
        if current_shard:
            temp_path = model_dir / f"model-{shard_index:05d}-temp.safetensors"
            safetensors_torch.save_file(current_shard, str(temp_path))
            temp_shard_files.append((temp_path, current_shard.copy()))
        
        # Now rename all files with correct total count
        total_shards = len(temp_shard_files)
        weight_map = {}
        for i, (temp_path, shard_dict) in enumerate(temp_shard_files, 1):
            final_name = f"model-{i:05d}-of-{total_shards:05d}.safetensors"
            final_path = model_dir / final_name
            temp_path.rename(final_path)
            # Update weight map
            for key in shard_dict.keys():
                weight_map[key] = final_name
        
        # Save index file
        index_data = {
            "metadata": {"total_size": sum(t.numel() * t.element_size() 
                                           for t in state_dict.values())},
            "weight_map": weight_map
        }
        index_path = model_dir / "model.safetensors.index.json"
        with open(index_path, 'w') as f:
            json.dump(index_data, f, indent=2)
    
    def _save_sharded_bin(self, state_dict: Dict[str, torch.Tensor], 
                          model_dir: Path, max_shard_size: int):
        """Save state dict as sharded PyTorch bin files."""
        current_shard = {}
        current_size = 0
        shard_index = 1
        temp_shard_files = []
        
        # Sort keys for deterministic ordering
        sorted_keys = sorted(state_dict.keys())
        
        for key in sorted_keys:
            tensor = state_dict[key]
            tensor_size = tensor.numel() * tensor.element_size()
            
            # If adding this tensor would exceed max size and current shard is not empty
            if current_size + tensor_size > max_shard_size and current_shard:
                # Save current shard to temporary file
                temp_path = model_dir / f"pytorch_model-{shard_index:05d}-temp.bin"
                torch.save(current_shard, str(temp_path))
                temp_shard_files.append((temp_path, current_shard.copy()))
                current_shard = {}
                current_size = 0
                shard_index += 1
            
            current_shard[key] = tensor
            current_size += tensor_size
        
        # Save last shard
        if current_shard:
            temp_path = model_dir / f"pytorch_model-{shard_index:05d}-temp.bin"
            torch.save(current_shard, str(temp_path))
            temp_shard_files.append((temp_path, current_shard.copy()))
        
        # Now rename all files with correct total count
        total_shards = len(temp_shard_files)
        weight_map = {}
        for i, (temp_path, shard_dict) in enumerate(temp_shard_files, 1):
            final_name = f"pytorch_model-{i:05d}-of-{total_shards:05d}.bin"
            final_path = model_dir / final_name
            temp_path.rename(final_path)
            # Update weight map
            for key in shard_dict.keys():
                weight_map[key] = final_name
        
        # Save index file
        index_data = {
            "metadata": {"total_size": sum(t.numel() * t.element_size() 
                                           for t in state_dict.values())},
            "weight_map": weight_map
        }
        index_path = model_dir / "pytorch_model.bin.index.json"
        with open(index_path, 'w') as f:
            json.dump(index_data, f, indent=2)

    def init_weights(self):
        raise NotImplementedError(
            "Subclass must implement init_weights method")

    def get_initializer(self, name: str) -> Callable[[torch.Tensor], None]:
        """Return a initializer function for the given name
        Args:
            name (str): The name of the parameter
        Returns:
            A callable function that takes a tensor(parameter weight) and initializes it,
            e.g,
                return partial(
                    nn.init.kaiming_normal_, 
                    a=0, mode='fan_in', nonlinearity='relu'
                )
        Examples:
            >>> model.get_initializer("layers.0.attention.q_proj.weight")(q_proj.weight)
        """
        raise NotImplementedError(
            "Subclass must implement get_initializer method")
    
    def train(self, mode: bool = True):
        """Set the model to training mode.
        
        This recursively sets all submodules to training mode.
        PyTorch's nn.Module.train() automatically handles recursion.
        
        Args:
            mode (bool): Whether to set training mode (True) or evaluation mode (False).
                Default: True.
        
        Returns:
            self: Returns self for method chaining.
        """
        # Call nn.Module.train() directly to recursively set all submodules
        nn.Module.train(self, mode)
        return self
    
    def eval(self):
        """Set the model to evaluation mode.
        
        This recursively sets all submodules to evaluation mode.
        PyTorch's nn.Module.eval() automatically handles recursion.
        
        Returns:
            self: Returns self for method chaining.
        """
        # Call nn.Module.eval() directly to recursively set all submodules
        nn.Module.eval(self)
        return self
    
    def get_checkpointable_module_classes(self):
        """Return a list of module classes that should be checkpointed"""
        raise NotImplementedError(
            "Subclass must implement get_checkpointable_module_classes method")

    def get_layers_to_shard(self):
        """Return a list of layers that should be sharded by FSDP"""
        raise NotImplementedError(
            "Subclass must implement get_layers_to_shard method")

    def get_optimizer_grouped_parameters(
            self,
            learning_rate: float,
            weight_decay: float) -> List[Dict[str, Any]]:
        """Get the optimizer grouped parameters for AdamW optimizer
        Args:
            learning_rate (float): The learning rate.
            weight_decay (float): The weight decay.
        Returns:
            A list of optimizer grouped parameters for AdamW optimizer.
        """
        optimizer_grouped_parameters = [
          {
            "params": [p for n, p in self.named_parameters() 
                       if p.requires_grad],
            "weight_decay": weight_decay,
            "lr": learning_rate,
          },
        ]
        return optimizer_grouped_parameters
    
    @classmethod
    def convert_hf_state_dict(cls,
                              hf_state_dict: Dict[str, torch.Tensor],
                              **kwargs) -> Dict[str, torch.Tensor]:
        """Convert a Hugging Face state dictionary to a model state dictionary
        Args:
            hf_state_dict (Dict[str, torch.Tensor]): The Hugging Face state dictionary.
            **kwargs: Additional keyword arguments.
        Returns:
            A dictionary of model state.
        """
        return hf_state_dict

    @classmethod
    def get_hf_state_dict(cls,
                          state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Convert the model state dict to the Hugging Face format"""
        return state_dict
    
    def generate(self, *args, **kwargs):
        """Generate text from the model"""
        raise NotImplementedError(
            "Subclass must implement generate method")
