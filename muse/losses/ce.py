"""
Cross-Entropy Loss Implementation.
@ccl
This module provides an optimized CrossEntropyLoss implementation that avoids
redundant calculations and offers flexible reduction modes.

The implementation computes per-token losses first, then applies reduction,
which is more efficient than computing the full loss matrix and allows for
token-level loss inspection when needed.

Classes:
    CrossEntropyLoss: Efficient cross-entropy loss with flexible options

Example:
    >>> criterion = CrossEntropyLoss(ignore_index=-100, reduction="mean")
    >>> logits = model(input_ids)  # Shape: (batch_size, seq_len, vocab_size)
    >>> labels = target_ids        # Shape: (batch_size, seq_len)
    >>> loss = criterion(logits, labels)
    >>> 
    >>> # With token-level loss
    >>> criterion = CrossEntropyLoss(return_token_loss=True)
    >>> loss, token_losses = criterion(logits, labels)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# =================================================================
# Block 1: Your High-Quality, Optimized Loss Function
# ===================================================================

class CrossEntropyLoss(nn.Module):
    """
    Efficient CrossEntropyLoss with per-token loss support.
    
    This implementation computes per-token losses first, then applies reduction,
    avoiding redundant calculations. It supports:
    - Label shifting for autoregressive models
    - Ignore index for padding tokens
    - Optional per-token loss return
    - Mean or sum reduction
    
    The loss handles batched inputs and properly accounts for ignored tokens
    in mean reduction.
    
    Args:
        ignore_index (int): Label value to ignore in loss calculation. Defaults to -100.
        return_token_loss (bool): If True, returns both reduced loss and per-token
            losses. Defaults to False.
        shift_labels (bool): If True, shifts labels left by 1 position (for autoregressive
            language modeling). Defaults to True.
        reduction (str): Reduction mode, "mean" or "sum". Defaults to "mean".
        
    Example:
        >>> # Standard usage for language modeling
        >>> criterion = CrossEntropyLoss(ignore_index=-100, shift_labels=True)
        >>> logits = model(input_ids)  # (batch, seq_len, vocab_size)
        >>> loss = criterion(logits, labels)
        >>> 
        >>> # Get per-token losses for analysis
        >>> criterion = CrossEntropyLoss(return_token_loss=True)
        >>> loss, token_losses = criterion(logits, labels)
        >>> # token_losses shape: (batch_size * seq_len,)
    """
    
    def __init__(self,
                 ignore_index: int = -100,
                 return_token_loss: bool = False,
                 shift_labels: bool = True,
                 reduction: str = "mean"):
        """
        Initialize CrossEntropyLoss.
        
        Args:
            ignore_index (int): Label value to ignore. Defaults to -100.
            return_token_loss (bool): Return per-token losses. Defaults to False.
            shift_labels (bool): Shift labels for autoregressive models. Defaults to True.
            reduction (str): "mean" or "sum". Defaults to "mean".
        """
        super().__init__()
        self.ignore_index = ignore_index
        self.return_token_loss = return_token_loss
        self.reduction = reduction
        self.shift_labels = shift_labels

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, per_token_loss_weight=1):
        """
        Compute cross-entropy loss.
        
        Args:
            logits (torch.Tensor): Model predictions with shape (..., vocab_size).
                Typically (batch_size, seq_len, vocab_size) for language models.
            labels (torch.Tensor): Ground truth labels with same shape as logits[:-1].
                Typically (batch_size, seq_len) for language models.
                
        Returns:
            torch.Tensor: Reduced loss (scalar)
            or
            Tuple[torch.Tensor, torch.Tensor]: (reduced_loss, per_token_losses) if
                return_token_loss=True. per_token_losses has shape (num_tokens,).
                
        Note:
            - If shift_labels=True, predicts token i+1 from position i
            - Ignored tokens (matching ignore_index) are excluded from loss
            - Mean reduction divides by number of non-ignored tokens
        """
        vocab_size = logits.shape[-1]
        
        if self.shift_labels:
          logits = logits[:, :-1, :]
          labels = labels[:, 1:]

        # Reshape for cross-entropy calculation
        logits_flat = logits.float().reshape(-1, vocab_size)
        labels_flat = labels.reshape(-1)


        # Step 1: Compute per-token loss.
        # This is the base for all other calculations.
        per_token_loss = F.cross_entropy(
            logits_flat,
            labels_flat,
            ignore_index=self.ignore_index,
            reduction="none"
        )
        
        # Step 2: Manually apply reduction to get the final loss.
        loss = (per_token_loss * per_token_loss_weight).sum()
        if self.reduction == "mean":
            # Ensure we divide by the number of valid (non-ignored) tokens
            total_elements = (labels_flat != self.ignore_index).sum()
            if total_elements > 0:
                loss /= total_elements
            else: # Handle case where all tokens are ignored
                loss.zero_()

        # Return what's requested
        if self.return_token_loss:
            return loss, per_token_loss
        
        return loss