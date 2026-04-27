"""
Distillation losses for Summary Attention training.
@ccl
Used in the warmup phase where the model's own full-attention output
acts as the teacher for its summary-attention output (self-distillation).
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class DistillConfig:
    """Configuration for distillation loss weights."""
    lm_factor: float = 1.0
    kl_factor: float = 5.0
    mse_factor: float = 5.0
    cos_factor: float = 0.0


def kl_divergence_loss(student_logits: torch.Tensor,
                       teacher_logits: torch.Tensor) -> torch.Tensor:
    """KL(student || teacher) on logits, temperature=1.0.

    Args:
        student_logits: [*, V] student output logits.
        teacher_logits: [*, V] teacher output logits (will be detached).

    Returns:
        Scalar KL divergence loss.
    """
    teacher_log_prob = torch.log_softmax(teacher_logits, dim=-1).detach()
    student_log_prob = torch.log_softmax(student_logits, dim=-1)
    kl = F.kl_div(
        student_log_prob, teacher_log_prob,
        reduction='none', log_target=True,
    ).sum(dim=-1)
    return kl.mean()


def attention_mse_loss(student_attn: torch.Tensor,
                       teacher_attn: torch.Tensor) -> torch.Tensor:
    """Per-element MSE between attention outputs, clamped to [0, 0.1].

    Args:
        student_attn: [*, D] student per-layer attention output.
        teacher_attn: [*, D] teacher per-layer attention output (will be detached).

    Returns:
        Scalar MSE loss.
    """
    teacher_attn = teacher_attn.detach()
    return torch.clamp(
        F.mse_loss(student_attn, teacher_attn, reduction='none'),
        0, 0.1,
    ).mean()


def attention_cosine_loss(student_attn: torch.Tensor,
                          teacher_attn: torch.Tensor) -> torch.Tensor:
    """1 - cosine_similarity between attention outputs.

    Args:
        student_attn: [*, D] student per-layer attention output.
        teacher_attn: [*, D] teacher per-layer attention output (will be detached).

    Returns:
        Scalar cosine loss.
    """
    teacher_attn = teacher_attn.detach()
    cos_sim = F.cosine_similarity(student_attn, teacher_attn, dim=-1)
    return (1 - cos_sim).mean()


def compute_distill_loss(
    student_attn_outputs: Optional[List[torch.Tensor]],
    teacher_attn_outputs: Optional[List[torch.Tensor]],
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    config: DistillConfig,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute combined distillation loss.

    Args:
        student_attn_outputs: Per-layer attention outputs from student (summary path).
        teacher_attn_outputs: Per-layer attention outputs from teacher (full-attn path).
        student_logits: Student output logits (summary filtered).
        teacher_logits: Teacher output logits.
        config: Distillation loss weight configuration.

    Returns:
        (total_distill_loss, loss_dict) where loss_dict contains detached
        per-component losses for logging.
    """
    device = student_logits.device
    distill_loss = torch.tensor(0.0, device=device)
    loss_dict: Dict[str, torch.Tensor] = {}

    # KL divergence on logits
    if config.kl_factor > 0:
        kl_loss = kl_divergence_loss(student_logits, teacher_logits)
        distill_loss = distill_loss + config.kl_factor * kl_loss
        loss_dict["kl_loss"] = kl_loss.detach()
    else:
        loss_dict["kl_loss"] = torch.tensor(0.0, device=device)

    # Attention MSE
    if (config.mse_factor > 0
            and student_attn_outputs is not None
            and teacher_attn_outputs is not None):
        attn_mse_losses = [
            attention_mse_loss(s, t)
            for s, t in zip(student_attn_outputs, teacher_attn_outputs)
        ]
        attn_mse = torch.stack(attn_mse_losses).mean()
        distill_loss = distill_loss + config.mse_factor * attn_mse
        loss_dict["attn_mse"] = attn_mse.detach()
        loss_dict["attn_mse_layers"] = torch.stack(attn_mse_losses).detach()
    else:
        loss_dict["attn_mse"] = torch.tensor(0.0, device=device)

    # Attention cosine similarity
    if (config.cos_factor > 0
            and student_attn_outputs is not None
            and teacher_attn_outputs is not None):
        attn_cos_losses = [
            attention_cosine_loss(s, t)
            for s, t in zip(student_attn_outputs, teacher_attn_outputs)
        ]
        attn_cos = torch.stack(attn_cos_losses).mean()
        distill_loss = distill_loss + config.cos_factor * attn_cos
        loss_dict["attn_cos"] = attn_cos.detach()
    else:
        loss_dict["attn_cos"] = torch.tensor(0.0, device=device)

    return distill_loss, loss_dict
