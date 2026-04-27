"""
Chunked Distillation Loss Computer for Summary Attention Training @ccl.

Memory-efficient two-stage backward controller that computes CE + KL distillation
loss in chunks along the sequence dimension. Each chunk only materializes
[b, chunk_size, vocab_size] logits, avoiding the full [b, seq_len, vocab_size] tensor.

This class handles CE + KL losses on logits. Attention MSE/cos distillation
(which operates on [b, s, d] attention outputs, not logits) should be computed
separately in the training script.

Usage:
    chunked_distill = ChunkedDistillLossComputer(
        lm_head=lm_head,
        minibatch_size=2048,
        lm_factor=1.0,
        kl_factor=5.0,
    )

    # In training loop:
    loss_dict = chunked_distill.forward_and_backward(
        student_hidden, teacher_hidden, labels,
        loss_scale=1.0 / grad_acc,
    )
    # backward is already done inside forward_and_backward!
    # loss_dict["ce_loss"], loss_dict["kl_loss"] are detached for logging only.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch.distributed.tensor import DTensor
except ImportError:
    DTensor = None


def _to_local(t):
    """Extract plain tensor from DTensor (no-op for Replicate placement)."""
    if DTensor is not None and isinstance(t, DTensor):
        return t.full_tensor()
    return t


class ChunkedDistillLossComputer:
    """
    分块计算 CE + KL distillation loss 的控制器。

    将 student/teacher 的 hidden states 按 seq 维度分块，
    每个 chunk 内过 lm_head 得到 logits，算完 CE+KL 后立即释放。

    【显存优化】
    峰值显存 = 2 × [b, chunk_size, vocab_size]（student + teacher logits per chunk）
    而非 2 × [b, seq_len, vocab_size]。

    【注意】
    返回的 loss 都是 detached 的，不可再 backward。
    所有 backward 操作在 forward_and_backward 内部完成。
    """

    def __init__(
        self,
        lm_head: nn.Module,
        minibatch_size: int,
        ignore_index: int = -100,
        lm_factor: float = 1.0,
        kl_factor: float = 5.0,
        enable_kl: bool = True,
    ):
        """
        Args:
            lm_head: 语言模型输出层 (nn.Linear or LMHeadWrapper)，需有 .out_features 属性。
            minibatch_size: 每个分块的序列长度。
            ignore_index: 标签中的忽略索引。
            lm_factor: CE loss 的权重。
            kl_factor: KL loss 的权重。
            enable_kl: 是否启用 KL distillation。False 时只算 CE，忽略 teacher_hidden。
        """
        self.lm_head = lm_head
        self.minibatch_size = minibatch_size
        self.ignore_index = ignore_index
        self.lm_factor = lm_factor
        self.kl_factor = kl_factor
        self.enable_kl = enable_kl

    def _apply_lm_head(self, x):
        """F.linear with DTensor-safe weights. Bypasses nn.Module dispatch to avoid
        mixed plain-tensor / DTensor errors after FSDP shard."""
        w = _to_local(self.lm_head.weight)
        b = self.lm_head.bias
        return F.linear(x, w, _to_local(b) if b is not None else None)

    def forward_and_backward(
        self,
        student_hidden: torch.Tensor,
        teacher_hidden: torch.Tensor = None,
        labels: torch.Tensor = None,
        gradient_accumulation_steps: int = 1,
    ) -> dict:
        """
        分块计算 CE (+ 可选 KL) loss，并完成 backward。

        当 teacher_hidden=None 或 kl_factor=0 时，仅计算 CE loss（continue 模式）。
        当 teacher_hidden 有值且 kl_factor>0 时，计算 CE + KL（distill 模式）。

        Args:
            student_hidden: [b, s, d] — 模型输出的 hidden states (skip_output_layer=True)
            teacher_hidden: [b, s, d] — teacher 的 hidden states (detached, no_grad)。
                            为 None 时只计算 CE loss。
            labels: [b, s] — pre-shifted labels (ignore_index=-100 for masked positions)
            gradient_accumulation_steps: 梯度累积步数，梯度会除以此值

        Returns:
            dict: {
                "ce_loss": detached scalar — 平均 CE loss (用于 logging),
                "kl_loss": detached scalar — 平均 KL loss (用于 logging, 无 teacher 时为 0),
                "total_loss": detached scalar — lm_factor * ce + kl_factor * kl,
            }
        """
        compute_kl = self.enable_kl and teacher_hidden is not None and self.kl_factor > 0
        device = student_hidden.device
        params = [p for p in self.lm_head.parameters() if p.requires_grad]
        grad_accs = [torch.zeros_like(p) for p in params]
        grad_input = torch.zeros_like(student_hidden)

        total_ce_sum = torch.tensor(0.0, device=device)
        total_kl_sum = torch.tensor(0.0, device=device)
        total_elements = (labels != self.ignore_index).sum()

        if total_elements.item() == 0:
            zero = torch.tensor(0.0, device=device)
            return {"ce_loss": zero, "kl_loss": zero, "total_loss": zero}

        seq_len = student_hidden.size(1)
        vocab_size = self.lm_head.out_features

        for i in range(0, seq_len, self.minibatch_size):
            s, e = i, min(i + self.minibatch_size, seq_len)
            h_chunk = student_hidden[:, s:e, :].detach().requires_grad_()

            # Student logits（需要梯度流过 h_chunk）
            student_logits = self._apply_lm_head(h_chunk)

            labels_chunk = labels[:, s:e]
            logits_flat = student_logits.reshape(-1, vocab_size)
            labels_flat = labels_chunk.reshape(-1)
            valid_mask = (labels_flat != self.ignore_index)

            valid_count = valid_mask.sum().item()
            if valid_count == 0:
                continue

            # CE loss (sum over valid tokens)
            ce_per_token = F.cross_entropy(
                logits_flat, labels_flat,
                ignore_index=self.ignore_index, reduction='none'
            )
            ce_sum = ce_per_token[valid_mask].sum()

            # KL loss (optional: only when teacher_hidden is provided and kl_factor > 0)
            if compute_kl:
                with torch.no_grad():
                    teacher_logits = self._apply_lm_head(teacher_hidden[:, s:e, :])
                teacher_flat = teacher_logits.reshape(-1, vocab_size)
                teacher_lp = torch.log_softmax(teacher_flat[valid_mask].float(), dim=-1)
                student_lp = torch.log_softmax(logits_flat[valid_mask].float(), dim=-1)
                kl_per_token = F.kl_div(
                    student_lp, teacher_lp,
                    reduction='none', log_target=True,
                ).sum(dim=-1)
                kl_sum = kl_per_token.sum()
                chunk_loss = self.lm_factor * ce_sum + self.kl_factor * kl_sum
            else:
                kl_sum = torch.tensor(0.0, device=device)
                chunk_loss = self.lm_factor * ce_sum

            # Manual gradient computation
            tensors_to_grad = params + [h_chunk]
            grads = torch.autograd.grad(chunk_loss, tensors_to_grad, retain_graph=False)

            for j in range(len(params)):
                grad_accs[j] += grads[j]
            grad_input[:, s:e, :] = grads[-1]

            total_ce_sum += ce_sum.detach()
            total_kl_sum += kl_sum.detach()

        # Apply accumulated gradients.
        # Order: backward FIRST, then lm_head grads.
        # For tied weights (lm_head = embedding), FSDP backward sets embedding grad
        # on p.grad via reduce-scatter. If we set p.grad first, FSDP would overwrite it.
        # By doing backward first, we then accumulate the lm_head grad on top.
        norm = total_elements * gradient_accumulation_steps

        # Backward to upstream model (gradients flow into summary params etc.)
        student_hidden.backward(gradient=grad_input / norm)

        for j, p in enumerate(params):
            if p.grad is None:
                p.grad = grad_accs[j] / norm
            else:
                p.grad += grad_accs[j] / norm

        # Compute reporting losses (mean over valid tokens, no loss_scale)
        ce_loss = (total_ce_sum / total_elements).detach()
        kl_loss = (total_kl_sum / total_elements).detach()
        total_loss = (self.lm_factor * ce_loss + self.kl_factor * kl_loss).detach()

        return {
            "ce_loss": ce_loss,
            "kl_loss": kl_loss,
            "total_loss": total_loss,
        }
