"""Head and backbone gradients must share one backward, including tied weights."""
import copy

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from muse.losses.chunked_distill_loss_computer import ChunkedDistillLossComputer


@pytest.mark.parametrize('tied', [False, True])
@pytest.mark.parametrize('accumulation', [1, 2])
def test_chunked_head_and_auxiliary_gradients_match_dense(tied, accumulation):
    torch.manual_seed(2026)
    model = nn.ModuleDict({'embed': nn.Embedding(17, 8), 'head': nn.Linear(8, 17, bias=False)})
    if tied:
        model['head'].weight = model['embed'].weight
    reference = copy.deepcopy(model)
    ids = torch.randint(17, (2, 7))
    labels = torch.randint(17, (2, 7))
    labels[0, -1] = -100
    computer = ChunkedDistillLossComputer(model['head'], minibatch_size=3, enable_kl=False)
    for _ in range(accumulation):
        hidden = model['embed'](ids).tanh()
        computer.forward_and_backward(
            hidden, labels=labels, gradient_accumulation_steps=accumulation,
            auxiliary_loss=hidden.square().mean() * 0.2,
        )
        expected_hidden = reference['embed'](ids).tanh()
        expected = F.cross_entropy(reference['head'](expected_hidden).flatten(0, 1), labels.flatten())
        ((expected + expected_hidden.square().mean() * 0.2) / accumulation).backward()
    for (_, actual), (_, expected) in zip(model.named_parameters(), reference.named_parameters()):
        assert actual.grad is not None
        torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-5, atol=1e-7)
