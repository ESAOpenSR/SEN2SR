"""Tests for the non-reference super-resolution wrapper."""

import torch
from torch import nn

from sen2sr.nonreference import srmodel


class AddConstraint(nn.Module):
    def __init__(self):
        super().__init__()
        self.offset = nn.Parameter(torch.tensor(0.25))
        self.low_resolution = None

    def forward(self, low_resolution, prediction):
        self.low_resolution = low_resolution
        return prediction + self.offset


def test_wrapper_clamps_prediction_before_applying_constraint():
    image = torch.tensor([[[[-2.0, 3.0]]]])
    constraint = AddConstraint()
    model = srmodel(nn.Identity(), constraint)

    result = model(image)

    torch.testing.assert_close(result, torch.tensor([[[[0.25, 3.25]]]]))
    torch.testing.assert_close(constraint.low_resolution, image)


def test_wrapper_registers_models_and_freezes_only_constraint():
    base = nn.Conv2d(1, 1, kernel_size=1)
    constraint = AddConstraint()

    model = srmodel(base, constraint, device="cpu")

    assert model.sr_model is base
    assert model.hard_constraint is constraint
    assert base.weight.requires_grad
    assert not constraint.offset.requires_grad
    assert not constraint.training
    assert constraint.offset.device.type == "cpu"
