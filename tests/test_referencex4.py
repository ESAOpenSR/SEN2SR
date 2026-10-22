"""Tests for the multi-stage 4x SR pipeline using deterministic models."""

import torch
from torch import nn

from sen2sr.referencex4 import srmodel


def test_pipeline_fuses_correct_bands_and_restores_sentinel2_order():
    class UpscaleRGBN(nn.Module):
        def forward(self, image):
            return (image + 100).repeat_interleave(4, -2).repeat_interleave(4, -1)

    class Resample20m(nn.Module):
        def forward(self, image):
            return image + 20

    class Fusion(nn.Module):
        def __init__(self):
            super().__init__()
            self.inputs = None

        def forward(self, image):
            self.inputs = image
            return image[:, :6] - 35

    class Constraint(nn.Module):
        def __init__(self):
            super().__init__()
            self.offset = nn.Parameter(torch.tensor(0.5))
            self.low_resolution = None

        def forward(self, low_resolution, prediction):
            self.low_resolution = low_resolution
            return prediction + self.offset

    image = torch.arange(10, 20, dtype=torch.float32).reshape(1, 10, 1, 1).expand(2, 10, 4, 6)
    fusion = Fusion()
    constraint = Constraint()
    model = srmodel(UpscaleRGBN(), Resample20m(), fusion, constraint)

    result = model(image)

    expected = torch.tensor([110, 111, 112, 0.5, 0.5, 0.5, 116, 2.5, 3.5, 4.5])
    torch.testing.assert_close(result, expected.reshape(1, 10, 1, 1).expand(2, 10, 16, 24))
    fusion_bands = torch.tensor([33, 34, 35, 37, 38, 39, 110, 111, 112, 116], dtype=image.dtype)
    torch.testing.assert_close(fusion.inputs, fusion_bands.reshape(1, 10, 1, 1).expand(2, 10, 16, 24))
    reference_bands = torch.tensor([33, 34, 35, 37, 38, 39], dtype=image.dtype)
    torch.testing.assert_close(constraint.low_resolution, reference_bands.reshape(1, 6, 1, 1).expand(2, 6, 4, 6))
    assert not constraint.training
    assert not constraint.offset.requires_grad
