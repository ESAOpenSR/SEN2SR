"""CPU-only tiling regressions; no pretrained weights or imagery are needed."""

import pytest
import torch
from torch import nn

from sen2sr.utils import predict_large


class Upscale(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, image):
        return image.repeat_interleave(self.scale, -2).repeat_interleave(self.scale, -1)


@pytest.mark.parametrize("scale", [1, 2, 4])
@pytest.mark.parametrize("overlap", [0, 1, 31, 32])
@pytest.mark.parametrize(
    "height,width",
    [
        (128, 128),
        (128, 256),
        (256, 128),
        (256, 256),
        (221, 161),
        (161, 221),
        (512, 512),
        (64, 64),
        (64, 256),
        (256, 64),
        (96, 160),
        (160, 96),
        (127, 128),
        (128, 127),
        (129, 129),
        (1, 1),
        (1, 129),
        (129, 1),
    ],
)
def test_predict_large_matches_full_image(height, width, scale, overlap):
    # Distinct values along both axes and channels reveal shifts and transposes.
    image = torch.arange(1, 2 * height * width + 1, dtype=torch.float32).reshape(2, height, width)
    model = Upscale(scale)

    result = predict_large(image, model, overlap=overlap)

    torch.testing.assert_close(result, model(image[None])[0], rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_predict_large_preserves_model_output_metadata(dtype):
    model = nn.Conv2d(2, 3, kernel_size=1).to(dtype)
    image = torch.ones(2, 128, 256, dtype=dtype, requires_grad=True)

    result = predict_large(image, model)

    torch.testing.assert_close(result, model(image[None])[0])
    assert result.dtype == dtype
    assert result.device.type == "cpu"
    assert not result.requires_grad


def test_predict_large_pads_small_tiles_for_fixed_size_models():
    class FixedSizeModel(nn.Module):
        def forward(self, image):
            assert image.shape[-2:] == (128, 128)
            return image.repeat_interleave(4, -2).repeat_interleave(4, -1)

    image = torch.arange(1, 65 * 221 + 1, dtype=torch.float32).reshape(1, 65, 221)

    result = predict_large(image, FixedSizeModel())

    expected = image.repeat_interleave(4, -2).repeat_interleave(4, -1)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


@pytest.mark.parametrize("scale", [1, 2, 4])
@pytest.mark.parametrize("height,width", [(240, 256), (256, 240), (256, 256)])
def test_predict_large_discards_internal_tile_edge_artifacts(height, width, scale):
    class EdgeArtifacts(Upscale):
        def forward(self, image):
            result = super().forward(image)
            margin = 8 * self.scale
            result[:, :, :margin] = -1
            result[:, :, -margin:] = -1
            result[:, :, :, :margin] = -1
            result[:, :, :, -margin:] = -1
            return result

    result = predict_large(torch.ones(1, height, width), EdgeArtifacts(scale), overlap=32)

    # Only physical image boundaries may retain the model's border artifacts.
    margin = 8 * scale
    interior = result[:, margin:-margin, margin:-margin]
    torch.testing.assert_close(interior, torch.ones_like(interior), rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(128, 128), (1, 1, 128, 128), (1, 0, 128), (1, 128, 0), (0, 128, 128)])
def test_predict_large_rejects_invalid_images(shape):
    with pytest.raises(ValueError, match=r"non-empty.*channels, height, width"):
        predict_large(torch.empty(shape), Upscale(4))


@pytest.mark.parametrize("overlap", [-1, 128, 129, 0.5])
def test_predict_large_rejects_invalid_overlap(overlap):
    with pytest.raises(ValueError, match="overlap"):
        predict_large(torch.ones(1, 128, 128), Upscale(4), overlap=overlap)


@pytest.mark.parametrize("shape", [(1, 128, 128), (2, 1, 128, 128)])
def test_predict_large_rejects_invalid_model_output_shape(shape):
    class InvalidShape(nn.Module):
        def forward(self, image):
            return image.new_ones(shape)

    with pytest.raises(ValueError, match="model must return a tensor"):
        predict_large(torch.ones(1, 128, 128), InvalidShape())


@pytest.mark.parametrize("height,width", [(64, 64), (129, 129), (256, 128)])
def test_predict_large_rejects_invalid_model_scale(height, width):
    class InvalidScale(nn.Module):
        def forward(self, image):
            return image.new_ones(1, 1, height, width)

    with pytest.raises(ValueError, match="same positive integer scale"):
        predict_large(torch.ones(1, 128, 128), InvalidScale())


@pytest.mark.parametrize("changed_shape", [(1, 2, 128, 128), (1, 1, 256, 256)])
def test_predict_large_rejects_inconsistent_model_outputs(changed_shape):
    class InconsistentOutput(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, image):
            shape = (1, 1, 128, 128) if self.calls == 0 else changed_shape
            self.calls += 1
            return image.new_ones(shape)

    with pytest.raises(ValueError, match="consistent across tiles"):
        predict_large(torch.ones(1, 128, 256), InconsistentOutput())
