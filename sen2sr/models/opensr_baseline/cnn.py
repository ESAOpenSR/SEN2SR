# I stole the code from here: https://github.com/hongyuanyu/SPAN
# The author of the code deserves all the  credit. I just make
# basic modifications to make it work with my codebase.


from collections import OrderedDict
from typing import List, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn as nn


def _make_pair(value: int) -> tuple:
    """
    Converts a single integer into a tuple of the same integer repeated twice.

    Args:
        value (int): Integer value to be converted.

    Returns:
        tuple: Tuple containing the integer repeated twice.
    """
    if isinstance(value, int):
        value = (value,) * 2
    return value


def conv_layer(
    in_channels: int, out_channels: int, kernel_size: int, bias: bool = True
) -> nn.Conv2d:
    """
    Creates a 2D convolutional layer with adaptive padding.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolution kernel.
        bias (bool, optional): Whether to include a bias term. Defaults to True.

    Returns:
        nn.Conv2d: 2D convolutional layer with calculated padding.
    """
    kernel_size = _make_pair(kernel_size)
    padding = (int((kernel_size[0] - 1) / 2), int((kernel_size[1] - 1) / 2))
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)


def activation(
    act_type: str, inplace: bool = True, neg_slope: float = 0.05, n_prelu: int = 1
) -> nn.Module:
    """
    Returns an activation layer based on the specified type.

    Args:
        act_type (str): Type of activation ('relu', 'lrelu', 'prelu').
        inplace (bool, optional): If True, performs the operation in-place. Defaults to True.
        neg_slope (float, optional): Negative slope for 'lrelu' and 'prelu'. Defaults to 0.05.
        n_prelu (int, optional): Number of parameters for 'prelu'. Defaults to 1.

    Returns:
        nn.Module: Activation layer.
    """
    act_type = act_type.lower()
    if act_type == "relu":
        layer = nn.ReLU(inplace)
    elif act_type == "lrelu":
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == "prelu":
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError(
            "activation layer [{:s}] is not found".format(act_type)
        )
    return layer


def sequential(*args) -> nn.Sequential:
    """
    Constructs a sequential container for the provided modules.

    Args:
        args: Modules in order of execution.

    Returns:
        nn.Sequential: A Sequential container.
    """
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError("sequential does not support OrderedDict input.")
        return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


def pixelshuffle_block(
    in_channels: int, out_channels: int, upscale_factor: int = 2, kernel_size: int = 3
) -> nn.Sequential:
    """
    Creates an upsampling block using pixel shuffle.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        upscale_factor (int, optional): Factor by which to upscale. Defaults to 2.
        kernel_size (int, optional): Size of the convolution kernel. Defaults to 3.

    Returns:
        nn.Sequential: Sequential block for upsampling.
    """
    conv = conv_layer(in_channels, out_channels * (upscale_factor**2), kernel_size)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return sequential(conv, pixel_shuffle)


class Conv3XC(nn.Module):
    def __init__(
        self,
        c_in: int,
        c_out: int,
        gain1: int = 1,
        s: int = 1,
        bias: bool = True,
        relu: bool = False,
        train_mode: bool = True,
    ):
        """
        Custom 3-stage convolutional block with optional ReLU activation and train/evaluation mode support.

        Args:
            c_in (int): Number of input channels.
            c_out (int): Number of output channels.
            gain1 (int, optional): Gain multiplier for intermediate layers. Defaults to 1.
            s (int, optional): Stride value for the convolutions. Defaults to 1.
            bias (bool, optional): Whether to include a bias term in the convolutions. Defaults to True.
            relu (bool, optional): If True, apply a LeakyReLU activation after the convolution. Defaults to False.
            train_mode (bool, optional): If True, use training mode with learnable parameters. Defaults to True.
        """
        super(Conv3XC, self).__init__()
        self.train_mode = train_mode
        self.weight_concat = None
        self.bias_concat = None
        self.update_params_flag = False
        self.stride = s
        self.has_relu = relu
        gain = gain1

        self.sk = nn.Conv2d(
            in_channels=c_in,
            out_channels=c_out,
            kernel_size=1,
            padding=0,
            stride=s,
            bias=bias,
        )
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=c_in,
                out_channels=c_in * gain,
                kernel_size=1,
                padding=0,
                bias=bias,
            ),
            nn.Conv2d(
                in_channels=c_in * gain,
                out_channels=c_out * gain,
                kernel_size=3,
                stride=s,
                padding=0,
                bias=bias,
            ),
            nn.Conv2d(
                in_channels=c_out * gain,
                out_channels=c_out,
                kernel_size=1,
                padding=0,
                bias=bias,
            ),
        )

        self.eval_conv = nn.Conv2d(
            in_channels=c_in,
            out_channels=c_out,
            kernel_size=3,
            padding=1,
            stride=s,
            bias=bias,
        )
        self.eval_conv.weight.requires_grad = False
        if self.eval_conv.bias is not None:
            self.eval_conv.bias.requires_grad = False
        self.update_params()

    def update_params(self):
        """
        Updates the parameters for evaluation mode by combining weights from the convolution layers.
        """
        w1 = self.conv[0].weight.data.clone().detach()
        b1 = self.conv[0].bias.detach().clone() if self.conv[0].bias is not None else w1.new_zeros(w1.shape[0])
        w2 = self.conv[1].weight.data.clone().detach()
        b2 = self.conv[1].bias.detach().clone() if self.conv[1].bias is not None else w2.new_zeros(w2.shape[0])
        w3 = self.conv[2].weight.data.clone().detach()
        b3 = self.conv[2].bias.detach().clone() if self.conv[2].bias is not None else w3.new_zeros(w3.shape[0])

        w = (
            F.conv2d(w1.flip(2, 3).permute(1, 0, 2, 3), w2, padding=2, stride=1)
            .flip(2, 3)
            .permute(1, 0, 2, 3)
        )
        b = (w2 * b1.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b2

        self.weight_concat = (
            F.conv2d(w.flip(2, 3).permute(1, 0, 2, 3), w3, padding=0, stride=1)
            .flip(2, 3)
            .permute(1, 0, 2, 3)
        )
        self.bias_concat = (w3 * b.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b3

        sk_w = self.sk.weight.data.clone().detach()
        sk_b = self.sk.bias.detach().clone() if self.sk.bias is not None else sk_w.new_zeros(sk_w.shape[0])
        target_kernel_size = 3

        H_pixels_to_pad = (target_kernel_size - 1) // 2
        W_pixels_to_pad = (target_kernel_size - 1) // 2
        sk_w = F.pad(
            sk_w, [H_pixels_to_pad, H_pixels_to_pad, W_pixels_to_pad, W_pixels_to_pad]
        )

        self.weight_concat = self.weight_concat + sk_w
        self.bias_concat = self.bias_concat + sk_b

        self.eval_conv.weight.data = self.weight_concat
        if self.eval_conv.bias is not None:
            self.eval_conv.bias.data = self.bias_concat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the convolution block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after convolution and optional activation.
        """
        if self.train_mode:
            pad = 1
            x_pad = F.pad(x, (pad, pad, pad, pad), "constant", 0)
            out = self.conv(x_pad) + self.sk(x)
        else:
            self.update_params()
            out = self.eval_conv(x)

        if self.has_relu:
            out = F.leaky_relu(out, negative_slope=0.05)
        return out


class SPAB(nn.Module):
    def __init__(
        self,
        in_channels: int,
        mid_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        train_mode: bool = True,
        bias: bool = False,
    ):
        """
        Self-parameterized attention block (SPAB) with multiple convolution layers.

        Args:
            in_channels (int): Number of input channels.
            mid_channels (Optional[int], optional): Number of middle channels. Defaults to in_channels.
            out_channels (Optional[int], optional): Number of output channels. Defaults to in_channels.
            train_mode (bool, optional): Indicates if the block is in training mode. Defaults to True.
            bias (bool, optional): Include bias in convolutions. Defaults to False.
        """
        super(SPAB, self).__init__()
        if mid_channels is None:
            mid_channels = in_channels
        if out_channels is None:
            out_channels = in_channels

        self.in_channels = in_channels
        self.c1_r = Conv3XC(
            in_channels, mid_channels, gain1=2, s=1, train_mode=train_mode
        )
        self.c2_r = Conv3XC(
            mid_channels, mid_channels, gain1=2, s=1, train_mode=train_mode
        )
        self.c3_r = Conv3XC(
            mid_channels, out_channels, gain1=2, s=1, train_mode=train_mode
        )
        self.act1 = torch.nn.SiLU(inplace=True)
        self.act2 = activation("lrelu", neg_slope=0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass of the SPAB block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple: (Output tensor, intermediate tensor, attention map).
        """
        out1 = self.c1_r(x)
        out1_act = self.act1(out1)

        out2 = self.c2_r(out1_act)
        out2_act = self.act1(out2)

        out3 = self.c3_r(out2_act)

        sim_att = torch.sigmoid(out3) - 0.5
        out = (out3 + x) * sim_att

        return out, out1, sim_att


class CNNSR(nn.Module):
    """
    Swift Parameter-free Attention Network (SPAN) for efficient super-resolution
    with deeper layers and channel attention.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        feature_channels: int = 48,
        upscale: int = 4,
        bias: bool = True,
        train_mode: bool = True,
        num_blocks: int = 10,
        **kwargs,
    ):
        """
        Initializes the CNNSR model.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            feature_channels (int, optional): Number of feature channels. Defaults to 48.
            upscale (int, optional): Upscaling factor. Defaults to 4.
            bias (bool, optional): Whether to include a bias term. Defaults to True.
            train_mode (bool, optional): If True, the model is in training mode. Defaults to True.
            num_blocks (int, optional): Number of attention blocks in the network. Defaults to 10.
        """
        super(CNNSR, self).__init__()

        # Initial Convolution
        self.conv_1 = Conv3XC(
            in_channels, feature_channels, gain1=2, s=1, train_mode=train_mode
        )

        # Deeper Blocks
        self.blocks = nn.ModuleList(
            [
                SPAB(feature_channels, bias=bias, train_mode=train_mode)
                for _ in range(num_blocks)
            ]
        )

        # Convolution after attention blocks
        self.conv_cat = conv_layer(
            feature_channels * 4, feature_channels, kernel_size=1, bias=True
        )
        self.conv_2 = Conv3XC(
            feature_channels, feature_channels, gain1=2, s=1, train_mode=train_mode
        )

        # Upsampling
        self.upsampler = pixelshuffle_block(
            feature_channels, out_channels, upscale_factor=upscale
        )

    def forward(
        self, x: torch.Tensor, save_attentions: Optional[List[int]] = None
    ) -> Union[torch.Tensor, tuple]:
        """
        Forward pass of the CNNSR model.

        Args:
            x (torch.Tensor): Input tensor.
            save_attentions (Optional[List[int]], optional): List of block indices from which to save attention maps.

        Returns:
            torch.Tensor: Super-resolved output.
            tuple: If save_attentions is specified, returns (output tensor, attention maps).
        """
        # Initial Convolution
        out_feature = self.conv_1(x)

        # Pass through all blocks, accumulating attention outputs
        attentions = []
        for index, block in enumerate(self.blocks):
            out, out2, att = block(out_feature)

            # Save the first residual block output
            if index == 0:
                out_b1 = out

            # Save the last residual block output
            if index == len(self.blocks) - 1:
                out_blast = out2

            # Save attention if needed
            if save_attentions is not None and index in save_attentions:
                attentions.append(att)

        # Final Convolution and concatenation
        out_bn = self.conv_2(out)
        out = self.conv_cat(torch.cat([out_feature, out_bn, out_b1, out_blast], 1))

        # Upsample
        output = self.upsampler(out)

        if save_attentions is not None:
            return output, attentions
        return output
