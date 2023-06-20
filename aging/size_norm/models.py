import torch
import torch.nn as nn
import torch.nn.functional as F
from toolz import partial, concatv
from typing import Optional
from torchvision.ops import SqueezeExcitation


# From: https://github.com/FrancescoSaverioZuppichini/BottleNeck-InvertedResidual-FusedMBConv-in-PyTorch
class ConvNormAct(nn.Sequential):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: int = 3,
        activation: Optional[nn.Module] = nn.LeakyReLU,
        norm: Optional[nn.Module] = nn.BatchNorm2d,
        **kwargs
    ):
        super().__init__(
            nn.Conv2d(
                in_features,
                out_features,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                **kwargs
            ),
            norm(out_features) if norm else nn.Identity(),
            activation(inplace=True) if activation else nn.Identity(),
        )


class DepthwiseSeparableConv(nn.Sequential):
    def __init__(self, in_features: int, out_features: int):
        super().__init__(
            nn.Conv2d(
                in_features, in_features, kernel_size=3, padding=1, groups=in_features
            ),
            nn.Conv2d(in_features, out_features, kernel_size=1),
        )


class SeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # depth-wise separable convolutions
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, 1, bias=False)
        )

    def forward(self, x):
        return self.conv1(x)


class ResidualAdd(nn.Module):
    def __init__(self, block: nn.Module, shortcut: Optional[nn.Module] = None):
        super().__init__()
        self.block = block
        self.shortcut = shortcut

    def forward(self, x):
        return self.block(x) + (self.shortcut(x) if self.shortcut else x)


# optimized/compressed form of an inverted residual block
class MBConv(nn.Sequential):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        expansion: int = 4,
        se_ratio: float = 0.25,
    ):
        expanded_features = in_features * expansion
        residual = ResidualAdd if in_features == out_features else nn.Sequential
        super().__init__(
            residual(
                nn.Sequential(
                    ConvNormAct(in_features, expanded_features, kernel_size=1),
                    # depthwise separable convolution
                    ConvNormAct(
                        expanded_features,
                        expanded_features,
                        kernel_size=3,
                        groups=expanded_features,
                    ),
                    SqueezeExcitation(
                        expanded_features, int(expanded_features * se_ratio)
                    ),
                    ConvNormAct(
                        expanded_features, out_features, kernel_size=1, activation=None
                    ),
                )
            ),
            nn.LeakyReLU(inplace=True),
        )


class FusedMBConv(nn.Sequential):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        expansion: int = 4,
        se_ratio: float = 0.25,
    ):
        expanded_features = in_features * expansion
        residual = ResidualAdd if in_features == out_features else nn.Sequential
        super().__init__(
            residual(
                nn.Sequential(
                    ConvNormAct(in_features, expanded_features, kernel_size=3),
                    SqueezeExcitation(
                        expanded_features, int(expanded_features * se_ratio)
                    ),
                    ConvNormAct(
                        expanded_features, out_features, kernel_size=1, activation=None
                    ),
                )
            ),
            nn.LeakyReLU(inplace=True),
        )


class EfficientDown(nn.Sequential):
    def __init__(self, block: nn.Module):
        super().__init__(
            nn.MaxPool2d(2),
            block,
        )


class EfficientUp(nn.Sequential):
    def __init__(snlf, block: nn.Module):
        super().__init__(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True), block
        )


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, separable=True):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        conv = (
            SeparableConv
            if separable
            else partial(nn.Conv2d, kernel_size=3, padding=1, bias=False)
        )
        self.double_conv = nn.Sequential(
            conv(in_channels, mid_channels),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(inplace=True),
            conv(mid_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, separable=True):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, separable=separable),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, cat_channels=None, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels, kernel_size=2, stride=2
            )
            addition = 0 if cat_channels is None else cat_channels
            self.conv = DoubleConv(in_channels + addition, out_channels)

    def forward(self, x1, x2=None):
        x1 = self.up(x1)

        if x2 is not None:
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]

            x1 = F.pad(
                x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2]
            )

            x1 = torch.cat([x2, x1], dim=1)
        return self.conv(x1)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


# ----- model classes ----- #


class UNet(nn.Module):
    def __init__(self, channels: list, separable=True):
        """Channels must include the input channel count (usually 1) as well"""
        super().__init__()
        self.channels = channels
        bilinear = False

        self.down = [DoubleConv(channels[0], channels[1], separable=separable)]
        # downsample
        for _in, _out in zip(channels[1:], channels[2:]):
            self.down.append(Down(_in, _out, separable))
        self.down = nn.ModuleList(self.down)

        self.up = []
        # upsample
        for _in, _out in zip(channels[::-1], channels[::-1][1:-1]):
            self.up.append(Up(_in, _out, bilinear=bilinear, cat_channels=_out))

        self.up.append(OutConv(channels[::-1][-2], channels[0]))
        self.up = nn.ModuleList(self.up)

    def forward(self, x):
        down = {}
        down[0] = self.down[0](x)
        for i, layer in enumerate(self.down[1:], start=1):
            down[i] = layer(down[i - 1])
        keys = list(down)[::-1]
        out = self.up[0](down[keys[0]], down[keys[1]])
        for layer, k in zip(self.up[1:-1], keys[2:]):
            out = layer(out, down[k])
        out = self.up[-1](out)
        return F.sigmoid(out)


class Autoencoder(nn.Module):
    def __init__(self, channels: list, separable=True):
        """Channels must include the input channel count (usually 1) as well"""
        super().__init__()
        bilinear = False
        self.channels = channels

        self.down = [DoubleConv(channels[0], channels[1], separable=separable)]
        # downsample
        for _in, _out in zip(channels[1:], channels[2:]):
            self.down.append(Down(_in, _out, separable))
        self.down = nn.ModuleList(self.down)

        self.up = []
        # upsample
        for _in, _out in zip(channels[::-1], channels[::-1][1:-1]):
            self.up.append(Up(_in, _out, bilinear=bilinear))

        self.up.append(OutConv(channels[::-1][-2], channels[0]))
        self.up = nn.ModuleList(self.up)

    def forward(self, x):
        for layer in concatv(self.down, self.up):
            x = layer(x)
        return F.sigmoid(x)


class EfficientAutoencoder(nn.Module):
    """EfficientNet inspired autoencoder"""

    def __init__(
        self,
        depth: int,
        init_channel: int,
        init_depth: int,
        channel_scaling: float,
        depth_scaling: float,
        input_channel: int = 16,
        fused: int = 2,
        expansion: int = 4,
    ):
        super().__init__()
        assert (
            channel_scaling >= 1
        ), "Channel scaling must be greater than or equal to 1"
        assert depth_scaling >= 1, "Depth scaling must be greater than or equal to 1"
        channels = [int(init_channel * channel_scaling**i) for i in range(depth)]
        depths = [int(init_depth * depth_scaling**i) for i in range(depth)]
        channels = [input_channel] + channels

        self.down = nn.ModuleList(
            [
                ConvNormAct(1, input_channel, kernel_size=3),
            ]
        )
        for i in range(depth):
            _Module = FusedMBConv if i < fused else MBConv
            to_add = []
            for j in range(depths[i]):
                mod = _Module(
                    channels[i] if j == 0 else channels[i + 1],
                    channels[i + 1],
                    expansion,
                )
                to_add.append(mod)
            to_add[-1] = EfficientDown(to_add[-1])
            self.down.extend(to_add)

        self.up = nn.ModuleList([])
        for i in range(depth - 1, -1, -1):
            _Module = FusedMBConv if i < fused else MBConv
            to_add = []
            for j in range(depths[i]):
                mod = _Module(
                    channels[i + 1] if j == 0 else channels[i], channels[i], expansion
                )
                to_add.append(mod)
            to_add[-1] = EfficientUp(to_add[-1])
            self.up.extend(to_add)

        self.up.append(
            nn.Sequential(
                nn.Conv2d(channels[0], 1, kernel_size=3, padding=1), nn.Sigmoid()
            )
        )

    def forward(self, x):
        for layer in self.down:
            x = layer(x)
        for layer in self.up:
            x = layer(x)
        return x
