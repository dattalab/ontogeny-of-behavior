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
        separable=False,
        **kwargs
    ):
        conv = (
            partial(DepthwiseSeparableConv, kernel_size=kernel_size)
            if separable
            else partial(
                nn.Conv2d, kernel_size=kernel_size, padding=kernel_size // 2, **kwargs
            )
        )
        super().__init__(
            conv(in_features, out_features),
            norm(out_features) if norm else nn.Identity(),
            activation() if activation else nn.Identity(),
        )


class DepthwiseSeparableConv(nn.Sequential):
    def __init__(self, in_features: int, out_features: int, kernel_size=3):
        super().__init__(
            nn.Conv2d(
                in_features,
                in_features,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                groups=in_features,
            ),
            nn.Conv2d(in_features, out_features, kernel_size=1),
        )


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
        activation: nn.Module = nn.LeakyReLU,
    ):
        expanded_features = in_features * expansion
        residual = ResidualAdd if in_features == out_features else nn.Sequential
        super().__init__(
            residual(
                nn.Sequential(
                    ConvNormAct(
                        in_features,
                        expanded_features,
                        kernel_size=1,
                        activation=activation,
                    ),
                    # like depthwise separable convolution
                    ConvNormAct(
                        expanded_features,
                        expanded_features,
                        kernel_size=3,
                        groups=expanded_features,
                        activation=activation,
                    ),
                    SqueezeExcitation(
                        expanded_features, int(expanded_features * se_ratio)
                    ),
                    ConvNormAct(
                        expanded_features, out_features, kernel_size=1, activation=None
                    ),
                )
            ),
            activation(),
        )
        self.out_channels = out_features


class FusedMBConv(nn.Sequential):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        expansion: int = 4,
        se_ratio: float = 0.25,
        activation: nn.Module = nn.LeakyReLU,
    ):
        expanded_features = in_features * expansion
        residual = ResidualAdd if in_features == out_features else nn.Sequential
        super().__init__(
            residual(
                nn.Sequential(
                    ConvNormAct(
                        in_features,
                        expanded_features,
                        kernel_size=3,
                        activation=activation,
                    ),
                    SqueezeExcitation(
                        expanded_features, int(expanded_features * se_ratio)
                    ),
                    ConvNormAct(
                        expanded_features, out_features, kernel_size=1, activation=None
                    ),
                )
            ),
            activation(),
        )
        self.out_channels = out_features


class EfficientDown(nn.Sequential):
    def __init__(self, block: nn.Module):
        super().__init__(
            nn.MaxPool2d(2),
            block,
        )


class EfficientUp(nn.Sequential):
    def __init__(self, block: nn.Module):
        super().__init__(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True), block
        )


class EfficientUNetUp(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.block = block

    def forward(self, x1, x2=None):
        x1 = self.up(x1)
        if x2 is not None:
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]

            x1 = F.pad(
                x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2]
            )

            x1 = torch.cat([x2, x1], dim=1)
        return self.block(x1)


class DoubleConv(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        mid_channels=None,
        separable=True,
        activation=nn.LeakyReLU,
    ):
        if mid_channels is None:
            mid_channels = out_channels
        super().__init__(
            ConvNormAct(
                in_channels, mid_channels, separable=separable, activation=activation
            ),
            ConvNormAct(
                mid_channels, out_channels, separable=separable, activation=activation
            ),
        )


class Down(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        separable=True,
        double_conv=True,
        activation=nn.LeakyReLU,
    ):
        super().__init__()
        conv = DoubleConv if double_conv else ConvNormAct
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            conv(in_channels, out_channels, separable=separable, activation=activation),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        cat_channels=None,
        bilinear=True,
        double_conv=True,
        separable=True,
        activation=nn.LeakyReLU,
    ):
        super().__init__()
        conv = DoubleConv if double_conv else ConvNormAct
        addition = 0 if cat_channels is None else cat_channels
        self.conv = conv(
            in_channels + addition,
            out_channels,
            separable=separable,
            activation=activation,
        )

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels, kernel_size=2, stride=2
            )

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
    def __init__(
        self,
        depth: int,
        init_channel: int,
        init_depth: int,
        channel_scaling: float,
        depth_scaling: float,
        input_channel: int = 16,
        separable=True,
        double_conv=True,
        activation: nn.Module = nn.LeakyReLU,
    ):
        """Channels must include the input channel count (usually 1) as well"""
        super().__init__()
        channels = [int(init_channel * channel_scaling**i) for i in range(depth)]
        depths = [int(init_depth * depth_scaling**i) for i in range(depth)]
        channels = [input_channel] + channels

        conv = DoubleConv if double_conv else ConvNormAct

        self.down = nn.ModuleList([conv(1, input_channel, separable=separable, activation=activation)])
        # downsample
        for i in range(depth):
            to_add = []
            for j in range(depths[i]):
                in_ch = channels[i] if j == 0 else channels[i + 1]
                out_ch = channels[i + 1]
                if j == depths[i] - 1:
                    to_add.append(
                        Down(
                            in_ch, out_ch, separable=separable, double_conv=double_conv, activation=activation
                        )
                    )
                else:
                    to_add.append(conv(in_ch, out_ch, separable=separable, activation=activation))
            self.down.extend(to_add)

        self.up = nn.ModuleList([])
        # upsample
        for i in range(depth - 1, -1, -1):
            to_add = []
            for j in range(depths[i]):
                in_ch = channels[i + 1] if j == 0 else channels[i]
                out_ch = channels[i]
                if j == depths[i] - 1:
                    to_add.append(
                        Up(
                            in_ch,
                            out_ch,
                            separable=separable,
                            double_conv=double_conv,
                            cat_channels=out_ch,
                            activation=activation,
                        )
                    )
                else:
                    to_add.append(conv(in_ch, out_ch, separable=separable, activation=activation))
            self.up.extend(to_add)

        self.up.append(OutConv(channels[0], 1))

    def forward(self, x):
        down = {}
        down[0] = self.down[0](x)
        for i, layer in enumerate(self.down[1:], start=1):
            down[i] = layer(down[i - 1])
        keys = list(down)[::-1]
        if isinstance(self.up[0], Up):
            out = self.up[0](down[keys[0]], down[keys[1]])
        else:
            out = self.up[0](down[keys[0]])
        for layer, k in zip(self.up[1:-1], keys[2:]):
            if isinstance(layer, Up):
                out = layer(out, down[k])
            else:
                out = layer(out)
        out = self.up[-1](out)
        return F.sigmoid(out)


class Autoencoder(nn.Module):
    def __init__(
        self,
        depth: int,
        init_channel: int,
        init_depth: int,
        channel_scaling: float,
        depth_scaling: float,
        input_channel: int = 16,
        separable=True,
        double_conv=True,
        activation: nn.Module = nn.LeakyReLU,
    ):
        """Channels must include the input channel count (usually 1) as well"""
        super().__init__()
        channels = [int(init_channel * channel_scaling**i) for i in range(depth)]
        depths = [int(init_depth * depth_scaling**i) for i in range(depth)]
        channels = [input_channel] + channels

        conv = DoubleConv if double_conv else ConvNormAct

        self.down = nn.ModuleList([conv(1, input_channel, separable=separable, activation=activation)])
        # downsample
        for i in range(depth):
            to_add = []
            for j in range(depths[i]):
                # if last conv of this layer, downsample
                in_ch = channels[i] if j == 0 else channels[i + 1]
                out_ch = channels[i + 1]
                if j == depths[i] - 1:
                    to_add.append(
                        Down(
                            in_ch, out_ch, separable=separable, double_conv=double_conv, activation=activation
                        )
                    )
                else:
                    to_add.append(conv(in_ch, out_ch, separable=separable, activation=activation))
            self.down.extend(to_add)

        self.up = nn.ModuleList([])
        # upsample
        for i in range(depth - 1, -1, -1):
            to_add = []
            for j in range(depths[i]):
                in_ch = channels[i + 1] if j == 0 else channels[i]
                out_ch = channels[i]
                if j == depths[i] - 1:
                    to_add.append(
                        Up(in_ch, out_ch, separable=separable, double_conv=double_conv, activation=activation)
                    )
                else:
                    to_add.append(conv(in_ch, out_ch, separable=separable, activation=activation))
            self.up.extend(to_add)

        self.up.append(OutConv(channels[0], 1))

    def forward(self, x):
        for layer in self.down:
            x = layer(x)
        for layer in self.up:
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
        activation: nn.Module = nn.LeakyReLU,
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
                ConvNormAct(1, input_channel, kernel_size=3, activation=activation),
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
                    activation=activation,
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
                    channels[i + 1] if j == 0 else channels[i], channels[i], expansion, activation=activation
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


class EfficientUNet(nn.Module):
    """EfficientNet inspired UNet"""

    def __init__(
        self,
        depth: int,
        init_channel: int,
        init_depth: int,
        channel_scaling: float,
        depth_scaling: float,
        input_channel: int = 16,
        fused: int = 1,
        expansion: int = 4,
        activation: nn.Module = nn.LeakyReLU,
    ):
        super().__init__()
        assert (
            channel_scaling >= 1
        ), "Channel scaling must be greater than or equal to 1"
        assert depth_scaling >= 1, "Depth scaling must be greater than or equal to 1"
        channels = [int(init_channel * channel_scaling**i) for i in range(depth)]
        depths = [int(init_depth * depth_scaling**i) for i in range(depth)]
        channels = [input_channel] + channels

        self.down = nn.ModuleList()
        self.down.append(ConvNormAct(1, input_channel, kernel_size=3, activation=activation))

        for i, (depth, (in_ch, out_ch)) in enumerate(zip(depths, zip(channels[:-1], channels[1:]))):
            _Module = FusedMBConv if i < fused else MBConv
            for j in range(depth):
                if j == depth - 1:
                    mod = EfficientDown(_Module(
                        in_ch,
                        out_ch,
                        expansion,
                        activation=activation,
                    ))
                else:
                    mod = _Module(
                        in_ch,
                        in_ch,
                        expansion,
                        activation=activation,
                    )
                self.down.append(mod)

        depths = depths[::-1]
        channels = channels[::-1]

        self.up = nn.ModuleList()
        for i, (depth, (in_ch, out_ch)) in enumerate(zip(depths, zip(channels[:-1], channels[1:]))):
            _Module = FusedMBConv if i > (len(channels) - 1 - fused) else MBConv
            for j in range(depth):
                if j == 0:
                    mod = EfficientUNetUp(_Module(
                        in_ch + out_ch,
                        out_ch,
                        expansion,
                        activation=activation,
                    ))
                else:
                    mod = _Module(
                        out_ch,
                        out_ch,
                        expansion,
                        activation=activation,
                    )
                self.up.append(mod)

        self.final_conv = nn.Conv2d(channels[-1], 1, kernel_size=3, padding=1)

    def forward(self, x):
        down = {}
        down[0] = self.down[0](x)
        for i, layer in enumerate(self.down[1:], start=1):
            down[i] = layer(down[i - 1])
        keys = list(down)[::-1]
        if isinstance(self.up[0], EfficientUNetUp):
            out = self.up[0](down[keys[0]], down[keys[1]])
        else:
            out = self.up[0](down[keys[0]])
        for layer, k in zip(self.up[1:], keys[2:]):
            if isinstance(layer, EfficientUNetUp):
                out = layer(out, down[k])
            else:
                out = layer(out)
        return F.sigmoid(self.final_conv(out))