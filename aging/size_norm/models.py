import torch
import torch.nn as nn
import torch.nn.functional as F
from toolz import partial
from typing import Optional
from torchvision.ops import SqueezeExcitation
from collections import OrderedDict


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
        dropout_p: float = 0.0,
        **kwargs
    ):
        conv = (
            partial(DepthwiseSeparableConv, kernel_size=kernel_size)
            if separable
            else partial(
                nn.Conv2d, kernel_size=kernel_size, padding=kernel_size // 2, **kwargs
            )
        )
        layers = OrderedDict()
        layers["conv"] = conv(in_features, out_features)
        if norm:
            layers["norm"] = norm(out_features)
        if dropout_p > 0 and out_features > 1:
            layers["dropout"] = nn.Dropout2d(dropout_p)
        if activation:
            layers["activation"] = activation()
        super().__init__(layers)


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


class FFTBranchedConv(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: nn.Module = nn.LeakyReLU,
        dropout_p: float = 0.0,
        separable: bool = True,
        double_conv: bool = True,
        residual: bool = False,
        **kwargs
    ):
        super().__init__()

        if double_conv:
            conv = partial(
                DoubleConv,
                residual=residual,
                separable=separable,
                activation=activation,
                dropout_p=dropout_p,
            )
        else:
            conv = partial(
                ConvNormAct,
                activation=activation,
                separable=separable,
                dropout_p=dropout_p,
                **kwargs
            )

        fft_conv = partial(
            DoubleConv,
            residual=residual,
            separable=separable,
            activation=activation,
            dropout_p=dropout_p,
            use_fft=True,
        )

        self.conv = conv(in_features, out_features)
        self.fft_conv = fft_conv(in_features, out_features)

    def forward(self, x):
        return self.conv(x) + self.fft_conv(x)


class ResidualAdd(nn.Module):
    def __init__(self, block: nn.Module, shortcut: Optional[nn.Module] = None):
        super().__init__()
        self.block = block
        self.shortcut = shortcut

    def forward(self, x):
        return self.block(x) + (self.shortcut(x) if self.shortcut is not None else x)


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


class DoubleConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        mid_channels=None,
        separable=True,
        residual=False,
        activation=nn.LeakyReLU,
        dropout_p: float = 0.0,
        use_fft=False,
    ):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.use_fft = use_fft
        self.block = nn.Sequential(
            ConvNormAct(
                in_channels,
                mid_channels,
                separable=separable,
                activation=activation,
                dropout_p=dropout_p,
            ),
            ConvNormAct(
                mid_channels,
                out_channels,
                separable=separable,
                activation=activation,
                dropout_p=dropout_p,
            ),
        )
        if residual:
            self.block = ResidualAdd(
                self.block,
                shortcut=ConvNormAct(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    activation=activation,
                    dropout_p=dropout_p,
                ),
            )

    def forward(self, x):
        if self.use_fft:
            return torch.fft.irfft2(self.block(torch.fft.rfft2(x.to(torch.float)).real.to(x.dtype)).to(torch.float), x.size()[-2:]).to(x.dtype)
        return self.block(x)


class Down(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        separable=True,
        double_conv=True,
        residual=False,
        dropout_p: float = 0.0,
        activation=nn.LeakyReLU,
        use_fft_branch=False,
    ):
        super().__init__()
        if use_fft_branch:
            conv = partial(
                FFTBranchedConv,
                residual=residual,
                double_conv=double_conv,
            )
        else:
            conv = partial(DoubleConv, residual=residual) if double_conv else ConvNormAct
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            conv(
                in_channels,
                out_channels,
                separable=separable,
                activation=activation,
                dropout_p=dropout_p,
            ),
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
        residual=False,
        dropout_p: float = 0.0,
        activation=nn.LeakyReLU,
    ):
        super().__init__()
        conv = partial(DoubleConv, residual=residual) if double_conv else ConvNormAct
        addition = 0 if cat_channels is None else cat_channels
        self.conv = conv(
            in_channels + addition,
            out_channels,
            separable=separable,
            activation=activation,
            dropout_p=dropout_p,
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


class Encoder(nn.Module):
    def __init__(
        self,
        channels,
        depths,
        double_conv,
        residual,
        separable,
        activation,
        dropout_p,
        use_fft_branch,
    ):
        super().__init__()
        if use_fft_branch:
            conv = partial(
                FFTBranchedConv,
                double_conv=double_conv,
                residual=residual,
            )
        else:
            conv = partial(DoubleConv, residual=residual) if double_conv else ConvNormAct

        self.down = nn.ModuleList(
            [
                conv(
                    1,
                    channels[0],
                    separable=separable,
                    activation=activation,
                    dropout_p=dropout_p,
                )
            ]
        )
        # downsample
        for i in range(len(depths)):
            to_add = []
            for j in range(depths[i]):
                # if last conv of this layer, downsample
                in_ch = channels[i] if j == 0 else channels[i + 1]
                out_ch = channels[i + 1]
                if j == depths[i] - 1:
                    to_add.append(
                        Down(
                            in_ch,
                            out_ch,
                            separable=separable,
                            double_conv=double_conv,
                            activation=activation,
                            dropout_p=dropout_p,
                            use_fft_branch=use_fft_branch,
                        )
                    )
                else:
                    to_add.append(
                        conv(
                            in_ch,
                            out_ch,
                            separable=separable,
                            activation=activation,
                            dropout_p=dropout_p,
                        )
                    )
            self.down.extend(to_add)

    def forward(self, x):
        for layer in self.down:
            x = layer(x)
        return x


class Bottleneck(nn.Module):
    def __init__(
        self,
        img_size,
        depth,
        final_channel,
        bottleneck_channels,
        latent_dim,
        separable,
        activation,
        dropout_p,
        is_variational=False,
    ):
        super().__init__()
        self.is_variational = is_variational
        final_img_size = img_size // (2**depth)
        bottle_in = bottleneck_channels * (final_img_size**2)
        down_layers = [
            ConvNormAct(
                final_channel,
                bottleneck_channels,
                separable=separable,
                activation=activation,
                dropout_p=dropout_p,
            ),
            nn.Flatten(),
        ]
        if is_variational:
            self.mu = nn.Linear(bottle_in, latent_dim)
            self.logvar = nn.Linear(bottle_in, latent_dim)
        else:
            down_layers.append(nn.Linear(bottle_in, latent_dim))

        self.down = nn.Sequential(*down_layers)

        up_layers = [
            nn.Linear(latent_dim, bottle_in),
            activation(),
            nn.Unflatten(1, (bottleneck_channels, final_img_size, final_img_size)),
            ConvNormAct(
                bottleneck_channels,
                final_channel,
                separable=separable,
                activation=activation,
                dropout_p=dropout_p,
            ),
        ]
        if not is_variational:
            up_layers.insert(0, nn.Tanh())

        self.up = nn.Sequential(*up_layers)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def pre_decoder(self, latents):
        return self.up(latents)

    def forward(self, x):
        x = self.down(x)
        if self.is_variational:
            mu = self.mu(x)
            logvar = self.logvar(x) + 1e-6
            # resampled latent space
            x = self.reparameterize(mu, logvar)
            return self.pre_decoder(x), x, mu, logvar
        return self.pre_decoder(x)


class Decoder(nn.Module):
    def __init__(
        self, channels, depths, double_conv, residual, separable, activation, dropout_p
    ):
        super().__init__()
        conv = partial(DoubleConv, residual=residual) if double_conv else ConvNormAct

        self.up = nn.ModuleList([])
        # upsample
        for i in range(len(depths) - 1, -1, -1):
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
                            activation=activation,
                            dropout_p=dropout_p,
                        )
                    )
                else:
                    to_add.append(
                        conv(
                            in_ch,
                            out_ch,
                            separable=separable,
                            activation=activation,
                            dropout_p=dropout_p,
                        )
                    )
            self.up.extend(to_add)

        self.up.append(OutConv(channels[0], 1))

    def forward(self, x):
        for layer in self.up:
            x = layer(x)
        return x


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
        residual=False,
        activation: nn.Module = nn.LeakyReLU,
    ):
        """Channels must include the input channel count (usually 1) as well"""
        super().__init__()
        channels = [int(init_channel * channel_scaling**i) for i in range(depth)]
        depths = [int(init_depth * depth_scaling**i) for i in range(depth)]
        channels = [input_channel] + channels

        conv = partial(DoubleConv, residual=residual) if double_conv else ConvNormAct

        self.down = nn.ModuleList(
            [conv(1, input_channel, separable=separable, activation=activation)]
        )
        # downsample
        for i in range(depth):
            to_add = []
            for j in range(depths[i]):
                in_ch = channels[i] if j == 0 else channels[i + 1]
                out_ch = channels[i + 1]
                if j == depths[i] - 1:
                    to_add.append(
                        Down(
                            in_ch,
                            out_ch,
                            separable=separable,
                            double_conv=double_conv,
                            activation=activation,
                        )
                    )
                else:
                    to_add.append(
                        conv(in_ch, out_ch, separable=separable, activation=activation)
                    )
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
                    to_add.append(
                        conv(in_ch, out_ch, separable=separable, activation=activation)
                    )
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
        residual=False,
        dropout_p: float = 0.0,
        activation: nn.Module = nn.LeakyReLU,
        bottleneck: Optional[int] = None,
        bottleneck_channels: Optional[int] = 30,
        img_size: Optional[int] = None,
        use_fft_branch: bool = False,
    ):
        """Channels must include the input channel count (usually 1) as well"""
        super().__init__()
        channels = [int(init_channel * channel_scaling**i) for i in range(depth)]
        depths = [int(init_depth * depth_scaling**i) for i in range(depth)]
        channels = [input_channel] + channels

        self.has_bottleneck = bottleneck is not None and bottleneck > 0
        if self.has_bottleneck:
            assert img_size is not None, "Must provide image size for bottleneck"
            self.bottleneck = Bottleneck(
                img_size,
                depth,
                channels[-1],
                bottleneck_channels,
                bottleneck,
                separable,
                activation,
                dropout_p,
            )

        self.encoder = Encoder(
            channels,
            depths,
            double_conv,
            residual,
            separable,
            activation,
            dropout_p,
            use_fft_branch,
        )
        self.decoder = Decoder(
            channels, depths, double_conv, residual, separable, activation, dropout_p
        )

    def forward(self, x):
        x = self.encoder(x)
        if self.has_bottleneck:
            x = self.bottleneck(x)
        x = self.decoder(x)
        return F.sigmoid(x)


class VariationalAutoencoder(nn.Module):
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
        residual=False,
        dropout_p: float = 0.0,
        activation: nn.Module = nn.LeakyReLU,
        bottleneck: Optional[int] = None,
        bottleneck_channels: Optional[int] = 30,
        img_size: Optional[int] = None,
        use_fft_branch: bool = False,
    ):
        """Channels must include the input channel count (usually 1) as well"""
        super().__init__()
        channels = [int(init_channel * channel_scaling**i) for i in range(depth)]
        depths = [int(init_depth * depth_scaling**i) for i in range(depth)]
        channels = [input_channel] + channels

        self.encoder = Encoder(
            channels,
            depths,
            double_conv,
            residual,
            separable,
            activation,
            dropout_p,
            use_fft_branch,
        )

        self.bottleneck = Bottleneck(
            img_size,
            depth,
            channels[-1],
            bottleneck_channels,
            bottleneck,
            separable,
            activation,
            dropout_p,
            is_variational=True,
        )

        self.decoder = Decoder(
            channels, depths, double_conv, residual, separable, activation, dropout_p
        )

    def decode(self, z):
        return self.decoder(self.bottleneck.pre_decoder(z))

    def forward(self, x):
        x = self.encoder(x)

        x, z, mu, logvar = self.bottleneck(x)

        x = self.decoder(x)
        return F.sigmoid(x), z, mu, logvar


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
                    channels[i + 1] if j == 0 else channels[i],
                    channels[i],
                    expansion,
                    activation=activation,
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
        self.down.append(
            ConvNormAct(1, input_channel, kernel_size=3, activation=activation)
        )

        for i, (depth, (in_ch, out_ch)) in enumerate(
            zip(depths, zip(channels[:-1], channels[1:]))
        ):
            _Module = FusedMBConv if i < fused else MBConv
            for j in range(depth):
                if j == depth - 1:
                    mod = EfficientDown(
                        _Module(
                            in_ch,
                            out_ch,
                            expansion,
                            activation=activation,
                        )
                    )
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
        for i, (depth, (in_ch, out_ch)) in enumerate(
            zip(depths, zip(channels[:-1], channels[1:]))
        ):
            _Module = FusedMBConv if i > (len(channels) - 1 - fused) else MBConv
            for j in range(depth):
                if j == 0:
                    mod = EfficientUNetUp(
                        _Module(
                            in_ch + out_ch,
                            out_ch,
                            expansion,
                            activation=activation,
                        )
                    )
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


class Regression(nn.Module):
    def __init__(self, input_size, linear=False):
        super().__init__()
        if linear:
            self.layer = nn.Sequential(
                nn.Flatten(),
                nn.Linear(input_size, 1),
            )
        else:
            self.layer = nn.Sequential(
                ConvNormAct(1, 4, kernel_size=3, separable=True, activation=nn.GELU),
                Down(4, 8, double_conv=False, activation=nn.GELU),
                Down(8, 16, double_conv=False, activation=nn.GELU),
                Down(16, 32, double_conv=False, activation=nn.GELU),
                Down(32, 64, double_conv=False, activation=nn.GELU),
                nn.Flatten(),
                nn.Linear(64 * 5 * 5, 1),
            )

    def forward(self, x):
        return self.layer(x)


class LogisticRegression(nn.Module):
    def __init__(self, n_classes, linear=False):
        super().__init__()
        if linear:
            self.layer = nn.Sequential(
                nn.Flatten(),
                nn.Linear(80 * 80, n_classes),
            )
        else:
            self.layer = nn.Sequential(
                ConvNormAct(1, 4, kernel_size=3, separable=True, activation=nn.GELU),
                Down(4, 8, double_conv=False, activation=nn.GELU),
                Down(8, 16, double_conv=False, activation=nn.GELU),
                Down(16, 32, double_conv=False, activation=nn.GELU),
                Down(32, 64, double_conv=False, activation=nn.GELU),
                nn.Flatten(),
                nn.Linear(64 * 5 * 5, n_classes),
            )

    def forward(self, x):
        return self.layer(x)
