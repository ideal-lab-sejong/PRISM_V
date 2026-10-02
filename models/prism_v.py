from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_conv(in_ch: int, out_ch: int, kernel_size: int) -> nn.Module:
    p = kernel_size // 2
    return nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=p,
                     bias=False)


def _norm_layer(norm: str, ch: int) -> nn.Module | None:
    if norm == "none":
        return None
    if norm == "instance":
        return nn.InstanceNorm2d(ch, affine=True, track_running_stats=False)
    raise ValueError(f"unknown norm: {norm!r}")


def _act_layer(activation: str, negative_slope: float) -> nn.Module:
    if activation == "relu":
        return nn.ReLU(inplace=True)
    if activation == "leaky_relu":
        return nn.LeakyReLU(negative_slope, inplace=True)
    if activation == "tanh":
        return nn.Tanh()
    if activation == "sigmoid":
        return nn.Sigmoid()
    raise ValueError(f"unknown activation: {activation!r}")


def _conv_block(in_ch: int, out_ch: int, kernel_size: int = 3,
                norm: str = "instance", activation: str = "leaky_relu",
                negative_slope: float = 0.01) -> nn.Sequential:
    layers: list[nn.Module] = [_make_conv(in_ch, out_ch, kernel_size)]
    nl = _norm_layer(norm, out_ch)
    if nl is not None:
        layers.append(nl)
    layers.append(_act_layer(activation, negative_slope))
    return nn.Sequential(*layers)


_BLUR_KERNELS_1D: dict[str, tuple[float, ...]] = {
    "tri3": (1.0, 2.0, 1.0),                      # Tri-3  (sigma~0.71)
    "bin5": (1.0, 4.0, 6.0, 4.0, 1.0),            # Bin-5  (sigma~1.00)
    "bin7": (1.0, 6.0, 15.0, 20.0, 15.0, 6.0, 1.0),  # Bin-7  (sigma~1.22, main)
    "bin9": (1.0, 8.0, 28.0, 56.0, 70.0, 56.0, 28.0, 8.0, 1.0),  # Bin-9 (sigma~1.41)
}


class BlurPool2d(nn.Module):
    def __init__(self, kernel: str = "bin7") -> None:
        super().__init__()
        if kernel not in _BLUR_KERNELS_1D:
            raise ValueError(
                f"unknown blur kernel: {kernel!r} "
                f"(choices: {sorted(_BLUR_KERNELS_1D)})")
        k = torch.tensor(_BLUR_KERNELS_1D[kernel])
        kernel2d = (k[:, None] * k[None, :])
        kernel2d = kernel2d / kernel2d.sum()
        self.pad = k.numel() // 2
        self.register_buffer("kernel", kernel2d[None, None])  # (1,1,K,K)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = x.shape[1]
        ksz = self.kernel.shape[-1]
        w = self.kernel.to(dtype=x.dtype, device=x.device).expand(c, 1, ksz, ksz)
        p = self.pad
        x = F.pad(x, (p, p, p, p), mode="reflect")
        return F.conv2d(x, w, stride=2, groups=c)


@dataclass
class RandomCNNConfig:
    in_channels: int = 3
    depth: int = 4 
    channels: tuple[int, ...] = (128, 128, 128, 128, 128)
    seed: int = 7
    output_layers: tuple[str, ...] = ("stage2", "stage3", "stage4", "stage5")
    norm: str = "instance"     
    downsample: str = "blur"       
    blur_kernel: str = "bin7"
    kernel_size: int | tuple[int, ...] = (3, 3, 3, 1, 1)
    activation: str = "leaky_relu"
    negative_slope: float = 0.01


class RandomCNN(nn.Module):
    def __init__(self, config: RandomCNNConfig):
        super().__init__()
        self.config = config
        if len(config.channels) != config.depth + 1:
            raise ValueError(
                f"channels tuple must have length depth+1 = "
                f"{config.depth + 1}, got {len(config.channels)}: "
                f"{config.channels}")
        ch = list(config.channels)

        # Resolve per-stage kernel schedule.
        if isinstance(config.kernel_size, int):
            kernels = [config.kernel_size] * len(ch)
        else:
            if len(config.kernel_size) != len(ch):
                raise ValueError(
                    f"kernel_size tuple must have length depth+1 = {len(ch)}, "
                    f"got {len(config.kernel_size)}: {config.kernel_size}")
            kernels = list(config.kernel_size)

        self.blocks = nn.ModuleList()
        in_ch = config.in_channels
        for c, k in zip(ch, kernels):
            self.blocks.append(_conv_block(
                in_ch, c, kernel_size=k, norm=config.norm,
                activation=config.activation,
                negative_slope=float(config.negative_slope)))
            in_ch = c
        if config.downsample == "blur":
            self.pool = BlurPool2d(config.blur_kernel)
        else:
            self.pool = nn.MaxPool2d(2, 2)

        self._init_weights(config.seed)
        self._freeze()

    def _init_weights(self, seed: int) -> None:
        g = torch.Generator(device="cpu").manual_seed(seed)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                with torch.no_grad():
                    fan_in = m.in_channels * m.kernel_size[0] * m.kernel_size[1]
                    std = (2.0 / fan_in) ** 0.5
                    m.weight.copy_(torch.randn(m.weight.shape, generator=g) * std)
                    if m.bias is not None:
                        m.bias.zero_()
            elif isinstance(m, nn.InstanceNorm2d):
                if m.weight is not None:
                    with torch.no_grad():
                        m.weight.fill_(1.0)
                        m.bias.zero_()

    def _freeze(self) -> None:
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):  
        return super().train(False)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feats: dict[str, torch.Tensor] = {}
        cur = x
        for i, block in enumerate(self.blocks):
            cur = block(cur if i == 0 else self.pool(cur))
            feats[f"stage{i + 1}"] = cur
        return {k: feats[k] for k in self.config.output_layers if k in feats}


def build_random_cnn_extractor(
        seed: int = 7, depth: int = 4,
        output_layers: tuple[str, ...] = ("stage2", "stage3", "stage4",
                                          "stage5"),
        norm: str = "instance", downsample: str = "blur",
        blur_kernel: str = "bin7",
        kernel_size: int | tuple[int, ...] = (3, 3, 3, 1, 1),
        channels: tuple[int, ...] = (128, 128, 128, 128, 128),
        activation: str = "leaky_relu",
        negative_slope: float = 0.01) -> RandomCNN:
    cfg = RandomCNNConfig(seed=seed, depth=depth,
                          output_layers=output_layers, norm=norm,
                          downsample=downsample, blur_kernel=blur_kernel,
                          kernel_size=kernel_size,
                          channels=channels, activation=activation,
                          negative_slope=negative_slope)
    return RandomCNN(cfg)
