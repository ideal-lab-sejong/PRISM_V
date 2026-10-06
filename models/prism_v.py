"""Frozen CNN with the fixed PRISM-V main architecture."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.main_spec import CNN_KERNEL, MAIN_CONFIG


class BlurPool2d(nn.Module):
    def __init__(self):
        super().__init__()
        kernel = torch.tensor((1.0, 6.0, 15.0, 20.0, 15.0, 6.0, 1.0))
        kernel2d = kernel[:, None] * kernel[None, :]
        self.register_buffer("kernel", (kernel2d / kernel2d.sum())[None, None])
        self.pad = 3

    @torch.no_grad()
    def forward(self, images):
        channels = images.shape[1]
        weights = self.kernel.to(dtype=images.dtype, device=images.device).expand(channels, 1, 7, 7)
        padded = F.pad(images, (self.pad,) * 4, mode="reflect")
        return F.conv2d(padded, weights, stride=2, groups=channels)


class RandomCNN(nn.Module):
    def __init__(self, seed, channels, output_layers):
        super().__init__()
        if len(channels) != len(CNN_KERNEL):
            raise ValueError("PRISM-V requires five CNN stages")
        self.output_layers = tuple(output_layers)
        self.blocks = nn.ModuleList()
        incoming = 3
        for outgoing, kernel in zip(channels, CNN_KERNEL):
            self.blocks.append(nn.Sequential(
                nn.Conv2d(incoming, outgoing, kernel, padding=kernel // 2, bias=False),
                nn.InstanceNorm2d(outgoing, affine=True, track_running_stats=False),
                nn.LeakyReLU(negative_slope=0.01, inplace=True)))
            incoming = outgoing
        self.pool = BlurPool2d()
        generator = torch.Generator(device="cpu").manual_seed(seed)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                fan_in = module.in_channels * module.kernel_size[0] * module.kernel_size[1]
                with torch.no_grad():
                    module.weight.copy_(torch.randn(module.weight.shape, generator=generator) * (2.0 / fan_in) ** 0.5)
            elif isinstance(module, nn.InstanceNorm2d):
                with torch.no_grad():
                    module.weight.fill_(1.0)
                    module.bias.zero_()
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        return super().train(False)

    def forward(self, images):
        features = {}
        current = images
        for index, block in enumerate(self.blocks):
            current = block(current if index == 0 else self.pool(current))
            features[f"stage{index + 1}"] = current
        return {name: features[name] for name in self.output_layers}


def build_random_cnn_extractor(seed=1, *, channels=tuple(MAIN_CONFIG["cnn_channels"]),
                               output_layers=tuple(MAIN_CONFIG["cnn_layers"])):
    return RandomCNN(seed, channels, output_layers)
