from __future__ import annotations

import torch
import torch.nn.functional as F


class RawPatchExtractor(torch.nn.Module):
    def __init__(self, factor: int = 2):
        super().__init__()
        self.factor = factor

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"raw": F.pixel_unshuffle(x, self.factor)}

    def train(self, mode: bool = True) -> "RawPatchExtractor":
        return super().train(False)


def build_raw_patch_extractor(factor: int = 2) -> RawPatchExtractor:
    return RawPatchExtractor(factor=factor)