"""Full-channel Gaussian fitting and scoring for the PRISM-V main model."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from scripts.main_spec import COV_RIDGE


def concat_features(feature_dict, layer_names):
    selected = [feature_dict[name] for name in layer_names]
    target = (max(t.shape[-2] for t in selected), max(t.shape[-1] for t in selected))
    aligned = [F.interpolate(t, size=target, mode="bilinear", align_corners=False)
               if t.shape[-2:] != target else t for t in selected]
    return torch.cat(aligned, dim=1)


@dataclass
class PrismState:
    mean: torch.Tensor
    cov_inv: torch.Tensor
    feat_h: int
    feat_w: int
    image_size: int
    layer_names: list[str]


def _pool_cov_spatial(cov, h, w, chunk=64):
    """Apply the main protocol's valid-neighbor 3×3 covariance average."""
    channels = cov.shape[-1]
    cov_hw = cov.view(h, w, channels, channels)
    pooled = torch.empty_like(cov_hw)
    for start in range(0, channels, chunk):
        stop = min(start + chunk, channels)
        width = stop - start
        block = cov_hw[:, :, start:stop, :].permute(2, 3, 0, 1).reshape(1, width * channels, h, w)
        block = F.avg_pool2d(block.contiguous(), kernel_size=3, stride=1,
                             padding=1, count_include_pad=False)
        pooled[:, :, start:stop, :] = block.view(width, channels, h, w).permute(2, 3, 0, 1)
    return pooled.view(h * w, channels, channels)


class PrismHead:
    def __init__(self, extractor, layer_names, image_size=224, device="cuda", cov_dtype="float32"):
        if cov_dtype not in ("float32", "float64"):
            raise ValueError("Covariance dtype must be float32 or float64")
        self.extractor = extractor.to(device)
        self.layer_names = list(layer_names)
        self.image_size = image_size
        self.device = device
        self.cov_dtype = torch.float64 if cov_dtype == "float64" else torch.float32
        self.state: PrismState | None = None

    @torch.no_grad()
    def _embed(self, images):
        return concat_features(self.extractor(images.to(self.device)), self.layer_names)

    @torch.no_grad()
    def fit(self, train_loader):
        self.state = None
        iterator = iter(train_loader)
        first = next(iterator, None)
        if first is None:
            raise ValueError("Cannot fit PrismHead on an empty training loader")
        full = self._embed(first["image"])
        _, channels, h, w = full.shape
        positions = h * w
        sum_x = torch.zeros(positions, channels, dtype=self.cov_dtype, device=self.device)
        sum_xx = torch.zeros(positions, channels, channels, dtype=self.cov_dtype, device=self.device)
        count = 0

        def accumulate(features):
            nonlocal count
            batch = features.shape[0]
            values = features.permute(2, 3, 0, 1).reshape(positions, batch, channels).to(self.cov_dtype)
            sum_x.add_(values.sum(dim=1))
            sum_xx.add_(torch.einsum("pbc,pbd->pcd", values, values))
            count += batch

        # Continue the same iterator so every training batch contributes once.
        accumulate(full)
        del first, full
        for batch in iterator:
            features = self._embed(batch["image"])
            accumulate(features)
            del features
        mean = sum_x / count
        cov = sum_xx / count - torch.einsum("pc,pd->pcd", mean, mean)
        del sum_xx
        cov = _pool_cov_spatial(cov, h, w)
        eye = torch.eye(channels, dtype=self.cov_dtype, device=self.device).expand_as(cov)
        cov = cov + COV_RIDGE * eye
        inverse = torch.linalg.inv(cov)
        self.state = PrismState(mean.to(torch.float32), inverse.to(torch.float32),
                                h, w, self.image_size, list(self.layer_names))

    @torch.no_grad()
    def score(self, images):
        if self.state is None:
            raise RuntimeError("PrismHead.fit must be called before score().")
        features = self._embed(images).to(self.state.mean.dtype)
        batch, channels, h, w = features.shape
        values = features.permute(2, 3, 0, 1).reshape(h * w, batch, channels)
        difference = values - self.state.mean.to(values.device).unsqueeze(1)
        inverse = self.state.cov_inv.to(values.device)
        projected = torch.einsum("pbc,pcd->pbd", difference, inverse)
        distance = (projected * difference).sum(dim=-1)
        distance = distance.transpose(0, 1).reshape(batch, 1, h, w).clamp_min(0.0).sqrt()
        maps = F.interpolate(distance, size=(self.image_size, self.image_size),
                             mode="bilinear", align_corners=False)
        return _gaussian_blur(maps).squeeze(1)


def _gaussian_blur(values, sigma=4.0):
    size = max(3, int(4 * sigma + 1) | 1)
    coordinates = torch.arange(size, device=values.device, dtype=values.dtype) - size // 2
    kernel = torch.exp(-(coordinates ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    weights = (kernel[:, None] * kernel[None, :]).expand(values.shape[1], 1, size, size)
    return F.conv2d(values, weights, padding=size // 2, groups=values.shape[1])
