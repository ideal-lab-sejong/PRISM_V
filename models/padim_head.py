from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


def concat_features(feature_dict: dict[str, torch.Tensor], layer_names: list[str]) -> torch.Tensor:
    selected = [feature_dict[n] for n in layer_names]
    target_h = max(t.shape[-2] for t in selected)
    target_w = max(t.shape[-1] for t in selected)
    out = []
    for t in selected:
        if t.shape[-2:] != (target_h, target_w):
            t = F.interpolate(t, size=(target_h, target_w), mode="bilinear",
                              align_corners=False)
        out.append(t)
    return torch.cat(out, dim=1)


@dataclass
class PaDiMState:
    mean: torch.Tensor
    cov_inv: torch.Tensor
    feat_h: int
    feat_w: int
    image_size: int
    channel_subset: torch.Tensor
    layer_names: list[str]


def _pool_cov_spatial(cov: torch.Tensor, h: int, w: int, radius: int, chunk: int = 64) -> torch.Tensor:
    c = cov.shape[-1]
    k = 2 * radius + 1
    cov_hw = cov.view(h, w, c, c)
    out = torch.empty_like(cov_hw)
    for i0 in range(0, c, chunk):
        i1 = min(i0 + chunk, c)
        nblk = i1 - i0
        x = cov_hw[:, :, i0:i1, :].permute(2, 3, 0, 1).reshape(1, nblk * c, h, w)
        x = F.avg_pool2d(x.contiguous(), kernel_size=k, stride=1,
                         padding=radius, count_include_pad=False)
        out[:, :, i0:i1, :] = x.view(nblk, c, h, w).permute(2, 3, 0, 1)
    return out.view(h * w, c, c)


class PaDiM:
    def __init__(self, extractor: torch.nn.Module, layer_names: list[str],
                 image_size: int = 256, device: str = "cuda",
                 cov_dtype: str = "float64", cov_pool_radius: int = 0,
                 cov_ridge: float = 0.01):
        self.extractor = extractor.to(device)
        self.layer_names = layer_names
        self.image_size = image_size
        self.device = device
        self.cov_dtype = torch.float64 if cov_dtype == "float64" else torch.float32
        self.cov_pool_radius = cov_pool_radius
        self.cov_ridge = cov_ridge
        self.state: PaDiMState | None = None

    @torch.no_grad()
    def _embed(self, image: torch.Tensor) -> torch.Tensor:
        feats = self.extractor(image.to(self.device))
        x = concat_features(feats, self.layer_names)
        if self.state is not None:
            idx = self.state.channel_subset.to(x.device)
            x = x.index_select(dim=1, index=idx)
        return x

    @torch.no_grad()
    def fit(self, train_loader) -> None:
        first = next(iter(train_loader))
        probe = self.extractor(first["image"].to(self.device))
        full = concat_features(probe, self.layer_names)
        _, c_full, h, w = full.shape
        subset_idx = torch.arange(c_full, device=self.device)
        c = subset_idx.numel()
        n_pix = h * w

        acc_dtype = self.cov_dtype
        sum_x = torch.zeros(n_pix, c, dtype=acc_dtype, device=self.device)
        sum_xx = torch.zeros(n_pix, c, c, dtype=acc_dtype, device=self.device)
        n_samples = 0

        def accumulate(batch_imgs: torch.Tensor) -> None:
            nonlocal n_samples
            feats = self.extractor(batch_imgs.to(self.device))
            x = concat_features(feats, self.layer_names).index_select(1, subset_idx)
            # x: (B, C, H, W) -> (H*W, B, C)
            b = x.shape[0]
            x = x.permute(2, 3, 0, 1).reshape(n_pix, b, c).to(acc_dtype)
            sum_x.add_(x.sum(dim=1))
            sum_xx.add_(torch.einsum("pbc,pbd->pcd", x, x))
            n_samples += b

        accumulate(first["image"])
        for batch in train_loader:
            if batch is first:
                continue
            accumulate(batch["image"])

        mean = sum_x / n_samples                                           
        cov = sum_xx / n_samples - torch.einsum("pc,pd->pcd", mean, mean)   
        del sum_xx                                      
        if self.cov_pool_radius != 0:
            cov = _pool_cov_spatial(cov, h, w, self.cov_pool_radius)
        eye = torch.eye(c, dtype=acc_dtype, device=self.device).expand_as(cov)
        cov = cov + self.cov_ridge * eye
        cov_inv = torch.linalg.inv(cov)

        self.state = PaDiMState(
            mean=mean.float(),
            cov_inv=cov_inv.float(),
            feat_h=h,
            feat_w=w,
            image_size=self.image_size,
            channel_subset=subset_idx.cpu(),
            layer_names=list(self.layer_names),
        )

    @torch.no_grad()
    def score(self, image_batch: torch.Tensor) -> torch.Tensor:
        if self.state is None:
            raise RuntimeError("PaDiM.fit must be called before score().")
        x = self._embed(image_batch)                                        # (B, C, H, W)
        b, c, h, w = x.shape
        n_pix = h * w
        x = x.permute(2, 3, 0, 1).reshape(n_pix, b, c)                      # (P, B, C)
        diff = x - self.state.mean.to(x.device).unsqueeze(1)                # (P, B, C)
        # m^2 = diff^T  Σ^-1  diff  per pixel
        cov_inv = self.state.cov_inv.to(x.device)                           # (P, C, C)
        # (P,B,C) @ (P,C,C) -> (P,B,C); then sum * diff
        tmp = torch.einsum("pbc,pcd->pbd", diff, cov_inv)
        m2 = (tmp * diff).sum(dim=-1)                                       # (P, B)
        m2 = m2.transpose(0, 1).reshape(b, 1, h, w).clamp_min(0.0).sqrt()
        amap = F.interpolate(m2, size=(self.image_size, self.image_size),
                             mode="bilinear", align_corners=False)
        # Mild Gaussian smoothing reduces noise — standard PaDiM postproc.
        amap = _gaussian_blur(amap, sigma=4.0)
        return amap.squeeze(1)


def _gaussian_blur(x: torch.Tensor, sigma: float = 4.0) -> torch.Tensor:
    k = max(3, int(4 * sigma + 1) | 1)
    half = k // 2
    coords = torch.arange(k, device=x.device, dtype=x.dtype) - half
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel_2d = (g[:, None] * g[None, :]).expand(x.shape[1], 1, k, k)
    return F.conv2d(x, kernel_2d, padding=half, groups=x.shape[1])
