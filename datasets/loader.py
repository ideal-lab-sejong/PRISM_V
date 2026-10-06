from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class Sample:
    image: torch.Tensor
    mask: torch.Tensor
    label: int
    defect_type: str
    path: str


def _build_transform(image_size: int, resize: int | None = None):
    return transforms.Compose([
        transforms.Resize(resize or image_size,
                          interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def _build_mask_transform(image_size: int, resize: int | None = None):
    return transforms.Compose([
        transforms.Resize(resize or image_size,
                          interpolation=transforms.InterpolationMode.NEAREST),
        transforms.CenterCrop(image_size),
    ])


class CategoryDataset(Dataset):
    def __init__(self, root: str | Path, category: str, split: str,
                 image_size: int = 224, resize: int | None = None):
        if split not in {"train", "test"}:
            raise ValueError(f"split must be train|test, got {split}")
        self.root = Path(root)
        self.category = category
        self.split = split
        self.image_size = image_size
        self.tx_image = _build_transform(image_size, resize)
        self.tx_mask = _build_mask_transform(image_size, resize)
        self.samples: list[tuple[Path, Path | None, int, str]] = []
        self._scan()
        if not self.samples:
            raise FileNotFoundError(
                f"No samples found for {category}/{split} under {self.root}"
            )

    def _category_root(self, kind: str) -> Path:
        return {
            "train": self.root / "images" / "train" / self.category,
            "test": self.root / "images" / "test" / self.category,
            "mask": self.root / "masks" / "test" / self.category,
        }[kind]

    def _scan(self) -> None:
        if self.split == "train":
            good_dir = self._category_root("train") / "good"
            for p in sorted(good_dir.glob("*.png")):
                self.samples.append((p, None, 0, "good"))
            return
        test_root = self._category_root("test")
        gt_root = self._category_root("mask")
        for defect_dir in sorted(test_root.iterdir()):
            if not defect_dir.is_dir():
                continue
            defect = defect_dir.name
            for p in sorted(defect_dir.glob("*.png")):
                if defect == "good":
                    self.samples.append((p, None, 0, defect))
                else:
                    mask_path = gt_root / defect / f"{p.stem}_mask.png"
                    if not mask_path.is_file():
                        raise FileNotFoundError(f"Missing anomaly mask: {mask_path}")
                    self.samples.append((p, mask_path, 1, defect))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Sample:
        img_path, mask_path, label, defect = self.samples[index]
        with Image.open(img_path) as im:
            image = self.tx_image(im.convert("RGB"))
        if mask_path is not None:
            with Image.open(mask_path) as mk:
                mask_pil = self.tx_mask(mk.convert("L"))
                mask_np = np.array(mask_pil, dtype=np.uint8)
            mask = torch.from_numpy((mask_np > 0).astype(np.float32))[None]
        else:
            mask = torch.zeros(1, self.image_size, self.image_size, dtype=torch.float32)
        return Sample(
            image=image,
            mask=mask,
            label=label,
            defect_type=defect,
            path=str(img_path),
        )


def collate(batch: list[Sample]) -> dict:
    return {
        "image": torch.stack([b.image for b in batch], dim=0),
        "mask": torch.stack([b.mask for b in batch], dim=0),
        "label": torch.tensor([b.label for b in batch], dtype=torch.long),
        "defect_type": [b.defect_type for b in batch],
        "path": [b.path for b in batch],
    }
