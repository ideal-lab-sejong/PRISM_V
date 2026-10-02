# Data

The datasets are not distributed with this package. Both are public
benchmarks; download them and place them here:

```
data/
├── mvtec/     MVTec AD  (15 categories)
└── visa/      VisA, official 1-class split (12 categories)
```

Nothing else needs to be configured: `scripts/run_prism_v.py` takes the root
as `--root` and auto-detects the categories under it.

## "raw" vs. the prepared roots

Each benchmark ships in its own vendor format (**raw**), while this package
reads a single directory convention:

```
<root>/images/train/<category>/good/<stem>.png     # normal training images
<root>/images/test/<category>/<defect>/<stem>.png  # test images ("good" = normal)
<root>/masks/test/<category>/<defect>/<stem>_mask.png
```

| | raw distribution | how it reaches the prepared root |
|---|---|---|
| MVTec AD | official archive `mvtec_anomaly_detection.tar.xz`: `<category>/train/good/`, `<category>/test/<defect>/`, `<category>/ground_truth/<defect>/*_mask.png`, plus `readme.txt` / `license.txt` | `datasets/dataset_prepare_mvtec.py` moves each image into the convention above; bytes are copied verbatim and the two text files are skipped |
| VisA | `VisA_20220922.tar`: `<category>/Data/Images/{Normal,Anomaly}/*.JPG`, `<category>/Data/Masks/Anomaly/*.png`, and the split defined **only** in `split_csv/1cls.csv` | `datasets/dataset_prepare_visa.py` rewrites it into the convention above |

The two conversions differ in cost. MVTec's is pure relocation — no pixel
touches it, and with `--src` it hard-links rather than copies, so it is
instant and free. VisA's has to re-encode JPG → PNG, which is why it takes
tens of minutes and inflates a 1.93 GB tar to ~15 GB.

`datasets/loader.py` reads only the convention above — both roots look the
same to it, and an unprepared root fails in `detect_categories` before any
computation starts.

## MVTec AD

Download `mvtec_anomaly_detection.tar.xz` from the official MVTec site
(<https://www.mvtec.com/company/research/datasets/mvtec-ad>, CC BY-NC-SA
4.0, licence acceptance required), then convert:

```bash
python datasets/dataset_prepare_mvtec.py --tar mvtec_anomaly_detection.tar.xz \
    --out data/mvtec
```

The download itself has no command form: the licence form issues a
one-off link rather than a fixed URL, so it is a browser step.

The script streams the archive and writes each of the 6612 images straight
to its destination — bytes copied verbatim, never decoded and re-encoded.
The archive's `readme.txt` / `license.txt` (at the root and in every
category) are not images and are skipped; they stay in the archive, which
is where the attribution and licence notice belong, so keep the archive.

Already extracted it? Point `--src` at that root and the script hard-links
instead of copying — instant, no extra disk:

```bash
python datasets/dataset_prepare_mvtec.py --src <extracted-root> --out data/mvtec
```

## VisA (1-class split)

Download `VisA_20220922.tar` from the official VisA release, then convert:

```bash
python datasets/dataset_prepare_visa.py \
  --tar /path/to/VisA_20220922.tar \
  --extract-dir /tmp/visa_extract \
  --out data/visa
```

The script reads `split_csv/1cls.csv` (columns `object,split,label,image,mask`)
and routes every row to its destination: `split=train` to
`images/train/<cat>/good/`, `split=test` to `images/test/<cat>/good/` or
`.../bad/` by label, with anomaly masks to `masks/test/<cat>/bad/`. Two
format conversions happen while copying, both required by the loader's
conventions and neither of which alters pixel values:

- images are re-encoded JPG → PNG (lossless) so the loader's `*.png` glob applies;
- masks are binarized (`> 0` → 255) and renamed to `<stem>_mask.png`.

Image resolution is **not** changed — VisA images stay at their native size
(e.g. 1284×1168), as do MVTec's (typically 900×900). When the script
finishes it re-reads the output with the loader and fails loudly unless all
12 categories are present.

## Where resizing and cropping happen

Nowhere on disk. Both roots hold **full-resolution** images; the input
pipeline is applied in memory every time an image is read, in
`datasets/loader.py`:

| | images (`_build_transform`) | masks (`_build_mask_transform`) |
|---|---|---|
| 1 | `Resize(256)`, bilinear | `Resize(256)`, nearest |
| 2 | `CenterCrop(224)` | `CenterCrop(224)` |
| 3 | `ToTensor()` | binarize `> 0` → {0,1} float |
| 4 | `Normalize(ImageNet mean/std)` | — |

Masks use nearest-neighbour interpolation and are re-binarized after
cropping so that resizing never introduces intermediate label values.

The two sizes come from `scripts/run_prism_v.py` (`--resize-size 256`,
`--image-size 224`), whose defaults are the adopted protocol; they are passed
straight to the transforms above. Both benchmarks therefore go through an
identical pipeline, and the only difference between an MVTec run and a VisA
run is the `--root` directory.
