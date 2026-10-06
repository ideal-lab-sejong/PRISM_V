"""The fixed PRISM-V main protocol for MVTec AD and VisA."""
from __future__ import annotations

SEEDS = [1, 2, 3, 4, 5]
PIPELINE = "prism_v"
DATASETS = {
    "mvtec": ["bottle", "cable", "capsule", "carpet", "grid", "hazelnut",
              "leather", "metal_nut", "pill", "screw", "tile", "toothbrush",
              "transistor", "wood", "zipper"],
    "visa": ["candle", "capsules", "cashew", "chewinggum", "fryum", "macaroni1",
             "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum"],
}
MAIN_CONFIG = {
    "image_size": 224, "resize_size": 256, "cnn_norm": "instance",
    "cnn_downsample": "blur", "cnn_blur_kernel": "bin7",
    "cnn_channels": [128] * 5, "cnn_activation": "leaky_relu",
    "cnn_negative_slope": 0.01, "cnn_layers": ["stage2", "stage3", "stage4", "stage5"],
    "prism_cov_pool": "r1",
}
CNN_KERNEL = (3, 3, 3, 1, 1)
COV_RIDGE = 0.01
