"""The four evaluation metrics for the fixed PRISM-V main experiment."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import label
from sklearn.metrics import auc, average_precision_score, roc_auc_score


def pixel_auroc(masks: np.ndarray, scores: np.ndarray) -> float:
    y = masks.reshape(-1).astype(np.uint8)
    s = scores.reshape(-1).astype(np.float64)
    if y.max() == 0:
        return float("nan")
    return float(roc_auc_score(y, s))


def image_auroc_tvar(labels: np.ndarray, scores: np.ndarray,
                     rho: float = 0.01) -> float:
    y = labels.astype(np.uint8)
    if y.min() == y.max():
        return float("nan")
    flat = scores.reshape(scores.shape[0], -1).astype(np.float64)
    n_pix = flat.shape[1]
    k = max(1, int(round(rho * n_pix)))
    topk = np.partition(flat, n_pix - k, axis=1)[:, n_pix - k:]
    s = topk.mean(axis=1)
    return float(roc_auc_score(y, s))


def image_auprc_tvar(labels: np.ndarray, scores: np.ndarray,
                     rho: float = 0.01) -> float:
    y = labels.astype(np.uint8)
    if y.min() == y.max():
        return float("nan")
    flat = scores.reshape(scores.shape[0], -1).astype(np.float64)
    n_pix = flat.shape[1]
    k = max(1, int(round(rho * n_pix)))
    topk = np.partition(flat, n_pix - k, axis=1)[:, n_pix - k:]
    s = topk.mean(axis=1)
    return float(average_precision_score(y, s))


def aupro(masks: np.ndarray, scores: np.ndarray, fpr_limit: float = 0.3) -> float:
    masks = masks.astype(np.uint8)
    s = scores.astype(np.float64).ravel()
    if masks.max() == 0:
        return float("nan")

    fp_w = (masks == 0).astype(np.float64).ravel()
    n_normal = fp_w.sum()
    if n_normal == 0:
        return float("nan")

    pro_w = np.zeros(masks.size, dtype=np.float64)
    num_regions = 0
    per_img = masks.shape[1] * masks.shape[2]
    for i, m in enumerate(masks):
        if m.max() == 0:
            continue
        lbl, n = label(m)
        num_regions += n
        flat = lbl.ravel()
        areas = np.bincount(flat)
        nz = flat > 0
        w = np.zeros(per_img, dtype=np.float64)
        w[nz] = 1.0 / areas[flat[nz]]
        pro_w[i * per_img:(i + 1) * per_img] = w

    order = np.argsort(s)[::-1]
    s_sorted = s[order]
    fprs = np.cumsum(fp_w[order]) / n_normal
    pros = np.cumsum(pro_w[order]) / num_regions
    keep = np.append(np.diff(s_sorted) != 0, True)
    fprs = np.concatenate(([0.0], fprs[keep]))
    pros = np.concatenate(([0.0], pros[keep]))

    idx = np.searchsorted(fprs, fpr_limit, side="right")
    if idx >= len(fprs):
        x, y = fprs, pros
    else:
        x0, x1 = fprs[idx - 1], fprs[idx]
        y0, y1 = pros[idx - 1], pros[idx]
        y_lim = y0 if x1 <= x0 else y0 + (y1 - y0) * (fpr_limit - x0) / (x1 - x0)
        x = np.concatenate((fprs[:idx], [fpr_limit]))
        y = np.concatenate((pros[:idx], [y_lim]))
    area = auc(x, y)
    return float(area / fpr_limit)
