"""Usage::

    python scripts/aggregate_ablation.py                 # all six tables
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PKG_ROOT = Path(__file__).resolve().parents[1]
PIPE = "random_cnn_padim"
SEEDS = [1, 2, 3, 4, 5]
METRICS = [("pixel_auroc", "pix-AUROC"), ("aupro", "AUPRO"),
           ("image_auroc_tvar", "img-AUROC"), ("image_auprc_tvar", "img-AUPRC")]
DATASETS = [("MVTec AD", "mvtec"), ("VisA", "visa")]



def load(res: Path, arm_path: str, ds: str):
    p = res / arm_path / ds / "tables" / "seeds.json"
    return json.loads(p.read_text()) if p.exists() else None


def seed_vals_any(rec, metric):
    """Per-seed values of one category record, or None (any seeds present)."""
    if PIPE not in rec:
        return None
    v = [float(m[metric]) for m in rec[PIPE].get("per_seed", [])
         if "error" not in m and metric in m]
    return v or None


def seed_vals_strict(rec, metric):
    """Per-seed values (SEEDS order) of one category record, or None."""
    if PIPE not in rec:
        return None
    by_seed = {m.get("seed"): m for m in rec[PIPE].get("per_seed", [])
               if "error" not in m}
    if not all(s in by_seed for s in SEEDS):
        return None
    return [float(by_seed[s][metric]) for s in SEEDS]


def cat_mean_std(R, metric, seed_vals):
    """Dataset mean/std: per-category 5-seed mean -> mean over categories;
    std = mean over categories of the per-category seed std."""
    means, stds = [], []
    for r in R:
        v = seed_vals(r, metric)
        if v is None:
            continue
        means.append(np.mean(v))
        stds.append(np.std(v))
    if not means:
        return float("nan"), float("nan")
    return float(np.mean(means)) * 100, float(np.mean(stds)) * 100


def per_cat(R, metric, seed_vals):
    """{category: (mean, std)} over seeds, × 100."""
    out = {}
    for r in R:
        v = seed_vals(r, metric)
        out[r["category"]] = ((float(np.mean(v)) * 100, float(np.std(v)) * 100)
                              if v is not None else (float("nan"), float("nan")))
    return out


def fmt(x):
    return "-" if x != x else f"{x:.1f}"


def fmt_ms(m, s):
    return "-" if m != m else f"{m:.1f}±{s:.1f}"


def fmt_delta(x, base):
    if x != x or base != base:
        return "-"
    return f"{x - base:+.1f}"


def per_cat_aupro_section(md, ds_label, present, seed_vals):
    """Append the per-category AUPRO table across the present arms."""
    md.append(f"### {ds_label} — per-category AUPRO (5-seed mean±std)\n")
    md.append("| Category | " + " | ".join(l for l, _ in present) + " |")
    md.append("|---|" + "---:|" * len(present))
    pcs = [per_cat(R, "aupro", seed_vals) for _, R in present]
    for c in pcs[0]:
        md.append(f"| {c} | "
                  + " | ".join(fmt_ms(*pc.get(c, (float('nan'),) * 2))
                               for pc in pcs)
                  + " |")
    md.append("| **avg** | "
              + " | ".join(fmt_ms(np.nanmean([m for m, _ in pc.values()]),
                                  np.nanmean([s for _, s in pc.values()]))
                           for pc in pcs)
              + " |")
    md.append("")


def render_first_arm_base(res: Path, head: list[str], arms_spec, col_label: str):
    """Delta vs the FIRST arm; arms lacking the fullest category set are
    dropped (blur_kernel / cov_pool convention)."""
    md = list(head)
    for ds_label, ds in DATASETS:
        arms = [(lbl, load(res, path, ds)) for lbl, path in arms_spec]
        ref_n = max((len(R) for _, R in arms if R is not None), default=0)
        arms = [(lbl, R if (R is not None and len(R) >= ref_n) else None)
                for lbl, R in arms]
        md.append(f"## {ds_label}\n")
        md.append(f"| {col_label} | " + " | ".join(l for _, l in METRICS) + " |")
        md.append("|---|" + "---:|" * len(METRICS))
        base = None
        for i, (lbl, R) in enumerate(arms):
            if R is None:
                md.append(f"| {lbl} | " + " | ".join("-" for _ in METRICS) + " |")
                continue
            cells = [cat_mean_std(R, m, seed_vals_any) for m, _ in METRICS]
            if i == 0:
                base = [c for c, _ in cells]
                md.append(f"| {lbl} | "
                          + " | ".join(fmt_ms(c, s) for c, s in cells) + " |")
            else:
                md.append(f"| {lbl} | "
                          + " | ".join(f"{fmt_ms(c, s)} ({fmt_delta(c, b)})"
                                       for (c, s), b in zip(cells, base)) + " |")
        md.append("")

        present = [(lbl, R) for lbl, R in arms if R is not None]
        if present:
            per_cat_aupro_section(md, ds_label, present, seed_vals_any)
    return md


def render_ref_arm_base(res: Path, head: list[str], arms_spec):
    """Delta vs the arm flagged as reference (building_block / hidden_dim /
    activation_fn convention; strict 5-seed check)."""
    md = list(head)
    for ds_label, ds in DATASETS:
        arms = [(lbl, load(res, path, ds), is_ref)
                for lbl, path, is_ref in arms_spec]
        ref_R = next((R for _, R, is_ref in arms if is_ref), None)
        base = ([cat_mean_std(ref_R, m, seed_vals_strict)[0] for m, _ in METRICS]
                if ref_R is not None else [float("nan")] * len(METRICS))

        md.append(f"## {ds_label}\n")
        md.append("| Model | " + " | ".join(l for _, l in METRICS) + " |")
        md.append("|---|" + "---:|" * len(METRICS))
        for lbl, R, is_ref in arms:
            if R is None:
                md.append(f"| {lbl} | " + " | ".join("-" for _ in METRICS) + " |")
                continue
            cells = [cat_mean_std(R, m, seed_vals_strict) for m, _ in METRICS]
            row = []
            for (c, s), b in zip(cells, base):
                cell = f"{fmt(c)}±{fmt(s)}"
                if not is_ref and c == c and b == b:
                    cell += f" ({c - b:+.1f})"
                row.append(cell)
            md.append(f"| {lbl} | " + " | ".join(row) + " |")
        md.append("")

        present = [(lbl, R) for lbl, R, _ in arms if R is not None]
        if present:
            per_cat_aupro_section(md, ds_label, present, seed_vals_strict)
    return md


# --------------------------------------------------------------------------
# Renderers, one per ablation.
# --------------------------------------------------------------------------

def render_building_block(res: Path) -> list[str]:
    head = ["# Ablation — building blocks\n"]
    arms = [
        ("PRISM-V (main)", "main", True),
        ("w/o BlurPool", "ablation/building_block/no_blur", False),
        ("w/o InstanceNorm", "ablation/building_block/no_in", False),
        ("BASE", "ablation/building_block/base", False),
    ]
    return render_ref_arm_base(res, head, arms)


def render_blur_kernel(res: Path) -> list[str]:
    head = ["# Ablation — BlurPool binomial kernel width\n"]
    arms = [
        ("tri3 [1,2,1]", "ablation/blur_kernel/tri3"),
        ("bin5 [1,4,6,4,1]", "ablation/blur_kernel/bin5"),
        ("bin7 (main)", "main"),
        ("bin9 [1,8,...,8,1]", "ablation/blur_kernel/bin9"),
    ]
    return render_first_arm_base(res, head, arms, "Kernel")


def render_cov_pool(res: Path) -> list[str]:
    head = ["# Ablation — PaDiM covariance spatial pooling\n"]
    arms = [
        ("r0 (none, base)", "ablation/cov_pool/covnone"),
        ("r1 (3x3, main)", "main"),
        ("r2 (5x5)", "ablation/cov_pool/covr2"),
    ]
    return render_first_arm_base(res, head, arms, "Model")


def render_hidden_dim(res: Path) -> list[str]:
    head = ["# Ablation — hidden size / per-stage channel width\n"]
    arms = [
        ("d32 (fused 128)", "ablation/hidden_dim/d32", False),
        ("d64 (fused 256)", "ablation/hidden_dim/d64", False),
        ("d128 (fused 512, main)", "main", True),
        ("d192 (fused 768)", "ablation/hidden_dim/d192", False),
    ]
    return render_ref_arm_base(res, head, arms)


def render_activation_fn(res: Path) -> list[str]:
    head = ["# Ablation — conv-block activation function\n"]
    arms = [
        ("LeakyReLU(0.01) (main)", "main", True),
        ("ReLU", "ablation/activation_fn/relu", False),
        ("Tanh", "ablation/activation_fn/tanh", False),
        ("Sigmoid", "ablation/activation_fn/sigmoid", False),
    ]
    return render_ref_arm_base(res, head, arms)



COMPONENT_ROWS = [
    ('Raw Mahal.',      'ablation/component/raw_mahalanobis', 'raw_mahalanobis'),
    ('w/o Act.',        'ablation/component/prism_v_no_act',  'random_cnn_padim'),
    ('**PRISM-V**',     'main',                               'random_cnn_padim'),
]
COMPONENT_COLS = ['img-AUROC', 'img-AUPRC', 'P-AUROC', 'AUPRO',
                  'G_img', 'G_pix']


def _component_per_category(base: Path, pipeline: str, seeds: set[int]):
    cats = {}
    sel_by_cat = {}
    for entry in json.loads((base / 'tables' / 'seeds.json').read_text()):
        per_seed = entry[pipeline]['per_seed']
        sel = [r for r in per_seed if r['seed'] in seeds] or per_seed 
        sel_by_cat[entry['category']] = sel
        cats[entry['category']] = [
            [float(r[k]) for r in sel]
            for k in ('image_auroc_tvar', 'image_auprc_tvar', 'pixel_auroc')]
    aupro_fp = base / 'tables' / 'aupro_official.json'
    if aupro_fp.exists():
        aupro = json.loads(aupro_fp.read_text())['per_cat_seed']
        for c in cats:
            sv = {int(s): float(v) for s, v in aupro[c].items()}
            keep = {s: v for s, v in sv.items() if s in seeds} or sv
            cats[c].append([keep[s] for s in sorted(keep)])
    else:
        for c in cats:
            cats[c].append([float(r['aupro']) for r in sel_by_cat[c]])
    for fname, keys in (('g_metrics.json', ('g_img', 'g_pix')),):
        per_file = json.loads(
            (base / 'scores' / pipeline / fname).read_text())['per_file']
        for key in keys:
            by_cat = {}
            for r in per_file:
                s = r['seed']
                by_cat.setdefault(r['category'], {})[
                    None if s is None else int(s)] = float(r[key])
            for c in cats:
                keep = ({s: v for s, v in by_cat[c].items() if s in seeds}
                        or by_cat[c])
                cats[c].append([keep[s] for s in sorted(keep)])
    return cats


def _component_table(title, rows_cats):
    lines = [f'### {title}', '',
             '| Row | ' + ' | '.join(COMPONENT_COLS) + ' |',
             '|---|' + '---:|' * len(COMPONENT_COLS)]
    for name, cats in rows_cats:
        cells = []
        for i in range(len(COMPONENT_COLS)):
            means = [float(np.mean(v[i])) for v in cats.values()]
            stds = [float(np.std(v[i])) for v in cats.values()]
            scale, prec = (100, 1) if i < 4 else (1, 3)
            cell = f'{float(np.mean(means)) * scale:.{prec}f}'
            if any(len(v[i]) > 1 for v in cats.values()):
                cell += f'±{float(np.mean(stds)) * scale:.{prec}f}'
            cells.append(cell)
        lines.append(f'| {name} | ' + ' | '.join(cells) + ' |')
    lines.append('')
    return lines


def render_component(res: Path) -> list[str]:
    seeds = set(SEEDS)
    data = {name: {ds: _component_per_category(res / rel / ds, pipe, seeds)
                   for ds in ('mvtec', 'visa')}
            for name, rel, pipe in COMPONENT_ROWS}

    md = ['# Component analysis — nonlinearity ablation\n']
    md += _component_table('Pooled — all 27 categories (MVTec 15 + VisA 12)',
                           [(n, {f'{ds}/{c}': v
                                 for ds, cats in d.items()
                                 for c, v in cats.items()})
                            for n, d in data.items()])
    md += _component_table('MVTec AD (15 categories)',
                           [(n, d['mvtec']) for n, d in data.items()])
    md += _component_table('VisA (12 categories)',
                           [(n, d['visa']) for n, d in data.items()])
    return md



# --------------------------------------------------------------------------

ABLATIONS = {
    "component": render_component,
    "building_block": render_building_block,
    "blur_kernel": render_blur_kernel,
    "cov_pool": render_cov_pool,
    "hidden_dim": render_hidden_dim,
    "activation_fn": render_activation_fn,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", metavar="NAME",
                    help="Ablations to aggregate: "
                         f"{', '.join(ABLATIONS)} (default: all).")
    ap.add_argument("--results-root", default=str(PKG_ROOT / "results"),
                    help="Root results directory holding main/ + ablation/.")
    ap.add_argument("--out-dir", default=None,
                    help="Write all tables as <out-dir>/<name>.md instead of "
                         "<results-root>/ablation/<name>/comparison.md.")
    args = ap.parse_args()
    res = Path(args.results_root)
    unknown = [n for n in args.names if n not in ABLATIONS]
    if unknown:
        ap.error(f"unknown ablation(s): {', '.join(unknown)} "
                 f"(choose from {', '.join(ABLATIONS)})")
    names = args.names or list(ABLATIONS)

    for name in names:
        try:
            md = ABLATIONS[name](res)
        except FileNotFoundError as exc:
            print(f"skipping {name}: missing input ({exc.filename})")
            continue
        out = (Path(args.out_dir) / f"{name}.md" if args.out_dir
               else res / "ablation" / name / "comparison.md")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(md), encoding="utf-8")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
