# PRISM-V — official implementation

Code to reproduce the PRISM-V results of the paper: the main experiment,
the component analysis and the ablation studies.


## 1. Data

Datasets and results are not shipped — you create them: `data/mvtec`
(~5 GB), `data/visa` (~15 GB) and `results/` (~21 GB for the full
reproduction).

Each benchmark ships in its own vendor format, and a `datasets/` script
converts it into the one layout both runs read:

```
<root>/images/train/<category>/good/<stem>.png     # normal training images
<root>/images/test/<category>/<defect>/<stem>.png  # test images ("good" = normal)
<root>/masks/test/<category>/<defect>/<stem>_mask.png
```



### 1.1 MVTec AD → `data/mvtec/`

No download command for this one: the official page
(<https://www.mvtec.com/company/research/datasets/mvtec-ad>) serves
`mvtec_anomaly_detection.tar.xz` only behind a licence-acceptance form, and
the link it issues is not a fixed URL — so this step is a browser step.
Accept the licence (CC BY-NC-SA 4.0: research / non-commercial,
attribution, share-alike; cite Bergmann et al., CVPR 2019) and download the
archive, then convert it:

```bash
python datasets/dataset_prepare_mvtec.py --tar mvtec_anomaly_detection.tar.xz \
    --out data/mvtec
```


### 1.2 VisA (1-class split) → `data/visa/`

VisA needs a conversion step, because its train/test split exists only as CSV rows (`split_csv/1cls.csv`) 

```bash
curl -L -O https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar
python datasets/dataset_prepare_visa.py --tar VisA_20220922.tar \
    --extract-dir /tmp/visa_extract --out data/visa
```


## 2. Running the experiments

None of the drivers below download or convert anything, 
so `data/mvtec` and `data/visa` must already exist; a
missing or unprepared root aborts in `detect_categories` before any
computation starts.

Run them **one at a time** 

```bash
bash scripts/run_main.sh     mvtec    
bash scripts/run_main.sh     visa
bash scripts/run_ablation.sh all mvtec
bash scripts/run_ablation.sh all visa
python scripts/aggregate_ablation.py  
```

Or run everything with one command — it chains exactly the five steps above in the same order:

```bash
bash scripts/run_all.sh
```

`run_all.sh` is a plain sequential driver: no scheduling, no retry, no
resume logic of its own. Everything below applies equally whether you run
it or the five commands by hand.

### 2.1 Runtime

A full reproduction takes about **19 h** on the reference GPU (single
GB10). The main experiment and the six ablations come to 34 arm × dataset
runs: 32 seeded ones at roughly 2 min per category for all 5 seeds — so
~33 min per MVTec arm (15 categories) and ~37 min per VisA arm (12
categories) — plus the two deterministic Raw-Mahalanobis runs of the
component analysis, ~2 min each.

### 2.2 Resuming after an interruption

Just re-run the same command. An arm that already finished is skipped and
the run picks up where it stopped — the check is simply whether that arm's
`<out>/tables/seeds.json` exists.

One thing to clean up first. The arm that was still running when the job
died also left a `seeds.json` behind, holding only the categories it got
through. That makes it look finished, so it gets skipped too and stays
half-done. Delete that one file before re-running:

```bash
grep -L COMPLETE results/main/master_*.log results/ablation/*/master_*.log
rm results/ablation/<name>/<arm>/<ds>/tables/seeds.json
```

The `grep` lists the groups that were cut off; in that log, the arm to
delete is the last `START` with no `DONE`.


## 3. Results

```
<out>/tables/seeds.json                       per-category metrics: per_seed + mean/std
<out>/scores/<pipeline>/<cat>_seed<N>.npz     maps (float16), masks, labels
<out>/logs/run_prism_v.log                    arguments + per-category timing
```

`<pipeline>` is `random_cnn_padim` — the internal identifier for PRISM-V,
kept as-is in the `--pipelines` flag, the `scores/` directory names, the
`seeds.json` record keys, and the logs. It denotes the same method the
paper and the comparison tables call PRISM-V.

Comparison tables land at `results/ablation/<name>/comparison.md`: one
summary table per dataset (img-AUROC, img-AUPRC, pix-AUROC, AUPRO) plus a
per-category AUPRO breakdown, cells as 5-seed mean±std with Δ against the
arm marked `(main)` or `(base)` in each table's first column.

The shell drivers keep a master log next to the run
(`results/main/master_<ds>.log`,
`results/ablation/<name>/master_<ds>.log`) with a `START` / `DONE rc=` /
`SKIP` line per arm — check it first if a table row prints `-`, which means
that arm's `seeds.json` is missing.

Since the raw anomaly maps are saved, a new metric is a recompute rather
than a re-run (`evaluation/score_io.py` loads an npz;
`scripts/compute_g.py` is an example that writes the G-metric sidecar).


## 4. Repository layout

```
models/      prism_v.py (the PRISM-V encoder), padim_head.py (PaDiM
             Gaussian head), raw_patches.py (the component-analysis
             Raw-Mahalanobis reference extractor)
evaluation/  metrics.py (AUROC/AUPRC, AUPRO, G metrics),
             score_io.py (raw-score persistence)
datasets/    loader.py — one loader for both datasets;
             dataset_prepare_mvtec.py and dataset_prepare_visa.py convert
             the official releases into the layout it reads
scripts/     run_prism_v.py (the single experiment engine; defaults = the
             adopted main config), run_main.sh (main experiment),
             run_ablation.sh (all ablations), run_all.sh (both, in order),
             aggregate_ablation.py (all comparison tables),
             compute_g.py (G-metric sidecar)
data/        create it — datasets are not distributed
results/     created by the run scripts: main/ and ablation/<name>/
```


## 5. Acknoledgement

This work was partially supported in part by the IITP (Institute of Information & Communications Technology Planning & Evaluation) - ICAN (ICT Challenge and Advanced Network of HRD) grant funded by the Korea government (Ministry of Science and ICT) (IITP-2026-RS-2024-00436528), and in part by the National Research Foundation of Korea(NRF) grant funded by the Korea government(MSIT) (RS-2024-00454514)
