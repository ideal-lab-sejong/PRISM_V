# PRISM-V

How Far Can Anomaly Detection
Go Before Learning?
Random Neural Features from Tables to Pixels

![Architecture](./images/PRISM_Framework.png)

Authors: Hyojun Yi, Geunyoung Kim, Hogeon Seo, and Seunghyoung Ryu

paper link: 


## Introduction

This repository contains the implementation of PRISM-V, the visual anomaly detection module of PRISM. PRISM-V is designed to investigate how effectively a convolutional neural network with randomly initialized weights can perform both image-level anomaly detection and pixel-level localization of anomalous regions.

## Data

Datasets and results are not shipped — you create them: `data/mvtec`, `data/visa` and `results/` .

Each benchmark ships in its own vendor format, and a `datasets/` script
converts it into the one layout both runs read:

```
<root>/images/train/<category>/good/<stem>.png     # normal training images
<root>/images/test/<category>/<defect>/<stem>.png  # test images ("good" = normal)
<root>/masks/test/<category>/<defect>/<stem>_mask.png
```



###  MVTec AD

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


### VisA 

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


```bash
bash scripts/run_main.sh     mvtec    
bash scripts/run_main.sh     visa
```



## Repository layout

```
models/      prism_v.py (the PRISM-V encoder), padim_head.py (PaDiM
             Gaussian head)
evaluation/  metrics.py (AUROC/AUPRC, AUPRO)
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


## Acknoledgement

This work was partially supported in part by the IITP (Institute of Information & Communications Technology Planning & Evaluation) - ICAN (ICT Challenge and Advanced Network of HRD) grant funded by the Korea government (Ministry of Science and ICT) (IITP-2026-RS-2024-00436528), and in part by the National Research Foundation of Korea(NRF) grant funded by the Korea government(MSIT) (RS-2024-00454514)


## Citation
