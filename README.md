# How Far Can Anomaly Detection Go Before Learning? Random Neural Features from Tables to Pixels

**PRISM FrameWork**

![Architecture](./images/PRISM_Framework.png)

Authors: Hyojun Yi, Geunyoung Kim, Hogeon Seo, and Seunghyoung Ryu

paper: 


## Introduction

This repository contains the implementation of **PRISM-V**, the visual anomaly detection module of PRISM. PRISM-V is designed to investigate how effectively a convolutional neural network with randomly initialized weights can perform both image-level anomaly detection and pixel-level localization of anomalous regions.


## Data

Download and prepare each dataset as follows.

### MVTec AD

Download `mvtec_anomaly_detection.tar.xz` from the [official MVTec AD website](https://www.mvtec.com/company/research/datasets/mvtec-ad), then run:

```bash
python datasets/dataset_prepare_mvtec.py --tar mvtec_anomaly_detection.tar.xz \
    --out data/mvtec
```

### VisA

Download and prepare VisA by running:

```bash
curl -L -O https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar

python datasets/dataset_prepare_visa.py --tar VisA_20220922.tar \
    --extract-dir /tmp/visa_extract --out data/visa
```

After preparation, the datasets will be available at:

```text
data/
├── mvtec/
└── visa/
```


## Running the experiments

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
