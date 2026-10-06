# How Far Can Anomaly Detection Go Before Learning? Random Neural Features from Tables to Pixels

**PRISM FrameWork**

![Architecture](./images/PRISM_Framework.png)

Authors: Hyojun Yi, Geunyoung Kim, Hogeon Seo, and Seunghyoung Ryu

paper: 


## Introduction

This repository contains the implementation of **PRISM-T**, the tabular anomaly detection module of PRISM. PRISM-T investigates how effectively a neural network with randomly initialized weights can detect anomalies in tabular data.


## Data

Download and prepare the dataset as follows.

### ADBench

Download the 47 Classical datasets from [ADBench](https://github.com/Minqi824/ADBench) by running:

```bash
python datasets/download_data.py
```

After preparation, the datasets will be available at:

```text
data/
└── Classical/
```


## Running the experiments

None of the drivers below download or convert anything,
so `data/Classical` must already exist; a missing data
root aborts before any computation starts.


```bash
python scripts/run_prism_t.py
```



## Repository layout

```
models/      prism_t.py (the PRISM-T encoder), mahalanobis.py
             (shrinkage Mahalanobis fit/score)
evaluation/  metrics.py (AUROC/AUPRC)
datasets/    loader.py (semi-supervised split + StandardScaler);
             download_data.py fetches the 47 ADBench Classical
             datasets into the layout it reads
scripts/     run_prism_t.py (the single experiment engine; defaults = the
             adopted main config)
data/        create it — datasets are not distributed
results/     created by the run script
```


## Acknowledgement

This work was partially supported in part by the IITP (Institute of Information & Communications Technology Planning & Evaluation) - ICAN (ICT Challenge and Advanced Network of HRD) grant funded by the Korea government (Ministry of Science and ICT) (IITP-2026-RS-2024-00436528), and in part by the National Research Foundation of Korea(NRF) grant funded by the Korea government(MSIT) (RS-2024-00454514)


## License / citation

To be added upon publication. The 47 ADBench "Classical" datasets remain under
their own licences; this package redistributes neither the datasets nor derived
weights (there are none — nothing is trained).
