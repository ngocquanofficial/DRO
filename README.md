# Promoting Ensemble Diversity with Interactive Bayesian Distributional Robustness for Fine-tuning Foundation Models

Official PyTorch implementation for the paper: "Promoting Ensemble Diversity with Interactive Bayesian Distributional Robustness for Fine-tuning Foundation Models" (ICML 2025).

## Overview
This repository contains the official implementation of Interactive Bayesian Distributional Robustness (IBDR). Our method focuses on enhancing the diversity of ensembles when fine-tuning foundation models.

## Installation
We recommend using a conda environment for dependency management.

```
conda create -n ibdr python==3.10 -y
conda activate ibdr

pip install -r requirements.txt
```

## Usage

### Model & Data Preparation: 
Download the [Vision Transformers (ViT-B/16)](https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz) checkpoints and place them in the checkpoints/ directory.

Download the TAB-1K datasets and extract them into the data/ directory. Please refer to [SSF](https://github.com/dongzelian/SSF) or [VPT](https://github.com/KMnP/vpt/blob/main/VTAB_SETUP.md) for preparing the 19 datasets included in VTAB-1K. For convenience, you can download the extracted file (VTAB.zip)[https://box.nju.edu.cn/f/57d5913e680243fca32b/?dl=1] to easily access the datasets.

### Training
All configurations are managed via YAML files and located in the configs/ folder. To start training (e.g., using the DRO configuration for CIFAR-100), run:

```
python main.py fit --config configs/dro/cifar100.yaml
```

### Citing

If you use this repository in your research, please consider citing our paper:

```
@inproceedings{phampromoting,
  title={Promoting Ensemble Diversity with Interactive Bayesian Distributional Robustness for Fine-tuning Foundation Models},
  author={Pham, Ngoc-Quan and Truong, Tuan and Tran, Quyen and Nguyen, Tan Minh and Phung, Dinh and Le, Trung},
  booktitle={Forty-second International Conference on Machine Learning},
  year={2025}
}
```

