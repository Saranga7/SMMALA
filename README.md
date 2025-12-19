# SMMALA 🦠 Detection of Malaria Infection from Parasite-Free Blood Smears

## Introduction

This project is a collaboration between the [IRD](https://www.ird.fr) and [ENS Ulm](https://www.ens.psl.eu).

---

## Setup Instructions

### 1️⃣ Clone the repository (with submodules)

The project depends on the [`dinov3`](https://github.com/facebookresearch/dinov3) submodule. To ensure reproducibility, clone the repository **with submodules**:

```bash
git clone --recurse-submodules https://github.com/Saranga7/SMMALA.git
```

If you already cloned without submodules, initialize them with:

```bash
git submodule update --init --recursive
```

### 2️⃣ Download dinov3 weights
After cloning, download the pretrained dinov3 weights and place them in `weights/pretrained`. Depending on which dinov3 encoder backbones you want to experiment with, it should look somthing like this:

```
weights/pretrained
├── dinov3_vit7b16.pth
├── dinov3_vitb16.pth
├── dinov3_vith16plus.pth
├── dinov3_vitl16.pth
├── dinov3_vits16plus.pth
└── dinov3_vits16.pthn
```

### 2️⃣ Download data

Download the images from [here](https://huggingface.co/datasets/nicoboou/smmala/tree/main). Preferably store the images at `preprocessed_data/dataset`, otherwise mandatorily set data.dataset_path in the config.

### 3️⃣ Setup virtual environment

### 4️⃣ Generate embeddings

Adjust the config file: `configs/get_image_embeddings.yaml`. Then,

```bash
cd get_embeddings
bash run_get_image_embeddings.sh
cd ..
```

### 5️⃣ Train and Test

Adjust the config file: `configs/emb_subVneg.yaml`.
Then,

```bash
python train.py --config-path="configs/train" --config-name="emb_subVneg.yaml" 
```

An example script that uses `dinov3_vitb16` with a `mean` aggregation strategy:

```bash
bash run_training.sh
```

