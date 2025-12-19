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

After cloning, download the pretrained dinov3 weights and place them in weights/pretrained (or as configured in your cfg). Depending on which dinov3 encoder backbones you want to experiment with, it should look somthing like this:

```
weights/pretrained
├── dinov3_vit7b16.pth
├── dinov3_vitb16.pth
├── dinov3_vith16plus.pth
├── dinov3_vitl16.pth
├── dinov3_vits16plus.pth
└── dinov3_vits16.pth
```
