# SMMALA 🦠 Detection of Malaria Infection from Parasite-Free Blood Smears

## Introduction

This project is a collaboration between the [IRD](https://www.ird.fr) and [ENS Ulm](https://www.ens.psl.eu).

---

## Setup Instructions

### 1️⃣ Clone the repository (with submodules)

The project depends on the `dinov3` submodule. To ensure reproducibility, clone the repository **with submodules**:

```bash
git clone --recurse-submodules <your-repo-url>


If you already cloned without submodules, initialize them with:

```bash
git submodule update --init --recursive

### 2️⃣ Download dinov3 weights

After cloning, download the pretrained dinov3 weights and place them in the following path (or as configured in your cfg):
