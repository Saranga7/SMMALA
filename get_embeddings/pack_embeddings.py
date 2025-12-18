# create_embeddings_mmap.py
import numpy as np
from pathlib import Path
import json
import tqdm

model_name = "resnet50" # dinov3_vith16plus
img_or_aug_embeddings = "aug_img_embeddings"  # "img_embeddings or "aug_img_embeddings", 

emb_dir = Path(f"/projects/smala3/Saranga/preprocessed_data/resized_embeddings/{img_or_aug_embeddings}/{model_name}")
out_dir = Path(f"/projects/smala3/Saranga/preprocessed_data/resized_embeddings/{img_or_aug_embeddings}/packed")
model_name = emb_dir.name  # e.g. dinov3_vitb16
out_dir.mkdir(parents=True, exist_ok=True)

# collect .npy files (including aug0..aug9 and originals)
files = sorted(emb_dir.glob("*.npy"))
print(f"Found {len(files)} files")

# load first to get shape/dtype
first = np.load(files[0])
D = first.shape[0] if first.ndim == 1 else first.shape
dtype = first.dtype
print("Embedding shape per file:", D, "dtype:", dtype)

# allocate array of shape (N, D)
N = len(files)
# If embeddings are 1D per file, stack into 2D
embeddings = np.empty((N, first.shape[0]), dtype=np.float32)

keys = []
for i, p in enumerate(tqdm.tqdm(files, desc="Loading embeddings")):
    a = np.load(p)
    if a.dtype != np.float32:
        a = a.astype(np.float32)
    embeddings[i] = a
    keys.append(p.name)  # store filename (e.g., ASY..._aug0.npy)

# save embeddings as a single .npy (uncompressed) — allows mmap_mode='r'
np.save(out_dir / f"{model_name}_embeddings.npy", embeddings)
# save keys as object array (or json)
np.save(out_dir / f"{model_name}_keys.npy", np.array(keys, dtype=object))
with open(out_dir / f"{model_name}_keys_to_index.json", "w") as f:
    json.dump({k: i for i, k in enumerate(keys)}, f)

print("Saved embeddings.npy and keys.npy and keys_to_index.json to", out_dir)
