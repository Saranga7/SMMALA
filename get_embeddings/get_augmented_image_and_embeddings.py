import logging
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from tqdm import tqdm
import random
import numpy as np
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import json
import sys
# Add project root (parent of src) to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.utils.utils_config import setup_device
from src.utils.utils_model import setup_seed

logger = logging.getLogger(__name__)


    
class RandomRotate90:
    """Randomly rotate by 0°, 90°, 180°, or 270° without interpolation."""
    def __call__(self, img: Image.Image):
        k = random.choice([0, 1, 2, 3])
        if k == 0:
            return img
        ops = {1: Image.ROTATE_90, 2: Image.ROTATE_180, 3: Image.ROTATE_270}
        return img.transpose(ops[k])

class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, exts=(".tif", ".tiff", ".TIF", ".TIFF")):
        """
        root_dir: folder with images
        transform: torchvision transforms to apply
        exts: allowed file extensions
        """
        self.root_dir = Path(root_dir)
        # gather files first
        self.files = sorted([p for p in self.root_dir.rglob("*") if p.suffix in exts])
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        p = self.files[idx]
        img = Image.open(p)
        if self.transform:
            img = self.transform(img)
        return img, str(p)



def load_backbone(cfg, device):

    if cfg.model.name.startswith("dinov3"):

        model = torch.hub.load(cfg.model.dinov3_repo_path, 
                               cfg.model.name, 
                               source = 'local', 
                               weights = str(Path(cfg.model.weights_path) / "pretrained" / Path(cfg.model.name + ".pth")))

        logger.info(f"{cfg.model.name} backbone loaded successfully.")
        if cfg.model.freeze_layers:
            for param in model.parameters():
                param.requires_grad = False
            logger.info("Backbone frozen")


    elif cfg.model.name == "resnet50":
        model = models.resnet50(pretrained=True)
        logger.info(f"{cfg.model.name} backbone loaded successfully.")
        if cfg.model.freeze_layers:
            for param in model.parameters():
                param.requires_grad = False
            logger.info("Backbone frozen")
    else:
        raise ValueError(f"Unsupported model name: {cfg.model.name}")

    return model.to(device)



def get_features_from_backbone(cfg, model, inputs):

    model.eval()
    with torch.no_grad():
        if cfg.model.name.startswith("dino"):
            features_dict = model.forward_features(inputs)
            features = features_dict["x_norm_clstoken"]
            features = torch.nn.functional.normalize(features, p = 2, dim = 1)
        elif cfg.model.name.startswith("resnet"):
            features = model.avgpool(model.layer4(model.layer3(model.layer2(model.layer1(model.maxpool(model.conv1(inputs)))))))
            features = torch.flatten(features, 1)
            features = torch.nn.functional.normalize(features, p = 2, dim = 1)

    return features



# to visualize the augmentations
def save_batch_images(imgs_tensor, paths, save_dir: Path, aug_idx: int = None):
    """Convert normalized tensors back to PIL images and save them.

    imgs_tensor: (B, C, H, W) tensor (may be on GPU)
    paths: list of original file paths (used for naming)
    save_dir: folder to save images
    aug_idx: optional augmentation index to append to filename
    """
    # mean/std used in your transforms
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    imgs_cpu = imgs_tensor.detach().cpu().clone()
    # unnormalize: x = x * std + mean
    imgs_cpu = imgs_cpu * std + mean
    imgs_cpu = imgs_cpu.clamp(0.0, 1.0)

    to_pil = transforms.ToPILImage()

    for img_t, p in zip(imgs_cpu, paths):
        pil = to_pil(img_t)
        stem = Path(p).stem
        if aug_idx is not None:
            fname = stem + f"_aug{aug_idx}.jpg"
        else:
            fname = stem + ".jpg"
        save_path = save_dir / fname
        pil.save(save_path)




@hydra.main(version_base="1.2")
def main(cfg: DictConfig):
 
    device = setup_device(cfg)
    setup_seed(cfg)

    model = load_backbone(cfg, device)
    model_name = cfg.model.name
    out_root_dir = cfg.data.out_dir

    img_dir = cfg.data.img_dir
    logger.info(model)

    if cfg.augmentations.enable_augmentation:

        logger.info("Data augmentation is enabled.")
        transform = transforms.Compose(
            [
                RandomRotate90(),
                transforms.Resize(size = (512, 512)),
                transforms.RandomHorizontalFlip(p = 0.5),
                transforms.RandomVerticalFlip(p = 0.5),
                transforms.RandomApply([transforms.ColorJitter(brightness = 0.1, contrast = 0.3, saturation = 0.3, hue = 0.3)], p = 0.4),
                transforms.ToTensor(),
                transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
            ]
        )

        k = int(cfg.augmentations.num_augmentations)
        out_prefix = "aug_img_embeddings"

        assert k >= 1, "num_augmentations must be >= 1"

    else:
        logger.info("Data augmentation is disabled")
        assert cfg.data.test_set_img_dir is not None and Path(cfg.data.test_set_img_dir).exists(), f"test_set_img_dir {cfg.data.test_set_img_dir} does not exist"

        logger.info(f"Using test set image directory {cfg.data.test_set_img_dir}")
        img_dir = cfg.data.test_set_img_dir

        transform = transforms.Compose(
            [
                transforms.Resize(size = (512, 512)),
                transforms.ToTensor(),
                transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
            ]
        )

        k = 1
        out_prefix = "img_embeddings"

    out_dir = Path(out_root_dir) / out_prefix / model_name
    out_dir.mkdir(parents = True, exist_ok = True)
    

    # Use cfg.augmentations.save_augmented_images = True in your config to enable saving.
    save_augmented = getattr(cfg.augmentations, "save_augmented_images", False)
    if save_augmented:
        if k == 1:
            aug_img_dir = Path(out_root_dir) / "img_embeddings"/ "augmented_images" 
        else:
            aug_img_dir = Path(out_root_dir) / "aug_img_embeddings"/ "augmented_images"  / f"{k}_augmentations"
        aug_img_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Augmented images will be saved to: {aug_img_dir}")
    else:
        aug_img_dir = None
    

    for i in range(k):

        logger.info(f"Starting augmentation {i+1}/{k}")
        ds = ImageDataset(img_dir, transform = transform)

        loader = DataLoader(
                ds,
                batch_size = cfg.data.batch_size,
                shuffle = False,
                num_workers = cfg.data.num_workers,
                pin_memory = True,
            )    

        pbar = tqdm(loader, total=len(loader), desc=f"Encoding with {model_name}")
                
        for imgs, paths in pbar:
            if save_augmented and aug_img_dir is not None:
                try:
                    save_batch_images(imgs, paths, aug_img_dir, i if k > 1 else None)
                except Exception as e:
                    logger.exception(f"Failed to save augmented images for batch: {e}")
            imgs = imgs.to(device)
            feats = get_features_from_backbone(cfg, model, imgs)  # (B, C)
            feats = feats.detach().cpu().numpy()
            

            for p, vec in zip(paths, feats):
                if k > 1:
                    fname = Path(p).stem + f"_aug{i}.npy"
                else:
                    fname = Path(p).stem + ".npy"
                save_path = out_dir / fname
                np.save(save_path, vec)

        pbar.close()
    logger.info(f"Saved embeddings to {out_dir}")


    # ---------------------------Pack embeddings into single .npy and keys files------------------
    emb_dir = out_dir
    packed_out_dir = Path(out_root_dir) / out_prefix / "packed"
    packed_out_dir.mkdir(parents = True, exist_ok = True)
    
    files = sorted(emb_dir.glob("*.npy"))
    N = len(files)
    print(f"Found {N} files")

    first = np.load(files[0])
    D = first.shape[0] if first.ndim == 1 else first.shape
    dtype = first.dtype
    print("Embedding shape per file:", D, "dtype:", dtype)

    embeddings = np.empty((N, first.shape[0]), dtype=np.float32)
    keys = []
    for i, p in enumerate(tqdm(files, desc="Loading embeddings")):
        a = np.load(p)
        if a.dtype != np.float32:
            a = a.astype(np.float32)
        embeddings[i] = a
        keys.append(p.name)  

    # save embeddings as a single .npy (uncompressed) 
    np.save(packed_out_dir / f"{model_name}_embeddings.npy", embeddings)
    # save keys as object array (or json)
    np.save(packed_out_dir / f"{model_name}_keys.npy", np.array(keys, dtype=object))
    with open(packed_out_dir / f"{model_name}_keys_to_index.json", "w") as f:
        json.dump({k: i for i, k in enumerate(keys)}, f)

    print("Saved embeddings.npy and keys.npy and keys_to_index.json to", packed_out_dir)
        


if __name__ == "__main__":
    main()
