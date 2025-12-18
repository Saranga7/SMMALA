from typing import Callable, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from omegaconf import DictConfig
import random

from src.data.custom_datasets import RawMicroscope
from src.data.collate import collate_func, collate_func_embeddings



def prepare_datasets(
    dataset: str,
    T_train: Callable,
    T_val: Callable,
    train_data: list,
    val_data: list,
    kwargs: Optional[dict] = None,
) -> Tuple[Dataset, Dataset]:
    """
    Prepares train and val datasets.

    Args:
        dataset (str): dataset name.
        T_train (Callable): pipeline of transformations for training dataset.
        T_val (Callable): pipeline of transformations for validation dataset.
        train_data_path (Optional[Union[str, Path]], optional): path where the
            training data is located. Defaults to None.
        val_data_path (Optional[Union[str, Path]], optional): path where the
            validation data is located. Defaults to None.
        data_format (Optional[str]): format of the data. Defaults to "image_folder".
            Possible values are "image_folder" and "h5".
        data_fraction (Optional[float]): percentage of data to use. Use all data when set to -1.0.
            Defaults to -1.0.

    Returns:
        Tuple[Dataset, Dataset]: training dataset and validation dataset.
    """


    skip_validation = kwargs.get("skip_validation", False) or not val_data or len(val_data) == 0


    train_dataset = RawMicroscope(
            root_dir=kwargs["root_dir"],
            data=train_data,
            train_type="train",
            transform=T_train,
            num_imgs_per_slide=kwargs["num_imgs_per_slide"],
            data_collection_method=kwargs["data_collection_method"],
            shuffle_labels=kwargs.get("shuffle_labels", False),
            use_monte_carlo=kwargs.get("use_monte_carlo", False),
            use_hard_balancing=kwargs.get("use_hard_balancing", False),
            num_classes=kwargs.get("num_classes", 3),
            reference_image_path=kwargs.get("reference_image_path", None),
            class_idx=kwargs.get("class_idx", None),
            merge_classes_for_binary=kwargs.get("merge_classes_for_binary", False),
            class_merge_mapping=kwargs.get("class_merge_mapping", None),

            embeddings_path=kwargs.get("embeddings_path", None),
            aug_embeddings_path=kwargs.get("aug_embeddings_path", None),
        )
    if not skip_validation:
        val_dataset = RawMicroscope(
            root_dir=kwargs["root_dir"],
            data=val_data,
            train_type="val",
            transform=T_val,
            num_imgs_per_slide=kwargs["num_imgs_per_slide"],
            data_collection_method=kwargs["data_collection_method"],
            shuffle_labels=False,
            use_monte_carlo=False,
            use_hard_balancing=False,
            num_classes=kwargs.get("num_classes", 3),
            reference_image_path=kwargs.get("reference_image_path", None),
            class_idx=kwargs.get("class_idx", None),
            merge_classes_for_binary=kwargs.get("merge_classes_for_binary", False),
            class_merge_mapping=kwargs.get("class_merge_mapping", None),

            embeddings_path=kwargs.get("embeddings_path", None),
        )
    else:
        val_dataset = None

    return train_dataset, val_dataset


def prepare_dataloaders(
    cfg: DictConfig,
    train_dataset: Dataset,
    val_dataset: Dataset,
    batch_size: int = 64,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    """Wraps a train and a validation dataset with a DataLoader.

    Args:
        train_dataset (Dataset): object containing training data.
        val_dataset (Dataset): object containing validation data.
        batch_size (int): batch size.
        num_workers (int): number of parallel workers.
    Returns:
        Tuple[DataLoader, DataLoader]: training dataloader and validation dataloader.
    """

    if cfg.model.use_imgs_or_embeddings == "embeddings":
        collate_fn = collate_func_embeddings
    else:
        collate_fn = collate_func
            
    seed = cfg.data.random_seed
    g = torch.Generator()
    g.manual_seed(seed)
    print("Num workers for dataloader:", num_workers)

    def worker_init_fn(worker_id):
        # derive unique seed for each worker from global SEED
        worker_seed = seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)


    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,  # Store batches directly on the pinned memory for faster GPU transfer
        persistent_workers=True,  # Avoid reinitializing the workers at each epoch
        drop_last=True,  # Drop the last batch if it is not complete
        collate_fn=collate_fn,

        # ensure reproducibility when using multiple workers
        worker_init_fn = worker_init_fn,
        generator = g,
    )
    if val_dataset is not None:
        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=False,  # Store batches directly on the pinned memory for faster GPU transfer
            persistent_workers=True,  # Avoid reinitializing the workers at each epoch
            collate_fn = collate_fn,

            # ensure reproducibility when using multiple workers
            worker_init_fn=worker_init_fn,
            generator = g,
        )
    else:
        val_loader = None

    return train_loader, val_loader
