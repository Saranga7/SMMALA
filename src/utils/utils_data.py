import logging
import pandas as pd
import numpy as np
import torch
from omegaconf import DictConfig
from torchvision import transforms
from torch.utils.data import DataLoader
from pathlib import Path
from PIL import Image
import random

from src.data.splitters.microscope_kfold_splitter import MicroscopeKFoldSplitter
from src.data.classification_dataloader import (
    prepare_datasets,
    prepare_transforms,
)
from src.data.custom_datasets import RawMicroscope 
from src.data.collate import collate_func, collate_func_embeddings

logger = logging.getLogger(__name__)


class RandomRotate90:
    """Randomly rotate by 0°, 90°, 180°, or 270° without interpolation."""
    def __call__(self, img: Image.Image):
        k = random.choice([0, 1, 2, 3])
        if k == 0:
            return img
        ops = {1: Image.ROTATE_90, 2: Image.ROTATE_180, 3: Image.ROTATE_270}
        return img.transpose(ops[k])


def prepare_datasets_and_transforms(cfg):
    if cfg.training.use_advanced_augmentation:
        T_train, T_val = get_advanced_transforms()
    else:
        T_train, T_val = prepare_transforms(cfg.data.dataset)

    train_path = Path(cfg.data.root_dir) / Path(cfg.data.data_train_filepath)
    metadata_path = Path(cfg.data.root_dir) / Path(cfg.data.metadata_filepath)

    if hasattr(cfg.data, "skip_validation") and cfg.data.skip_validation:
        main_data = pd.read_csv(train_path)
        metadata = pd.read_csv(metadata_path)
        merged_data = main_data.merge(metadata, left_on="slide_id", right_on="NUM DOSS", how="left")
        train_data = merged_data.to_dict(orient="records")
        val_data = []
        logger.info("Using all data for training, skipping validation.")
    else:
        splitter = MicroscopeKFoldSplitter(
            csv_file_path=train_path,
            metadata_file_path=metadata_path,
            num_folds=cfg.data.num_folds,
            random_state=cfg.data.random_seed,
        )
        train_data, val_data = splitter.get_fold()

    return prepare_datasets(
        dataset="rawmicroscope",
        T_train=T_train,
        T_val=T_val,
        train_data=train_data,
        val_data=val_data,
        kwargs=cfg.data,
    )


def get_test_dataset(
    cfg: DictConfig,
    T_test: transforms.Compose,
):
    csv_file_path = Path(cfg.data.root_dir) / Path(cfg.data.data_test_filepath)
    metadata_file_path = Path(cfg.data.root_dir) / Path(cfg.data.metadata_filepath)

    main_data = pd.read_csv(csv_file_path)
    metadata = pd.read_csv(metadata_file_path)
    merged_data = main_data.merge(metadata, left_on="slide_id", right_on="NUM DOSS", how="left")
    test_data = merged_data.to_dict(orient="records")

    slide_ids = set([row["slide_id"] for row in test_data])
    logger.info(f"Unique slide ids in test data: {len(slide_ids)}")

   
    test_dataset = RawMicroscope(
        root_dir=cfg.data.root_dir,
        data=test_data,
        train_type="test",
        transform=T_test,
        num_imgs_per_slide=cfg.data.num_imgs_per_slide,
        data_collection_method=cfg.data.data_collection_method,
        shuffle_labels=cfg.data.shuffle_labels,
        use_monte_carlo=False,
        use_hard_balancing=False,
        num_classes=cfg.data.num_classes,
        reference_image_path=getattr(cfg.data, "reference_image_path", None),
        class_idx=getattr(cfg.data, "class_idx", None),
        merge_classes_for_binary=getattr(cfg.data, "merge_classes_for_binary", False),
        class_merge_mapping=getattr(cfg.data, "class_merge_mapping", None),

        embeddings_path=getattr(cfg.data, "embeddings_path", None),
        aug_embeddings_path=getattr(cfg.data, "aug_embeddings_path", None),
        )



    return test_dataset


def get_test_dataloader(cfg: DictConfig):

    T_test = transforms.Compose(
        [
            transforms.Resize(size = (512, 512)),
            # transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0), ratio=(1.0, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


    test_dataset = get_test_dataset(
        cfg=cfg,
        T_test=T_test,
    )


    if cfg.model.use_imgs_or_embeddings == "embeddings":
        collate_fn = collate_func_embeddings
    else:
        collate_fn = collate_func

    seed = cfg.data.random_seed
    g = torch.Generator()
    g.manual_seed(seed)

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=cfg.optimizer.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=False,
        drop_last=False,
        collate_fn=collate_fn,

        # ensure reproducibility when using multiple workers
        worker_init_fn = lambda worker_id: np.random.seed(seed + worker_id),
        generator = g,
    )

    return test_loader


def get_advanced_transforms():


    T_train = transforms.Compose(
        [   
            RandomRotate90(),
            # transforms.RandomResizedCrop(size=256, scale=(0.4, 1.0), ratio=(1.0, 1.0)),
            transforms.Resize(size = (512, 512)),
            transforms.RandomHorizontalFlip(p = 0.5),
            transforms.RandomVerticalFlip(p = 0.5),
            transforms.RandomApply([transforms.ColorJitter(brightness = 0.1, contrast = 0.3, saturation = 0.3, hue = 0.3)], p = 0.4),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

 

    T_val = transforms.Compose(
        [
            # transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0), ratio=(1.0, 1.0)),
            transforms.Resize(size = (512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return T_train, T_val


def compute_class_weights(class_weight, classes, y, num_classes=3, merge_classes=None):
    """
    Calculate class weights for classification tasks, mirroring scikit-learn's compute_class_weight.

    Parameters
    ----------
    class_weight : str, dict, or None
        If 'balanced', class weights are computed as:
            n_samples / (n_classes * np.bincount(y))
        If a dictionary is provided, keys are class labels and values are corresponding class weights.
        If None, all classes are assigned a weight of 1.

    classes : array-like of shape (n_classes,)
        Array of class labels. For example, [0, 1, 2] for multi-class or [0, 1] for binary.

    y : array-like of shape (n_samples,)
        Array of class labels for each sample.

    num_classes : int, default=3
        Number of classes in the classification task. Can be 2 or 3 based on the scenario.

    merge_classes : dict, optional
        Mapping for merging classes when `num_classes=2`. For example:
            {'Positive': [1, 2]}
        This will merge classes 1 and 2 into a single class labeled as 'Positive'.

    Returns
    -------
    class_weight_tensor : torch.Tensor
        Tensor containing the weight for each class. For binary classification, this will be a scalar.
    """
    y = np.array(y)
    classes = np.array(classes)

    if num_classes == 2:
        if merge_classes is None:
            raise ValueError("merge_classes must be provided for binary classification.")
        if "Positive" not in merge_classes:
            raise ValueError("merge_classes must contain a 'Positive' key for merging.")

        positive_classes = merge_classes["Positive"]
        # Verify that all classes to be merged are in the original classes
        if not set(positive_classes).issubset(set(classes)):
            missing = set(positive_classes) - set(classes)
            raise ValueError(f"Classes to merge {missing} are not present in `classes`.")

        # Create a new y where Positive classes are merged
        y_binary = np.where(np.isin(y, positive_classes), 1, 0)
        unique_binary_classes = np.unique(y_binary)
        if len(unique_binary_classes) != 2:
            raise ValueError("After merging, there should be exactly two classes for binary classification.")

        # Define binary classes
        binary_classes = [0, 1]  # 0: Negative, 1: Positive
        y_mapped = y_binary
    elif num_classes == 3:
        y_mapped = y
    else:
        raise ValueError("num_classes must be either 2 or 3.")

    if class_weight == "balanced":
        # Compute weights as n_samples / (n_classes * count_per_class)
        class_counts = np.bincount(y_mapped, minlength=num_classes).astype(np.float64)
        total_samples = len(y_mapped)
        weights = total_samples / (num_classes * class_counts + 1e-6)  # Avoid division by zero
    elif isinstance(class_weight, dict):
        # Assign weights based on the dictionary
        weights = np.ones(num_classes, dtype=np.float64)
        class_weight_keys = list(class_weight.keys())
        # Map original class labels to indices
        if num_classes == 2:
            class_label_to_index = {0: 0, 1: 1}
        else:
            class_label_to_index = {cls: idx for idx, cls in enumerate(classes)}

        for cls_label, weight in class_weight.items():
            if num_classes == 2:
                if cls_label in merge_classes["Positive"]:
                    cls_index = 1  # Positive class index
                elif cls_label == 0:
                    cls_index = 0  # Negative class index
                else:
                    raise ValueError(f"Class label {cls_label} is not valid for binary classification.")
            else:
                if cls_label not in class_label_to_index:
                    raise ValueError(f"Class label {cls_label} provided in class_weight is not in `classes`.")
                cls_index = class_label_to_index[cls_label]
            weights[cls_index] = class_weight[cls_label]

        # Check if all classes in `classes` are covered in `class_weight`
        if num_classes == 2:
            required_labels = [0] + merge_classes["Positive"]
            if not set(required_labels).issubset(set(class_weight_keys)):
                missing = set(required_labels) - set(class_weight_keys)
                raise ValueError(f"The following classes are missing in class_weight dict: {missing}")
        else:
            missing = set(classes) - set(class_weight_keys)
            if missing:
                raise ValueError(f"The following classes are missing in class_weight dict: {missing}")
    else:
        # Uniform weights
        weights = np.ones(num_classes, dtype=np.float64)

    # Convert to tensor
    if num_classes == 2:
        # For binary classification, return a scalar for pos_weight
        class_weight_tensor = torch.tensor(weights[1], dtype=torch.float32)
    else:
        # For multi-class, return a tensor of weights
        class_weight_tensor = torch.tensor(weights, dtype=torch.float32)

    return class_weight_tensor


def get_class_weights(dataset, class_weight, num_classes, merge_classes={"Positive": [1, 2]}):
    all_labels = torch.tensor([int(label) for label in dataset.labels])
    logger.info(f"unique labels: {torch.unique(all_labels)}")
    logger.info(f"number of labels: {len(all_labels)}")
    class_weights = compute_class_weights(
        class_weight=class_weight,
        classes=np.unique(all_labels.numpy()),
        y=all_labels.numpy(),
        num_classes=num_classes,
        merge_classes=merge_classes,
    )
    # compute_class_weights already returns a tensor, no need to convert again
    return class_weights
