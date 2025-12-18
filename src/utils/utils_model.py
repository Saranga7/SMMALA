import logging
import math
import random
import os
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.models as models
import numpy as np
from pathlib import Path
from torchvision.models.resnet import ResNet

from src.common.layers.residual_gated_attention import ResidualGatedAttention
from src.common.layers.lora import add_lora
from src.utils.utils_data import get_class_weights
from src.common.losses.focal import FocalLoss
from src.common.losses.sosr import SOSRLoss
from src.utils.misc import omegaconf_select
from src.models.slide_aggregator import SlideAggregator
from src.models.cnn import CustomCNN

logger = logging.getLogger(__name__)


def get_model(cfg, num_classes, device):
    """
    Get model based on configuration and data collection method (image or slide-level)
    """
    # Determine output features based on effective class setup
    effective_num_classes, _, _ = get_effective_class_setup(cfg, num_classes)
    out_features = 1 if effective_num_classes == 2 else effective_num_classes

    model = _load_backbone(cfg, out_features)

    model = _apply_lora(model, cfg)

    if cfg.data.data_collection_method == "slide":
        if cfg.model.slide_aggregator_method == "cdf_mlp":
            logger.info("Using CDF_MLP slide aggregator")
            # this method needs the trained classifier head to output FoV probabilities
            model = _add_head(model, cfg, out_features, device)
            assert cfg.model.fov_classifier_head_weights_path is not None, "Path to FoV classifier head weights must be provided for CDF_MLP slide aggregator"
            model.load_state_dict(torch.load(cfg.model.fov_classifier_head_weights_path, map_location=device))
            logger.info("FoV classifier head weights loaded successfully.")
            # freeze the entire model 
            model.eval()
            for param in model.parameters():
                param.requires_grad = False

        model = _add_slide_aggregator(model, cfg, out_features, 
                                        use_embed_transform = False)

        for name, p in model.named_parameters():
            if "psi" in name:
                assert p.requires_grad, f"{name} should be trainable"
            else:
                assert not p.requires_grad, f"{name} should be frozen"

    elif cfg.data.data_collection_method == "image":
        model = _add_head(model, cfg, out_features, device)
    else:
        raise ValueError(f"Data collection method {cfg.data.data_collection_method} not supported")

    return model


def _add_slide_aggregator(model, cfg, out_features, use_embed_transform = True):
    """
    Wrap model with SlideAggregator for slide-level classification
    """
    model = SlideAggregator(model, cfg, out_features = out_features, use_transform = use_embed_transform)
    return model


def _load_backbone(cfg, out_features):

    if cfg.model.name.startswith("dinov3"):

        model = torch.hub.load(cfg.model.dinov3_repo_path, 
                               cfg.model.name, 
                               source = 'local', 
                               weights = str(Path(cfg.model.weights_path) / "pretrained" / Path(cfg.model.name + ".pth")))

        print(f"{cfg.model.name} backbone loaded successfully.")
        if cfg.model.freeze_layers:
            for param in model.parameters():
                param.requires_grad = False
            print("Backbone frozen")


    elif cfg.model.name.startswith("dinov2"):
        if cfg.slurm.enabled:
            model = torch.hub.load("facebookresearch/dinov2", cfg.model.name, pretrained=False)
            model.load_state_dict(
                torch.load(Path(cfg.model.weights_path) / "pretrained" / Path(cfg.model.name + ".pth"))
            )
        else:
            model = torch.hub.load("facebookresearch/dinov2", cfg.model.name, pretrained=True)
        if cfg.model.freeze_layers:
            for param in model.parameters():
                param.requires_grad = False

    elif cfg.model.name == "resnet50":
        if cfg.slurm.enabled:
            model = models.resnet50(pretrained = False)
            model.load_state_dict(torch.load(Path(cfg.model.weights_path) / "pretrained" / "resnet50-0676ba61.pth"))
        else:
            model = models.resnet50(pretrained = True)
        if cfg.model.freeze_layers:
            for param in model.parameters():
                param.requires_grad = False


    elif cfg.model.name.startswith("custom_cnn"):
        model = CustomCNN(name = cfg.model.name, num_classes = out_features)
    else:
        raise ValueError(f"Model {cfg.model.name} not supported")
    

    return model


def _apply_lora(model, cfg):
    if cfg.model.use_lora:
        assert cfg.model.freeze_layers, "LoRA can only be used with frozen encoder weights"
        model = add_lora(
            model,
            torch.nn.Linear,
            rank=cfg.model.lora_rank,
            alpha=cfg.model.lora_alpha,
            num_blocks_to_modify=cfg.model.lora_num_blocks,
        )
    return model




def _add_head(model, cfg, out_features, device):
    # no need to add head for custom_cnn models
    if cfg.model.name.startswith("custom_cnn"):
        return model

    if cfg.model.name.startswith("dino"):
        if "convnext" in cfg.model.name:
            print("Convnext Model detected")
            num_features = model.embed_dim
        else:
            num_features = model.num_features

    elif cfg.model.name == "resnet50":
        num_features = model.fc.in_features


    if cfg.model.encoder_weights_path is not None:
        logger.info(f"Loading backbone weights from: {cfg.model.encoder_weights_path}")
        ckpt = torch.load(cfg.model.encoder_weights_path, map_location = device)
        model.load_state_dict(ckpt["online_encoder"], strict = True)
        logger.info("Backbone weights loaded successfully.")
        
    
    # Add classification head based on how complex you want it
    if cfg.model.use_complex_head == 1:
        model.head = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, out_features),
        )
    elif cfg.model.use_complex_head == 1.5:
        model.head = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, out_features),
        )

    elif cfg.model.use_complex_head == 2:
        model.head = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.PReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1024),
            nn.PReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.PReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, out_features),
        )
    elif cfg.model.use_complex_head == 3:
        model.head = nn.Sequential(
            ResidualGatedAttention(num_features, num_heads = 4),
            nn.Dropout(0.3),
            nn.Linear(num_features, out_features),
        )
    else:
        model.head = nn.Linear(num_features, out_features)

    for param in model.head.parameters():
        param.requires_grad = True

    # rename head to fc for compatibility with the rest of the code
    if isinstance(model, ResNet):
        model.fc = model.head
        del model.head

    if cfg.model.use_imgs_or_embeddings == "embeddings":
        if isinstance(model, ResNet):
            model = model.fc
        else:
            model = model.head
            model.num_features = num_features


    return model


def setup_seed(cfg):
    seed = omegaconf_select(cfg, "data.random_seed", np.random.randint(0, 10000))
    seed = int(seed)
    logger.info(f"Setting random seed to {seed}")
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return seed


def setup_model(cfg, num_classes, device):
    """Enhanced setup_model to handle both image and slide-level classification"""

    model = get_model(cfg, num_classes, device)

    return model.to(device)


def setup_criterion(cfg, num_classes, train_dataset, device, merge_classes={"Positive": [1, 2]}):
    """
    Sets up the criterion (loss function) based on the training configuration.

    Args:
        cfg: Training configuration.
        num_classes (int): Number of classes in the classification task.
        train_dataset: The training dataset for potential class weight calculation.
        device: The device for training (CPU/GPU).
        merge_classes (dict, optional): Mapping of new classes to original classes for binary classification.

    Returns:
        nn.Module: The criterion (loss function).
    """
    loss_type = cfg.training.loss_type.lower()

    # Get effective number of classes based on merging and filtering
    effective_num_classes, class_idx, merge_mapping = get_effective_class_setup(cfg, num_classes)

    if cfg.data.use_wce:
        # Use the merge_mapping if available, otherwise fall back to the old merge_classes
        if merge_mapping is not None:
            # Convert new merge_mapping format {0: [0], 1: [1, 2]} to old format {"Positive": [1, 2]}
            # Find which original classes are merged into the positive class (class 1)
            positive_classes = []
            for new_class, old_classes in merge_mapping.items():
                if new_class == 1:  # Positive class
                    positive_classes = old_classes
                    break
            merge_for_weights = {"Positive": positive_classes} if positive_classes else None
        else:
            merge_for_weights = merge_classes if effective_num_classes == 2 else None

        class_weights = get_class_weights(
            dataset=train_dataset,
            class_weight="balanced",
            num_classes=effective_num_classes,  # Use effective_num_classes instead of num_classes
            merge_classes=merge_for_weights,
        ).to(device)

        # No need to filter class weights since get_class_weights returns weights for effective classes
        logger.info(f"Computed class weights: {class_weights}")
    else:
        class_weights = None

    if loss_type == "ce":
        if effective_num_classes == 2:
            if class_weights is not None:
                # For binary case, class_weights is already a scalar (pos_weight for positive class)
                pos_weight = class_weights
                logger.info(f"Using BCEWithLogitsLoss with pos_weight: {pos_weight}")
                criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            else:
                logger.info("Using BCEWithLogitsLoss without class weights.")
                criterion = nn.BCEWithLogitsLoss()
        elif effective_num_classes == 3:
            if class_weights is not None:
                logger.info(f"Using CrossEntropyLoss with class weights: {class_weights}")
                criterion = nn.CrossEntropyLoss(
                    weight=class_weights, label_smoothing=cfg.training.get("label_smoothing", 0.0)
                )
            else:
                logger.info("Using CrossEntropyLoss without class weights.")
                criterion = nn.CrossEntropyLoss(label_smoothing=cfg.training.get("label_smoothing", 0.0))

    elif loss_type == "mse":
        logger.info("Using MSELoss.")
        criterion = nn.MSELoss()

    elif loss_type == "focal":
        logger.info("Using FocalLoss.")
        criterion = FocalLoss(
            num_classes=effective_num_classes,
            alpha=cfg.training.get("focal_alpha", 1),
            gamma=cfg.training.get("focal_gamma", 2),
        )
    elif loss_type == "sosr":
        logger.info("Using SOSRLoss.")
        if hasattr(cfg.training, "cost_matrix") and cfg.training.cost_matrix is not None:
            cost_matrix = torch.tensor(cfg.training.cost_matrix, device=device, dtype=torch.float32)
        else:
            if effective_num_classes == 3:
                cost_matrix = torch.tensor(
                    [
                        [0.0, 1.0, 1.0],  # Costs for true class 0
                        [1.0, 0.0, 1.0],  # Costs for true class 1
                        [1.0, 1.0, 0.0],  # Costs for true class 2
                    ],
                    device=device,
                    dtype=torch.float32,
                )
            elif effective_num_classes == 2:
                cost_matrix = torch.tensor(
                    [
                        [0.0, 1.0],  # Costs for true class 0
                        [1.0, 0.0],  # Costs for true class 1 (Positive)
                    ],
                    device=device,
                    dtype=torch.float32,
                )
        criterion = SOSRLoss(cost_matrix=cost_matrix)
    else:
        raise ValueError(f"Unknown loss type: {cfg.training.loss_type}")

    return criterion


def setup_optimizer(cfg, model):
    if cfg.training.use_weight_decay:
        return AdamW(
            model.parameters(),
            lr=cfg.optimizer.lr,
            weight_decay=cfg.optimizer.weight_decay,
        )
    else:
        return Adam(model.parameters(), lr=cfg.optimizer.lr)


def setup_scheduler(cfg, optimizer):
    if cfg.training.use_lr_scheduler:
        return CosineAnnealingLR(optimizer, T_max=cfg.training.max_epochs, eta_min=cfg.optimizer.eta_min)
    return None


def print_model_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    # ANSI escape codes for colors
    purple = "\033[95m"
    red = "\033[91m"
    reset = "\033[0m"

    logger.info(f"{purple}Total parameters: {red}{total_params}{reset}")
    logger.info(f"{purple}Trainable parameters: {red}{trainable_params}{reset}")
    logger.info(f"{purple}Non-trainable parameters: {red}{non_trainable_params}{reset}")


def get_all_layers(model):
    return [module for name, module in model.named_modules() if len(list(module.children())) == 0]


def get_unfrozen_layers(model, epoch, cfg):
    all_layers = get_all_layers(model)
    num_layers = len(all_layers) - 1  # -1 to exclude the final layer (head)

    if cfg.training.unfreeze_strategy == "fixed":
        layers_to_unfreeze = min(epoch // cfg.training.unfreeze_epoch_interval + 1, num_layers)
    elif cfg.training.unfreeze_strategy == "adaptive":
        if epoch == 0:
            layers_to_unfreeze = 1  # unfreeze 1st layer
        else:
            layers_to_unfreeze = math.ceil(math.log(epoch * cfg.training.unfreeze_adaptive_decay_rate + 1, 2))

    num_layers_to_unfreeze = min(layers_to_unfreeze, num_layers)
    return num_layers_to_unfreeze


def unfreeze_layers(model, epoch, cfg):
    all_layers = get_all_layers(model)
    num_layers_to_unfreeze = get_unfrozen_layers(model, epoch, cfg)

    for i, layer in enumerate(reversed(all_layers[:-1])):
        if i < num_layers_to_unfreeze:
            for param in layer.parameters():
                param.requires_grad = True
        else:
            break

    return num_layers_to_unfreeze


def load_model_weights(model, weights_path):
    logger.info(f"Loading model weights from: {weights_path}")
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))


def map_labels_to_class_indices(labels, class_idx):
    """
    Map original labels to new labels based on selected class indices.

    Args:
        labels: Original labels
        class_idx: List of class indices to keep (e.g., [0, 1, 2] or [0, 2])

    Returns:
        mapped_labels: New labels mapped to range [0, len(class_idx)-1]
        valid_mask: Boolean mask indicating which labels are valid
    """
    labels = np.array(labels)
    valid_mask = np.isin(labels, class_idx)

    # Create mapping from original indices to new indices
    idx_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(class_idx)}

    # Map labels
    mapped_labels = np.zeros_like(labels)
    for old_idx, new_idx in idx_mapping.items():
        mapped_labels[labels == old_idx] = new_idx

    return mapped_labels, valid_mask


def filter_predictions_and_probs(predictions, probs, labels, class_idx):
    """
    Filter predictions and probabilities based on selected class indices.

    Args:
        predictions: Model predictions
        probs: Model probabilities
        labels: True labels
        class_idx: List of class indices to keep

    Returns:
        filtered_predictions, filtered_probs, mapped_labels, valid_mask
    """
    # Map labels to new indices
    mapped_labels, valid_mask = map_labels_to_class_indices(labels, class_idx)

    # Filter probabilities to only include selected classes
    if probs.ndim > 1 and probs.shape[1] > len(class_idx):
        filtered_probs = probs[:, class_idx]
    else:
        filtered_probs = probs

    # Re-compute predictions from filtered probabilities
    if len(class_idx) == 2 and filtered_probs.shape[1] == 1:
        # Binary case with single output
        filtered_predictions = (filtered_probs > 0.5).astype(int).flatten()
    else:
        # Multi-class case
        filtered_predictions = np.argmax(filtered_probs, axis=1)

    return filtered_predictions, filtered_probs, mapped_labels, valid_mask


def merge_classes_by_mapping(labels, class_merge_mapping):
    """
    Merge classes according to a mapping dictionary.

    Args:
        labels: Original labels (numpy array or tensor)
        class_merge_mapping: Dict mapping new_class -> [old_classes]
                           e.g., {0: [0], 1: [1, 2]} means:
                           - old class 0 becomes new class 0
                           - old classes 1,2 become new class 1

    Returns:
        merged_labels: New labels after merging
    """
    labels = np.array(labels)
    merged_labels = np.zeros_like(labels)

    for new_class, old_classes in class_merge_mapping.items():
        for old_class in old_classes:
            merged_labels[labels == old_class] = new_class

    return merged_labels


def should_merge_classes(cfg):
    """
    Check if class merging should be applied based on configuration.

    Args:
        cfg: Configuration object

    Returns:
        bool: True if classes should be merged
    """
    return getattr(cfg.data, "merge_classes_for_binary", False)


def get_effective_class_setup(cfg, num_classes):
    """
    Get the effective class setup considering merging and filtering.

    Args:
        cfg: Configuration object
        num_classes: Original number of classes

    Returns:
        tuple: (effective_num_classes, class_idx, merge_mapping)
    """
    # Check if we should merge classes first
    if should_merge_classes(cfg):
        merge_mapping = getattr(cfg.data, "class_merge_mapping", {0: [0], 1: [1, 2]})
        effective_num_classes = len(merge_mapping)
        class_idx = list(range(effective_num_classes))
        return effective_num_classes, class_idx, merge_mapping
    else:
        # Use the existing class_idx filtering
        class_idx = getattr(cfg.data, "class_idx", list(range(num_classes)))
        effective_num_classes = len(class_idx)
        return effective_num_classes, class_idx, None


def process_labels_for_training(labels, cfg, num_classes):
    """
    Process labels according to configuration (merging and/or filtering).

    Args:
        labels: Original labels
        cfg: Configuration object
        num_classes: Original number of classes

    Returns:
        tuple: (processed_labels, valid_mask, effective_num_classes)
    """
    labels = np.array(labels)

    # Step 1: Merge classes if required
    if should_merge_classes(cfg):
        merge_mapping = getattr(cfg.data, "class_merge_mapping", {0: [0], 1: [1, 2]})
        labels = merge_classes_by_mapping(labels, merge_mapping)
        # Update num_classes after merging
        effective_num_classes = len(merge_mapping)
    else:
        effective_num_classes = num_classes

    # Step 2: Filter classes if class_idx is specified and different from full range
    class_idx = getattr(cfg.data, "class_idx", list(range(effective_num_classes)))

    if set(class_idx) != set(range(effective_num_classes)):
        # Apply class filtering
        mapped_labels, valid_mask = map_labels_to_class_indices(labels, class_idx)
        return mapped_labels, valid_mask, len(class_idx)
    else:
        # No filtering needed, all samples are valid
        valid_mask = np.ones(len(labels), dtype=bool)
        return labels, valid_mask, effective_num_classes
