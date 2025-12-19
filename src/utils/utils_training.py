from pathlib import Path
from typing import Optional, Tuple, Any
import logging
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import nn

# Import utility functions
from src.common.losses.focal import FocalLoss
from src.common.losses.sosr import SOSRLoss
from src.utils.utils_eval import extract_image_features
from src.models.slide_aggregator import SlideAggregator
from src.utils.utils_model import should_merge_classes, merge_classes_by_mapping

# Initialize logger
logger = logging.getLogger(__name__)


def process_batch(
    model: torch.nn.Module,
    batch: Tuple[Any],
    device: torch.device,
    criterion: torch.nn.Module,
    num_classes: int,
    compute_feats: bool = False,
    temperature_layer: Optional[torch.Tensor] = None,
    class_idx: Optional[list] = None,
    cfg: Optional[DictConfig] = None,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Any,
    torch.Tensor,
]:
    """
    Process a single batch of data.

    Args:
        model (torch.nn.Module): The model to evaluate.
        batch (Tuple[Any]): A batch of data.
        device (torch.device): The device to run computations on.
        criterion (torch.nn.Module): The loss function.
        num_classes (int): Number of classes.
        class_idx (list, optional): List of class indices being used.
        compute_feats (bool): Whether to compute features.
        temperature_layer (Optional[torch.Tensor]): Temperature layer.

    Returns:
        Tuple containing loss, outputs, feats, probs, preds, labels, original_labels, slide_ids, indices.
    """


    if cfg.model.use_imgs_or_embeddings == "embeddings":
    
        (
            inputs,
            labels,
            slide_ids,
            original_labels,
            indices,
            embeddings,
        ) = batch

        # inputs = embeddings.to(device)
    else:
        
        inputs, labels, slide_ids, original_labels, indices = batch

    if cfg.model.use_imgs_or_embeddings == "embeddings":
        inputs = embeddings.to(device)
    else:
        inputs = inputs.to(device)
    labels = labels.to(device)
    original_labels = original_labels.to(device)

    # Process labels according to configuration (merging only, filtering is done in dataset)
    if cfg is not None and should_merge_classes(cfg):
        # Convert to numpy for processing, then back to tensor
        labels_np = labels.cpu().numpy()
        merge_mapping = getattr(cfg.data, "class_merge_mapping", {0: [0], 1: [1, 2]})
        merged_labels_np = merge_classes_by_mapping(labels_np, merge_mapping)
        labels = torch.tensor(merged_labels_np, device=device, dtype=labels.dtype)

    feats = None
    
    with torch.no_grad():
        if compute_feats and cfg.model.use_imgs_or_embeddings != "embeddings":
            feats = (
                model.extract_features(inputs)
                if isinstance(model, SlideAggregator)
                else extract_image_features(model, inputs, device, before_last_layer=True)
            )
            feats = feats.detach().cpu().numpy()
    outputs = model(inputs)

    # calibrate
    outputs = temperature_layer(outputs) if temperature_layer is not None else outputs

    loss, probs, preds = get_loss_and_prediction(criterion, outputs, labels, num_classes, class_idx, cfg)

    return (
        loss,
        outputs,
        feats,
        probs,
        preds,
        labels,
        original_labels,
        slide_ids,
        indices,
    )


def get_loss_and_prediction(
    criterion: torch.nn.Module,
    outputs: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    class_idx: Optional[list] = None,
    cfg: Optional[DictConfig] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute loss, probabilities, and predictions based on the criterion.

    Args:
        criterion (torch.nn.Module): Loss function.
        outputs (torch.Tensor): Model outputs.
        labels (torch.Tensor): Ground truth labels.
        num_classes (int): Number of classes.
        class_idx (list, optional): List of class indices being used.
        cfg (DictConfig, optional): Configuration object for determining effective classes.

    Returns:
        Tuple containing loss, probabilities, and predictions.
    """
    # Use get_effective_class_setup if cfg is available, otherwise fallback to class_idx
    if cfg is not None:
        from src.utils.utils_model import get_effective_class_setup

        effective_num_classes, _, _ = get_effective_class_setup(cfg, num_classes)
    else:
        effective_num_classes = len(class_idx) if class_idx is not None else num_classes

    if isinstance(criterion, torch.nn.MSELoss):
        if effective_num_classes == 2:
            labels = labels.float().unsqueeze(1)
            loss = criterion(outputs, labels)
            probs_sigmoid = torch.sigmoid(outputs)
            # Convert 1D probabilities to 2D for consistency with the rest of the pipeline
            probs = torch.cat([1 - probs_sigmoid, probs_sigmoid], dim=1)
            preds = (outputs > 0.5).int().squeeze()
        else:
            labels_one_hot = F.one_hot(labels, num_classes=effective_num_classes).float()
            probs = F.softmax(outputs, dim=1)
            loss = criterion(outputs, labels_one_hot)
            preds = torch.argmax(outputs, dim=1)

    elif isinstance(criterion, (torch.nn.CrossEntropyLoss, torch.nn.BCEWithLogitsLoss)):
        if effective_num_classes == 2:
            loss = criterion(outputs.squeeze(), labels.float())
            probs_sigmoid = torch.sigmoid(outputs)
            # Convert 1D probabilities to 2D for consistency with the rest of the pipeline
            probs = torch.cat([1 - probs_sigmoid, probs_sigmoid], dim=1)
            preds = (probs_sigmoid > 0.5).int().squeeze()
        else:
            loss = criterion(outputs, labels)
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

    elif isinstance(criterion, FocalLoss):
        loss = criterion(outputs, labels)
        probs = F.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)

    elif isinstance(criterion, SOSRLoss):
        loss = criterion(outputs, labels)
        probs = criterion.compute_risks_and_probs(outputs, cost_matrix=criterion.cost_matrix)
        preds = torch.argmax(probs, dim=1)

    else:
        raise NotImplementedError(f"Criterion {criterion} not implemented.")

    return loss, probs, preds




def save_best_model(
    model: torch.nn.Module,
    current_accuracy: float,
    best_accuracy: float,
    cfg: DictConfig,
):
    """
    Save the model if the current accuracy is better than the best accuracy.

    Args:
        model (torch.nn.Module): The model to save.
        current_accuracy (float): Current validation accuracy.
        best_accuracy (float): Best validation accuracy so far.
        cfg (DictConfig): Configuration object.

    Returns:
        float: Updated best accuracy.
    """
    if current_accuracy > best_accuracy:
        best_accuracy = current_accuracy
        save_dir = Path(cfg.model.weights_path) / "custom"
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            model.state_dict(),
            save_dir / cfg.training.weights_filename,
        )
        logger.info(f"Best model saved with accuracy: {best_accuracy:.4f}")
    return best_accuracy
