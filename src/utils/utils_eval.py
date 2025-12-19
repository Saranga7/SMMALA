import os
import numpy as np
import logging
import wandb
import torch
import torch.nn as nn
import torchvision.models as models
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from tqdm import tqdm
from pathlib import Path

from src.models.cnn import CustomCNN
from src.utils.utils_visualization import (
    plot_embedding_space,
    plot_all_embedding_spaces,
)

logger = logging.getLogger(__name__)


def calculate_auc_scores(all_labels, all_probs, num_classes=3):
    auc_scores = []
    if num_classes == 2:
        # Ensure probabilities are 1D (class 1)
        if all_probs.ndim == 2 and all_probs.shape[1] == 2:
            all_probs = all_probs[:, 1]  # Extract positive class
        else:
            all_probs = all_probs.flatten()

        if len(np.unique(all_labels)) < 2:
            logger.warning("AUC undefined: Only one class present.")
            auc_scores = [float("nan")]
        else:
            try:
                auc = roc_auc_score(all_labels, all_probs)
                auc_scores = [auc]
            except Exception as e:
                logger.error(f"AUC Error: {e}")
                auc_scores = [float("nan")]
    else:
        for cls in range(num_classes):
            cls_probs = all_probs[:, cls] if all_probs.ndim > 1 else all_probs
            try:
                auc = roc_auc_score((all_labels == cls).astype(int), cls_probs)
                auc_scores.append(auc)
            except ValueError:
                logger.warning(f"Class {cls}: AUC undefined.")
                auc_scores.append(float("nan"))
    return auc_scores


def calculate_specificity(labels, preds, num_classes):
    """
    Calculate specificity for binary and multiclass classification.

    Args:
        labels (np.ndarray): Ground truth labels, shape [n_samples].
        preds (np.ndarray): Predicted labels, shape [n_samples].
        num_classes (int): Number of classes.

    Returns:
        float: Weighted specificity for multiclass, or specificity for binary classification.
    """
    if num_classes == 2:
        tn = np.sum((labels == 0) & (preds == 0))
        fp = np.sum((labels == 0) & (preds == 1))
        return tn / (tn + fp) if (tn + fp) > 0 else 0.0

    specificities = []
    weights = []
    total_negatives = 0

    for i in range(num_classes):
        binary_labels = labels == i
        binary_preds = preds == i

        tn = np.sum(~binary_labels & ~binary_preds)
        fp = np.sum(~binary_labels & binary_preds)
        tn_fp = tn + fp

        spec = tn / tn_fp if tn_fp > 0 else 0.0
        specificities.append(spec)

        weights.append(tn_fp)
        total_negatives += tn_fp

    # edge case: no negatives across all classes
    if total_negatives == 0:
        return 0.0

    weights = np.array(weights) / total_negatives
    weighted_specificity = np.sum(np.array(specificities) * weights)

    return weighted_specificity


def calculate_metrics(labels, preds, probs, num_classes, all_original_labels, class_idx=None, cfg=None):
    """
    Calculate metrics with support for class index filtering and class merging.

    Args:
        labels: True labels
        preds: Predictions
        probs: Probabilities
        num_classes: Original number of classes
        all_original_labels: Original labels before any mapping
        class_idx: List of class indices being used (e.g., [0,1] or [0,1,2])
        cfg: Configuration object for handling class merging
    """

    # Process labels/preds/probs according to configuration
    if cfg is not None:
        processed_labels, processed_preds, processed_probs, effective_num_classes, merge_mapping = (
            process_predictions_for_metrics(all_original_labels, preds, probs, cfg, num_classes)
        )

        # For confusion matrix in merge case: keep original labels (rows) vs merged predictions (cols)
        if merge_mapping is not None:
            # Create custom confusion matrix for merge case
            conf_matrix = create_confusion_matrix_with_merge(
                all_original_labels,  # Original labels (0, 1, 2)
                processed_preds,  # Merged predictions (0, 1)
                num_classes,  # Number of original classes (3)
                merge_mapping,  # Merge mapping
            )

            # For other metrics calculations, we need to merge the labels too
            from src.utils.utils_model import merge_classes_by_mapping

            metrics_labels = merge_classes_by_mapping(all_original_labels, merge_mapping)
        else:
            # No merging, standard confusion matrix for filtering case
            conf_matrix = confusion_matrix(
                processed_labels, processed_preds, labels=list(range(effective_num_classes))
            )
            metrics_labels = processed_labels

        # Use processed labels/preds for other metrics calculation
        labels = metrics_labels  # Use merged labels for accuracy, precision, etc.
        preds = processed_preds
        probs = processed_probs

    elif class_idx is not None:
        # Legacy mode: apply class filtering
        labels, preds, probs, valid_mask = apply_class_filtering(labels, preds, probs, all_original_labels, class_idx)
        effective_num_classes = len(class_idx)

        # Standard confusion matrix for filtering case
        conf_matrix = confusion_matrix(labels, preds, labels=list(range(effective_num_classes)))
        original_classes = class_idx
    else:
        effective_num_classes = num_classes
        original_classes = list(range(num_classes))

        # Standard confusion matrix
        conf_matrix = confusion_matrix(labels, preds, labels=original_classes)

    # Metrics based on mapped labels
    try:
        metrics = {
            "accuracy": accuracy_score(labels, preds),
            "accuracy_weighted": balanced_accuracy_score(labels, preds),
            "specificity_weighted": calculate_specificity(labels, preds, effective_num_classes),
            "precision": precision_score(labels, preds, zero_division=0),
            "recall": recall_score(labels, preds, zero_division=0),
            "f1": f1_score(labels, preds, zero_division=0),
            "f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0),
            "auc": calculate_auc_scores(labels, probs, effective_num_classes),
            "confusion_matrix": conf_matrix,
        }
    except ValueError as e:
        metrics = {
            "accuracy": accuracy_score(labels, preds),
            "accuracy_weighted": balanced_accuracy_score(labels, preds),
            "specificity_weighted": calculate_specificity(labels, preds, effective_num_classes),
            "precision": precision_score(labels, preds, average="macro", zero_division=0),
            "recall": recall_score(labels, preds, average="macro", zero_division=0),
            "f1": f1_score(labels, preds, average="macro", zero_division=0),
            "f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0),
            "auc": calculate_auc_scores(labels, probs, effective_num_classes),
            "confusion_matrix": conf_matrix,
        }


    # Add legacy metrics for backward compatibility
    if effective_num_classes == 2:
        metrics.update(
            {
                "accuracy_2classes": metrics["accuracy"],
                "sensitivity_2classes": metrics["recall"],  # recall is sensitivity
                "specificity_2classes": metrics["specificity_weighted"],
                "f1_2classes": metrics["f1"],
                "precision_2classes": metrics["precision"],
            }
        )
    elif effective_num_classes == 3:
        metrics.update(
            {
                "accuracy_3classes": metrics["accuracy"],
                "sensitivity_3classes": metrics["recall"],
                "specificity_3classes": metrics["specificity_weighted"],
                "f1_3classes": metrics["f1"],
                "precision_3classes": metrics["precision"],
            }
        )

    return metrics


def calculate_metrics_by_slide(cfg, labels, preds, probs, slide_ids, num_classes, all_original_labels, class_idx=None):
    """
    Calculate metrics aggregated at the slide level, supporting dynamic class indices.

    Args:
        cfg: Configuration object
        labels: Image-level labels (mapped)
        preds: Image-level predictions
        probs: Image-level probabilities
        slide_ids: Slide IDs for each image
        num_classes: Original number of classes
        all_original_labels: Original labels before mapping
        class_idx: List of class indices to use (e.g., [0, 1] for binary, [0, 1, 2] for 3-class)

    Returns:
        dict: Slide-level metrics
    """
    # If class_idx is not provided, use all classes
    if class_idx is None:
        class_idx = list(range(num_classes))

    effective_num_classes = len(class_idx)

    unique_slides = np.unique(slide_ids)
    slide_labels, slide_preds, slide_probs, slide_original_labels = [], [], [], []

    # Ensure 2D probabilities for aggregation
    if effective_num_classes == 2:
        if probs.ndim == 1:
            probs = np.column_stack((1 - probs, probs))
        elif probs.shape[1] == 1:
            probs = np.column_stack((1 - probs.flatten(), probs.flatten()))

    for slide in unique_slides:
        slide_mask = slide_ids == slide
        if np.sum(slide_mask) == 0:
            continue  # Skip empty slides

        # True label (mapped for binary)
        slide_labels.append(labels[slide_mask][0])
        slide_original_labels.append(all_original_labels[slide_mask][0])

        # Aggregation method
        if cfg.model.slide_aggregator_method == "mean":
            slide_prob = np.mean(probs[slide_mask], axis=0)
            slide_pred = np.argmax(slide_prob)

        elif cfg.model.slide_aggregator_method == "median":
            slide_prob = np.median(probs[slide_mask], axis=0)
            slide_pred = np.argmax(slide_prob)

        else:  # Majority voting
            # Convert tile preds to votes
            tile_preds = preds[slide_mask].astype(int)
            class_counts = np.bincount(tile_preds, minlength=effective_num_classes)
            slide_pred = np.argmax(class_counts)
            # Probabilities = vote proportions
            slide_prob = class_counts / len(tile_preds)

        slide_probs.append(slide_prob)
        slide_preds.append(slide_pred)

    # Convert to arrays
    slide_labels = np.array(slide_labels)
    slide_preds = np.array(slide_preds)
    slide_probs = np.array(slide_probs)
    slide_original_labels = np.array(slide_original_labels)

    # Use the adapted calculate_metrics function for consistency
    slide_metrics = calculate_metrics(
        slide_labels, slide_preds, slide_probs, num_classes, slide_original_labels, class_idx=class_idx, cfg=cfg
    )

    # Add legacy metrics for backward compatibility
    if effective_num_classes == 2:
        # For binary classification, add specific 2-class metrics
        slide_metrics.update(
            {
                "accuracy_2classes": slide_metrics["accuracy"],
                "sensitivity_2classes": slide_metrics["recall"],  # recall is sensitivity
                "specificity_2classes": slide_metrics["specificity_weighted"],
                "f1_2classes": slide_metrics["f1"],
                "precision_2classes": slide_metrics["precision"],
            }
        )
    elif effective_num_classes == 3:
        # For 3-class classification, add specific 3-class metrics
        slide_metrics.update(
            {
                "accuracy_3classes": slide_metrics["accuracy"],
                "sensitivity_3classes": slide_metrics["recall"],
                "specificity_3classes": slide_metrics["specificity_weighted"],
                "f1_3classes": slide_metrics["f1"],
                "precision_3classes": slide_metrics["precision"],
            }
        )

    return slide_metrics


def extract_image_features(model, inputs, device, before_last_layer=False):
    inputs = inputs.to(device)

    
    base_model = model

    if isinstance(base_model, CustomCNN):
        features = base_model.features_extraction(inputs)
    elif isinstance(base_model, models.ResNet):
        features = base_model.avgpool(
            base_model.layer4(
                base_model.layer3(
                    base_model.layer2(
                        base_model.layer1(
                            base_model.maxpool(base_model.relu(base_model.bn1(base_model.conv1(inputs))))
                        )
                    )
                )
            )
        )
        features = torch.flatten(features, 1)
    elif "dino" in base_model.__class__.__name__.lower():
        features_dict = base_model.forward_features(inputs)
        features = features_dict["x_norm_clstoken"]
    elif hasattr(base_model, "forward_features") and "dino" not in base_model.__class__.__name__.lower():
        features = base_model.forward_features(inputs)

    else:
        features = base_model(inputs)

    if before_last_layer:
        if isinstance(base_model, models.ResNet):
            features = features  # ResNet features are already extracted before `fc`
        elif hasattr(base_model, "head") and isinstance(base_model.head, nn.Sequential):
            # If the head is a sequential model, remove the last linear layer(s)
            head_layers = list(base_model.head.children())[:-1]
            head_without_last = nn.Sequential(*head_layers)
            features = head_without_last(features)
        elif hasattr(base_model, "fc") and isinstance(base_model.fc, nn.Sequential):
            # Similar for ResNet if the `fc` is a sequential head
            fc_layers = list(base_model.fc.children())[:-1]
            fc_without_last = nn.Sequential(*fc_layers)
            features = fc_without_last(features)

    if isinstance(features, dict):
        features = next(iter(features.values()))

    if len(features.shape) > 2:
        features = features.mean([2, 3])  # Global average pooling for 2D features

    return features


def get_embeddings(model, dataloader, device, metadata_filepath=None, get_slide_embeddings=False):
    model.eval()
    embeddings = []
    labels = []
    etudes = []
    age_groups = []
    genders = []
    fevers = []

    if metadata_filepath:
        metadata_df = pd.read_csv(metadata_filepath)

        # Create lookup dictionaries from slide ID to each attribute
        slide_to_etude = dict(zip(metadata_df["NUM DOSS"], metadata_df["Etude"]))
        slide_to_age_group = dict(zip(metadata_df["NUM DOSS"], metadata_df["age_group"]))
        slide_to_gender = dict(zip(metadata_df["NUM DOSS"], metadata_df["gender"]))
        slide_to_fever = dict(zip(metadata_df["NUM DOSS"], metadata_df["fever"]))

    with torch.no_grad():
        if get_slide_embeddings:
            for batch in tqdm(dataloader, desc="Embeddings", leave=False):
                inputs = batch[0].to(device)
                batch_labels = batch[1]
                slide_ids = batch[2] if len(batch) == 5 else batch[3]

                features = model.extract_features(inputs)

                for feat, label, slide_id in zip(features, batch_labels, slide_ids):
                    embeddings.append(feat.cpu().numpy())
                    labels.append(label.item())
                    if metadata_filepath:
                        etudes.append(slide_to_etude.get(slide_id, None))
                        age_groups.append(slide_to_age_group.get(slide_id, None))
                        genders.append(slide_to_gender.get(slide_id, None))
                        fevers.append(slide_to_fever.get(slide_id, None))
        else:
            for batch in tqdm(dataloader, desc="Embeddings", leave=False):
                inputs = batch[0].to(device)
                batch_labels = batch[1]
                slide_ids = batch[2] if len(batch) == 5 else batch[3]

                features = extract_image_features(model, inputs, device)
                embeddings.extend(features.cpu().numpy())
                labels.extend(batch_labels.numpy())
                if metadata_filepath:
                    for slide_id in slide_ids:
                        etudes.append(slide_to_etude.get(slide_id, None))
                        age_groups.append(slide_to_age_group.get(slide_id, None))
                        genders.append(slide_to_gender.get(slide_id, None))
                        fevers.append(slide_to_fever.get(slide_id, None))

    return (
        np.array(embeddings),
        np.array(labels),
        np.array(etudes) if metadata_filepath else None,
        np.array(age_groups) if metadata_filepath else None,
        np.array(genders) if metadata_filepath else None,
        np.array(fevers) if metadata_filepath else None,
    )


def perform_analysis(
    cfg,
    model,
    train_loader,
    val_loader,
    device,
    epoch,
    train_conf_over_time,
    val_conf_over_time,
    val_metrics,
):
    """Analysis function remains largely the same but adds handling for slide-level embeddings"""

    (
        train_embeddings,
        train_labels,
        train_etudes,
        train_age_groups,
        train_genders,
        train_fevers,
    ) = get_embeddings(
        model,
        train_loader,
        device,
        metadata_filepath=Path(cfg.data.root_dir) / Path(cfg.data.metadata_filepath),
        get_slide_embeddings=cfg.data.data_collection_method == "slide",
    )

    (
        val_embeddings,
        val_labels,
        val_etudes,
        val_age_groups,
        val_genders,
        val_fevers,
    ) = get_embeddings(
        model,
        val_loader,
        device,
        metadata_filepath=Path(cfg.data.root_dir) / Path(cfg.data.metadata_filepath),
        get_slide_embeddings=cfg.data.data_collection_method == "slide",
    )

    embed_space_fig_labels = plot_embedding_space(train_embeddings, train_labels, val_embeddings, val_labels, epoch)
    embed_space_fig_etudes = plot_embedding_space(train_embeddings, train_etudes, val_embeddings, val_etudes, epoch)

    embed_space_fig_all = plot_all_embedding_spaces(
        train_embeddings,
        val_embeddings,
        train_labels,
        val_labels,
        train_etudes,
        val_etudes,
        train_age_groups,
        val_age_groups,
        train_genders,
        val_genders,
        train_fevers,
        val_fevers,
        epoch,
    )

    if cfg.wandb.enabled and not cfg.slurm.enabled:
        wandb.log(
            {
                "embedding_space_labels_epoch": wandb.Image(embed_space_fig_labels),
                "embedding_space_etudes_epoch": wandb.Image(embed_space_fig_etudes),
                "embedding_space_all_epoch": wandb.Image(embed_space_fig_all),
            }
        )
    elif cfg.slurm.enabled:
        image_folder = os.path.join(cfg.training.log_dir, "log_images")
        os.makedirs(image_folder, exist_ok=True)

        embed_space_fig_labels.savefig(os.path.join(image_folder, f"embedding_space_labels_epoch_{epoch}.png"))
        embed_space_fig_etudes.savefig(os.path.join(image_folder, f"embedding_space_etudes_epoch_{epoch}.png"))

        logger.info(f"Analysis plots saved to: {image_folder}")

    plt.close(embed_space_fig_labels)
    plt.close(embed_space_fig_etudes)
    plt.close(embed_space_fig_all)


def apply_class_filtering(labels, preds, probs, original_labels, class_idx):
    """
    Apply class filtering based on selected class indices.

    Args:
        labels: Labels (possibly already mapped)
        preds: Predictions
        probs: Probabilities
        original_labels: Original unfiltered labels
        class_idx: List of class indices to keep

    Returns:
        filtered_labels, filtered_preds, filtered_probs, valid_mask
    """
    if class_idx is None:
        return labels, preds, probs, np.ones(len(labels), dtype=bool)

    # Import the utility function
    from src.utils.utils_model import map_labels_to_class_indices, filter_predictions_and_probs

    # Map original labels to new indices and get valid mask
    mapped_labels, valid_mask = map_labels_to_class_indices(original_labels, class_idx)

    # Filter predictions and probabilities
    filtered_preds, filtered_probs, _, _ = filter_predictions_and_probs(
        preds[valid_mask], probs[valid_mask], original_labels[valid_mask], class_idx
    )

    return mapped_labels[valid_mask], filtered_preds, filtered_probs, valid_mask


def apply_consistent_class_filtering(labels, preds, probs, class_idx):
    """
    Apply consistent filtering across labels, predictions, and probabilities.

    Args:
        labels: Original labels
        preds: Predictions
        probs: Probabilities
        class_idx: List of class indices to keep

    Returns:
        filtered_labels, filtered_preds, filtered_probs
    """
    labels = np.array(labels)
    preds = np.array(preds)
    probs = np.array(probs)

    # Create mask for samples that have labels in class_idx
    valid_mask = np.isin(labels, class_idx)

    # Filter all arrays using the same mask
    filtered_labels = labels[valid_mask]
    filtered_preds = preds[valid_mask]
    filtered_probs = probs[valid_mask]

    # Map filtered labels to new indices [0, 1, ...]
    label_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(class_idx)}

    mapped_labels = np.array([label_mapping[label] for label in filtered_labels])

    # For binary case, also remap predictions if needed
    if preds.max() >= len(class_idx):
        # Predictions might be in original space, remap them
        mapped_preds = np.array([label_mapping.get(pred, pred) for pred in filtered_preds])
    else:
        mapped_preds = filtered_preds

    # For probabilities, select only relevant columns if needed
    if probs.ndim > 1 and probs.shape[1] > len(class_idx):
        filtered_probs = filtered_probs[:, class_idx]

    return mapped_labels, mapped_preds, filtered_probs


def process_predictions_for_metrics(labels, preds, probs, cfg, num_classes):
    """
    Process predictions and labels for metrics calculation, handling class merging.

    Args:
        labels: Original labels
        preds: Original predictions
        probs: Original probabilities
        cfg: Configuration object
        num_classes: Original number of classes

    Returns:
        tuple: (processed_labels, processed_preds, processed_probs, effective_num_classes, class_mapping)
    """
    from src.utils.utils_model import should_merge_classes, merge_classes_by_mapping

    labels = np.array(labels)
    preds = np.array(preds)
    probs = np.array(probs)

    # Apply class merging if configured
    if should_merge_classes(cfg):
        merge_mapping = getattr(cfg.data, "class_merge_mapping", {0: [0], 1: [1, 2]})

        # For merge case: keep original labels but merge predictions
        # This allows for confusion matrix: rows=original classes (3), cols=merged predictions (2)

        # DON'T merge labels - keep them original for confusion matrix
        processed_labels = labels

        # For predictions, we need to handle the case where they might already be in the merged space
        # or if they're in the original space and need merging
        if preds.max() >= len(merge_mapping):
            # Predictions are in original space, need merging
            merged_preds = merge_classes_by_mapping(preds, merge_mapping)
        else:
            # Predictions are already in merged space
            merged_preds = preds

        effective_num_classes = len(merge_mapping)

        # For probabilities, if they have more classes than effective, we need to merge them
        if probs.ndim > 1 and probs.shape[1] > effective_num_classes:
            merged_probs = np.zeros((len(probs), effective_num_classes))
            for new_class, old_classes in merge_mapping.items():
                merged_probs[:, new_class] = probs[:, old_classes].sum(axis=1)
        else:
            merged_probs = probs

        return processed_labels, merged_preds, merged_probs, effective_num_classes, merge_mapping
    else:
        # No merging, just apply regular class filtering if needed
        class_idx = getattr(cfg.data, "class_idx", list(range(num_classes)))
        if set(class_idx) != set(range(num_classes)):
            # Apply consistent filtering
            mapped_labels, filtered_preds, filtered_probs = apply_consistent_class_filtering(
                labels, preds, probs, class_idx
            )
            return mapped_labels, filtered_preds, filtered_probs, len(class_idx), None
        else:
            return labels, preds, probs, num_classes, None


def create_confusion_matrix_with_merge(true_labels, pred_labels, num_original_classes, merge_mapping):
    """
    Create a confusion matrix for the merge case where we want:
    - Rows: original classes (e.g., [0, 1, 2])
    - Columns: merged predictions (e.g., [0, 1])

    Args:
        true_labels: Original true labels
        pred_labels: Merged predictions
        num_original_classes: Number of original classes
        merge_mapping: Mapping of merged classes to original classes

    Returns:
        Confusion matrix of shape (num_original_classes, len(merge_mapping))
    """
    # Create the matrix with proper dimensions
    n_merged_classes = len(merge_mapping)
    conf_matrix = np.zeros((num_original_classes, n_merged_classes), dtype=int)

    # Fill the matrix manually
    for i in range(num_original_classes):
        mask = true_labels == i
        if mask.sum() > 0:
            preds_for_class_i = pred_labels[mask]
            for j in range(n_merged_classes):
                conf_matrix[i, j] = (preds_for_class_i == j).sum()

    return conf_matrix
