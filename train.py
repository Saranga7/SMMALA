import logging
from pathlib import Path
from typing import Optional, Dict, Any
import traceback

import time


import hydra
import torch
from omegaconf import DictConfig
from tqdm import tqdm
import numpy as np
import wandb
import matplotlib.pyplot as plt
import pandas as pd

from src.utils.utils_training import save_best_model, process_batch
from src.utils.utils_logging import (
    TqdmToLogger,
    log_metrics,
    send_logs_to_wandb,
)
from src.data.classification_dataloader import prepare_dataloaders
from src.utils.utils_data import prepare_datasets_and_transforms, get_test_dataloader
from src.utils.utils_config import setup_environment, setup_device
from src.utils.utils_eval import (
    calculate_metrics,
    calculate_metrics_by_slide,
    perform_analysis,
)
from src.utils.utils_model import (
    get_all_layers,
    unfreeze_layers,
    setup_seed,
    setup_model,
    setup_optimizer,
    setup_scheduler,
    setup_criterion,
    load_model_weights,
)

logger = logging.getLogger(__name__)


def train_and_validate(
    cfg: DictConfig,
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    device: torch.device,
    num_classes: int,
    class_idx: list = None,
) -> None:
    """
    Train and validate the model across epochs.

    Args:
        cfg (DictConfig): Configuration object.
        model (torch.nn.Module): The model to train and validate.
        train_loader (torch.utils.data.DataLoader): Training data loader.
        val_loader (torch.utils.data.DataLoader): Validation data loader.
        criterion (torch.nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        scheduler (Optional[torch.optim.lr_scheduler._LRScheduler]): Learning rate scheduler.
        device (torch.device): Device to run computations on.
        num_classes (int): Number of classes.
    """
    train_conf_over_time = []
    val_conf_over_time = []
    best_val_accuracy = 0.0
    num_unfrozen_layers = 0

    skip_validation = val_loader is None
    if skip_validation:
        logger.info("Validation skipped: training on full dataset without validation")

    tqdm_logger = TqdmToLogger(logger)
    for epoch in tqdm(range(cfg.training.max_epochs), desc="Epochs", file=tqdm_logger):
        if cfg.training.use_gradual_unfreezing:
            num_unfrozen_layers = unfreeze_layers(model, epoch, cfg)

        train_metrics = train_one_epoch(cfg, model, train_loader, criterion, optimizer, device, num_classes, class_idx)

        if not skip_validation:
            val_metrics = validate_one_epoch(
                cfg, model, val_loader, criterion, device, num_classes, class_idx=class_idx
            )
            val_metrics["num_frozen_layers"] = len(get_all_layers(model)) - num_unfrozen_layers
            log_metrics(val_metrics, epoch=epoch, phase="val", cfg=cfg, num_classes=num_classes)
        else:
            val_metrics = {"accuracy": np.nan, "accuracy_weighted": np.nan}

        log_metrics(
            {"lr": optimizer.param_groups[0]["lr"]},
            epoch,
            "learning_rate",
            cfg=cfg,
            num_classes=num_classes,
        )
        log_metrics(train_metrics, epoch=epoch, phase="train", cfg=cfg, num_classes=num_classes)

        if scheduler:
            scheduler.step()

        if epoch % cfg.training.analysis_interval == 0 and epoch > 0:
            if not skip_validation:
                perform_analysis(
                    cfg,
                    model,
                    train_loader,
                    val_loader,
                    device,
                    epoch,
                    train_conf_over_time,
                    val_conf_over_time,
                    val_metrics,
                )
            else:
                logger.info("Skipping analysis requiring validation data")

        if not skip_validation:
            best_val_accuracy = save_best_model(model, val_metrics.get("accuracy", 0.0), best_val_accuracy, cfg)
        else:
            best_val_accuracy = save_best_model(model, train_metrics.get("accuracy", 0.0), best_val_accuracy, cfg)

    logger.info("Training complete.")


def train_one_epoch(
    cfg: DictConfig,
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_classes: int,
    class_idx: list = None,
) -> Dict[str, float]:
    """
    Train the model for one epoch.

    Args:
        cfg (DictConfig): Configuration object.
        model (torch.nn.Module): The model to train.
        dataloader (torch.utils.data.DataLoader): Training data loader.
        criterion (torch.nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        device (torch.device): Hardware device to run computations on.
        num_classes (int): Number of classes.

    Returns:
        Dict[str, float]: Training metrics.
    """

    metadata_path = Path(cfg.data.root_dir) / Path(cfg.data.metadata_filepath)
    metadata_df = pd.read_csv(metadata_path)
    slide_to_etude = dict(zip(metadata_df["NUM DOSS"].astype(str), metadata_df["Etude"]))

    (
        running_loss,
        embeddings,
        logits,
        labels,
        slide_ids,
        probs,
        predictions,
        etudes,
        original_labels,
        correct_probs,
        incorrect_probs,
        indices,
    ) = one_epoch(
        model=model,
        cfg=cfg,
        device=device,
        dataloader=dataloader,
        optimizer=optimizer,
        criterion=criterion,
        num_classes=num_classes,
        slide_to_etude=slide_to_etude,
        mode="train",
        temperature_layer=None,
        return_embeddings=False,
        return_correct_incorrect_probs=False,
        class_idx=class_idx,
    )

    epoch_loss = running_loss / len(dataloader)
    metrics = calculate_metrics(
        np.array(labels),
        np.array(predictions),
        np.array(probs),
        num_classes,
        all_original_labels=np.array(original_labels),
        class_idx=class_idx,
        cfg=cfg,
    )

    if dataloader.dataset.use_monte_carlo:
        dataloader.dataset.on_epoch_end()

    return {
        "loss": epoch_loss,
        "accuracy": metrics.get("accuracy", 0.0),
        "accuracy_weighted": metrics.get("accuracy_weighted", 0.0),
        "precision": metrics.get("precision", 0.0),
        "recall": metrics.get("recall", 0.0),
        "specificity": metrics.get("specificity_weighted", 0.0),
        "f1": metrics.get("f1", 0.0),
        "f1_weighted": metrics.get("f1_weighted", 0.0),
    }


def validate_one_epoch(
    cfg: DictConfig,
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    num_classes: int,
    phase: str = "val",
    class_idx: list = None,
) -> Dict[str, Any]:
    """
    Validate the model for one epoch.

    Args:
        cfg (DictConfig): Configuration object.
        model (torch.nn.Module): The model to validate.
        dataloader (torch.utils.data.DataLoader): Validation data loader.
        criterion (torch.nn.Module): Loss function.
        device (torch.device): Device to run computations on.
        num_classes (int): Number of classes.

    Returns:
        Dict[str, Any]: Validation metrics.
    """

    metadata_path = Path(cfg.data.root_dir) / Path(cfg.data.metadata_filepath)
    metadata_df = pd.read_csv(metadata_path)
    slide_to_etude = dict(zip(metadata_df["NUM DOSS"].astype(str), metadata_df["Etude"]))

    (
        running_loss,
        embeddings,
        logits,
        labels,
        slide_ids,
        probs,
        predictions,
        etudes,
        original_labels,
        correct_probs,
        incorrect_probs,
        indices,
    ) = one_epoch(
        model=model,
        cfg=cfg,
        device=device,
        dataloader=dataloader,
        optimizer=None,
        criterion=criterion,
        num_classes=num_classes,
        slide_to_etude=slide_to_etude,
        mode=phase,
        temperature_layer=None,
        return_embeddings=False,
        return_correct_incorrect_probs=True,
        class_idx=class_idx,
    )

    val_loss = running_loss / len(dataloader)

    # --------------------------- Slide-level CSV Logging --------------------------- #
    # Activer le logging CSV au niveau slide pendant le test
    if phase == "test":  # Toujours activer pendant la validation
        logger.info(f"Creating slide-level CSV during {phase}...")

        # Utiliser output_dir si défini, sinon utiliser le répertoire du workspace
        if hasattr(cfg.model, "output_dir"):
            output_dir = Path(cfg.model.output_dir) / "csv_logs"
        else:
            output_dir = Path("./results") / "csv_logs"

        output_dir.mkdir(parents=True, exist_ok=True)

        val_csv_path = output_dir / "slide_level_results_val.csv"

        try:
            create_slide_level_csv(
                cfg=cfg,
                num_classes=num_classes,
                indices=indices,
                slide_ids=slide_ids,
                predictions=predictions,
                labels=labels,
                probs=probs,
                original_labels=original_labels,
                output_path=val_csv_path,
                phase=phase,
            )
        except Exception as e:
            logger.error(f"Error creating slide-level CSV: {e}")

    metrics = calculate_metrics(
        np.array(labels),
        np.array(predictions),
        np.array(probs),
        num_classes,
        all_original_labels=np.array(original_labels),
        class_idx=class_idx,
        cfg=cfg,
    )

    metrics_by_slide = calculate_metrics_by_slide(
        cfg,
        np.array(labels),
        np.array(predictions),
        np.array(probs),
        slide_ids,
        num_classes,
        all_original_labels=np.array(original_labels),
        class_idx=class_idx,
    )

    return {
        "loss": val_loss,
        "accuracy": metrics["accuracy"],
        "accuracy_weighted": metrics["accuracy_weighted"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "specificity_weighted": metrics["specificity_weighted"],
        "f1": metrics["f1"],
        "f1_weighted": metrics["f1_weighted"],
        "conf_matrix": metrics["confusion_matrix"],
        "auc_scores": metrics["auc"],
        "accuracy_slides": metrics_by_slide["accuracy"],
        "accuracy_weighted_slides": metrics_by_slide["accuracy_weighted"],
        "precision_slides": metrics_by_slide["precision"],
        "recall_slides": metrics_by_slide["recall"],
        "specificity_weighted_slides": metrics_by_slide["specificity_weighted"],
        "f1_slides": metrics_by_slide["f1"],
        "f1_weighted_slides": metrics_by_slide["f1_weighted"],
        "conf_matrix_slides": metrics_by_slide["confusion_matrix"],
        "auc_scores_slides": metrics_by_slide["auc"],
        "correct_confidences": correct_probs,
        "incorrect_confidences": incorrect_probs,
        "all_labels": np.array(labels),
        "all_probs": np.array(probs),
    }


def test_one_epoch(
    cfg: DictConfig,
    model: torch.nn.Module,
    device: torch.device,
    num_classes: int,
    class_idx: list = None,
):
    """
    Test the model on the test set.

    Args:
        cfg (DictConfig): Configuration object.
        model (torch.nn.Module): Trained PyTorch model.
        device (torch.device): Device to run the model on ('cuda' or 'cpu').
        num_classes (int): Number of classes.
    """
    # Load metadata to map slide IDs to studies
    metadata_path = Path(cfg.data.root_dir) / Path(cfg.data.metadata_filepath)
    metadata_df = pd.read_csv(metadata_path)
    slide_to_etude = dict(zip(metadata_df["NUM DOSS"].astype(str), metadata_df["Etude"]))

    model.eval()
    model.to(device)

    # --------------------------------------------------------------------------------------------------------------- #
    # --------------------------------------------------- Test Phase ------------------------------------------------ #
    # --------------------------------------------------------------------------------------------------------------- #

    test_loader = get_test_dataloader(cfg)
    criterion_test = setup_criterion(cfg, num_classes, test_loader.dataset, device)
    logger.info(f"Test criterion: {criterion_test}")

    initial_metrics = validate_one_epoch(
        cfg, model, test_loader, criterion_test, device, num_classes, phase="test", class_idx=class_idx
    )
    log_metrics(initial_metrics, epoch=0, phase="test_initial", cfg=cfg, num_classes=num_classes)


    logger.info("Testing complete.")
    plt.close("all")


def create_slide_level_csv(
    cfg,
    num_classes,
    indices,
    slide_ids,
    predictions,
    labels,
    probs,
    original_labels,
    output_path,
    phase="test",
):
    """
    Crée un fichier CSV avec les informations au niveau slide en agrégeant les images.

    Args:
        cfg: Configuration object contenant les paramètres d'agrégation
        num_classes: Nombre de classes dans le modèle
        indices: Les indices des images dans le dataset
        slide_ids: Les identifiants des slides
        predictions: Les prédictions du modèle au niveau image
        labels: Les vraies étiquettes au niveau image
        probs: Les probabilités prédites au niveau image
        original_labels: Les étiquettes originales au niveau image
        output_path: Chemin de sortie pour le fichier CSV
        phase: Phase du test ("train", "val", "test")
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Création du CSV au niveau slide pour la phase {phase}...")

    slide_ids = np.array(slide_ids)
    predictions = np.array(predictions)
    labels = np.array(labels)
    probs = np.array(probs)
    original_labels = np.array(original_labels)
    indices = np.array(indices)

    # Ensure 2D probabilities for binary (needed for mean/median)
    if num_classes == 2:
        if probs.ndim == 1:
            probs = np.column_stack((1 - probs, probs))
        elif probs.shape[1] == 1:
            probs = np.column_stack((1 - probs.flatten(), probs.flatten()))

    unique_slides = np.unique(slide_ids)

    # Préparer les listes pour stocker les données agrégées au niveau slide
    slide_data = []

    for slide_idx, slide in enumerate(unique_slides):
        slide_mask = slide_ids == slide
        if np.sum(slide_mask) == 0:
            continue  # Skip empty slides

        # Récupérer les données pour ce slide
        slide_indices = indices[slide_mask]
        slide_predictions = predictions[slide_mask]
        slide_labels_array = labels[slide_mask]
        slide_probs_array = probs[slide_mask]
        slide_original_labels_array = original_labels[slide_mask]

        # Prendre le premier label (tous devraient être identiques pour un slide)
        slide_true_label = slide_labels_array[0]
        slide_original_label = slide_original_labels_array[0]

        # Agrégation selon la méthode configurée
        if cfg.model.slide_aggregator_method == "mean":
            slide_prob = np.mean(slide_probs_array, axis=0)
            slide_prediction = np.argmax(slide_prob)

        elif cfg.model.slide_aggregator_method == "median":
            slide_prob = np.median(slide_probs_array, axis=0)
            slide_prediction = np.argmax(slide_prob)

        else:  # Majority voting
            tile_preds = slide_predictions.astype(int)
            class_counts = np.bincount(tile_preds, minlength=num_classes)
            slide_prediction = np.argmax(class_counts)
            slide_prob = class_counts / len(tile_preds)

        max_probability = np.max(slide_prob)

        slide_row = {
            "slide_id": slide,
            "slide_index": slide_idx,
            "slide_prediction": slide_prediction,
            "slide_true_label": slide_true_label,
            "slide_original_label": slide_original_label,
            "max_probability": max_probability,
            "num_images_in_slide": len(slide_indices),
        }

        for class_idx in range(num_classes):
            slide_row[f"probability_class_{class_idx}"] = slide_prob[class_idx] if class_idx < len(slide_prob) else 0.0

        slide_data.append(slide_row)

    df = pd.DataFrame(slide_data)

    df.to_csv(output_path, index=False)
    logger.info(f"CSV slide-level saved: {output_path}")
    logger.info(f"Number of slides saved: {len(df)}")

    return df


def one_epoch(
    model,
    cfg,
    device,
    dataloader,
    optimizer,
    criterion,
    num_classes,
    slide_to_etude,
    mode="train",
    temperature_layer=None,
    return_embeddings=False,
    return_correct_incorrect_probs=False,
    class_idx=None,
):
    """
    Runs one epoch of training, validation, or testing.

    Args:
        model (nn.Module): Model to be trained/evaluated.
        cfg (Namespace or dict): Configuration containing (at least) `use_mc_dropout` and `mc_samples`.
        device (torch.device): Device to use.
        dataloader (DataLoader): DataLoader of the dataset split.
        optimizer (torch.optim.Optimizer): Optimizer (used only in train mode).
        criterion (callable): Loss function.
        num_classes (int): Number of classes for classification.
        slide_to_etude (dict): Mapping from slide ID to 'etude' ID/label.
        mode (str): One of ['train', 'val', 'test'].
        temperature_layer (callable, optional): A temperature scaling layer or None.
        return_embeddings (bool, optional): If True, concatenates and returns embeddings.
        return_correct_incorrect_probs (bool, optional): If True, returns probability distributions
                                                        for correct/incorrect predictions.

    Returns:
        tuple:
            (running_loss,
            embeddings,
            logits,
            labels,
            slide_ids,
            probs,
            predictions,
            etudes,
            original_labels,
            correct_probs,
            incorrect_probs
        )
    """
    do_mc_dropout = (mode != "train") and cfg.confidence.use_mc_dropout

    if mode == "train" or do_mc_dropout:
        model.train()
    else:
        model.eval()

    all_embeddings = []
    all_logits = []
    all_probs = []
    all_preds = []
    all_labels = []
    all_original_labels = []
    all_slide_ids = []
    all_etudes = []
    all_indices = []
    correct_probs = []
    incorrect_probs = []

    running_loss = 0.0

    tqdm_logger = TqdmToLogger(logger)
    for batch in tqdm(dataloader, leave=False, file=tqdm_logger):
        if mode == "train":
            optimizer.zero_grad()

        # --- Monte Carlo Dropout (Val/Test) ---
        if do_mc_dropout:
            mc_probs_list = []
            mc_outputs_list = []

            num_mc_samples = cfg.confidence.mc_samples

            for mc_sample in range(num_mc_samples):
                # compute embeddings on the last MC sample if we actually want embeddings
                compute_feats = (mc_sample == num_mc_samples - 1) if return_embeddings else False

                (
                    loss,
                    outputs,
                    feats,
                    probs,
                    preds,
                    labels,
                    original_labels,
                    slide_ids,
                    indices,
                ) = process_batch(
                    model=model,
                    batch=batch,
                    device=device,
                    criterion=criterion,
                    num_classes=num_classes,
                    compute_feats=compute_feats,
                    temperature_layer=temperature_layer,
                    class_idx=class_idx,
                    cfg=cfg,
                )
                mc_probs_list.append(probs)
                mc_outputs_list.append(outputs)

            mc_probs = torch.stack(mc_probs_list)  # [mc_samples, batch_size, num_classes]
            mean_probs = mc_probs.mean(dim=0)  # [batch_size, num_classes]
            preds = mean_probs.argmax(dim=1)

        # --- Standard Train/Val/Test ---
        else:
            compute_feats = return_embeddings
            (
                loss,
                outputs,
                feats,
                probs,
                preds,
                labels,
                original_labels,
                slide_ids,
                indices,
            ) = process_batch(
                model=model,
                batch=batch,
                device=device,
                criterion=criterion,
                num_classes=num_classes,
                compute_feats=compute_feats,
                temperature_layer=temperature_layer,
                class_idx=class_idx,
                cfg=cfg,
            )
            mean_probs = probs

            if mode == "train":
                loss.backward()
                optimizer.step()

        running_loss += loss.item()

        if return_embeddings and (feats is not None):
            all_embeddings.append(feats)

        all_logits.extend(outputs.detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_slide_ids.extend(slide_ids)
        all_probs.extend(mean_probs.detach().cpu().numpy())
        all_preds.extend(preds.detach().cpu().numpy())
        all_original_labels.extend(original_labels.detach().cpu().numpy())
        all_indices.extend(indices.detach().cpu().numpy())

        for sid in slide_ids:
            all_etudes.append(slide_to_etude.get(str(sid), None))

        if return_correct_incorrect_probs:
            correct_mask = preds.eq(labels.view_as(preds))
            correct_probs_batch = mean_probs[correct_mask].detach().cpu().numpy()
            incorrect_probs_batch = mean_probs[~correct_mask].detach().cpu().numpy()
            correct_probs.extend(correct_probs_batch)
            incorrect_probs.extend(incorrect_probs_batch)

    embeddings = None
    if return_embeddings and len(all_embeddings) > 0:
        embeddings = np.concatenate(all_embeddings, axis=0)

    logits = np.array(all_logits)
    labels = np.array(all_labels)
    slide_ids = np.array(all_slide_ids)
    probs = np.array(all_probs)
    predictions = np.array(all_preds)
    etudes = np.array(all_etudes)
    original_labels = np.array(all_original_labels)
    indices = np.array(all_indices)

    # print for each image, its slide_id, prediction, true label, and probability
    # for i in range(len(labels)):
    #     # get image id
    #     image_id = indices[i] if indices is not None else i
    #     logger.info(
    #         f"Slide ID: {slide_ids[i]}, Image_Id: {image_id}, Prediction: {predictions[i]}, True Label: {labels[i]}, Probability: {probs[i]}"
    #     )

    return (
        running_loss,
        embeddings,
        logits,
        labels,
        slide_ids,
        probs,
        predictions,
        etudes,
        original_labels,
        correct_probs,
        incorrect_probs,
        indices,
    )


@hydra.main(version_base="1.2")
def main(cfg: DictConfig):
    """
    Main function to orchestrate training, validation, and testing.

    Args:
        cfg (DictConfig): Configuration object.
    """
    start_time = time.time()

    flat_config = setup_environment(cfg)
    device = setup_device(cfg)
    num_classes = int(cfg.data.num_classes)

    # Extract class indices from configuration
    class_idx = getattr(cfg.data, "class_idx", list(range(num_classes)))

    setup_seed(cfg)

    model = setup_model(cfg, num_classes, device)

    print(model)

    if cfg.testing.enabled and not cfg.training.enabled:
        load_model_weights(model, cfg.testing.weights_filename)
        test_one_epoch(cfg, model, device, num_classes, class_idx)
    else:
        train_dataset, val_dataset = prepare_datasets_and_transforms(cfg)
        train_loader, val_loader = prepare_dataloaders(
            cfg,
            train_dataset,
            val_dataset,
            batch_size=cfg.optimizer.batch_size,
            num_workers=cfg.data.num_workers,
        )

        
        optimizer = setup_optimizer(cfg, model)
        scheduler = setup_scheduler(cfg, optimizer)
        criterion = setup_criterion(cfg, num_classes, train_dataset, device)

        train_and_validate(
            cfg,
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            scheduler,
            device,
            num_classes,
            class_idx,
        )

        if cfg.testing.enabled:
            best_weights_path = Path(cfg.model.weights_path) / "custom" / cfg.training.weights_filename
            print(f"Loading best weights from {best_weights_path}")
            load_model_weights(model, best_weights_path)
            test_one_epoch(cfg, model, device, num_classes, class_idx)

    send_logs_to_wandb(cfg, flat_config)

    end_time = time.time()
    elapsed = int(end_time - start_time)
    hours = elapsed // 3600
    mins = (elapsed % 3600) // 60
    secs = elapsed % 60

    print(f"Total runtime: {hours:02d}h {mins:02d}m {secs:02d}s")
    logger.info(f"Total runtime: {hours:02d}h {mins:02d}m {secs:02d}s")


    plt.close("all")


if __name__ == "__main__":
    main()
