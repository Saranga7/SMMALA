import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import ConnectionPatch
from scipy.stats import norm
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.manifold import TSNE
from sklearn.calibration import calibration_curve
from collections import defaultdict
import logging
from itertools import combinations
from scipy import stats

from src.utils.utils_model import get_all_layers


def plot_confusion_matrix(conf_matrix, num_classes, title="Confusion Matrix", cmap="Purples"):
    plt.figure(figsize=(10, 8))

    # Determine actual matrix dimensions
    matrix_rows, matrix_cols = conf_matrix.shape

    # Define labels based on actual matrix dimensions, not num_classes parameter
    if matrix_rows == 2 and matrix_cols == 2:
        # Binary classification case (2x2 matrix)
        x_labels = ["Negative", "Submicro"]
        y_labels = ["Negative", "Submicro"]
    elif matrix_rows == 3 and matrix_cols == 2:
        # Merge case (3x2 matrix: 3 original classes → 2 merged classes)
        x_labels = ["Negative", "Positive"]
        y_labels = ["Negative", "Submicro", "Micro"]
    elif matrix_rows == 2 and matrix_cols == 3:
        # Reverse merge case (2x3 matrix)
        x_labels = ["Negative", "Submicro", "Micro"]
        y_labels = ["Negative", "Positive"]
    else:
        # 3x3 matrix or other cases
        x_labels = ["Negative", "Submicro", "Micro"]
        y_labels = ["Negative", "Submicro", "Micro"]

    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap=cmap, annot_kws={"fontsize": 16})
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")

    plt.xticks(ticks=np.arange(len(x_labels)) + 0.5, labels=x_labels, rotation=45, ha="right")
    plt.yticks(ticks=np.arange(len(y_labels)) + 0.5, labels=y_labels, rotation=0)

    return plt.gcf()


def plot_roc_curve(labels, probs, num_classes):
    plt.figure(figsize=(10, 8))

    # Determine the actual number of classes from the probabilities shape
    probs = np.array(probs)
    if probs.ndim == 1:
        # Single probability column, assume binary classification
        effective_num_classes = 2
        # Convert to 2-column format for binary case
        probs_binary = np.column_stack([1 - probs, probs])
        probs = probs_binary
    else:
        effective_num_classes = min(num_classes, probs.shape[1])

    if effective_num_classes == 2:
        # For binary classification, use the positive class probabilities
        positive_probs = probs[:, 1] if probs.shape[1] > 1 else probs.flatten()
        fpr, tpr, _ = roc_curve(labels, positive_probs)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
    else:
        for i in range(effective_num_classes):
            if i < probs.shape[1]:  # Only plot if column exists
                fpr, tpr, _ = roc_curve(labels == i, probs[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f"ROC curve (Class {i}) (AUC = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.grid()
    return plt.gcf()


def plot_precision_recall_curve(labels, probabilities, num_classes):
    """
    Plot precision-recall curves for multi-class classification.

    Args:
        labels: Ground truth labels
        probabilities: Model prediction probabilities
        num_classes: Number of classes

    Returns:
        matplotlib.Figure: The precision-recall curve plot
    """
    plt.figure(figsize=(10, 8))

    # Determine the actual number of classes from the probabilities shape
    probabilities = np.array(probabilities)
    if probabilities.ndim == 1:
        # Single probability column, assume binary classification
        effective_num_classes = 2
        positive_probs = probabilities
    else:
        effective_num_classes = min(num_classes, probabilities.shape[1])

    if effective_num_classes == 2:
        # Binary classification case
        if probabilities.ndim > 1 and probabilities.shape[1] > 1:
            positive_probs = probabilities[:, 1]
        else:
            positive_probs = probabilities.flatten()

        precision, recall, _ = precision_recall_curve(labels, positive_probs)
        ap = average_precision_score(labels, positive_probs)

        plt.plot(recall, precision, lw=2, label=f"AP = {ap:.3f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend(loc="lower left")

    else:
        # Multi-class case
        for i in range(effective_num_classes):
            if i < probabilities.shape[1]:  # Only plot if column exists
                precision, recall, _ = precision_recall_curve(labels == i, probabilities[:, i])
                ap = average_precision_score(labels == i, probabilities[:, i])
                plt.plot(recall, precision, lw=2, label=f"Class {i} (AP = {ap:.3f})")

        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curves (One-vs-Rest)")
        plt.legend(loc="lower left")

    plt.grid(True)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    return plt.gcf()


def plot_confidence_distribution(
    train_confidences,
    val_confidences,
    epoch,
    train_conf_over_time,
    val_conf_over_time,
    analysis_interval,
    val_correct_confidences,
    val_incorrect_confidences,
    all_probs=None,
    all_labels=None,
    num_classes=None,
):
    """
    Plot comprehensive confidence distribution analysis including:
    - Train vs Val confidence distributions
    - Mean confidence over time
    - Correct vs Incorrect predictions confidence
    - Per-class confidence distributions (if all_probs, all_labels, and num_classes are provided)
    """

    # Fixed palette for up to 3 classes
    val_to_categs = {0: "Negative", 1: "Submicroscopic", 2: "Microscopic"}
    palette = ["#E8D2CD", "#A06A8D", "#2B1E3C"][:num_classes]
    hue_map = {i: palette[i] for i in range(num_classes)}

    # Determine number of rows based on whether we have class-specific data
    n_rows = 2 if all_probs is not None and all_labels is not None and num_classes is not None else 1
    fig = plt.figure(figsize=(20, 10 * n_rows))

    # Create a grid of subplots
    if n_rows == 2:
        gs = fig.add_gridspec(2, 3)
    else:
        gs = fig.add_gridspec(1, 3)

    # First row - Original plots
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    # Plot train vs val confidence distribution
    ax1.hist(train_confidences, bins=20, alpha=0.5, label="Train")
    ax1.hist(val_confidences, bins=20, alpha=0.5, label="Validation")
    ax1.set_title(f"Confidence Distribution (Epoch {epoch + 1})")
    ax1.set_xlabel("Confidence")
    ax1.set_ylabel("Frequency")
    ax1.legend()

    # Plot confidence over time
    num_points = (epoch // analysis_interval) + 1
    x = list(range(0, epoch + 1, analysis_interval))
    y_train = train_conf_over_time[:num_points]
    y_val = val_conf_over_time[:num_points]

    if len(x) > 0 and len(y_train) > 0:
        x_plot = x[: len(y_train)]

        ax2.plot(x_plot, y_train, label="Train", marker="o")
        ax2.plot(x_plot, y_val, label="Validation", marker="o")

        y_train_arr = np.array(y_train)
        y_val_arr = np.array(y_val)

        train_std = np.std(y_train_arr)
        ax2.fill_between(x_plot, y_train_arr - train_std, y_train_arr + train_std, alpha=0.2)

        val_std = np.std(y_val_arr)
        ax2.fill_between(x_plot, y_val_arr - val_std, y_val_arr + val_std, alpha=0.2)

        ax2.set_title("Mean Confidence Over Time")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Mean Confidence")
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, "Not enough data points yet", ha="center", va="center")

    # Plot correct vs incorrect confidence distribution
    ax3.hist(val_correct_confidences, bins=20, alpha=0.5, label="Correct", color="green")
    ax3.hist(val_incorrect_confidences, bins=20, alpha=0.5, label="Incorrect", color="red")
    ax3.set_title(f"Confidence Distribution: Correct vs Incorrect (Epoch {epoch + 1})")
    ax3.set_xlabel("Confidence")
    ax3.set_ylabel("Frequency")
    ax3.legend()

    # Second row - Per-class confidence distributions (if data is provided)
    if all_probs is not None and all_labels is not None and num_classes is not None:
        ax4 = fig.add_subplot(gs[1, 0])  # KDE plot
        ax5 = fig.add_subplot(gs[1, 1])  # Violin plot

        # Prepare data for binary case
        if num_classes == 2:
            if all_probs.ndim == 1 or all_probs.shape[1] == 1:
                all_probs = np.column_stack((1 - all_probs, all_probs))

        # KDE plots for each class
        for i in range(num_classes):
            class_probs = all_probs[all_labels == i][:, i]
            if len(class_probs) > 0:
                categ = val_to_categs[i]
                sns.kdeplot(class_probs, fill=True, label=categ, ax=ax4, clip=(0, 1), color=hue_map[i], alpha=0.5)

        ax4.axvline(0.5, color="gray", linestyle="--", alpha=0.7)  # Decision threshold
        ax4.set_xlabel("Confidence")
        ax4.set_ylabel("Density")
        ax4.set_title("Per-Class Confidence Distribution")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # Violin plot for per-class confidence
        data_list = []
        class_list = []
        for i in range(num_classes):
            categ = val_to_categs[i]
            class_probs = all_probs[all_labels == i][:, i]
            data_list.extend(class_probs)
            class_list.extend([categ] * len(class_probs))

        sns.violinplot(x=class_list, y=data_list, ax=ax5, palette=palette)
        ax5.set_xlabel("Class")
        ax5.set_ylabel("Confidence")
        ax5.set_title("Per-Class Confidence")

    plt.tight_layout()
    return fig


def plot_embedding_space(train_embeddings, train_labels, val_embeddings, val_labels, epoch):
    tsne = TSNE(n_components=2, random_state=42)
    train_tsne = tsne.fit_transform(train_embeddings)
    val_tsne = tsne.fit_transform(val_embeddings)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    sns.scatterplot(x=train_tsne[:, 0], y=train_tsne[:, 1], hue=train_labels, ax=ax1)
    ax1.set_title(f"Train Embedding Space (Epoch {epoch + 1})")
    sns.scatterplot(x=val_tsne[:, 0], y=val_tsne[:, 1], hue=val_labels, ax=ax2)
    ax2.set_title(f"Validation Embedding Space (Epoch {epoch + 1})")

    return fig


def plot_all_embedding_spaces(
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
):
    tsne = TSNE(n_components=2, random_state=42)
    train_tsne = tsne.fit_transform(train_embeddings)
    val_tsne = tsne.fit_transform(val_embeddings)

    fig, axs = plt.subplots(5, 2, figsize=(20, 40))
    fig.suptitle(f"Embedding Spaces at Epoch {epoch + 1}", fontsize=16)

    # Define unique palettes for each pair of plots
    palettes = [
        ["#2B1E3C", "#A06A8D", "#E8D2CD"],  # Palette for "Label"
        ["#FF6F61", "#6B5B95", "#88B04B"],  # Palette for "Etude"
        ["#D65076", "#45B8AC", "#EFC050"],  # Palette for "Age Group"
        ["#5B5EA6", "#9B2335", "#DFCFBE"],  # Palette for "Gender"
        ["#BC243C", "#6A5ACD", "#FFB347"],  # Palette for "Fever"
    ]

    # Function to enforce class-to-color mapping
    def get_palette(hue_data, palette):
        unique_classes = sorted(set(hue_data))  # Ensure consistent order
        mapped_palette = {cls: palette[i % len(palette)] for i, cls in enumerate(unique_classes)}
        return mapped_palette

    # Plot function with custom palettes
    def plot_tsne(ax, tsne_data, hue_data, title, palette):
        palette = get_palette(hue_data, palette)
        sns.scatterplot(
            x=tsne_data[:, 0],
            y=tsne_data[:, 1],
            hue=hue_data,
            ax=ax,
            palette=palette,
            legend="full",
        )
        ax.set_title(title)
        ax.set_xlabel("TSNE-1")
        ax.set_ylabel("TSNE-2")

    # Plot t-SNE visualizations with corresponding palettes
    plot_tsne(axs[0, 0], train_tsne, train_labels, "Train by Label", palettes[0])
    plot_tsne(axs[0, 1], val_tsne, val_labels, "Val by Label", palettes[0])

    plot_tsne(axs[1, 0], train_tsne, train_etudes, "Train by Etude", palettes[1])
    plot_tsne(axs[1, 1], val_tsne, val_etudes, "Val by Etude", palettes[1])

    plot_tsne(axs[2, 0], train_tsne, train_age_groups, "Train by Age Group", palettes[2])
    plot_tsne(axs[2, 1], val_tsne, val_age_groups, "Val by Age Group", palettes[2])

    plot_tsne(axs[3, 0], train_tsne, train_genders, "Train by Gender", palettes[3])
    plot_tsne(axs[3, 1], val_tsne, val_genders, "Val by Gender", palettes[3])

    plot_tsne(axs[4, 0], train_tsne, train_fevers, "Train by Fever", palettes[4])
    plot_tsne(axs[4, 1], val_tsne, val_fevers, "Val by Fever", palettes[4])

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    return fig


def plot_subset_performance(subset_metrics, confidence_thresholds, epoch):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    for threshold in confidence_thresholds:
        ax1.plot(subset_metrics[threshold]["loss"], label=f"Threshold {threshold}")
        ax2.plot(subset_metrics[threshold]["accuracy"], label=f"Threshold {threshold}")

    ax1.set_title("Validation Loss by Confidence Subset")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    ax2.set_title("Validation Accuracy by Confidence Subset")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()

    return fig


def plot_before_after_comparison(before_value, after_value, metric_name):
    fig, ax = plt.subplots(figsize=(8, 6))

    labels = ["Before Calibration", "After Calibration"]
    values = [before_value, after_value]

    ax.bar(labels, values, color=["blue", "green"])
    ax.set_ylabel(metric_name)
    ax.set_title(f"{metric_name} Before and After Calibration")

    for i, v in enumerate(values):
        ax.text(i, v, f"{v:.4f}", ha="center", va="bottom")

    plt.tight_layout()
    return fig


def plot_calibration_curve(y_true, y_prob, n_bins=10, title="Calibration Curve"):
    fig, ax = plt.subplots()

    if y_prob.ndim == 1:
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
        ax.plot(prob_pred, prob_true, marker="o", label="Model")
    else:
        for i in range(y_prob.shape[1]):
            y_true_binary = (y_true == i).astype(int)
            prob_true, prob_pred = calibration_curve(y_true_binary, y_prob[:, i], n_bins=n_bins)
            ax.plot(prob_pred, prob_true, marker="o", label=f"Class {i}")

    ax.plot([0, 1], [0, 1], linestyle="--", label="Perfectly calibrated")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("True probability")
    ax.set_title(title)
    ax.legend()

    return fig


def plot_epochs_to_layers(epochs_to_layers, model):
    num_layers = len(get_all_layers(model))
    epochs_to_frozen_layers = {epoch: num_layers - layers for epoch, layers in epochs_to_layers.items()}
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        list(epochs_to_frozen_layers.keys()),
        list(epochs_to_frozen_layers.values()),
        marker="o",
    )
    ax.set_xlabel("Epochs")
    ax.set_ylabel("# Frozen Layers")
    ax.set_title("Adaptive Gradual Unfreezing")
    ax.grid(True)
    plt.tight_layout()
    return fig


def plot_joint_with_marginals(df, x_col="std_conf", y_col="mean_conf", hue_col="true_label"):
    """
    Create a joint scatter plot with marginal distributions for multiple categories.

    Args:
        df (pd.DataFrame): DataFrame containing at least x_col, y_col, and hue_col.
        x_col (str): Column name for x-axis (e.g., 'std_conf').
        y_col (str): Column name for y-axis (e.g., 'mean_conf').
        hue_col (str): Column name that defines categories (e.g., 'true_label' or 'correct').

    Returns:
        fig: The created matplotlib Figure object.
    """
    if hue_col == "true_label":
        hue_categories = [0, 1, 2]
        val_to_categs = {0: "Negative", 1: "Submicroscopic", 2: "Microscopic"}
        palette = ["#E8D2CD", "#A06A8D", "#2B1E3C"]
        hue_map = {i: palette[i] for i in hue_categories}

    elif hue_col == "correct":
        hue_categories = [True, False]
        val_to_categs = {True: "Correct", False: "Incorrect"}
        palette = ["#FF6F61", "#88E04F"]
        hue_map = {i: palette[i] for i in hue_categories}

    # Create a JointGrid
    g = sns.JointGrid(data=df, x=x_col, y=y_col, height=8, space=0)

    # Plot each category separately on the main scatter
    for hue_val in hue_categories:
        subset = df[df[hue_col] == hue_val]
        g.ax_joint.scatter(
            subset[x_col],
            subset[y_col],
            color=hue_map[hue_val],
            label=val_to_categs[hue_val],
            edgecolor="black",
            alpha=0.98,
        )

    # Plot marginal distributions (KDE) for each category
    for hue_val in hue_categories:
        subset = df[df[hue_col] == hue_val]
        sns.kdeplot(x=subset[x_col], ax=g.ax_marg_x, color=hue_map[hue_val], fill=True, alpha=0.68, linewidth=1)
        sns.kdeplot(y=subset[y_col], ax=g.ax_marg_y, color=hue_map[hue_val], fill=True, alpha=0.68, linewidth=1)

    # Add a legend
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", label=str(hv), markerfacecolor=hue_map[hv], markersize=8)
        for hv in hue_categories
    ]
    g.ax_joint.legend(handles=handles, title=hue_col.capitalize(), loc="upper right")

    # Set labels and grid on the joint plot
    g.ax_joint.set_xlabel("St. Deviation of Confidence")
    g.ax_joint.set_ylabel("Mean Confidence")
    g.ax_joint.grid(True, alpha=0.3)

    plt.tight_layout()
    return g.figure


def create_linear_confidence_plot(
    confidences, slide_id, prediction, true_label, image_indices, mean_conf, std_conf, get_image_by_index, num_images=3
):
    """
    Create a linear confidence plot for a single slide, highlight some representative images.

    Args:
        confidences (list): List of confidence scores for the slide's images.
        image_indices (list): Corresponding indices for each image in the dataset.
        mean_conf (float): Mean confidence for the slide.
        std_conf (float): Std confidence for the slide.
        get_image_by_index (callable): Function that takes an index and returns an image array (H x W x C).
        num_images (int): How many representative images to display.

    Returns:
        fig (matplotlib.figure.Figure): The resulting figure.
    """
    fig, ax = plt.subplots(figsize=(12, 6))  # Larger figure for more space

    # Sort confidences and remember sorting order
    sorted_indices = np.argsort(confidences)
    sorted_conf = np.array(confidences)[sorted_indices]

    # Plot baseline and confidences
    ax.axhline(0, color="black", linewidth=1, alpha=0.7)
    ax.scatter(sorted_conf, np.zeros_like(sorted_conf), s=20, color="grey", alpha=0.7)

    # Add mean and std lines
    ax.axvline(mean_conf, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_conf:.2f}")
    ax.axvspan(mean_conf - std_conf, mean_conf + std_conf, color="red", alpha=0.1, label=f"±1 STD: {std_conf:.2f}")

    ax.set_yticks([])
    ax.set_title(f"Slide {slide_id}\nPred: {prediction}, True: {true_label}")
    ax.set_xlabel("Confidence")
    ax.set_xlim(0, 1)

    # Choose representative images
    if num_images == 3:
        candidates = [sorted_indices[0], sorted_indices[len(sorted_indices) // 2], sorted_indices[-1]]
    else:
        candidates = sorted_indices[:num_images]

    # Distribute images so they don't overlap:
    # We'll place them above the baseline line (y=0), at different vertical offsets.
    # For example, place them at y=0.5, 0.9, 1.3, ... based on how many images we have
    vertical_spacing = 0.4
    base_y = 0.5
    vertical_positions = [base_y + i * vertical_spacing for i in range(num_images)]

    # If confidences are close, shift them slightly horizontally
    # Just a simple approach: spread them around their original conf_val
    horizontal_shifts = np.linspace(-0.05, 0.05, num_images)

    # Ensure y-limit so images are visible inside the plot
    max_y = vertical_positions[-1] + 0.5
    ax.set_ylim(-0.5, max_y)

    for i, c_idx in enumerate(candidates):
        conf_val = confidences[c_idx]
        img_idx = image_indices[c_idx]

        img = get_image_by_index(img_idx)

        # Assign positions:
        x_pos = min(max(conf_val + horizontal_shifts[i], 0.05), 0.95)  # Keep inside x-limits
        y_pos = vertical_positions[i]

        # Create the image annotation
        imagebox = OffsetImage(img, zoom=0.045)  # Moderate zoom
        ab = AnnotationBbox(imagebox, (x_pos, y_pos), frameon=False, xycoords="data")
        ax.add_artist(ab)

        # Connect with a dotted line
        # Draw from (conf_val,0) baseline point to (x_pos,y_pos)
        # This line can be diagonal
        ax.plot([conf_val, x_pos], [0, y_pos], color="gray", linestyle="--", linewidth=1)

    # Add legend after all elements are drawn
    ax.legend()

    plt.tight_layout()
    return fig


def generate_composite_umap_with_confidence(
    embeddings,
    labels,
    slide_ids,
    confidences,
    predictions,
    slide_to_etude,
    get_image_by_index,
    cfg,
    num_images_per_slide=3,
    random_state=42,
):
    """
    Generate a composite figure showing UMAP embeddings and class-specific slide-level confidence plots,
    with connections between selected embeddings and their slide plots.

    Args:
        embeddings (np.ndarray): Array of image embeddings.
        labels (np.ndarray): Array of true labels.
        slide_ids (np.ndarray): Array of slide IDs corresponding to each image.
        confidences (np.ndarray): Array of confidence scores for each image.
        predictions (np.ndarray): Array of predicted labels for each image.
        slide_to_etude (dict): Mapping from slide IDs to studies (etudes).
        get_image_by_index (callable): Function to retrieve an image by index.
        cfg (Config): Configuration object.
        num_images_per_slide (int): Number of images to highlight per slide.
        random_state (int): Random state for reproducibility.

    Returns:
        matplotlib.figure.Figure: The composite figure.
    """
    logger = logging.getLogger(__name__)

    # Step 1: Compute slide-level statistics
    slide_stats = defaultdict(lambda: {"confidences": [], "correct": []})

    for idx, slide_id in enumerate(slide_ids):
        slide_stats[slide_id]["confidences"].append(confidences[idx])
        correct = int(predictions[idx] == labels[idx])
        slide_stats[slide_id]["correct"].append(correct)

    # Calculate mean confidence, std confidence, and accuracy per slide
    slide_summary = {}
    for slide_id, stats in slide_stats.items():
        confs = np.array(stats["confidences"])
        mean_conf = np.mean(confs)
        std_conf = np.std(confs)
        accuracy = np.mean(stats["correct"])  # Proportion correct

        # Assign true label (assumes all images in a slide have the same label)
        unique_labels = np.unique(labels[slide_ids == slide_id])
        if len(unique_labels) == 1:
            true_label = unique_labels[0]
        else:
            # Assign the most frequent label if multiple exist
            true_label = np.bincount(labels[slide_ids == slide_id]).argmax()

        slide_summary[slide_id] = {
            "mean_conf": mean_conf,
            "std_conf": std_conf,
            "accuracy": accuracy,
            "true_label": true_label,
        }

    # Step 2: Group slides by class
    class_to_slides = defaultdict(list)
    for slide_id, summary in slide_summary.items():
        class_label = summary["true_label"]
        class_to_slides[class_label].append(
            {
                "slide_id": slide_id,
                "mean_conf": summary["mean_conf"],
                "std_conf": summary["std_conf"],
                "accuracy": summary["accuracy"],
            }
        )

    # Step 3: Select slides per class based on criteria
    selected_slides = []
    for class_label, slides in class_to_slides.items():
        # Correctly predicted slides: accuracy == 1.0 (all correct)
        correct_slides = [s for s in slides if s["accuracy"] == 1.0]
        if correct_slides:
            # Slide with lowest mean confidence and highest std_conf
            selected_correct = min(correct_slides, key=lambda x: (x["mean_conf"], -x["std_conf"]))
            selected_slides.append(selected_correct)
        else:
            logger.warning(f"No fully correct slides found for class {class_label}.")

        # Incorrectly predicted slides: accuracy < 1.0 (some or all incorrect)
        incorrect_slides = [s for s in slides if s["accuracy"] < 1.0]
        if incorrect_slides:
            # Slide with highest mean confidence and lowest std_conf
            selected_incorrect = max(incorrect_slides, key=lambda x: (x["mean_conf"], -x["std_conf"]))
            selected_slides.append(selected_incorrect)
        else:
            logger.warning(f"No incorrect slides found for class {class_label}.")

    if not selected_slides:
        logger.error("No slides selected based on the provided criteria.")
        return None

    # Step 4: Reduce embeddings to 2D using t-SNE or UMAP
    tsne = TSNE(n_components=2, random_state=random_state, n_jobs=-1)
    reduced_embeddings = tsne.fit_transform(embeddings)

    # Step 5: Create the composite figure
    num_selected_slides = len(selected_slides)
    fig_width = 5 + 8 * num_selected_slides  # Adjust width based on number of slides
    fig = plt.figure(figsize=(fig_width, 10))
    gs = plt.GridSpec(2, 1 + num_selected_slides, figure=fig, width_ratios=[2] + [4] * num_selected_slides)

    # UMAP Embedding Plot
    ax_umap = fig.add_subplot(gs[:, 0])
    scatter = ax_umap.scatter(
        reduced_embeddings[:, 0],
        reduced_embeddings[:, 1],
        c=labels,
        cmap="viridis",
        alpha=0.6,
        s=20,
        edgecolors="none",
    )
    ax_umap.set_title("Validation Embedding Space")
    ax_umap.set_xlabel("t-SNE 1")
    ax_umap.set_ylabel("t-SNE 2")
    plt.colorbar(scatter, ax=ax_umap, label="True Labels")

    # Dictionary to store slide embedding coordinates
    slide_embedding_coords = {}

    # Iterate over all slides to map their embedding coordinates
    for slide in selected_slides:
        slide_id = slide["slide_id"]
        # Find all indices corresponding to this slide
        indices = np.where(slide_ids == slide_id)[0]
        # Compute the average embedding for the slide
        slide_embedding = reduced_embeddings[indices]
        avg_embedding = np.mean(slide_embedding, axis=0)
        slide_embedding_coords[slide_id] = avg_embedding

    # Iterate over selected slides and create confidence plots with connections
    for i, slide in enumerate(selected_slides):
        slide_id = slide["slide_id"]
        class_label = slide["true_label"]
        mean_conf = slide["mean_conf"]
        std_conf = slide["std_conf"]
        accuracy = slide["accuracy"]

        # Determine prediction status
        prediction_status = "Correct" if accuracy == 1.0 else "Incorrect"

        # Extract indices for the current slide
        slide_mask = slide_ids == slide_id
        slide_confidences = confidences[slide_mask]
        slide_labels = labels[slide_mask]
        slide_indices = np.where(slide_mask)[0]

        # Create subplot for the slide's confidence plot
        ax_slide = fig.add_subplot(gs[:, i + 1])

        # Create linear confidence plot on the current axis
        create_linear_confidence_plot(
            confidences=slide_confidences,
            slide_id=slide_id,
            prediction=prediction_status,
            true_label=class_label,
            image_indices=slide_indices,
            mean_conf=mean_conf,
            std_conf=std_conf,
            get_image_by_index=get_image_by_index,
            num_images=num_images_per_slide,
            ax=ax_slide,  # Pass the axis to plot on
        )

        # Step 6: Draw connections between UMAP and slide plots
        # Get embedding coordinates
        embed_x, embed_y = slide_embedding_coords[slide_id]
        # Get the position of the slide plot in the figure
        # Transform the slide plot's axes to figure coordinates
        transFigure = fig.transFigure.inverted()
        slide_bbox = ax_slide.get_window_extent()
        slide_bbox = transFigure.transform(slide_bbox)
        slide_center = slide_bbox.mean(axis=0)

        # Transform the embedding coordinates to figure coordinates
        embed_display = ax_umap.transData.transform((embed_x, embed_y))
        embed_display_fig = fig.transFigure.inverted().transform(embed_display)

        # Create a ConnectionPatch
        con = ConnectionPatch(
            xyA=embed_display_fig,
            coordsA="figure fraction",
            xyB=slide_center,
            coordsB="figure fraction",
            axesA=fig,
            axesB=fig,
            color="gray",
            linestyle="--",
            linewidth=1,
            alpha=0.5,
        )
        fig.add_artist(con)

    plt.tight_layout()
    return fig


def plot_thresholds(cal_df, tau_mean, tau_std, title):
    """
    Plots mean confidence vs. std deviation with thresholds.
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=cal_df, x="std_conf", y="mean_conf", hue="correct", palette=["red", "green"], alpha=0.6)
    plt.axhline(y=tau_mean, color="purple", linestyle="--", label="Mean Confidence Threshold")
    plt.axvline(x=tau_std, color="blue", linestyle="--", label="Std Dev Threshold")
    plt.xlabel("Std Dev. of Confidence")
    plt.ylabel("Mean Confidence")
    plt.title(title)
    plt.legend()
    plt.show()

    return plt.gcf()


def plot_lda_decision_boundary(
    lda, df, feature_cols=["mean_conf", "std_conf"], label_col="correct", threshold=0.5, title="LDA Decision Boundary"
):
    """
    Plots the data points and the LDA decision boundary for a specified threshold.

    Args:
        lda (LinearDiscriminantAnalysis): Trained LDA model.
        df (pd.DataFrame): DataFrame containing features and labels.
        feature_cols (list, optional): List of feature column names. Defaults to ['mean_conf', 'std_conf'].
        label_col (str, optional): Column name for labels. Defaults to 'correct'.
        threshold (float, optional): Threshold for the decision boundary. Defaults to 0.5.
        title (str, optional): Plot title. Defaults to "LDA Decision Boundary".

    Returns:
        matplotlib.figure.Figure: The generated plot.
    """
    plt.figure(figsize=(10, 6))

    # Scatter plot of the two classes
    sns.scatterplot(
        data=df,
        x=feature_cols[1],
        y=feature_cols[0],
        hue=label_col,
        palette={True: "green", False: "red"},
        alpha=0.6,
    )

    # Calculate the discriminant score threshold:
    d_threshold = norm.ppf(threshold)

    # LDA coefficients
    coef = lda.coef_[0]
    intercept = lda.intercept_[0]

    # Decision boundary: w1*x1 + w2*x2 + b = d_threshold
    # Solve for x2:
    x_vals = np.linspace(df[feature_cols[1]].min(), df[feature_cols[1]].max(), 100)
    y_vals = (d_threshold - coef[0] * x_vals - intercept) / coef[1]

    # Plot the decision boundary
    plt.plot(x_vals, y_vals, color="blue", linestyle="--", label=f"LDA Decision Boundary (Threshold={threshold})")

    plt.xlabel("Standard Deviation of Confidence (std_conf)")
    plt.ylabel("Mean Confidence (mean_conf)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    fig = plt.gcf()

    return fig


def plot_lda_different_thresholds(thresholds, accuracy_list, share_und):
    # Normalize thresholds for color mapping
    norm_thresholds = (thresholds - thresholds.min()) / (thresholds.max() - thresholds.min())

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot accuracy vs share of undiagnosticable slides
    sc = ax.scatter(share_und, accuracy_list, c=norm_thresholds, cmap="viridis", edgecolor="k", s=40)

    # Add colorbar to show LDA threshold
    cbar = plt.colorbar(sc)
    cbar.set_label("LDA Probability Threshold", fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    # Annotate specific points (e.g., thresholds 0.25, 0.5, 0.75)
    for idx, th in enumerate(thresholds):
        if th in [0.25, 0.5, 0.75]:  # Adjust threshold points of interest
            ax.annotate(
                f"{th:.2f}",
                (share_und[idx], accuracy_list[idx]),
                textcoords="offset points",
                xytext=(-10, 10),
                fontsize=10,
                color="black",
                arrowprops=dict(arrowstyle="->", color="gray", lw=1),
            )

    # Add labels and title
    ax.set_xlabel("Share of Undiagnosticable Slides (%)", fontsize=12)
    ax.set_ylabel("Accuracy on Diagnosticable Slides", fontsize=12)
    ax.set_title("Accuracy vs % Undiagnosticable Slides\n(LDA Threshold as Color)", fontsize=14, weight="bold")

    # Adjust grid and layout
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    return fig


def plot_qda_decision_boundary(
    qda, df, feature_cols=["mean_conf", "std_conf"], label_col="correct", threshold=0.5, title="QDA Decision Boundary"
):
    """
    Plots data points and the QDA decision boundary (contour) for a specified threshold in a 2D feature space.

    Args:
        qda (QuadraticDiscriminantAnalysis): A trained QDA model (binary classification).
        df (pd.DataFrame): DataFrame containing at least the two features and the label column.
        feature_cols (list, optional): Names of the two feature columns, e.g. ['mean_conf','std_conf'].
        label_col (str, optional): Column name for the class label (True/False, 0/1, etc.). Defaults to 'correct'.
        threshold (float, optional): Probability threshold for class 1. Defaults to 0.5.
        title (str, optional): Title of the plot. Defaults to "QDA Decision Boundary".

    Returns:
        matplotlib.figure.Figure: The generated plot as a Figure object.
    """
    # For consistency with your LDA plot, we'll put "std_conf" on the X-axis, "mean_conf" on the Y-axis
    # i.e., feature_cols[1] is X, feature_cols[0] is Y
    x_col, y_col = feature_cols[1], feature_cols[0]

    plt.figure(figsize=(10, 6))

    # 1) Scatter-plot of the data
    sns.scatterplot(
        data=df,
        x=x_col,
        y=y_col,
        hue=label_col,
        palette={True: "green", False: "red"} if df[label_col].dtype == bool else "Set1",
        alpha=0.6,
    )

    # 2) Create a 2D mesh grid covering the range of x_col (std_conf) and y_col (mean_conf)
    x_min, x_max = df[x_col].min(), df[x_col].max()
    y_min, y_max = df[y_col].min(), df[y_col].max()

    # Add a small margin around the plot (optional)
    margin_frac = 0.1
    x_margin = (x_max - x_min) * margin_frac
    y_margin = (y_max - y_min) * margin_frac
    x_min -= x_margin
    x_max += x_margin
    y_min -= y_margin
    y_max += y_margin

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

    # 3) For each point in the grid, compute the probability of class 1
    # QDA was trained on [mean_conf, std_conf], i.e. feature_cols[0], feature_cols[1]
    # So we must feed the model in that order: first column = mean_conf, second column = std_conf
    # But in our grid, xx is std_conf, yy is mean_conf -> feed as np.c_[yy, xx]
    grid_points = np.c_[yy.ravel(), xx.ravel()]  # shape (200*200, 2)
    probs_1 = qda.predict_proba(grid_points)[:, 1]  # Probability of class-1
    probs_1 = probs_1.reshape(xx.shape)  # reshape back to (200, 200)

    # 4) Draw a contour line for the given threshold
    # We'll draw a dashed line where P(class=1) == threshold
    contour = plt.contour(xx, yy, probs_1, levels=[threshold], colors="blue", linestyles="--")
    # Label for the contour (matplotlib doesn't automatically add legends for contours)
    contour.collections[0].set_label(f"QDA boundary (thr={threshold})")

    # 5) Plot formatting
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    fig = plt.gcf()
    return fig


def plot_qda_different_thresholds(thresholds, accuracy_list, share_und):
    """
    Plots Accuracy vs. Share of Undiagnosticable Slides, with color indicating QDA threshold.

    Args:
        thresholds (array-like): List/array of threshold values used for QDA acceptance.
        accuracy_list (list): Balanced accuracy at each threshold.
        share_und (list): Fraction of slides labeled "undiagnosticable" (i.e., not accepted) at each threshold.

    Returns:
        matplotlib.figure.Figure: The generated plot as a Figure object.
    """
    # Normalize thresholds for color mapping
    norm_thresholds = (thresholds - thresholds.min()) / (thresholds.max() - thresholds.min())

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot accuracy vs. share of undiagnosticable slides
    sc = ax.scatter(share_und, accuracy_list, c=norm_thresholds, cmap="viridis", edgecolor="k", s=40)

    # Add colorbar to show QDA threshold
    cbar = plt.colorbar(sc)
    cbar.set_label("QDA Probability Threshold", fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    # Annotate specific points (e.g., thresholds 0.25, 0.5, 0.75)
    for idx, th in enumerate(thresholds):
        if th in [0.25, 0.5, 0.75]:  # Adjust threshold points of interest
            ax.annotate(
                f"{th:.2f}",
                (share_und[idx], accuracy_list[idx]),
                textcoords="offset points",
                xytext=(-10, 10),
                fontsize=10,
                color="black",
                arrowprops=dict(arrowstyle="->", color="gray", lw=1),
            )

    # Axis labels and title
    ax.set_xlabel("Share of Undiagnosticable Slides", fontsize=12)
    ax.set_ylabel("Accuracy on Diagnosticable Slides", fontsize=12)
    ax.set_title("Accuracy vs. % Undiagnosticable Slides\n(QDA Threshold as Color)", fontsize=14, weight="bold")

    # Grid and layout
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    return fig




def add_stat_annotations(ax, df, feature_col, group_col, test="ttest", offset=0.1):
    """
    Adds statistical test annotations using precomputed summary statistics for faster computation.
    """

    # Precompute mean, std, and count for each group
    print(f"Performing {test} tests for pairwise group comparisons...")
    group_stats = df.groupby(group_col)[feature_col].agg(["mean", "std", "count"]).reset_index()

    # Get unique groups (sorted for consistent order)
    print(f"Unique groups: {group_stats[group_col].tolist()}")
    groups = group_stats[group_col].tolist()
    pairs = list(combinations(groups, 2))

    # Start above the global max of the data
    print(f"Feature range: {df[feature_col].min()} - {df[feature_col].max()}")
    y_max = df[feature_col].max()
    current_height = y_max + offset

    for g1, g2 in pairs:
        # Extract summary stats for the two groups
        stats1 = group_stats[group_stats[group_col] == g1]
        stats2 = group_stats[group_stats[group_col] == g2]

        mean1, std1, n1 = stats1["mean"].values[0], stats1["std"].values[0], stats1["count"].values[0]
        mean2, std2, n2 = stats2["mean"].values[0], stats2["std"].values[0], stats2["count"].values[0]

        print(f"Group {g1}: mean={mean1:.5f}, std={std1:.5f}, n={n1}")
        print(f"Group {g2}: mean={mean2:.5f}, std={std2:.5f}, n={n2}")

        if test == "ttest":
            t_stat, pval = stats.ttest_ind_from_stats(mean1, std1, n1, mean2, std2, n2, equal_var=False)
        elif test == "mannwhitney":
            data1 = df.loc[df[group_col] == g1, feature_col]
            data2 = df.loc[df[group_col] == g2, feature_col]
            _, pval = stats.mannwhitneyu(data1, data2, alternative="two-sided")
        else:
            raise ValueError("Unsupported test type. Choose 'mannwhitney' or 'ttest'.")

        print(f"Comparing {g1} vs. {g2}: p-value = {pval:.8f}")

        # Determine significance label
        star_label = (
            "****" if pval < 1e-4 else "***" if pval < 1e-3 else "**" if pval < 1e-2 else "*" if pval < 0.05 else "ns"
        )

        # Plot the significance bracket
        x1, x2 = groups.index(g1), groups.index(g2)
        y = current_height
        h = 0.02 * (y_max - df[feature_col].min())  # Bracket height

        ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.2, c="k")
        ax.text((x1 + x2) * 0.5, y + h, star_label, ha="center", va="bottom", color="k", fontsize=10)

        # Move up for next annotation
        current_height += offset


def add_stat_annotations_by_pairs(ax, df, feature_col, group_col, test="ttest", offset=0.1, selected_pairs=None):
    """
    Ajoute des annotations de tests statistiques.

    Si selected_pairs est fourni (liste de tuples), seules ces paires seront comparées.
    Sinon, toutes les combinaisons de groupes sont comparées.
    """
    # Pré-calcul des statistiques par groupe
    group_stats = df.groupby(group_col)[feature_col].agg(["mean", "std", "count"]).reset_index()

    # Liste des groupes (triés pour un ordre cohérent)
    groups = group_stats[group_col].tolist()

    # Utiliser les paires sélectionnées si fournies, sinon toutes les combinaisons
    if selected_pairs is None:
        from itertools import combinations

        selected_pairs = list(combinations(groups, 2))

    y_max = df[feature_col].max()
    current_height = y_max + offset

    for g1, g2 in selected_pairs:
        stats1 = group_stats[group_stats[group_col] == g1]
        stats2 = group_stats[group_stats[group_col] == g2]

        mean1, std1, n1 = stats1["mean"].values[0], stats1["std"].values[0], stats1["count"].values[0]
        mean2, std2, n2 = stats2["mean"].values[0], stats2["std"].values[0], stats2["count"].values[0]

        print(f"Group {g1}: mean={mean1:.5f}, std={std1:.5f}, n={n1}")
        print(f"Group {g2}: mean={mean2:.5f}, std={std2:.5f}, n={n2}")

        if test == "ttest":
            t_stat, pval = stats.ttest_ind_from_stats(mean1, std1, n1, mean2, std2, n2, equal_var=False)
        elif test == "mannwhitney":
            data1 = df.loc[df[group_col] == g1, feature_col]
            data2 = df.loc[df[group_col] == g2, feature_col]
            _, pval = stats.mannwhitneyu(data1, data2, alternative="two-sided")
        else:
            raise ValueError("Type de test non supporté. Choisissez 'mannwhitney' ou 'ttest'.")

        print(f"Comparing {g1} vs. {g2}: p-value = {pval:.8f}")

        star_label = (
            "****" if pval < 1e-4 else "***" if pval < 1e-3 else "**" if pval < 1e-2 else "*" if pval < 0.05 else "ns"
        )

        x1, x2 = groups.index(g1), groups.index(g2)
        y = current_height
        h = 0.02 * (y_max - df[feature_col].min())

        ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.2, c="k")
        ax.text((x1 + x2) / 2, y + h, star_label, ha="center", va="bottom", color="k", fontsize=10)

        current_height += offset



def plot_auc_evolution(auc_evolution, step=0.05):
    """
    Plot AUC evolution as uncertain slides are removed.

    Parameters:
        auc_evolution (list): AUC values as a function of the remaining slides.
        step (float): Fraction of data removed at each step.
    """
    x_vals = [1 - i * step for i in range(len(auc_evolution))]
    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, auc_evolution, marker="o", label="AUC Evolution")
    plt.xlabel("Fraction of Slides Remaining")
    plt.ylabel("Mean AUC")
    plt.title("AUC Evolution as Uncertain Slides are Removed")
    plt.legend()
    plt.grid()
    plt.show()
