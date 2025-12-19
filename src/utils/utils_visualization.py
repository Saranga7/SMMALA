import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.manifold import TSNE


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


