import numpy as np
import logging
import ast
import wandb
import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew

from scipy.interpolate import interp1d
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
    classification_report,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset

try:
    import idr_torch
except ImportError:
    _idr_torch_available = False
else:
    _idr_torch_available = True

# check torch version
if torch.__version__ >= "2.2.0" and not _idr_torch_available:
    from tabpfn import (
        TabPFNClassifier,
    )  # TODO: comment if on JZ due to Jean Zay dependency issue => reimplement when not on Jean Zay


from tqdm import tqdm
from src.utils.utils_visualization import (
    create_linear_confidence_plot,
)


logger = logging.getLogger(__name__)


def get_confidence_distribution(model, dataloader, device, num_classes):
    model.eval()
    confidences = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Confidence Distribution", leave=False):
            inputs = batch[0].to(device)
            outputs = model(inputs)
        probs = F.softmax(outputs, dim=1) if num_classes > 2 else torch.sigmoid(outputs)
        max_probs = probs.max(dim=1)[0].cpu().numpy() if num_classes > 2 else probs.squeeze().cpu().numpy()
        confidences.extend(max_probs)
    return confidences


def collect_slide_confidences(model, dataloader, device, num_classes):
    """Collect confidence scores for all images grouped by slide.

    Args:
        model: The trained model
        dataloader: DataLoader containing test data
        device: Device to run the model on
        num_classes: Number of classes in the classification task

    Returns:
        dict: Dictionary containing confidence scores and metadata per slide
    """
    model.eval()
    slide_data = {}

    with torch.no_grad():

        for batch in tqdm(dataloader, desc="Collecting slide confidences"):
            inputs, labels, slide_ids, original_labels, indices = batch

            inputs = inputs.to(device)
            outputs = model(inputs)

            probs = F.softmax(outputs, dim=1) if num_classes > 2 else torch.sigmoid(outputs)
            confidences = probs.max(dim=1)[0].cpu().numpy() if num_classes > 2 else probs.squeeze().cpu().numpy()

            # Store data for each slide
            for i, slide_id in enumerate(slide_ids):
                if slide_id not in slide_data:
                    slide_data[slide_id] = {
                        "confidences": [],
                        "true_label": labels[i].item(),
                        "original_label": original_labels[i].item(),
                        "predictions": [],
                        "image_indices": [],
                    }

                slide_data[slide_id]["confidences"].append(confidences[i])
                pred = (probs[i] > 0.5).int().item() if num_classes == 2 else probs[i].argmax().item()
                slide_data[slide_id]["predictions"].append(pred)
                if indices is not None:
                    slide_data[slide_id]["image_indices"].append(indices[i].item())

    # Compute final slide-level predictions
    for slide_id in slide_data:
        predictions = slide_data[slide_id]["predictions"]
        slide_data[slide_id]["final_prediction"] = max(set(predictions), key=predictions.count)

    return slide_data


def create_wandb_linear_confidence_table(slide_data, get_image_by_index):
    """
    Create a wandb.Table with a linear confidence plot and representative images for each slide.

    Args:
        slide_data (dict): Returned by collect_slide_confidences.
        get_image_by_index (callable): Function to retrieve an image array given an index.

    Returns:
        wandb.Table
    """
    columns = [
        "slide_id",
        "prediction",
        "true_label",
        "linear_confidence_plot",
        "mean_confidence",
        "std_confidence",
    ]

    table = wandb.Table(columns=columns)

    for slide_id, data in slide_data.items():
        mean_conf = np.mean(data["confidences"])
        std_conf = np.std(data["confidences"])

        fig = create_linear_confidence_plot(
            confidences=data["confidences"],
            slide_id=slide_id,
            prediction=data["final_prediction"],
            true_label=data["true_label"],
            image_indices=data["image_indices"],
            mean_conf=mean_conf,
            std_conf=std_conf,
            get_image_by_index=get_image_by_index,
            num_images=3,
        )

        confidence_plot = wandb.Image(fig)
        plt.close(fig)

        table.add_data(
            slide_id,
            data["final_prediction"],
            data["true_label"],
            confidence_plot,
            mean_conf,
            std_conf,
        )

    return table


def create_slide_confidence_histogram(confidences, slide_id, prediction, true_label, num_classes):
    """Create a histogram for a single slide's confidence distribution.

    Args:
        confidences: Array of confidence scores for images in the slide
        slide_id: Identifier for the slide
        prediction: Model's prediction for this slide
        true_label: True label for this slide
        num_classes: Number of classes in the classification task

    Returns:
        fig: matplotlib figure object
    """
    mean_val = np.mean(confidences)
    std_val = np.std(confidences)

    plt.figure(figsize=(6, 4))
    plt.hist(confidences, bins=20, alpha=0.7, color="blue", edgecolor="black")

    # Add a vertical line at the mean
    plt.axvline(
        mean_val,
        color="red",
        linestyle="--",
        linewidth=2,
        label="Mean: {:.2f}".format(mean_val),
    )

    # Shade the region within one standard deviation
    plt.axvspan(
        mean_val - std_val,
        mean_val + std_val,
        alpha=0.2,
        color="grey",
        label="±1 STD: {:.2f}".format(std_val),
    )

    # Add title and labels
    plt.title(f"Slide {slide_id}\nPred: {prediction}, True: {true_label}")
    plt.xlabel("Confidence")
    plt.ylabel("Count")
    plt.xlim(0.0, 1.0)

    # # Add a text box with mean and std info
    # text_str = f"Mean: {mean_val:.2f}\nSTD: {std_val:.2f}"
    # plt.text(
    #     0.05,
    #     0.95,
    #     text_str,
    #     transform=plt.gca().transAxes,
    #     fontsize=10,
    #     verticalalignment="top",
    #     bbox=dict(boxstyle="round", facecolor="white", edgecolor="black", alpha=0.7),
    # )

    # Add legend
    plt.legend()

    return plt.gcf()


def create_wandb_slide_confidence_table(slide_data, num_classes):
    """Create a wandb Table containing slide confidence histograms.

    Args:
        slide_data: Dictionary containing confidence scores and metadata per slide
        num_classes: Number of classes in the classification task

    Returns:
        wandb.Table: Table containing slide confidence visualizations
    """

    # Create columns for the table
    columns = [
        "slide_id",
        "prediction",
        "true_label",
        "confidence_histogram",
        "mean_confidence",
        "std_confidence",
    ]

    # Create table
    table = wandb.Table(columns=columns)

    # Add data for each slide
    for slide_id, data in slide_data.items():
        # Create histogram
        fig = create_slide_confidence_histogram(
            data["confidences"],
            slide_id,
            data["final_prediction"],
            data["true_label"],
            num_classes,
        )

        # Convert plot to wandb Image
        confidence_plot = wandb.Image(fig)
        plt.close(fig)

        # Calculate statistics
        mean_conf = np.mean(data["confidences"])
        std_conf = np.std(data["confidences"])

        # Add row to table
        table.add_data(
            slide_id,
            data["final_prediction"],
            data["true_label"],
            confidence_plot,
            mean_conf,
            std_conf,
        )

    return table


def compute_slide_stats_on_confidences(confidence_type, slide_ids, confidences, labels, predictions, slide_size=10):
    """
    Compute various statistics per slide based on confidence scores, retaining sorted confidence vectors.

    Args:
        slide_ids (list or array-like): List of slide IDs (length N), one per image.
        confidences (list or array-like): List/array of shape (N,) with per-image confidence scores.
        labels (list or array-like): List of true labels per image (length N).
        predictions (list or array-like): List of predicted labels per image (length N).
        slide_size (int, optional): Number of images per slide. Defaults to 10.

    Returns:
        pd.DataFrame: DataFrame with aggregated statistics per slide.
    """
    # 1. Verify that all input lists have the same length
    lengths = [len(slide_ids), len(confidences), len(labels), len(predictions)]
    if len(set(lengths)) != 1:
        min_length = min(lengths)
        logger.warning(
            f"Input lists have mismatched lengths: "
            f"slide_ids={len(slide_ids)}, "
            f"confidences={len(confidences)}, "
            f"labels={len(labels)}, "
            f"predictions={len(predictions)}. "
            f"Truncating all lists to the smallest length: {min_length}."
        )
        slide_ids = slide_ids[:min_length]
        confidences = confidences[:min_length]
        labels = labels[:min_length]
        predictions = predictions[:min_length]

    # 2. Create the initial DataFrame
    try:
        df = pd.DataFrame(
            {
                "slide_id": slide_ids,
                "confidence": confidences,  # Each entry is a scalar
                "true_label": labels,
                "predicted_label": predictions,
            }
        )
    except Exception as e:
        logger.error(f"Error creating DataFrame: {e}")
        raise

    # 3. Remove entries with missing data
    initial_length = len(df)
    df = df.dropna(subset=["slide_id", "confidence", "true_label", "predicted_label"])
    if len(df) < initial_length:
        logger.warning(f"Dropped {initial_length - len(df)} rows due to missing data.")

    # 4. Count images per slide
    counts = df["slide_id"].value_counts()
    valid_slides = counts[counts == slide_size].index
    invalid_slides = counts[counts != slide_size].index

    # 5. Log information about invalid slides
    for slide in invalid_slides:
        logger.info(f"Slide {slide} has {counts[slide]} images, which is not equal to slide_size={slide_size}.")

    logger.info(f"Number of valid slides: {len(valid_slides)}")
    logger.info(f"Number of invalid slides: {len(invalid_slides)}")

    # 6. Filter the DataFrame to include only valid slides
    df = df[df["slide_id"].isin(valid_slides)]

    # Verify that all slides now have exactly slide_size images
    post_filter_counts = df["slide_id"].value_counts()
    if not all(post_filter_counts == slide_size):
        logger.error("Post-filtering, some slides do not have the exact number of images.")
        raise ValueError("Data inconsistency detected after filtering slides.")
    else:
        logger.info("All slides have been successfully filtered to have exactly the specified number of images.")

    # 7. Group by slide_id
    grouped = df.groupby("slide_id")

    # 8. Initialize the per-slide stats DataFrame
    slide_stats = pd.DataFrame({"slide_id": grouped.size().index})

    # 9. Compute scalar statistics on confidence
    slide_stats["mean_confidence"] = grouped["confidence"].mean().values
    slide_stats["std_confidence"] = grouped["confidence"].std(ddof=1).fillna(0).values
    slide_stats["var_confidence"] = grouped["confidence"].var(ddof=1).fillna(0).values
    slide_stats["skewness_confidence"] = (
        grouped["confidence"].apply(lambda x: skew(x) if len(x) > 2 else 0).fillna(0).values
    )

    # 10. Aggregate true labels (assuming each slide has the same true label)
    slide_stats["true_label"] = (
        grouped["true_label"]
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
        .fillna(-1)  # Assuming -1 is not a valid label; adjust as needed
        .values
    )

    # 11. Final prediction by majority vote of predicted_label
    def vote(preds):
        arr = np.array([p.item() if isinstance(p, np.ndarray) else p for p in preds])
        if len(arr) == 0:
            return np.nan
        counts = np.bincount(arr)
        return np.argmax(counts) if len(counts) > 0 else np.nan

    slide_stats["final_prediction"] = grouped["predicted_label"].agg(vote).values

    # 12. Correctness
    slide_stats["correct"] = slide_stats["final_prediction"] == slide_stats["true_label"]

    # 13. Function to sort and flatten confidence scores
    def sort_and_flatten_confidences(confidences_1d):
        confidences_sorted = np.sort(confidences_1d)
        return confidences_sorted.tolist()

    slide_stats["sorted_flat_vector"] = (
        grouped["confidence"].apply(lambda x: sort_and_flatten_confidences(x.values)).values
    )

    # 15. Save the DataFrame to CSV
    try:
        slide_stats.to_csv(f"slide_stats_{confidence_type}.csv", index=False)
        logger.info(f"Slide statistics successfully saved to 'slide_stats_{confidence_type}.csv'")
    except Exception as e:
        logger.error(f"Error saving DataFrame to CSV: {e}")
        raise

    return slide_stats


def compute_slide_stats(confidence_type, slide_ids, outputs, labels, predictions, slide_size=10, phase="test"):
    """
    Compute various statistics per slide, storing statistical moments as 3D vectors.

    Args:
        slide_ids (list or array-like): List of slide IDs (length N), one per image.
        outputs (list or array-like): List/array of shape (N, 3) with per-image probability vectors or per-image logits.
        labels (list or array-like): List of true labels per image (length N).
        predictions (list or array-like): List of predicted labels per image (length N).
        slide_size (int, optional): Number of images per slide. Defaults to 10.

    Returns:
        pd.DataFrame: DataFrame with aggregated statistics per slide.
    """
    df = pd.DataFrame(
        {
            "slide_id": slide_ids,
            "prob_vector": list(outputs),  # Each entry is a list or array of length 3
            "true_label": labels,
            "predicted_label": predictions,
        }
    )

    counts = df["slide_id"].value_counts()
    valid_slides = counts[counts == slide_size].index
    invalid_slides = counts[counts != slide_size].index
    for slide in invalid_slides:
        logger.info(f"Slide {slide} has {counts[slide]} images, which is not equal to slide_size={slide_size}.")
    df = df[df["slide_id"].isin(valid_slides)]

    df["prob_norm"] = df["prob_vector"].apply(lambda v: np.linalg.norm(v))

    grouped = df.groupby("slide_id")

    slide_stats = pd.DataFrame({"slide_id": grouped.size().index})

    slide_stats["prob_vectors"] = grouped["prob_vector"].apply(lambda x: np.stack(x.values)).values

    slide_stats["mean_prob_norm"] = grouped["prob_norm"].mean().values
    slide_stats["std_prob_norm"] = grouped["prob_norm"].std(ddof=1).values
    slide_stats["var_prob_norm"] = grouped["prob_norm"].var(ddof=1).values
    slide_stats["skewness_prob_norm"] = grouped["prob_norm"].apply(lambda x: skew(x) if len(x) > 2 else 0).values

    def compute_stat(arr, stat_func):
        """
        Compute a statistical function per dimension on a 2D array.

        Args:
            arr (np.ndarray): Array of shape (slide_size, 3)
            stat_func (callable): Function to compute (e.g., np.mean, np.std)

        Returns:
            np.ndarray: Array of shape (3,) with the statistic for each dimension
        """
        return stat_func(arr, axis=0)

    slide_stats["mean_confidence"] = (
        grouped["prob_vector"].apply(lambda x: compute_stat(np.stack(x.values), np.mean)).values
    )

    slide_stats["std_confidence"] = (
        grouped["prob_vector"].apply(lambda x: compute_stat(np.stack(x.values), np.std)).values
    )

    slide_stats["var_confidence"] = (
        grouped["prob_vector"].apply(lambda x: compute_stat(np.stack(x.values), np.var)).values
    )

    slide_stats["skewness_confidence"] = (
        grouped["prob_vector"]
        .apply(lambda x: skew(np.stack(x.values), axis=0) if len(x) > 2 else np.array([0, 0, 0]))
        .values
    )

    slide_stats["true_label"] = (
        grouped["true_label"].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan).values
    )

    def vote(preds):
        arr = np.array([p.item() if isinstance(p, np.ndarray) else p for p in preds])
        counts = np.bincount(arr)
        return np.argmax(counts) if len(counts) > 0 else np.nan

    slide_stats["final_prediction"] = grouped["predicted_label"].agg(vote).values

    slide_stats["correct"] = slide_stats["final_prediction"] == slide_stats["true_label"]

    def sort_and_flatten_prob_vectors(prob_vectors_3d):
        prob_vectors_3d = np.array(prob_vectors_3d)
        if prob_vectors_3d.ndim != 2:
            raise ValueError(f"Expected a 2D array, but got shape {prob_vectors_3d.shape}")
        norms = np.linalg.norm(prob_vectors_3d, axis=1)  # shape (slide_size,)
        sort_idx = np.argsort(norms)
        sorted_vecs = prob_vectors_3d[sort_idx]  # still shape (slide_size, 3)
        flattened = sorted_vecs.flatten()  # shape (slide_size*3,)
        return flattened

    slide_stats["sorted_flat_vector"] = slide_stats["prob_vectors"].apply(
        lambda arr: sort_and_flatten_prob_vectors(arr)
    )

    slide_stats["prob_vectors"] = slide_stats["prob_vectors"].apply(lambda x: x.tolist())
    slide_stats["mean_confidence"] = slide_stats["mean_confidence"].apply(lambda x: x.tolist())
    slide_stats["std_confidence"] = slide_stats["std_confidence"].apply(lambda x: x.tolist())
    slide_stats["var_confidence"] = slide_stats["var_confidence"].apply(lambda x: x.tolist())
    slide_stats["skewness_confidence"] = slide_stats["skewness_confidence"].apply(lambda x: x.tolist())
    slide_stats["sorted_flat_vector"] = slide_stats["sorted_flat_vector"].apply(lambda x: x.tolist())

    # slide_stats["prob_vectors"] = slide_stats["prob_vectors"].apply(ast.literal_eval)
    # slide_stats["sorted_flat_vector"] = slide_stats["sorted_flat_vector"].apply(ast.literal_eval)

    # save to .csv
    slide_stats.to_csv(f"slide_stats_{phase}.csv", index=False)

    return slide_stats


def evaluate_confidence_thresholding(
    df: pd.DataFrame,
    confidence_col: str,
    label_col: str,
    prediction_col: str,
    num_thresholds: int = 100,
    plot: bool = True,
) -> dict:
    """
    Evaluate classification performance by confidence thresholding using Balanced Accuracy Score.
    Handles single-class cases by assigning NaN where the metric cannot be computed.

    Args:
        df (pd.DataFrame): DataFrame containing confidence scores, true labels, and predictions.
        confidence_col (str): Name of the column with confidence scores.
        label_col (str): Name of the column with true labels.
        prediction_col (str): Name of the column with predicted labels.
        num_thresholds (int, optional): Number of thresholds to evaluate. Defaults to 100.
        plot (bool, optional): Whether to generate a plot of the metric vs. % Slides Removed. Defaults to True.

    Returns:
        dict: A dictionary containing:
            - 'thresholds': List of confidence thresholds.
            - 'removed_slide_percentages': List of percentages of slides removed at each threshold.
            - 'metric_values': List of Balanced Accuracy Scores at each threshold.
            - 'figure': The matplotlib figure object if `plot=True`, else None.
    """
    df_sorted = df.sort_values(by=confidence_col, ascending=True).reset_index(drop=True)

    thresholds = np.linspace(
        df_sorted[confidence_col].min(),
        df_sorted[confidence_col].max(),
        num=num_thresholds,
    )

    metric_values = []
    removed_slide_percentages = []

    num_total = len(df_sorted)

    # metric at 0.0 threshold
    full_metric = balanced_accuracy_score(df_sorted[label_col], df_sorted[prediction_col])
    metric_values.append(full_metric)
    removed_slide_percentages.append(0.0)

    for threshold in thresholds:
        retained_df = df_sorted[df_sorted[confidence_col] >= threshold]
        num_retained = len(retained_df)

        if num_retained == 0:
            continue

        removed_percentage = (1 - num_retained / num_total) * 100
        removed_slide_percentages.append(removed_percentage)

        if retained_df[label_col].nunique() < 2:
            metric_values.append(np.nan)  # Metric undefined for single-class
            continue

        metric = balanced_accuracy_score(retained_df[label_col], retained_df[prediction_col])
        metric_values.append(metric)

    fig = None
    if plot:
        removed_percent = np.array(removed_slide_percentages)
        metrics = np.array(metric_values)

        valid_mask = ~np.isnan(metrics)
        removed_percent_valid = removed_percent[valid_mask]
        metrics_valid = metrics[valid_mask]

        if len(removed_percent_valid) < 2:
            raise ValueError("Not enough valid data points to generate a plot.")

        # remove duplicate x-axis values by averaging their corresponding metric values
        unique_removed, indices = np.unique(removed_percent_valid, return_index=True)
        unique_metrics = metrics_valid[indices]

        # interpolate for smooth plotting
        interpolation = interp1d(unique_removed, unique_metrics, kind="linear", fill_value="extrapolate")
        fine_removed = np.linspace(unique_removed.min(), unique_removed.max(), 500)
        fine_metrics = interpolation(fine_removed)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fine_removed, fine_metrics, color="blue", label="Balanced Accuracy")
        ax.set_xlabel("% Slides Removed From Testset", fontsize=12)
        ax.set_ylabel("Balanced Accuracy", fontsize=12)
        ax.set_title("Balanced Accuracy vs. % Slides Removed\nTest Set", fontsize=14)
        ax.grid(True)
        ax.legend()

    return {
        "thresholds": thresholds.tolist(),
        "removed_slide_percentages": removed_slide_percentages,
        "metric_values": metric_values,
        "figure": fig,
    }


def compute_performance_by_confidence(df):
    """
    Compute performance metrics stratified by confidence.

    Parameters:
        probabilities (np.ndarray): Probability matrix for each class.
        labels (np.ndarray): True class labels.
        confidence_scores (np.ndarray): Confidence scores for each slide.
        threshold (float): Confidence threshold for stratification.

    Returns:
        pd.DataFrame: DataFrame with performance metrics for each method and confidence subset.
    """

    labels = np.array(df["true_label"].values)
    confidence_scores = np.array(df["confidence_score"].values)
    predictions = np.array(df["final_prediction"].values)
    threshold = np.median(confidence_scores)

    high_confidence_mask = confidence_scores >= threshold
    low_confidence_mask = ~high_confidence_mask

    results = []

    for type in ["Raw"]:
        for subset, mask in zip(
            ["Low Confidence", "High Confidence"],
            [low_confidence_mask, high_confidence_mask],
        ):
            # Get subset data
            subset_labels = labels[mask]
            subset_predictions = predictions[mask]

            # Compute Weighted Accuracy (3 classes)
            weighted_accuracy = balanced_accuracy_score(subset_labels, subset_predictions)

            # Compute NPV (Submicro or Micro class, subset_labels == 0 or 1)
            binary_labels = (subset_labels > 0).astype(int)
            binary_predictions = (subset_predictions > 0).astype(int)
            tn, fp, fn, tp = confusion_matrix(binary_labels, binary_predictions, labels=[0, 1]).ravel()
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

            # Store results
            results.append(
                {
                    "Type": type,
                    "Confidence Subset": subset,
                    "Weighted Accuracy (3 classes)": weighted_accuracy,
                    "NPV (Submicro & Micro)": npv,
                }
            )

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Pivot table for a clean view
    metrics_df = results_df.pivot(index="Type", columns="Confidence Subset")
    metrics_df.columns = [" ".join(col).strip() for col in metrics_df.columns.values]

    # Compute Gaps
    metrics_df["Weighted Accuracy (3 classes) Gap (%)"] = 100 * (
        metrics_df["Weighted Accuracy (3 classes) High Confidence"]
        - metrics_df["Weighted Accuracy (3 classes) Low Confidence"]
    )
    metrics_df["NPV (Submicro & Micro) Gap (%)"] = 100 * (
        metrics_df["NPV (Submicro & Micro) High Confidence"] - metrics_df["NPV (Submicro & Micro) Low Confidence"]
    )

    # Reorder columns for final output
    metrics_df = metrics_df[
        [
            "Weighted Accuracy (3 classes) Low Confidence",
            "Weighted Accuracy (3 classes) High Confidence",
            "Weighted Accuracy (3 classes) Gap (%)",
            "NPV (Submicro & Micro) Low Confidence",
            "NPV (Submicro & Micro) High Confidence",
            "NPV (Submicro & Micro) Gap (%)",
        ]
    ]

    return metrics_df


def compute_confidences(
    confidence_metric,
    train_df,
    val_df,
    test_df,
    model,
    device,
    num_classes,
    num_channels_list=None,
):
    """
    Compute confidence scores for each dataset using the specified metric.

    Args:
        confidence_metric (str): Confidence metric to use.
        train_df (pd.DataFrame): Training set DataFrame.
        val_df (pd.DataFrame): Validation set DataFrame.
        test_df (pd.DataFrame): Test set DataFrame.
        model: The trained model.
        device: Device to run the model on.
        num_classes: Number of classes in the classification task.
        num_channels_list: List of number of channels for each image in the dataset.

    Returns:
        pd.DataFrame: DataFrame with confidence scores for each dataset.
    """
    if val_df is None:
        val_df = test_df.copy()
        
    if confidence_metric == "grade_sensitive":
        train_confidences = compute_grade_sensitive_confidence(train_df["prob_vectors"].values)
        val_confidences = compute_grade_sensitive_confidence(val_df["prob_vectors"].values)
        test_confidences = compute_grade_sensitive_confidence(test_df["prob_vectors"].values)

    elif confidence_metric == "entropy":
        train_confidences = compute_entropy_confidence(train_df["prob_vectors"].values)
        val_confidences = compute_entropy_confidence(val_df["prob_vectors"].values)
        test_confidences = compute_entropy_confidence(test_df["prob_vectors"].values)

    elif confidence_metric == "mc_dropout":
        train_confidences = compute_monte_carlo_dropout_confidence(train_df["mc_probs"].values)
        val_confidences = compute_monte_carlo_dropout_confidence(val_df["mc_probs"].values)
        test_confidences = compute_monte_carlo_dropout_confidence(test_df["mc_probs"].values)

    elif confidence_metric == "class_proximity":
        train_confidences = compute_class_proximity_score(train_df["mean_probs"].values)
        val_confidences = compute_class_proximity_score(val_df["mean_probs"].values)
        test_confidences = compute_class_proximity_score(test_df["mean_probs"].values)

    else:
        classifier, train_confidences, val_confidences = train_confidence_classifier(
            confidence_metric, train_df, val_df
        )
        test_confidences = infer_confidence_classifier(classifier, test_df)

    date = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    test_confidences.to_csv(f"slide_stats_test_{confidence_metric}_{date}.csv", index=False)
    train_confidences.to_csv(f"slide_stats_train_{confidence_metric}_{date}.csv", index=False)

    return train_confidences, val_confidences, test_confidences


# ------------------------------------------------------------------------------------------------------------------- #
# --------------------------------------------- Non Parametric Confidence Scores --------------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------- #


def compute_grade_sensitive_confidence(probs: torch.Tensor) -> torch.Tensor:
    """
    Compute grade-sensitive confidence scores from risk vectors.

    Args:
        probs (torch.Tensor): Probabilities vectors of shape [batch_size, num_classes].

    Returns:
        torch.Tensor: Grade-sensitive confidence scores of shape [batch_size].
    """
    # Get the top 2 probabilities
    top_probs, _ = probs.topk(2, dim=1)
    confidence_scores = top_probs[:, 0] - top_probs[:, 1]
    return confidence_scores


def compute_entropy_confidence(probs: torch.Tensor) -> torch.Tensor:
    """
    Compute entropy-based confidence scores.

    Args:
        probs (torch.Tensor): Probability vectors of shape [batch_size, num_classes].

    Returns:
        torch.Tensor: Confidence scores of shape [batch_size].
    """
    # Compute entropy
    entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=1)  # Avoid log(0) with 1e-12

    # Normalize by the maximum possible entropy (log of the number of classes)
    max_entropy = torch.log(torch.tensor(probs.size(1), dtype=torch.float32, device=probs.device))
    confidence_scores = 1 - (entropy / max_entropy)

    return confidence_scores


def compute_risk_variance_confidence(logits: torch.Tensor, cost_matrix) -> torch.Tensor:
    """
    Compute confidence scores based on the variance of risk vectors.

    Args:
        logits (torch.Tensor): Raw model outputs, shape [batch_size, num_classes].
        cost_matrix (torch.Tensor): Cost matrix of shape [num_classes, num_classes].

    Returns:
        torch.Tensor: Confidence scores of shape [batch_size].
    """
    risks = torch.matmul(logits, cost_matrix)
    variance = risks.var(dim=1)

    # Normalize variance to compute confidence
    max_variance = risks.max(dim=1).values.var()
    confidence_scores = 1 - (variance / max_variance)

    return confidence_scores


def compute_risk_confidence(logits, cost_matrix):
    """
    Compute risks and probabilities by applying softmax after calculating risks.

    Args:
        logits (torch.Tensor): Raw model outputs, shape [batch_size, num_classes].

    Returns:
        risks (torch.Tensor): Risk vectors, shape [batch_size, num_classes].
        probs (torch.Tensor): Probabilities, shape [batch_size, num_classes].
    """
    # Compute risk vectors
    risks = torch.matmul(logits, cost_matrix)  # [batch_size, num_classes]

    # Apply softmax to convert risks into probabilities
    probs = torch.softmax(-risks, dim=1)  # Negative sign to prioritize lower risks
    return probs


def compute_monte_carlo_dropout_confidence(mc_probs: torch.Tensor) -> torch.Tensor:
    """
    Args:
        mc_probs (torch.Tensor): shape [M, batch_size, num_classes]
    Returns:
        torch.Tensor: shape [batch_size], confidence in [0,1]
    """
    epsilon = 1e-8

    # Variance over MC samples for each (batch_item, class)
    per_class_var = mc_probs.var(dim=0, unbiased=True)
    per_class_std = per_class_var.sqrt()

    # Collapse class dimension into a single uncertainty value per sample
    uncertainty = per_class_std.mean(dim=1)

    # Scale uncertainty into [0,1] and invert to get confidence
    max_uncertainty = torch.clamp(uncertainty.max(), min=epsilon)
    confidence_scores = 1.0 - (uncertainty / max_uncertainty)

    return confidence_scores


def compute_class_proximity_score(mean_probs):
    # Compute Class-Wise Proximity Metrics
    top_competing_probs, _ = mean_probs.topk(2, dim=1)  # Top 2 probabilities
    confidences = top_competing_probs[:, 0] - top_competing_probs[:, 1]
    return confidences


# ------------------------------------------------------------------------------------------------------------------- #
# --------------------------------------- Parametric Confidence Scores ---------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------- #


def train_confidence_classifier(confidence_type, train_df, val_df):
    """
    Train a supervised classifier to predict slide classifiability.

    Parameters:
    - train_df (pd.DataFrame): Preprocessed training DataFrame.
    - val_df (pd.DataFrame): Preprocessed validation DataFrame.

    Returns:
    - RandomForestClassifier: Trained classifier.
    """

    X_train = np.vstack(train_df["sorted_flat_vector"].values)
    y_train = train_df["correct"]

    X_val = np.vstack(val_df["sorted_flat_vector"].values)
    y_val = val_df["correct"]

    if confidence_type == "random_forest":
        clf = RandomForestClassifier(n_estimators=100, random_state=42)

    elif confidence_type == "xgboost":
        clf = XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            use_label_encoder=False,
            eval_metric="logloss",
        )

    elif confidence_type == "svc":
        clf = SVC(probability=True, kernel="rbf", random_state=42)

    elif confidence_type == "logistic_regression":
        clf = LogisticRegression(random_state=42)

    elif confidence_type == "decision_tree":
        clf = DecisionTreeClassifier(random_state=42)

    elif confidence_type == "lda":
        clf = LinearDiscriminantAnalysis()

    elif confidence_type == "qda":
        clf = QuadraticDiscriminantAnalysis()

    elif confidence_type == "tabpfn":
        clf = (
            TabPFNClassifier()
        )  # TODO: commented due to Jean Zay dependency issue => reimplement when not on Jean Zay
        pass

    elif confidence_type == "mlp":
        clf, y_train_proba, y_val_proba = train_confidence_neural_net(X_train, y_train, X_val, y_val)

        train_df["confidence_score"] = y_train_proba
        val_df["confidence_score"] = y_val_proba

        roc_auc = roc_auc_score(y_val, y_val_proba)
        logger.info(f"Validation ROC-AUC: {roc_auc:.4f}")

        y_val_pred = (y_val_proba > 0.5).astype(int)
        logger.info("Validation Classification Report:")
        logger.info(classification_report(y_val, y_val_pred))

        logger.info("Validation Confusion Matrix:")
        logger.info(confusion_matrix(y_val, y_val_pred))
        return clf, train_df, val_df

    else:
        raise ValueError(f"Invalid confidence_type: {confidence_type}")

    print("fitting classifier...")
    clf.fit(X_train, y_train)

    y_train_proba = clf.predict_proba(X_train)[:, 1]
    train_df["confidence_score"] = y_train_proba

    # Evaluate on validation set
    y_val_proba = clf.predict_proba(X_val)[:, 1]
    val_df["confidence_score"] = y_val_proba
    roc_auc = roc_auc_score(y_val, y_val_proba)
    logger.info(f"Validation ROC-AUC: {roc_auc:.4f}")

    # Classification Report
    y_val_pred = clf.predict(X_val)
    logger.info("Validation Classification Report:")
    logger.info(classification_report(y_val, y_val_pred))

    # Confusion Matrix
    logger.info("Validation Confusion Matrix:")
    logger.info(confusion_matrix(y_val, y_val_pred))

    return clf, train_df, val_df


def infer_confidence_classifier(model, test_df):
    """
    Predict confidence scores for the test set using the trained model.

    Parameters:
    - model (RandomForestClassifier): Trained classifier.
    - test_df (pd.DataFrame): Preprocessed test DataFrame.

    Returns:
    - pd.DataFrame: Test DataFrame with added 'confidence_score' column.
    """
    X_test = np.vstack(test_df["sorted_flat_vector"].values)

    if model.__class__.__name__ == "MLP":
        test_proba = model.get_probabilities(model, X_test)
    else:
        test_proba = model.predict_proba(X_test)[:, 1]

    test_df["confidence_score"] = test_proba

    return test_df


def train_confidence_neural_net(X_train, y_train, X_val, y_val):
    class MLP(nn.Module):
        def __init__(self, input_size, hidden_sizes, num_classes):
            super(MLP, self).__init__()
            layers = []
            in_size = input_size
            for hidden_size in hidden_sizes:
                layers.append(nn.Linear(in_size, hidden_size))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.5))
                in_size = hidden_size
            layers.append(nn.Linear(in_size, num_classes))
            self.network = nn.Sequential(*layers)

        def forward(self, x):
            return self.network(x)

        @staticmethod
        def get_probabilities(model, X):
            model.eval()
            dataset = ConfidenceDataset(X, np.zeros(len(X)))  # Dummy labels
            loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
            probs = []
            with torch.no_grad():
                for batch_X, _ in loader:
                    batch_X = batch_X.to(device)
                    outputs = model(batch_X)
                    prob = torch.sigmoid(outputs).cpu().numpy()
                    probs.extend(prob.flatten())
            return np.array(probs)

    # Hyperparameters
    input_size = X_train.shape[1]
    hidden_sizes = [64, 64]
    output_size = 1  # Binary classification
    num_epochs = 200
    batch_size = 64
    learning_rate = 0.0005

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class ConfidenceDataset(Dataset):
        def __init__(self, X, y):
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

        def __len__(self):
            return len(self.y)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    train_dataset = ConfidenceDataset(X_train, y_train)
    val_dataset = ConfidenceDataset(X_val, y_val)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

    model = MLP(input_size, hidden_sizes, output_size).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=20, factor=0.5, verbose=True)

    # Early stopping
    patience = 60
    best_val_loss = float("inf")
    counter = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_X.size(0)

        epoch_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)

                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_X.size(0)

        val_loss /= len(val_loader.dataset)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(
                f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}"
            )

        # scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            best_model = model
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered!")
                break  # Stop training

    y_train_proba = MLP.get_probabilities(best_model, X_train)
    y_val_proba = MLP.get_probabilities(best_model, X_val)

    return best_model, y_train_proba, y_val_proba
