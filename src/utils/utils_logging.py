import json
import os
import logging
import wandb
import io
import matplotlib.pyplot as plt
from pathlib import Path
from src.utils.utils_visualization import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
)

logger = logging.getLogger(__name__)


def log_metrics(metrics, epoch=None, phase="train", log_file="training_logs.txt", cfg=None, num_classes=None):
    if cfg.wandb.enabled and not cfg.slurm.enabled:
        wandb_log_dict = {}
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                wandb_log_dict[f"{phase}/{k}"] = v
            else:
                continue

        wandb_log_dict["epoch"] = epoch

        if phase in ["test", "val", "test_initial"] and num_classes is not None:
            # Existing visualization code...
            conf_matrix_fig = plot_confusion_matrix(metrics["conf_matrix"], num_classes, cmap="Purples")
            conf_matrix_slides_fig = plot_confusion_matrix(
                metrics["conf_matrix_slides"], num_classes, title="Confusion Matrix (Slides)", cmap="Blues"
            )
            roc_fig = plot_roc_curve(metrics["all_labels"], metrics["all_probs"], num_classes)

            # save figs in .svg format
            conf_matrix_fig.savefig(f"conf_matrix_{phase}.svg")
            conf_matrix_slides_fig.savefig(f"conf_matrix_slides_{phase}.svg")
            roc_fig.savefig(f"roc_curve_{phase}.svg")

            # Add precision-recall curve
            pr_curve_fig = plot_precision_recall_curve(metrics["all_labels"], metrics["all_probs"], num_classes)

            wandb_log_dict.update(
                {
                    f"{phase}/confusion_matrix": wandb.Image(conf_matrix_fig),
                    f"{phase}/confusion_matrix_slides": wandb.Image(conf_matrix_slides_fig),
                    f"{phase}/roc_curve": wandb.Image(roc_fig),
                    f"{phase}/pr_curve": wandb.Image(pr_curve_fig),
                }
            )

            if num_classes == 2:
                wandb_log_dict[f"{phase}/auc_score"] = metrics["auc_scores"][0]
            else:
                wandb_log_dict[f"{phase}/auc_scores"] = {
                    f"class_{i}": auc for i, auc in enumerate(metrics["auc_scores"])
                }

            plt.close(conf_matrix_fig)
            plt.close(conf_matrix_slides_fig)
            plt.close(roc_fig)
            plt.close(pr_curve_fig)

        wandb.log(wandb_log_dict)

    else:
        # Handle non-wandb logging case
        slurm_log_dict = {}
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                slurm_log_dict[f"{phase}/{k}"] = v
            else:
                continue

        if "auc_scores" in metrics:
            if num_classes == 2:
                slurm_log_dict[f"{phase}/auc_score"] = metrics["auc_scores"][0]
            else:
                for i, auc in enumerate(metrics["auc_scores"]):
                    slurm_log_dict[f"{phase}/auc_score_class_{i}"] = auc

        with open(Path(cfg.training.log_dir) / log_file, "a+") as f:
            log_dict = {**slurm_log_dict, "epoch": epoch, "phase": phase}
            json.dump(log_dict, f)
            f.write("\n")

        if phase in ["test", "val"] and num_classes is not None:
            image_folder = Path(cfg.training.log_dir) / "log_images"
            os.makedirs(image_folder, exist_ok=True)

            # Existing visualization saving code...
            if "conf_matrix" in metrics:
                conf_matrix_fig = plot_confusion_matrix(metrics["conf_matrix"], num_classes, cmap="Purples")
                conf_matrix_fig.savefig(image_folder / f"{phase}_confusion_matrix_epoch.png")
                plt.close(conf_matrix_fig)

            if "conf_matrix_slides" in metrics:
                conf_matrix_slides_fig = plot_confusion_matrix(
                    metrics["conf_matrix_slides"], num_classes, title="Confusion Matrix (Slides)", cmap="Blues"
                )
                conf_matrix_slides_fig.savefig(image_folder / f"{phase}_confusion_matrix_slides_epoch.png")
                plt.close(conf_matrix_slides_fig)

            if "all_labels" in metrics and "all_probs" in metrics:
                roc_fig = plot_roc_curve(metrics["all_labels"], metrics["all_probs"], num_classes)
                roc_fig.savefig(image_folder / f"{phase}_roc_curve_epoch.png")
                plt.close(roc_fig)

                # Add precision-recall curve saving
                pr_curve_fig = plot_precision_recall_curve(metrics["all_labels"], metrics["all_probs"], num_classes)
                pr_curve_fig.savefig(image_folder / f"{phase}_pr_curve_epoch.png")
                plt.close(pr_curve_fig)


def send_logs_to_wandb(cfg, flat_config):
    if cfg.slurm.enabled:
        run = wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            config=flat_config,
            id=str(cfg.slurm.job_id),
            mode="offline",
        )
        logger.info(f"Initialized WandB, run name: {run.name}")
        logger.info(f"WANDB Run id: {run.id}")
        logger.info("Logging to WANDB...")

        with open(cfg.training.log_dir + "/training_logs.txt", "r") as f:
            for line in f:
                wandb.log(eval(line))

        image_folder = Path(cfg.training.log_dir) / "log_images"
        if image_folder.exists():
            for image_file in image_folder.glob("*.png"):
                parts = image_file.stem.split("_")
                try:
                    epoch = int(parts[-1])
                    image_name = "_".join(parts[:-1])
                    wandb.log({image_name: wandb.Image(str(image_file))})
                except ValueError:
                    image_name = image_file.stem
                    wandb.log({image_name: wandb.Image(str(image_file))})
                    logger.warning(f"Logged {image_file} without an epoch number.")

        wandb.finish()

    elif cfg.wandb.enabled and not cfg.slurm.enabled:
        wandb.finish()


class TqdmToLogger(io.StringIO):
    """
    Output stream for TQDM which will output to logger module instead of
    the StdOut.
    """

    def __init__(self, logger, level=logging.INFO):
        super(TqdmToLogger, self).__init__()
        self.logger = logger
        self.level = level

    def write(self, buf):
        # Log only if the buffer is not empty
        if buf.strip():
            self.logger.log(self.level, buf.strip())

    def flush(self):
        pass


class SLURMLogger:
    def __init__(self, cfg):
        self.cfg = cfg
        self.log_dir = cfg.training.log_dir
        os.makedirs(self.log_dir, exist_ok=True)

    def log_metrics(self, metrics, phase, epoch=None):
        log_file_path = os.path.join(self.log_dir, f"{phase}_metrics.txt")
        with open(log_file_path, "a") as log_file:
            log_line = f"Epoch: {epoch}, " if epoch is not None else ""
            log_line += ", ".join([f"{key}: {value:.4f}" for key, value in metrics.items()])
            log_file.write(log_line + "\n")

    def log_image(self, image, name):
        image_path = os.path.join(self.log_dir, f"{name}.png")
        image.savefig(image_path)
