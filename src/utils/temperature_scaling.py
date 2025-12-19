import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import logging
from pathlib import Path

from sklearn.metrics import roc_auc_score


logger = logging.getLogger(__name__)


class TemperatureScaling(nn.Module):
    def __init__(self, temperature=1.5):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * temperature)

    def forward(self, logits):
        return self.temperature_scale(logits)

    def __repr__(self):
        return f"TemperatureScaling(temperature={self.temperature})"

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature


class ECELoss(nn.Module):
    """
    Copyright to https://github.com/gpleiss/temperature_scaling/blob/master/temperature_scaling.py

    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).

    The input to this loss is the logits of a model, NOT the softmax scores.

    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:

    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |

    We then return a weighted average of the gaps, based on the number
    of samples in each bin

    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """

    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece


def calibrate_temperature(model, val_loader, criterion, device, num_classes):
    temperature = TemperatureScaling().to(device)
    nll_criterion = criterion
    ece_criterion = ECELoss().to(device)

    logits_list = []
    labels_list = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Collecting logits", leave=False):
            if isinstance(model, ChAdaViT):
                if len(batch) == 7:
                    (
                        inputs,
                        labels,
                        covariables,
                        slide_ids,
                        original_labels,
                        indices,
                        num_channels_list,
                    ) = batch
                    covariables = {k: v.to(device) for k, v in covariables.items()}
                else:
                    inputs, labels, slide_ids, original_labels, num_channels_list = batch
                    covariables = None
            else:
                if len(batch) == 6:
                    inputs, labels, covariables, slide_ids, original_labels, indices = batch
                    covariables = {k: v.to(device) for k, v in covariables.items()}
                else:
                    inputs, labels, slide_ids, original_labels, indices = batch
                    covariables = None

            inputs = inputs.to(device)
            labels = labels.to(device)
            original_labels = original_labels.to(device)

            if covariables is not None:
                logits = model(inputs, covariables)
            else:
                if isinstance(model, ChAdaViT):
                    model.mixed_channels = False
                    logits = model(x=inputs, index=0, list_num_channels=[num_channels_list])
                    logits = logits.flatten(start_dim=1)
                else:
                    logits = model(inputs)

            logits_list.append(logits)
            labels_list.append(labels)

    logits = torch.cat(logits_list).to(device)
    labels = torch.cat(labels_list).to(device)

    # Adjust labels for binary classification
    if num_classes == 2:
        labels = labels.float().unsqueeze(1)

    before_temperature_nll = nll_criterion(logits, labels).item()
    before_temperature_ece = ece_criterion(logits, labels).item()
    logger.info(f"Before temperature - NLL: {before_temperature_nll:.3f}, ECE: {before_temperature_ece:.3f}")

    optimizer = torch.optim.LBFGS([temperature.temperature], lr=0.01, max_iter=50)

    def eval():
        optimizer.zero_grad()
        loss = nll_criterion(temperature(logits), labels)
        loss.backward()
        return loss

    optimizer.step(eval)

    after_temperature_nll = nll_criterion(temperature(logits), labels).item()
    after_temperature_ece = ece_criterion(temperature(logits), labels).item()
    logger.info(f"Optimal temperature: {temperature.temperature.item():.3f}")
    logger.info(f"After temperature - NLL: {after_temperature_nll:.3f}, ECE: {after_temperature_ece:.3f}")

    # Calculate probabilities and predictions
    if num_classes == 2:
        before_probs = torch.sigmoid(logits)  # Use sigmoid for binary
        after_probs = torch.sigmoid(temperature(logits))
        before_preds = (before_probs > 0.5).int()  # Binary predictions
        after_preds = (after_probs > 0.5).int()
    else:
        before_probs = F.softmax(logits, dim=1)  # Use softmax for multi-class
        after_probs = F.softmax(temperature(logits), dim=1)
        before_preds = torch.argmax(before_probs, dim=1)
        after_preds = torch.argmax(after_probs, dim=1)

    before_accuracy = (before_preds == labels).float().mean().item()
    after_accuracy = (after_preds == labels).float().mean().item()

    labels_np = labels.cpu().numpy()
    before_probs_np = before_probs.detach().cpu().numpy()
    after_probs_np = after_probs.detach().cpu().numpy()

    if num_classes == 2:
        before_auc = roc_auc_score(labels_np, before_probs_np)
        after_auc = roc_auc_score(labels_np, after_probs_np)
    else:
        before_auc = roc_auc_score(labels_np, before_probs_np, multi_class="ovr", average="macro")
        after_auc = roc_auc_score(labels_np, after_probs_np, multi_class="ovr", average="macro")

    return temperature, {
        "before_calibration": {
            "nll": before_temperature_nll,
            "ece": before_temperature_ece,
            "accuracy": before_accuracy,
            "auc": before_auc,
            "probs": before_probs_np,
            "logits": logits.detach().cpu().numpy(),
        },
        "after_calibration": {
            "nll": after_temperature_nll,
            "ece": after_temperature_ece,
            "accuracy": after_accuracy,
            "auc": after_auc,
            "probs": after_probs_np,
            "logits": temperature(logits).detach().cpu().numpy(),
        },
        "labels": labels_np,
        "temperature": temperature.temperature.item(),
    }


def calibrate_model(cfg, model, val_loader, criterion, device, num_classes):
    logger.info("Calibrating model with temperature scaling...")
    temperature_layer, calibration_results = calibrate_temperature(model, val_loader, criterion, device, num_classes)

    save_dir = Path(cfg.model.weights_path) / "custom"
    save_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "temperature_state_dict": temperature_layer.state_dict(),
        },
        save_dir / cfg.training.calibrated_weights_filename,
    )

    logger.info("Calibration complete.")
    return calibration_results


def calibrate(model, val_loader, criterion, device, num_classes):
    """
    Perform simple temperature scaling calibration.

    Args:
        model (torch.nn.Module): The model to calibrate.
        val_loader (DataLoader): Validation data for calibration.
        criterion (Callable): Loss function for calibration.
        device (torch.device): Device to perform calibration on.
        num_classes (int): Number of output classes.

    Returns:
        temperature_layer (TemperatureScalingLayer): Trained temperature scaling layer.
        calibration_results (dict): Metrics from the calibration process.
    """
    model.eval()
    temperature_layer = TemperatureScaling().to(device)
    optimizer = torch.optim.LBFGS([temperature_layer.temperature], lr=0.01, max_iter=50)

    def eval_fn():
        loss = 0
        for batch in val_loader:
            if isinstance(model, ChAdaViT):
                if len(batch) == 7:
                    (
                        inputs,
                        labels,
                        covariables,
                        slide_ids,
                        original_labels,
                        indices,
                        num_channels_list,
                    ) = batch
                    covariables = {k: v.to(device) for k, v in covariables.items()}
                else:
                    (
                        inputs,
                        labels,
                        slide_ids,
                        original_labels,
                        indices,
                        num_channels_list,
                    ) = batch
                    covariables = None
            else:
                if len(batch) == 6:
                    inputs, labels, covariables, slide_ids, original_labels, indices = batch
                    covariables = {k: v.to(device) for k, v in covariables.items()}
                else:
                    inputs, labels, slide_ids, original_labels, indices = batch
                    covariables = None

            inputs = inputs.to(device)
            labels = labels.to(device)
            original_labels = original_labels.to(device)

            with torch.no_grad():
                if covariables is not None:
                    outputs = model(inputs, covariables)
                else:
                    if isinstance(model, ChAdaViT):
                        model.mixed_channels = False
                        feats = model(x=inputs, index=0, list_num_channels=[num_channels_list]).flatten(start_dim=1)
                        outputs = model.head(feats)
                    else:
                        outputs = model(inputs)

            scaled_logits = temperature_layer(outputs)
            loss += criterion(scaled_logits, labels).item()
        return loss

    optimizer.step(eval_fn)

    return temperature_layer
