import logging
import time
import wandb
import torch
import os
import omegaconf
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


def setup_environment(cfg):
    cfg.training.weights_filename = (
        cfg.training.weights_filename.split(".")[0]
        + "_"
        + cfg.model.name
        + "_"
        + time.strftime("%Y%m%d_%H%M%S")
        + ".pth"
    )
    cfg.training.calibrated_weights_filename = (
        cfg.training.calibrated_weights_filename.split(".")[0]
        + "_"
        + cfg.model.name
        + "_"
        + time.strftime("%Y%m%d_%H%M%S")
        + ".pth"
    )
    cfg.data.random_seed = int(cfg.data.random_seed)
    cfg.data.num_classes = int(cfg.data.num_classes)
    cfg.data.num_imgs_per_slide = int(cfg.data.num_imgs_per_slide)
    cfg.model.use_complex_head = cfg.model.use_complex_head    # saranga: redundant?
    cfg.optimizer.batch_size = int(cfg.optimizer.batch_size)
    cfg.training.max_epochs = int(cfg.training.max_epochs)
    flat_config = setup_wandb(cfg)
    return flat_config


def setup_wandb(cfg):
    flat_config = flatten_config(cfg)
    if cfg.wandb.enabled and not cfg.slurm.enabled:
        if cfg.wandb.run_name is not None:
            wandb.init(
                project=cfg.wandb.project,
                entity=cfg.wandb.entity,
                config=flat_config,
                name=cfg.wandb.run_name,
            )
        else:
            wandb.init(
                project=cfg.wandb.project,
                entity=cfg.wandb.entity,
                config=flat_config,
            )
    else:
        logger.warning("Wandb is not enabled or SLURM is enabled. Not initializing wandb.")
    return flat_config


def update_config_from_wandb(cfg):
    """Update the configuration with parameters from wandb.config if they exist."""
    for key in wandb.config.keys():
        last_element = key.split(".")[-1]
        if last_element in cfg:
            setattr(cfg, last_element, wandb.config[key])
            print(f"Updated {last_element} to {wandb.config[key]}")


def flatten_config(cfg):
    config_dict = OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)
    flat_config = {}
    for key, value in config_dict.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                flat_config[f"{key}.{sub_key}"] = sub_value
        else:
            flat_config[key] = value
    return flat_config


def setup_device(cfg):
    assert cfg.devices is not None, "Devices must be specified in the config file."

    if not torch.cuda.is_available():
        print("CUDA is not available. Using CPU.")
        return torch.device("cpu")

    def get_visible_devices():
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            return [int(x) for x in os.environ["CUDA_VISIBLE_DEVICES"].split(",")]
        return list(range(torch.cuda.device_count()))

    slurm_enabled = os.environ.get("SLURM_JOB_ID") is not None

    if slurm_enabled:
        num_devices = int(cfg.devices)
        visible_devices = get_visible_devices()

        if num_devices > len(visible_devices):
            raise ValueError(f"Requested {num_devices} devices, but only {len(visible_devices)} are available.")

        device_list = visible_devices[:num_devices]

    else:
        if isinstance(cfg.devices, list):
            device_list = cfg.devices
        elif isinstance(cfg.devices, omegaconf.listconfig.ListConfig):
            device_list = list(cfg.devices)
        elif isinstance(cfg.devices, (int, str)):
            device_list = [int(cfg.devices)]
        else:
            raise ValueError(f"Unsupported type for devices: {type(cfg.devices)}")

    for dev in device_list:
        if dev >= torch.cuda.device_count():
            raise ValueError(f"Invalid device ID {dev}. Available devices: 0 to {torch.cuda.device_count() - 1}")

    # For now, we'll still use only the first device if multiple are specified
    if len(device_list) > 1:
        logger.warning("Warning: Multiple devices specified, but only the first one will be used.")

    selected_device = f"cuda:{device_list[0]}"
    logger.info(f"Using device {selected_device}")
    return torch.device(selected_device)
