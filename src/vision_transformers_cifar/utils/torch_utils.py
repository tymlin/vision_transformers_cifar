import random

import lightning as L
import numpy as np
import torch

from vision_transformers_cifar.loggers import logger
from vision_transformers_cifar.models import Device


def seed_everything(seed: int) -> None:
    logger.info(f"Setting seed to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    L.seed_everything(seed, workers=True, verbose=False)


def check_torch_devices_availability(as_dict: bool = False) -> dict[Device, bool] | list[Device]:
    """Checks the availability of CPU, CUDA, and MPS devices in PyTorch.

    as_dict: bool: Whether to return the availability status as a dictionary.

    :return dict: A dictionary with the availability status of each device.
    """
    available_devices = {
        Device.CPU: True,
        Device.CUDA: torch.cuda.is_available(),
        Device.MPS: torch.backends.mps.is_available(),
    }
    if not as_dict:
        available_devices = [device for device, status in available_devices.items() if status]
    return available_devices


def auto_detect_device() -> Device:
    """Select an available device, preferring CUDA, then MPS, then CPU.

    :return Device: Selected device
    """
    available_devices = check_torch_devices_availability()

    if Device.CUDA in available_devices:
        return Device.CUDA

    if Device.MPS in available_devices:
        return Device.MPS

    return Device.CPU
