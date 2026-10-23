import random

import lightning as L
import numpy as np
import torch

from vision_transformers_cifar.loggers import logger


def seed_everything(seed: int) -> None:
    logger.info(f"Setting seed to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    L.seed_everything(seed, workers=True, verbose=False)
