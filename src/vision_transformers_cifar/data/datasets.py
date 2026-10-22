import textwrap
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torchvision
from PIL import Image
from torch import Tensor
from torchvision.datasets import VisionDataset

from vision_transformers_cifar.consts import DATASETS_DIRPATH
from vision_transformers_cifar.data.types import DatasetSplit, DatasetType
from vision_transformers_cifar.loggers import logger
from vision_transformers_cifar.utils.files import create_directory


class ImageDataset(VisionDataset, ABC):
    name: str
    dataset: VisionDataset
    classes_names: list[str]
    classes_int2str: dict[int, str]
    classes_str2int: dict[str, int]
    num_classes: int
    collate_fn: Callable | None = None

    def __init__(self, transform: Callable | None):
        super().__init__()

        self.transform = transform if transform is not None else torchvision.transforms.ToTensor()
        create_directory(DATASETS_DIRPATH, exist_ok=True)

    @abstractmethod
    def get_raw_image(self, index: int) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def get_raw_target(self, index: int) -> int:
        raise NotImplementedError()

    @abstractmethod
    def get_class_names(self) -> list[str]:
        raise NotImplementedError()

    @abstractmethod
    def get_dataset_len(self) -> int:
        raise NotImplementedError()

    def __len__(self) -> int:
        return self.get_dataset_len()

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        image = self.get_raw_image(index)
        target = self.get_raw_target(index)
        image = Image.fromarray(image)
        image = self.transform(image)
        return image, target

    def get_raw_sample(self, index: int) -> tuple[np.ndarray, int]:
        image = self.get_raw_image(index)
        target = self.get_raw_target(index)
        return image, target

    def initialize_classes_dicts(self) -> None:
        self.classes_names = self.get_class_names()
        self.classes_int2str = {k: v for k, v in enumerate(self.classes_names)}
        self.classes_str2int = {v: k for k, v in enumerate(self.classes_names)}
        self.num_classes = len(self.classes_names)

    def plot_images(
        self,
        indices: list[int],
        after_transforms: bool = False,
        save_filepath: str | Path | None = None,
        show_plot: bool = False,
    ) -> plt.Figure:
        if not indices:
            msg = "`indices` must contain at least one element."
            logger.error(msg)
            raise ValueError(msg)

        n = len(indices)
        fig, axes = plt.subplots(1, n, figsize=(3 * n, 4))

        if n == 1:
            axes = [axes]
        for idx, ax in zip(indices, axes):
            if after_transforms:
                image, target = self[idx]
                image = image.detach().cpu().permute(1, 2, 0).numpy()
                vmax = 1.0
            else:
                image, target = self.get_raw_sample(idx)
                vmax = 255
            ax.imshow(image, cmap="gray", vmin=0, vmax=vmax)
            ax_px_width = ax.get_window_extent().width
            char_pixel_ratio = 7
            char_width = int(ax_px_width / char_pixel_ratio)
            title = f"{self.classes_int2str[target]}"
            wrapped_title = "\n".join(textwrap.wrap(title, width=char_width))
            ax.set_title(wrapped_title, fontsize=14)
        plt.tight_layout()

        if save_filepath is not None:
            save_filepath = Path(save_filepath)
            create_directory(save_filepath.parent, parents=True, exist_ok=True)
            fig.savefig(save_filepath, dpi=200, bbox_inches="tight")
            logger.info(f"Saved figure to '{save_filepath}'")
        if show_plot:
            plt.show()
        plt.close(fig)

        return fig


class CIFAR10Dataset(ImageDataset):
    def __init__(self, split: DatasetSplit, transform: Callable | None = None) -> None:
        if split not in (DatasetSplit.TRAIN, DatasetSplit.TEST):
            msg = f"Unsupported dataset split: {split}. " "Expected 'train' or 'test'."
            logger.error(msg)
            raise ValueError(msg)

        super().__init__(transform=transform)

        split_bool = split == DatasetSplit.TRAIN
        dataset_dir = DATASETS_DIRPATH / "cifar10"
        create_directory(dataset_dir, exist_ok=True)
        self.dataset: torchvision.datasets.CIFAR10 = torchvision.datasets.CIFAR10(
            root=dataset_dir,
            train=split_bool,
            transform=None,
            download=True,
        )

        self.name = f"{self.dataset.__class__.__name__}-{split}"
        self.initialize_classes_dicts()

        logger.info(f"Initialized `{self.name}` dataset")

    def get_raw_image(self, index: int) -> np.ndarray:
        return self.dataset.data[index]

    def get_raw_target(self, index: int) -> int:
        return self.dataset.targets[index]

    def get_class_names(self) -> list[str]:
        return self.dataset.classes

    def get_dataset_len(self) -> int:
        return len(self.dataset.targets)
