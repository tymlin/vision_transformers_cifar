from typing import Callable

import torch
from torchvision.transforms import Compose

from vision_transformers_cifar.data.datasets import ImageDataset
from vision_transformers_cifar.data.types import DatasetSplit
from vision_transformers_cifar.loggers import logger


class DataModule:
    def __init__(
        self,
        train_ds: ImageDataset,
        val_ds: ImageDataset,
        test_ds: ImageDataset | None,
        inverse_transform: Compose | None = None,
        batch_size: int = 32,
        num_workers: int = 8,
        persistent_workers: bool = False,
        pin_memory: bool = True,
        drop_last: bool = True,
        shuffle: bool = True,
        seed: int = 42,
        collate_fn_train: Callable | None = None,
        collate_fn_validation: Callable | None = None,
        collate_fn_test: Callable | None = None,
    ):
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds
        self.inverse_transform = inverse_transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        self.train_generator = torch.Generator().manual_seed(seed)

        self.total_batches = {}
        self.train_dataloader = None
        self.val_dataloader = None
        self.test_dataloader = None
        self.collate_fns: dict[DatasetSplit, Callable | None] = {
            DatasetSplit.TRAIN: collate_fn_train,
            DatasetSplit.VALIDATION: collate_fn_validation,
            DatasetSplit.TEST: collate_fn_test,
        }

        self.setup_dataloaders()
        # log info at the end of the init
        self._log_info()

    def setup_dataloaders(self) -> None:
        self.train_dataloader = self._build_dataloader(DatasetSplit.TRAIN)
        self.val_dataloader = self._build_dataloader(DatasetSplit.VALIDATION)
        if self.test_ds is not None:
            self.test_dataloader = self._build_dataloader(DatasetSplit.TEST)

    def get_dataset(self, split: DatasetSplit) -> ImageDataset | None:
        if split == DatasetSplit.TRAIN:
            return self.train_ds
        elif split == DatasetSplit.VALIDATION:
            return self.val_ds
        elif split == DatasetSplit.TEST:
            return self.test_ds
        else:
            allowed_splits = " or ".join(f"'{split.value}'" for split in DatasetSplit)
            msg = f"Dataset split must be either {allowed_splits}"
            logger.error(msg)
            raise ValueError(msg)

    def get_dataloader(self, split: DatasetSplit) -> torch.utils.data.DataLoader | None:
        if split == DatasetSplit.TRAIN:
            return self.train_dataloader
        elif split == DatasetSplit.VALIDATION:
            return self.val_dataloader
        elif split == DatasetSplit.TEST:
            return self.test_dataloader
        else:
            allowed_splits = " or ".join(f"'{split.value}'" for split in DatasetSplit)
            msg = f"Dataset split must be either {allowed_splits}"
            logger.error(msg)
            raise ValueError(msg)

    def get_train_dataloader(self) -> torch.utils.data.DataLoader:
        if self.train_dataloader is None:
            msg = "Training dataloader is not initialized. " "Call `setup_dataloaders()` first."
            logger.error(msg)
            raise RuntimeError(msg)

        return self.train_dataloader

    def get_val_dataloader(self) -> torch.utils.data.DataLoader:
        if self.val_dataloader is None:
            msg = "Validation dataloader is not initialized. " "Call `setup_dataloaders()` first."
            logger.error(msg)
            raise RuntimeError(msg)

        return self.val_dataloader

    def get_test_dataloader(self) -> torch.utils.data.DataLoader | None:
        if self.test_ds is None:
            return None

        if self.test_dataloader is None:
            msg = "Test dataloader is not initialized. " "Call `setup_dataloaders()` first."
            logger.error(msg)
            raise RuntimeError(msg)

        return self.test_dataloader

    def _build_dataloader(self, split: DatasetSplit) -> torch.utils.data.DataLoader:
        shuffle = split == DatasetSplit.TRAIN and self.shuffle  # shuffle only for train
        drop_last = self.drop_last if split == DatasetSplit.TRAIN else False  # don't drop last for val/test

        dataset = self.get_dataset(split)
        if dataset is None:
            msg = f"Dataset for {split} split is None"
            logger.error(msg)
            raise ValueError(msg)

        collate_fn = self.collate_fns[split]
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn,
            generator=(self.train_generator if split == DatasetSplit.TRAIN else None),
        )

        self.total_batches[split] = len(dataloader)
        return dataloader

    def _log_info(self) -> None:
        msg = "Datasets summary: "
        for split in DatasetSplit:
            ds = self.get_dataset(split)
            ds_name = "None" if ds is None else ds.name
            num_samples = 0 if ds is None else len(ds)
            ds_info = f"{num_samples} ({ds_name})"
            msg += f"\n\t{split.rjust(10)}: {ds_info}"
        logger.info(msg)

        msg = "Transforms summary:"
        for split in DatasetSplit:
            dataset = self.get_dataset(split)
            if dataset is None:
                continue
            transform_info = "\n".join(f"\t\t{line}" for line in str(dataset.transform).splitlines())
            msg += f"\n\t{split.value}:\n{transform_info}"
        logger.info(msg)

        msg = "Dataloader batches summary:"
        for split in DatasetSplit:
            dl = self.get_dataloader(split)
            dl_info = "None" if dl is None else f"{len(dl)} batches"
            msg += f"\n\t{split.value.rjust(10)}: {dl_info}"
        logger.info(msg)

    def state_dict(self) -> dict:
        state = {
            "train_generator_state": self.train_generator.get_state(),
        }
        return state

    def load_state_dict(self, state_dict: dict) -> None:
        self.train_generator.set_state(state_dict["train_generator_state"])
        logger.info("Loaded datamodule state dict")
