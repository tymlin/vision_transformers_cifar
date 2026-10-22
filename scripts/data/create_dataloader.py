from vision_transformers_cifar.data import CIFAR10Dataset, DataModule, DatasetSplit
from vision_transformers_cifar.loggers import logger

if __name__ == "__main__":
    train_ds = CIFAR10Dataset(split=DatasetSplit.TRAIN)
    test_ds = CIFAR10Dataset(split=DatasetSplit.TEST)

    datamodule = DataModule(
        train_ds=train_ds,
        val_ds=test_ds,
        test_ds=test_ds,
        batch_size=32,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
        shuffle=True,
    )

    train_dataloader = datamodule.get_train_dataloader()

    loader_iter = iter(train_dataloader)
    images, targets = next(loader_iter)
    logger.info(f"Images shape: {images.shape}, targets shape: {targets.shape}")
