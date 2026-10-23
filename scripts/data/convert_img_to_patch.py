import matplotlib.pyplot as plt
import torchvision

from vision_transformers_cifar.data import CIFAR10Dataset, DataModule, DatasetSplit, build_basic_transform
from vision_transformers_cifar.loggers import logger
from vision_transformers_cifar.models import img_to_patch

if __name__ == "__main__":
    transform = build_basic_transform()
    train_ds = CIFAR10Dataset(split=DatasetSplit.TRAIN, transform=transform)
    test_ds = CIFAR10Dataset(split=DatasetSplit.TEST, transform=transform)

    datamodule = DataModule(
        train_ds=train_ds,
        val_ds=test_ds,
        test_ds=test_ds,
        batch_size=8,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
        shuffle=True,
    )

    train_dataloader = datamodule.get_train_dataloader()

    loader_iter = iter(train_dataloader)
    images, targets = next(loader_iter)
    logger.info(f"Images shape: {images.shape}, targets shape: {targets.shape}")

    patches = img_to_patch(images, 4, flatten_channels=False)

    fig, ax = plt.subplots(images.shape[0], 1, figsize=(10, 4))
    fig.suptitle("Images as input sequences of patches")
    for i in range(images.shape[0]):
        img_grid = torchvision.utils.make_grid(patches[i], nrow=64, normalize=True, pad_value=0.9)
        img_grid = img_grid.permute(1, 2, 0)
        ax[i].imshow(img_grid)
        ax[i].axis("off")
    plt.tight_layout()
    plt.show()
    plt.close()
