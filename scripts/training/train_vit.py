import lightning as L

from vision_transformers_cifar.callbacks import (
    FinalMetricsCallback,
    MetricsPlotCallback,
    ModuleSummaryCallback,
    RichModelSummaryCallback,
    RichProgressBarCallback,
)
from vision_transformers_cifar.consts import NOW, RUNS_DIRPATH
from vision_transformers_cifar.data import CIFAR10Dataset, DataModule, DatasetSplit, build_transform
from vision_transformers_cifar.loggers import configure_logger, logger
from vision_transformers_cifar.modules import VisionTransformerLightningModule
from vision_transformers_cifar.utils.torch_utils import auto_detect_device, seed_everything

BATCH_SIZE = 8
LR = 3e-4
MAX_EPOCHS = 10
RUN_NAME_SUFFIX = ""
MODEL_KWARGS = {
    "embed_dim": 256,
    "hidden_dim": 512,
    "num_heads": 8,
    "num_layers": 6,
    "patch_size": 4,
    "num_channels": 3,
    "num_patches": 64,
    "num_classes": 10,
    "dropout": 0.2,
}
SEED = 42

if __name__ == "__main__":
    # first configure the logger with run dirpath and file handler
    run_name_suffix = f"_{RUN_NAME_SUFFIX}" if RUN_NAME_SUFFIX else ""
    run_dirpath = RUNS_DIRPATH / f"{NOW}{run_name_suffix}"
    configure_logger(run_dirpath=run_dirpath)
    logger.info(f"Logging into directory: `{run_dirpath}`")

    seed_everything(SEED)

    transform = build_transform()
    train_ds = CIFAR10Dataset(split=DatasetSplit.TRAIN, transform=transform)
    test_ds = CIFAR10Dataset(split=DatasetSplit.TEST, transform=transform)

    datamodule = DataModule(
        train_ds=train_ds,
        val_ds=test_ds,  # use test as validation
        test_ds=test_ds,
        batch_size=32,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
        shuffle=True,
        persistent_workers=True,
        seed=SEED,
    )
    train_dataloader = datamodule.get_train_dataloader()
    validation_dataloader = datamodule.get_val_dataloader()
    test_dataloader = datamodule.get_test_dataloader()

    device = auto_detect_device()
    logger.info(f"Using device: `{device}`")

    trainer = L.Trainer(
        default_root_dir=run_dirpath,
        accelerator="auto",
        devices=1,
        max_epochs=MAX_EPOCHS,
        callbacks=[
            L.pytorch.callbacks.LearningRateMonitor("epoch"),
            ModuleSummaryCallback(),
            RichModelSummaryCallback(max_depth=2),
            RichProgressBarCallback(),
            MetricsPlotCallback(),
            FinalMetricsCallback(),
            L.pytorch.callbacks.ModelCheckpoint(
                dirpath=run_dirpath / "checkpoints",
                mode="max",
                monitor="val_acc",
                save_last=True,
            ),
        ],
        logger=[
            L.pytorch.loggers.TensorBoardLogger(
                save_dir=run_dirpath,
                name="tensorboard",
                version="",
                default_hp_metric=False,
            ),
            L.pytorch.loggers.CSVLogger(
                save_dir=run_dirpath,
                version="",
                name="csv",
            ),
        ],
        log_every_n_steps=1,
    )

    model = VisionTransformerLightningModule(model_kwargs=MODEL_KWARGS, lr=LR)
    trainer.fit(model, train_dataloader, validation_dataloader)
