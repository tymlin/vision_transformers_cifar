import lightning as L

from vision_transformers_cifar.callbacks import (
    FinalMetricsCallback,
    MetricsPlotCallback,
    ModuleSummaryCallback,
    RichModelSummaryCallback,
    RichProgressBarCallback,
)
from vision_transformers_cifar.config import RunConfig
from vision_transformers_cifar.consts import NOW
from vision_transformers_cifar.data import CIFAR10Dataset, DataModule, build_transform
from vision_transformers_cifar.loggers import configure_logger, logger
from vision_transformers_cifar.modules import VisionTransformerLightningModule
from vision_transformers_cifar.utils.files import save_json
from vision_transformers_cifar.utils.torch_utils import seed_everything

if __name__ == "__main__":
    config = RunConfig()

    run_name_suffix = f"_{config.run_name_suffix}" if config.run_name_suffix else ""
    run_dirpath = config.run_dirpath / f"{NOW}{run_name_suffix}"
    configure_logger(run_dirpath=run_dirpath)
    logger.info(f"Logging into directory: `{run_dirpath}`")

    config.run_dirpath = run_dirpath
    save_json(run_dirpath / "config.json", config.model_dump(mode="json"))
    logger.info(f"Training configuration:\n{config.model_dump_json(indent=4)}")

    seed_everything(config.seed)

    transform = build_transform(image_size=config.data.image_size)
    train_ds = CIFAR10Dataset(split=config.data.train_split, transform=transform)
    val_ds = CIFAR10Dataset(split=config.data.val_split, transform=transform)
    test_ds = (
        CIFAR10Dataset(split=config.data.test_split, transform=transform)
        if config.data.test_split is not None
        else None
    )

    datamodule = DataModule(
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        persistent_workers=config.data.persistent_workers,
        pin_memory=config.data.pin_memory,
        drop_last=config.data.drop_last,
        shuffle=config.data.shuffle,
        seed=config.seed,
    )

    loggers = []
    if config.logging.tensorboard:
        loggers.append(
            L.pytorch.loggers.TensorBoardLogger(
                save_dir=run_dirpath,
                name="tensorboard",
                version="",
                default_hp_metric=False,
            )
        )
    if config.logging.csv:
        loggers.append(L.pytorch.loggers.CSVLogger(save_dir=run_dirpath, name="csv", version=""))

    callbacks = []
    if loggers:
        callbacks.append(L.pytorch.callbacks.LearningRateMonitor("epoch"))
    callbacks.extend(
        [
            ModuleSummaryCallback(),
            RichModelSummaryCallback(max_depth=2),
            RichProgressBarCallback(),
        ]
    )
    if config.logging.csv:
        callbacks.append(MetricsPlotCallback())
    callbacks.extend(
        [
            FinalMetricsCallback(),
            L.pytorch.callbacks.ModelCheckpoint(
                dirpath=run_dirpath / "checkpoints",
                mode="max",
                monitor="val_acc",
                save_last=True,
            ),
        ]
    )

    trainer = L.Trainer(
        default_root_dir=run_dirpath,
        callbacks=callbacks,
        logger=loggers or False,
        **config.trainer.model_dump(),
    )
    logger.info(f"Using device: `{trainer.strategy.root_device}`")

    module = VisionTransformerLightningModule(
        model_config=config.model,
        optimizer_config=config.optimizer,
        scheduler_config=config.scheduler,
    )
    trainer.fit(
        module,
        train_dataloaders=datamodule.get_train_dataloader(),
        val_dataloaders=datamodule.get_val_dataloader(),
    )
