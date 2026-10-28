import lightning as L
from torch.utils.data import default_collate
from torchvision.transforms import v2

from vision_transformers_cifar.callbacks import (
    FinalMetricsCallback,
    MetricsPlotCallback,
    ModuleSummaryCallback,
    RichModelSummaryCallback,
    RichProgressBarCallback,
)
from vision_transformers_cifar.config import DataConfig, RunConfig, TrainerConfig, ViTModelConfig
from vision_transformers_cifar.consts import NOW
from vision_transformers_cifar.data import (
    CIFAR10Dataset,
    DataModule,
    build_augmix_transform,
    build_autoaugment_transform,
    build_basic_transform,
    build_batch_transform,
    build_eval_transform,
    build_randaugment_transform,
    build_standard_transform,
    build_threeaugment_transform,
    build_timm_randaugment_transform,
)
from vision_transformers_cifar.loggers import configure_logger, logger
from vision_transformers_cifar.modules import VisionTransformerLightningModule
from vision_transformers_cifar.utils.files import save_json
from vision_transformers_cifar.utils.torch_utils import seed_everything

EPOCHS = 500
BATCH_SIZE = 128

if __name__ == "__main__":
    config = RunConfig(
        run_name_suffix="vit-0.8M_augmix_label-smoothing-0.1_cutmix-mixup_init-weights",
        data=DataConfig(
            batch_size=BATCH_SIZE,
            cutmix_enabled=True,
            mixup_enabled=True,
        ),
        model=ViTModelConfig(
            embed_dim=128,
            hidden_dim=256,
            num_heads=4,
            dropout=0.1,
        ),
        trainer=TrainerConfig(max_epochs=EPOCHS),
    )

    run_name_suffix = f"_{config.run_name_suffix}" if config.run_name_suffix else ""
    run_dirpath = config.run_dirpath / f"{NOW}{run_name_suffix}"
    configure_logger(run_dirpath=run_dirpath)
    logger.info(f"Logging into directory: `{run_dirpath}`")

    config.run_dirpath = run_dirpath
    save_json(run_dirpath / "config.json", config.model_dump(mode="json"))
    logger.info(f"Training configuration:\n{config.model_dump_json(indent=4)}")

    seed_everything(config.seed)

    train_transforms = build_augmix_transform(image_size=config.data.image_size)
    eval_transforms = build_eval_transform(image_size=config.data.image_size)
    batch_transform = build_batch_transform(
        mixup_enabled=config.data.mixup_enabled,
        cutmix_enabled=config.data.cutmix_enabled,
        num_classes=config.model.num_classes,
    )

    train_ds = CIFAR10Dataset(split=config.data.train_split, transform=train_transforms)
    val_ds = CIFAR10Dataset(split=config.data.val_split, transform=eval_transforms)
    test_ds = (
        CIFAR10Dataset(split=config.data.test_split, transform=eval_transforms)
        if config.data.test_split is not None
        else None
    )

    train_collate_fn = default_collate
    if batch_transform is not None:
        train_collate_fn = v2.Compose(
            [
                default_collate,
                batch_transform,
            ]
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
        collate_fn_train=train_collate_fn,
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
