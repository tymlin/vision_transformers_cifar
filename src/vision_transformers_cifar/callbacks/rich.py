from lightning.pytorch.callbacks import RichProgressBar

from vision_transformers_cifar.loggers import logger
from vision_transformers_cifar.utils.progress import progress_to_str


class RichProgressBarCallback(RichProgressBar):
    def __init__(self):
        super().__init__(
            refresh_rate=10,
            console_kwargs={
                "force_terminal": True,
                "force_interactive": True,
            },
        )

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)

        if not trainer.is_global_zero or self.progress is None:
            return

        logger.bind(file_only=True).info(f"\n{progress_to_str(self.progress)}")
