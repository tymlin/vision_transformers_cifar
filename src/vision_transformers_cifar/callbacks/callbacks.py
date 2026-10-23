import textwrap

from lightning.pytorch.callbacks import Callback, RichModelSummary, RichProgressBar
from rich import get_console
from rich.text import Text

from vision_transformers_cifar.loggers import logger
from vision_transformers_cifar.utils.progress import progress_to_str


class ModuleSummaryCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return

        dataloader = trainer.train_dataloader
        image, _ = dataloader.dataset[0]
        channels, height, width = image.shape
        input_shape = (dataloader.batch_size, channels, height, width)

        model_structure = str(pl_module.model)
        model_summary = pl_module.model.summary(input_shape)
        optim_info = "\n\n".join(str(optimizer) for optimizer in trainer.optimizers)
        loss_fn_info = str(pl_module.loss_fn)

        module_info_dict = {
            "Model structure": model_structure,
            "Model summary": model_summary,
            "Optimizer": optim_info,
            "Loss function": loss_fn_info,
        }

        module_name = type(pl_module).__name__
        msg = f"`{module_name}` summary:\n"

        for name, info in module_info_dict.items():
            info = textwrap.indent(info, "\t\t")
            msg += f"\t{name}:\n{info}\n"

        logger.info(msg)


class RichModelSummaryCallback(RichModelSummary):
    @staticmethod
    def summarize(*args, **kwargs):
        with get_console().capture() as captured:
            RichModelSummary.summarize(*args, **kwargs)

        summary = Text.from_ansi(captured.get()).plain.rstrip()
        logger.info(f"\n{summary}")


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


class FinalMetricsCallback(Callback):
    def on_train_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return

        metrics = trainer.callback_metrics
        summary = "\n".join(
            f"\t{name}: {float(value):.4f}"
            for name, value in metrics.items()
            if name.startswith(("train_", "val_", "test_"))
            and not name.endswith("_step")
            and f"{name}_epoch" not in metrics
        )

        logger.info(f"Final epoch metrics:\n{summary}")
