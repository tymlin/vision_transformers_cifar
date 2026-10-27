import textwrap
from inspect import signature

from lightning.pytorch.callbacks import Callback, RichModelSummary
from rich import get_console
from rich.text import Text

from vision_transformers_cifar.loggers import logger


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
        scheduler_summaries = []
        for config in trainer.lr_scheduler_configs:
            scheduler = config.scheduler
            details = [f"{name}: {value}" for name, value in scheduler.state_dict().items()]
            details.extend(
                [
                    f"interval: {config.interval}",
                    f"frequency: {config.frequency}",
                ]
            )
            if config.monitor is not None:
                details.append(f"monitor: {config.monitor}")

            details_text = textwrap.indent("\n".join(details), "    ")
            scheduler_summaries.append(f"{type(scheduler).__name__}:\n{details_text}")
        scheduler_info = "\n\n".join(scheduler_summaries) or "None"
        loss_fn = pl_module.loss_fn
        loss_kwargs = {
            name: getattr(loss_fn, name) for name in signature(type(loss_fn)).parameters if hasattr(loss_fn, name)
        }
        loss_kwargs_info = ", ".join(f"{name}={value!r}" for name, value in loss_kwargs.items())
        loss_fn_info = f"{type(loss_fn).__name__}({loss_kwargs_info})"

        module_info_dict = {
            "Model structure": model_structure,
            "Model summary": model_summary,
            "Optimizer": optim_info,
            "LR scheduler": scheduler_info,
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
