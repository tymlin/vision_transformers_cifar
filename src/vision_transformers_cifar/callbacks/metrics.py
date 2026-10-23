import csv
from pathlib import Path

from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import CSVLogger

from vision_transformers_cifar.loggers import logger
from vision_transformers_cifar.visualization import plot_metric


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


class MetricsPlotCallback(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch > 0:
            self._plot_metrics(trainer)

    def on_train_end(self, trainer, pl_module):
        self._plot_metrics(trainer)

    def _plot_metrics(self, trainer):
        if not trainer.is_global_zero:
            return

        csv_logger = next(item for item in trainer.loggers if isinstance(item, CSVLogger))
        csv_filepath = Path(csv_logger.log_dir) / "metrics.csv"

        metrics_by_name: dict[str, dict[str, dict[int, float]]] = {}

        with csv_filepath.open(encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            columns = reader.fieldnames or []

            metric_columns = [
                name
                for name in columns
                if name.startswith(("train_", "val_")) and not name.endswith("_step") and f"{name}_epoch" not in columns
            ]

            for row in reader:
                if not row.get("epoch"):
                    continue

                epoch = int(row["epoch"]) + 1

                for name in metric_columns:
                    value = row.get(name)
                    if not value:
                        continue

                    stage, metric_name = name.split("_", 1)
                    metric_name = metric_name.removesuffix("_epoch")

                    metric_data = metrics_by_name.setdefault(metric_name, {})
                    stage_data = metric_data.setdefault(stage, {})
                    stage_data[epoch] = float(value)

        plots_dirpath = Path(trainer.default_root_dir) / "metrics_plots"

        for metric_name, metrics in metrics_by_name.items():
            plot_metric(
                metrics=metrics,
                metric_name=metric_name,
                save_filepath=plots_dirpath / f"{metric_name}.png",
            )
