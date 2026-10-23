from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def plot_metric(
    metrics: dict[str, dict[int, float]],
    metric_name: str,
    save_filepath: str | Path | None = None,
    show_plot: bool = False,
) -> plt.Figure:
    colors = {
        "train": "limegreen",
        "val": "royalblue",
    }

    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")

    for stage, color in colors.items():
        series = metrics.get(stage, {})
        if not series:
            continue

        points = sorted((int(epoch), value) for epoch, value in series.items())
        epochs, values = zip(*points)

        ax.plot(
            epochs,
            values,
            label=stage,
            color=color,
            marker="o",
            markersize=4,
        )

    if not ax.lines:
        plt.close(fig)
        raise ValueError(f"No values recorded for metric {metric_name!r}.")

    ax.set_title(metric_name.upper())
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric_name)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(alpha=0.3)
    ax.legend()

    if save_filepath is not None:
        save_filepath = Path(save_filepath)
        save_filepath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_filepath, dpi=200)

    if show_plot:
        plt.show()

    plt.close(fig)
    return fig
