from __future__ import annotations

import logging
import sys
from pathlib import Path

from loguru import logger as _logger

from vision_transformers_cifar.consts import LOGGER_FORMAT, LOGGER_LEVEL, RUNS_DIRPATH

logger = _logger


def get_log_filepath(
    run_name: str | None = None,
    run_dirpath: str | Path | None = None,
    log_filename: str | None = None,
) -> Path | None:
    """Resolve the logfile path for a run.

    :param run_name: Name of the run directory inside ``RUNS_DIRPATH``.
    :param run_dirpath: Explicit run directory path.
    :param log_filename: Filename for the log sink. If ``None``, file logging is disabled.
    :return: Path to the log file for the selected run, or ``None`` if file logging is disabled.
    :raises ValueError: If both ``run_name`` and ``run_dirpath`` are provided.
    """
    if run_name is not None and run_dirpath is not None:
        raise ValueError("Provide either 'run_name' or 'run_dirpath', not both.")

    if log_filename is None:
        return None

    if run_dirpath is not None:
        run_dir = Path(run_dirpath)
    elif run_name is not None:
        run_dir = RUNS_DIRPATH / run_name
    else:
        return None

    return run_dir / log_filename


class _LoguruHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        def preserve_source(loguru_record):
            loguru_record.update(
                name=record.name,
                function=record.funcName,
                line=record.lineno,
            )

        logger.patch(preserve_source).opt(
            exception=record.exc_info,
        ).log(level, record.getMessage())


def _configure_lightning_logging() -> None:
    handler = _LoguruHandler()

    for name in ("lightning", "lightning.pytorch", "lightning.fabric"):
        lightning_logger = logging.getLogger(name)

        for previous in lightning_logger.handlers[:]:
            lightning_logger.removeHandler(previous)
            previous.close()

        lightning_logger.addHandler(handler)
        lightning_logger.setLevel(logging.DEBUG)
        lightning_logger.propagate = False


def configure_logger(
    run_name: str | None = None,
    run_dirpath: str | Path | None = None,
    log_filename: str | None = "logs.log",
    level: str = LOGGER_LEVEL,
    format_: str = LOGGER_FORMAT,
    add_console_sink: bool = True,
) -> Path | None:
    """Configure the shared Loguru logger for console and optional per-run file logging.

    :param run_name: Name of the run directory inside ``RUNS_DIRPATH``.
    :param run_dirpath: Explicit run directory path.
    :param log_filename: Filename for the log sink. If ``None``, file logging is disabled.
    :param level: Logging level applied to configured sinks.
    :param format_: Loguru format string.
    :param add_console_sink: Whether to keep a stdout sink.
    :return: The resolved log filepath, or ``None`` if no file sink is configured.
    """
    log_filepath = get_log_filepath(
        run_name=run_name,
        run_dirpath=run_dirpath,
        log_filename=log_filename,
    )

    logger.remove()

    if add_console_sink:
        logger.add(
            sys.stdout,
            colorize=True,
            level=level,
            format=format_,
            filter=lambda record: not record["extra"].get("file_only", False),
        )

    if log_filepath is not None:
        log_filepath.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_filepath,
            colorize=False,
            level=level,
            format=format_,
        )

    _configure_lightning_logging()
    return log_filepath


configure_logger()

if __name__ == "__main__":
    logger.info("Logger configured")
    logger.debug("Debug example for {}", "video metadata")
    logger.warning("Warning example for {}", "missing optional field")
    logger.error("Error example for {}", "failed request")
