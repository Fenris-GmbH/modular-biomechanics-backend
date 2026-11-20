import logging
import os
from typing import Union


def setup_logging(
    logfile: str = "log_summary.log",
    log_dir: Union[str, os.PathLike] = "logs",
) -> logging.Logger:
    """Configure console and file handlers for the metrics logger.

    The log directory is created when needed. Existing handlers on the named
    logger are cleared to prevent duplicate messages across repeated setup.
    """
    # Ensure log directory exists
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, logfile)

    logger = logging.getLogger("metrics_logger")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # Prevent duplicate logs in Jupyter or reruns

    # Formatter
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    # File handler (saves log to disk)
    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setFormatter(formatter)

    # Console handler (prints log to console)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
