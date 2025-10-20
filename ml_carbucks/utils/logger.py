import logging
import sys
import os
from logging.handlers import RotatingFileHandler


def setup_logger(
    name: str | None = None,
    log_file: str = "logs/logs.log",
    console_level=logging.INFO,
    file_level=logging.DEBUG,
) -> logging.Logger:
    """Set up a logger with both console and rotating file handlers.

    Args:
        name (str | None, optional): Logger name, defaults to module name.
        log_file (str, optional): File path for log file, defaults to 'logs/logs.log'.
        console_level (int, optional): Log level for console output.
        file_level (int, optional): Log level for file output.

    Returns:
        logging.Logger: Configured logger instance.
    """

    logger = logging.getLogger(name or __name__)
    logger.setLevel(min(console_level, file_level))  # Set lowest level to capture all

    # Prevent adding handlers multiple times in interactive environments
    if logger.hasHandlers():
        return logger

    # Ensure log directory exists
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    # Rotating file handler to limit file size and backups
    file_handler = RotatingFileHandler(log_file, maxBytes=10_000_000, backupCount=5)
    file_handler.setLevel(file_level)
    file_formatter = logging.Formatter(
        "%(levelname)s %(name)s %(asctime)s [%(filename)s:%(funcName)s:%(lineno)d] | %(message)s",
        datefmt="%d/%m/%Y %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)

    # Console handler outputs to stdout
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setLevel(console_level)
    stream_formatter = logging.Formatter(
        "%(levelname)s %(name)s %(asctime)s | %(message)s", datefmt="%H:%M:%S"
    )
    stream_handler.setFormatter(stream_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    # Prevent logs from being propagated to root logger if undesired
    logger.propagate = False

    return logger
