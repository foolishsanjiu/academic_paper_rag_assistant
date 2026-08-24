"""Logging configuration for the application."""

import logging
from logging.handlers import RotatingFileHandler

from config import LOG_DIRECTORY, LOG_FILE

LOG_DIR = LOG_DIRECTORY


def setup_logging() -> None:
    """
    Configure console and rotating-file logging.

    The log file is limited to 1 MB. Up to three backup files are kept.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # 防止多次调用 setup_logging() 后重复输出日志
    root_logger.handlers.clear()

    formatter = logging.Formatter(
        fmt=(
            "%(asctime)s | %(levelname)s | "
            "%(name)s | %(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        filename=LOG_FILE,
        maxBytes=1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # 减少第三方 HTTP 库的底层日志噪声
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
