import logging
import sys
from logging.handlers import SMTPHandler  # Add this import
from pathlib import Path
from typing import Optional


class CustomFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels."""

    COLORS = {
        "DEBUG": "\033[0;36m",  # Cyan
        "INFO": "\033[0;32m",  # Green
        "WARNING": "\033[0;33m",  # Yellow
        "ERROR": "\033[0;31m",  # Red
        "CRITICAL": "\033[0;37;41m",  # White on Red
        "RESET": "\033[0m",  # Reset
    }

    def format(self, record: logging.LogRecord) -> str:
        # Save original values
        orig_levelname = record.levelname

        # Add color only for console handler
        if hasattr(self, "_style"):
            # Check if this formatter is being used by a StreamHandler
            for handler in logging.getLogger().handlers:
                if (
                    isinstance(handler, logging.StreamHandler)
                    and handler.formatter == self
                ):
                    record.levelname = (
                        f"{self.COLORS.get(record.levelname, '')}"
                        f"{record.levelname}"
                        f"{self.COLORS['RESET']}"
                    )
                    break

        # Format the message
        result = super().format(record)

        # Restore original values
        record.levelname = orig_levelname

        return result


def setup_logger(
    name: str,
    log_level: int = logging.INFO,
    log_file: Optional[Path] = None,
    console_output: bool = True,
    email_settings: Optional[dict] = None,
) -> logging.Logger:
    """
    Set up a logger with both file and console handlers.

    Args:
        name: Name of the logger
        log_level: Logging level (default: INFO)
        log_file: Optional path to log file
        console_output: Whether to output logs to console (default: True)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Prevent adding handlers multiple times
    if logger.handlers:
        return logger

    # Format string for log messages
    fmt = "%(asctime)s | %(levelname)-8s | " "%(filename)s:%(lineno)d | " "%(message)s"
    date_fmt = "%Y-%m-%d %H:%M:%S"

    # File handler
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(fmt, date_fmt))
        logger.addHandler(file_handler)

    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(CustomFormatter(fmt, date_fmt))
        logger.addHandler(console_handler)

    # Email handler
    if email_settings:
        email_handler = SMTPHandler(
            mailhost=email_settings["mailhost"],
            fromaddr=email_settings["fromaddr"],
            toaddrs=email_settings["toaddrs"],
            subject=email_settings["subject"],
            credentials=email_settings["credentials"],
            secure=(),
        )
        # Only send ERROR and CRITICAL logs via email
        email_handler.setLevel(logging.ERROR)
        email_handler.setFormatter(logging.Formatter(fmt, date_fmt))
        logger.addHandler(email_handler)

    return logger
