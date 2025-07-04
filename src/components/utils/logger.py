"""Simple logging system for the recommendation system.

Basic logging setup with file output and console output.
"""

import logging
from pathlib import Path


def setup_logger(
    name: str = "recommendation_system",
    log_level: str = "INFO",
    log_to_file: bool = True,
    log_dir: str = "logs",
) -> logging.Logger:
    """Set up a simple logger.

    Args:
        name: Logger name
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file: Whether to log to file
        log_dir: Directory for log files

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)

    # Clear existing handlers
    logger.handlers = []

    # Set log level
    log_level_obj = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(log_level_obj)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level_obj)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_to_file:
        # Create log directory
        log_path = Path(log_dir)
        log_path.mkdir(exist_ok=True)

        # Create file handler
        log_file = log_path / f"{name}.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(log_level_obj)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def log_error(logger: logging.Logger, error: Exception, context: str = ""):
    """Log an error with context.

    Args:
        logger: Logger instance
        error: Exception that occurred
        context: Additional context information
    """
    error_msg = f"{type(error).__name__}: {error!s}"
    if context:
        error_msg = f"[{context}] {error_msg}"

    logger.error(error_msg, exc_info=True)


def log_warning(logger: logging.Logger, message: str, context: str = ""):
    """Log a warning with context.

    Args:
        logger: Logger instance
        message: Warning message
        context: Additional context information
    """
    if context:
        message = f"[{context}] {message}"

    logger.warning(message)


def log_info(logger: logging.Logger, message: str, context: str = ""):
    """Log an info message with context.

    Args:
        logger: Logger instance
        message: Info message
        context: Additional context information
    """
    if context:
        message = f"[{context}] {message}"

    logger.info(message)


# Default logger for the application
default_logger = setup_logger()
