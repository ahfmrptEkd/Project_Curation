"""Simple logging system for the recommendation system.

Basic logging setup with file output and console output.
"""

import logging
from pathlib import Path
import re
from typing import Any
from typing import Dict


def sanitize_log_message(message: str) -> str:
    """Mask sensitive information in log messages.

    Masks the following patterns of sensitive information:
    - API keys (api_key, openai_key, etc.)
    - Email addresses
    - Passwords (password, passwd, etc.)
    - Tokens (token, auth_token, etc.)
    - User IDs (user_id when long)

    Args:
        message: Message to mask sensitive information

    Returns:
        Message with sensitive information masked
    """
    if not isinstance(message, str):
        message = str(message)

    # API 키 패턴 마스킹 (다양한 형태 지원)
    patterns = [
        # API 키 패턴
        (r'(api[_-]?key["\']?\s*[:=]\s*["\']?)[^"\'&\s,}]{8,}', r"\1***"),
        (r'(openai[_-]?key["\']?\s*[:=]\s*["\']?)[^"\'&\s,}]{8,}', r"\1***"),
        (r'(secret[_-]?key["\']?\s*[:=]\s*["\']?)[^"\'&\s,}]{8,}', r"\1***"),
        # 토큰 패턴
        (r'(token["\']?\s*[:=]\s*["\']?)[^"\'&\s,}]{16,}', r"\1***"),
        (r'(auth[_-]?token["\']?\s*[:=]\s*["\']?)[^"\'&\s,}]{16,}', r"\1***"),
        (r'(access[_-]?token["\']?\s*[:=]\s*["\']?)[^"\'&\s,}]{16,}', r"\1***"),
        # 비밀번호 패턴
        (r'(password["\']?\s*[:=]\s*["\']?)[^"\'&\s,}]{3,}', r"\1***"),
        (r'(passwd["\']?\s*[:=]\s*["\']?)[^"\'&\s,}]{3,}', r"\1***"),
        (r'(pwd["\']?\s*[:=]\s*["\']?)[^"\'&\s,}]{3,}', r"\1***"),
        # 이메일 패턴
        (r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", r"***@***.***"),
        # 긴 사용자 ID (16자리 이상)
        (r'(user[_-]?id["\']?\s*[:=]\s*["\']?)[^"\'&\s,}]{16,}', r"\1***"),
    ]

    for pattern, replacement in patterns:
        message = re.sub(pattern, replacement, message, flags=re.IGNORECASE)

    return message


def setup_logger(
    name: str = "recommendation_system",
    log_level: str = "INFO",
    log_to_file: bool = True,
    log_dir: str = "logs",
    enable_rotation: bool = True,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    enable_security: bool = True,
) -> logging.Logger:
    """Set up a logger with enhanced security and rotation features.

    Args:
        name: Logger name
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file: Whether to log to file
        log_dir: Directory for log files
        enable_rotation: Whether to enable log file rotation
        max_file_size: Maximum size of a log file in bytes
        backup_count: Number of backup files to keep
        enable_security: Whether to enable sensitive data masking

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

        # Create file handler with rotation support
        log_file = log_path / f"{name}.log"

        if enable_rotation:
            from logging.handlers import RotatingFileHandler

            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=max_file_size,
                backupCount=backup_count,
                encoding="utf-8",
            )
        else:
            file_handler = logging.FileHandler(log_file, encoding="utf-8")

        file_handler.setLevel(log_level_obj)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Store security setting in logger
    logger.security_enabled = enable_security

    return logger


def log_error(logger: logging.Logger, error: Exception, context: str = ""):
    """Log an error with context and security features.

    Args:
        logger: Logger instance
        error: Exception that occurred
        context: Additional context information
    """
    error_msg = f"{type(error).__name__}: {error!s}"
    if context:
        error_msg = f"[{context}] {error_msg}"

    # Apply security masking if enabled
    if hasattr(logger, "security_enabled") and logger.security_enabled:
        error_msg = sanitize_log_message(error_msg)

    logger.error(error_msg, exc_info=True)


def log_warning(logger: logging.Logger, message: str, context: str = ""):
    """Log a warning with context and security features.

    Args:
        logger: Logger instance
        message: Warning message
        context: Additional context information
    """
    if context:
        message = f"[{context}] {message}"

    # Apply security masking if enabled
    if hasattr(logger, "security_enabled") and logger.security_enabled:
        message = sanitize_log_message(message)

    logger.warning(message)


def log_info(logger: logging.Logger, message: str, context: str = ""):
    """Log an info message with context and security features.

    Args:
        logger: Logger instance
        message: Info message
        context: Additional context information
    """
    if context:
        message = f"[{context}] {message}"

    # Apply security masking if enabled
    if hasattr(logger, "security_enabled") and logger.security_enabled:
        message = sanitize_log_message(message)

    logger.info(message)


def log_debug(logger: logging.Logger, message: str, context: str = ""):
    """Log a debug message with context and security features.

    Args:
        logger: Logger instance
        message: Debug message
        context: Additional context information
    """
    if context:
        message = f"[{context}] {message}"

    # Apply security masking if enabled
    if hasattr(logger, "security_enabled") and logger.security_enabled:
        message = sanitize_log_message(message)

    logger.debug(message)


def log_structured(
    logger: logging.Logger,
    event_type: str,
    data: Dict[str, Any],
    context: str = "",
):
    """Log structured data with security features.

    Args:
        logger: Logger instance
        event_type: Type of event (e.g., 'api_call', 'user_action')
        data: Dictionary containing event data
        context: Additional context information
    """
    from datetime import datetime
    import json

    structured_data = {
        "event_type": event_type,
        "timestamp": datetime.now().isoformat(),
        "data": data,
    }

    if context:
        structured_data["context"] = context

    structured_msg = json.dumps(
        structured_data, ensure_ascii=False, indent=None
    )

    # Apply security masking if enabled
    if hasattr(logger, "security_enabled") and logger.security_enabled:
        structured_msg = sanitize_log_message(structured_msg)

    logger.info(structured_msg)


# Default logger for the application
default_logger = setup_logger()
