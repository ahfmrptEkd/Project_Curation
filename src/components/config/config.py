"""Simple configuration loader for the recommendation system.

Manages settings using YAML configuration with environment variable support.
"""

import os
from pathlib import Path
import re
from typing import Any
from typing import Dict
from typing import Optional

from dotenv import load_dotenv
import yaml

# Load environment variables from .env file
load_dotenv()

# 설정 캐시
_config_cache: Optional[Dict[str, Any]] = None


def substitute_env_vars(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Substitute environment variables in configuration values.

    Supports ${VAR_NAME} syntax for environment variable substitution.

    Args:
        config_dict: Configuration dictionary to process

    Returns:
        Dictionary with environment variables substituted
    """

    def substitute_value(value):
        if isinstance(value, str):
            # ${VAR_NAME} 패턴 찾기
            pattern = r"\$\{([^}]+)\}"
            matches = re.findall(pattern, value)

            for match in matches:
                env_value = os.getenv(match, "")
                value = value.replace(f"${{{match}}}", env_value)

            return value
        elif isinstance(value, dict):
            return {k: substitute_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [substitute_value(item) for item in value]
        else:
            return value

    return substitute_value(config_dict)


def load_config(force_reload: bool = False) -> Dict[str, Any]:
    """Load configuration from YAML file with environment variable support.

    Args:
        force_reload: Force reload configuration even if cached

    Returns:
        Dictionary containing configuration
    """
    global _config_cache

    # 캐시된 설정이 있고 강제 재로드가 아닌 경우 캐시 반환
    if _config_cache is not None and not force_reload:
        return _config_cache

    config_path = Path(__file__).parent / "settings.yaml"

    try:
        with open(config_path, encoding="utf-8") as file:
            config = yaml.safe_load(file)

        if config is None:
            raise ValueError("YAML file is empty or invalid")

        # 환경변수 치환
        config = substitute_env_vars(config)

        # 캐시에 저장
        _config_cache = config

        return config
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        ) from None
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML configuration: {e}") from e


def get_app_config() -> Dict[str, Any]:
    """Get application configuration."""
    config = load_config()
    return config.get("app", {})


def get_openai_config() -> Dict[str, Any]:
    """Get OpenAI configuration."""
    config = load_config()
    return config.get("openai", {})


def get_database_config() -> Dict[str, Any]:
    """Get database configuration."""
    config = load_config()
    return config.get("database", {})


def get_logging_config() -> Dict[str, Any]:
    """Get logging configuration with enhanced settings."""
    config = load_config()
    logging_config = config.get("logging", {})

    # 기본값 설정
    default_logging_config = {
        "level": "INFO",
        "to_file": True,
        "log_dir": "logs",
        "enable_rotation": True,
        "max_file_size": 10 * 1024 * 1024,  # 10MB
        "backup_count": 5,
        "enable_security": True,
        "sensitive_data_masking": True,
        "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        "date_format": "%Y-%m-%d %H:%M:%S",
        "structured_logging": False,
        "performance_logging": True,
        "console_output": True,
        "levels": {"file": "INFO", "console": "INFO"},
    }

    # 설정 병합 (기본값 + 사용자 설정)
    merged_config = {**default_logging_config, **logging_config}

    # levels 설정 특별 처리
    if "levels" in logging_config:
        merged_config["levels"] = {
            **default_logging_config["levels"],
            **logging_config["levels"],
        }

    return merged_config


def get_security_config() -> Dict[str, Any]:
    """Get security-related configuration."""
    logging_config = get_logging_config()

    return {
        "enable_security": logging_config.get("enable_security", True),
        "sensitive_data_masking": logging_config.get(
            "sensitive_data_masking", True
        ),
        "api_key_masking": True,  # 항상 활성화
        "email_masking": True,  # 항상 활성화
    }


def get_performance_config() -> Dict[str, Any]:
    """Get performance monitoring configuration."""
    logging_config = get_logging_config()

    return {
        "performance_logging": logging_config.get("performance_logging", True),
        "slow_query_threshold": 1000,  # ms
        "memory_usage_logging": True,
        "api_response_time_logging": True,
    }


def reload_config() -> Dict[str, Any]:
    """Reload configuration from file.

    Returns:
        Dictionary containing reloaded configuration
    """
    return load_config(force_reload=True)
