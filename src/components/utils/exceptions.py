"""Custom exceptions for the recommendation system.

This module defines a hierarchy of custom exceptions for better error handling
and debugging in the recommendation system.
"""

from typing import Dict
from typing import Optional


class ErrorCodes:
    """표준화된 에러 코드 상수들.

    카테고리별로 구분된 에러 코드를 정의합니다.
    형식: [CATEGORY]_[NUMBER] (예: DATA_001, API_002)
    """

    # Base error codes
    RECOMMENDATION_000 = "RECOMMENDATION_000"

    # Data-related error codes (DATA_001 ~ DATA_099)
    DATA_001 = "DATA_001"  # Data file not found
    DATA_002 = "DATA_002"  # Data format error
    DATA_003 = "DATA_003"  # Data validation error
    DATA_004 = "DATA_004"  # General data error

    # API-related error codes (API_001 ~ API_099)
    API_001 = "API_001"  # OpenAI API error
    API_002 = "API_002"  # Rate limit error
    API_003 = "API_003"  # Network error
    API_004 = "API_004"  # General API error

    # Configuration error codes (CONFIG_001 ~ CONFIG_099)
    CONFIG_001 = "CONFIG_001"  # Configuration file not found
    CONFIG_002 = "CONFIG_002"  # Invalid configuration
    CONFIG_003 = "CONFIG_003"  # Missing configuration
    CONFIG_004 = "CONFIG_004"  # General configuration error

    # Algorithm error codes (ALGO_001 ~ ALGO_099)
    ALGO_001 = "ALGO_001"  # Embedding generation error
    ALGO_002 = "ALGO_002"  # Similarity calculation error
    ALGO_003 = "ALGO_003"  # Insufficient data error
    ALGO_004 = "ALGO_004"  # General algorithm error

    # User input error codes (USER_001 ~ USER_099)
    USER_001 = "USER_001"  # Invalid user input
    USER_002 = "USER_002"  # User validation error
    USER_003 = "USER_003"  # General user input error

    # Cold start error codes (COLD_001 ~ COLD_099)
    COLD_001 = "COLD_001"  # New user error
    COLD_002 = "COLD_002"  # Insufficient user data
    COLD_003 = "COLD_003"  # General cold start error


class RecommendationError(Exception):
    """Base exception class for all recommendation system errors.

    This is the base class for all custom exceptions in the recommendation
    system. It provides a consistent interface for error handling.

    Args:
        message (str): Human-readable error message
        error_code (str): Unique error code for programmatic handling
        details (dict): Additional error details for debugging
    """

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict] = None,
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or ErrorCodes.RECOMMENDATION_000
        self.details = details or {}

    def __str__(self):
        return f"[{self.error_code}] {self.message}"

    def to_dict(self):
        """예외 정보를 딕셔너리로 변환합니다."""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details,
            "exception_type": self.__class__.__name__,
        }


# Data-related exceptions
class DataError(RecommendationError):
    """Base class for data-related errors."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict] = None,
    ):
        super().__init__(message, error_code or ErrorCodes.DATA_004, details)


class DataFileNotFoundError(DataError):
    """Raised when a required data file is not found."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict] = None,
    ):
        super().__init__(message, error_code or ErrorCodes.DATA_001, details)


class DataFormatError(DataError):
    """Raised when data format is invalid or corrupted."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict] = None,
    ):
        super().__init__(message, error_code or ErrorCodes.DATA_002, details)


class DataValidationError(DataError):
    """Raised when data validation fails."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict] = None,
    ):
        super().__init__(message, error_code or ErrorCodes.DATA_003, details)


# API-related exceptions
class APIError(RecommendationError):
    """Base class for API-related errors."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict] = None,
    ):
        super().__init__(message, error_code or ErrorCodes.API_004, details)


class OpenAIAPIError(APIError):
    """Raised when OpenAI API calls fail."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict] = None,
    ):
        super().__init__(message, error_code or ErrorCodes.API_001, details)


class RateLimitError(APIError):
    """Raised when API rate limits are exceeded."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict] = None,
    ):
        super().__init__(message, error_code or ErrorCodes.API_002, details)


class NetworkError(APIError):
    """Raised when network-related errors occur."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict] = None,
    ):
        super().__init__(message, error_code or ErrorCodes.API_003, details)


# Configuration-related exceptions
class ConfigurationError(RecommendationError):
    """Base class for configuration-related errors."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict] = None,
    ):
        super().__init__(message, error_code or ErrorCodes.CONFIG_004, details)


class ConfigFileNotFoundError(ConfigurationError):
    """Raised when configuration file is not found."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict] = None,
    ):
        super().__init__(message, error_code or ErrorCodes.CONFIG_001, details)


class InvalidConfigurationError(ConfigurationError):
    """Raised when configuration values are invalid."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict] = None,
    ):
        super().__init__(message, error_code or ErrorCodes.CONFIG_002, details)


class MissingConfigurationError(ConfigurationError):
    """Raised when required configuration is missing."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict] = None,
    ):
        super().__init__(message, error_code or ErrorCodes.CONFIG_003, details)


# Recommendation algorithm exceptions
class RecommendationAlgorithmError(RecommendationError):
    """Base class for recommendation algorithm errors."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict] = None,
    ):
        super().__init__(message, error_code or ErrorCodes.ALGO_004, details)


class EmbeddingGenerationError(RecommendationAlgorithmError):
    """Raised when embedding generation fails."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict] = None,
    ):
        super().__init__(message, error_code or ErrorCodes.ALGO_001, details)


class SimilarityCalculationError(RecommendationAlgorithmError):
    """Raised when similarity calculation fails."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict] = None,
    ):
        super().__init__(message, error_code or ErrorCodes.ALGO_002, details)


class InsufficientDataError(RecommendationAlgorithmError):
    """Raised when there's insufficient data for recommendations."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict] = None,
    ):
        super().__init__(message, error_code or ErrorCodes.ALGO_003, details)


# User input exceptions
class UserInputError(RecommendationError):
    """Base class for user input errors."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict] = None,
    ):
        super().__init__(message, error_code or ErrorCodes.USER_003, details)


class InvalidUserInputError(UserInputError):
    """Raised when user input is invalid."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict] = None,
    ):
        super().__init__(message, error_code or ErrorCodes.USER_001, details)


class UserValidationError(UserInputError):
    """Raised when user input validation fails."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict] = None,
    ):
        super().__init__(message, error_code or ErrorCodes.USER_002, details)


# Cold start problem exceptions
class ColdStartError(RecommendationError):
    """Base class for cold start related errors."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict] = None,
    ):
        super().__init__(message, error_code or ErrorCodes.COLD_003, details)


class NewUserError(ColdStartError):
    """Raised when handling new users with no history."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict] = None,
    ):
        super().__init__(message, error_code or ErrorCodes.COLD_001, details)


class InsufficientUserDataError(ColdStartError):
    """Raised when user has insufficient data for personalized recommendations."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict] = None,
    ):
        super().__init__(message, error_code or ErrorCodes.COLD_002, details)
