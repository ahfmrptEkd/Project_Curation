"""Custom exceptions for the recommendation system.

This module defines a hierarchy of custom exceptions for better error handling
and debugging in the recommendation system.
"""

from typing import Dict
from typing import Optional


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
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}

    def __str__(self):
        return f"[{self.error_code}] {self.message}"


# Data-related exceptions
class DataError(RecommendationError):
    """Base class for data-related errors."""

    pass


class DataFileNotFoundError(DataError):
    """Raised when a required data file is not found."""

    pass


class DataFormatError(DataError):
    """Raised when data format is invalid or corrupted."""

    pass


class DataValidationError(DataError):
    """Raised when data validation fails."""

    pass


# API-related exceptions
class APIError(RecommendationError):
    """Base class for API-related errors."""

    pass


class OpenAIAPIError(APIError):
    """Raised when OpenAI API calls fail."""

    pass


class RateLimitError(APIError):
    """Raised when API rate limits are exceeded."""

    pass


class NetworkError(APIError):
    """Raised when network-related errors occur."""

    pass


# Configuration-related exceptions
class ConfigurationError(RecommendationError):
    """Base class for configuration-related errors."""

    pass


class ConfigFileNotFoundError(ConfigurationError):
    """Raised when configuration file is not found."""

    pass


class InvalidConfigurationError(ConfigurationError):
    """Raised when configuration values are invalid."""

    pass


class MissingConfigurationError(ConfigurationError):
    """Raised when required configuration is missing."""

    pass


# Recommendation algorithm exceptions
class RecommendationAlgorithmError(RecommendationError):
    """Base class for recommendation algorithm errors."""

    pass


class EmbeddingGenerationError(RecommendationAlgorithmError):
    """Raised when embedding generation fails."""

    pass


class SimilarityCalculationError(RecommendationAlgorithmError):
    """Raised when similarity calculation fails."""

    pass


class InsufficientDataError(RecommendationAlgorithmError):
    """Raised when there's insufficient data for recommendations."""

    pass


# User input exceptions
class UserInputError(RecommendationError):
    """Base class for user input errors."""

    pass


class InvalidUserInputError(UserInputError):
    """Raised when user input is invalid."""

    pass


class UserValidationError(UserInputError):
    """Raised when user input validation fails."""

    pass


# Cold start problem exceptions
class ColdStartError(RecommendationError):
    """Base class for cold start related errors."""

    pass


class NewUserError(ColdStartError):
    """Raised when handling new users with no history."""

    pass


class InsufficientUserDataError(ColdStartError):
    """Raised when user has insufficient data for personalized recommendations."""

    pass
