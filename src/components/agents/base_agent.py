"""Base agent class for the book recommendation system.

This module defines the core Agent class that orchestrates the recommendation process
using a single-agent approach with MCP tools integration.
"""

from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
import logging
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

logger = logging.getLogger(__name__)


class UserType(Enum):
    """사용자 타입 정의."""

    COLD_START = "cold_start"  # 신규 사용자 (데이터 < 5권)
    WARM_UP = "warm_up"  # 데이터 축적 중 (5-20권)
    PERSONALIZED = "personalized"  # 완전 개인화 (20권+)


@dataclass
class RecommendationRequest:
    """추천 요청 데이터 구조."""

    user_id: str
    query: str
    context: Optional[str] = None
    max_recommendations: int = 3
    user_type: Optional[UserType] = None
    filters: Optional[Dict[str, Any]] = None


@dataclass
class RecommendationResult:
    """추천 결과 데이터 구조."""

    recommendations: List[Dict[str, Any]]
    reasoning: str
    confidence_score: float
    user_type: UserType
    processing_time: float
    metadata: Optional[Dict[str, Any]] = None


class BaseAgent(ABC):
    """Base class for book recommendation agents.

    This class provides a framework for Single-Agent based recommendation systems
    that integrate multiple MCP tools to generate personalized book recommendations.

    The BaseAgent class serves as the foundation for all recommendation agents
    in the system. It defines the core interfaces and workflow for processing
    user requests, determining user types, and generating recommendations.

    Core Responsibilities:
        - User type classification (Cold Start, Warm-up, Personalized)
        - Request processing and validation
        - Orchestration of data retrieval and analysis
        - Chain-of-Thought based recommendation generation
        - Result formatting and metadata collection
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the base agent with configuration.

        Args:
            config: Configuration dictionary containing agent settings
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self._setup_components()

    @abstractmethod
    def _setup_components(self) -> None:
        """Setup required components (DataManager, Recommender, etc.).

        This method should be implemented by subclasses to initialize
        their specific components like DataManager, RAGSystem, etc.
        """
        pass

    def process_request(
        self, request: RecommendationRequest
    ) -> RecommendationResult:
        """Main entry point for processing recommendation requests.

        This method orchestrates the entire recommendation process:
        1. Validates the request
        2. Determines user type
        3. Retrieves and analyzes user data
        4. Generates recommendations
        5. Formats and returns results

        Args:
            request: RecommendationRequest object containing user query and context

        Returns:
            RecommendationResult containing recommendations and metadata
        """
        try:
            start_time = self._get_current_time()

            # 1. Request validation
            self._validate_request(request)

            # 2. User type determination
            user_type = self._determine_user_type(request)
            request.user_type = user_type

            # 3. Data retrieval and analysis
            user_data = self._retrieve_user_data(request)
            context_data = self._retrieve_context_data(request)

            # 4. Recommendation generation
            recommendations = self._generate_recommendations(
                request, user_data, context_data
            )

            # 5. Result formatting
            processing_time = self._get_current_time() - start_time

            result = RecommendationResult(
                recommendations=recommendations,
                reasoning=self._get_recommendation_reasoning(),
                confidence_score=self._calculate_confidence_score(
                    recommendations
                ),
                user_type=user_type,
                processing_time=processing_time,
                metadata=self._collect_metadata(
                    request, user_data, context_data
                ),
            )

            self.logger.info(
                f"Successfully processed request for user {request.user_id}"
            )
            return result

        except Exception as e:
            self.logger.error(f"Error processing request: {e!s}")
            raise

    @abstractmethod
    def _determine_user_type(self, request: RecommendationRequest) -> UserType:
        """Determine user type based on available data.

        Args:
            request: RecommendationRequest object

        Returns:
            UserType enum value
        """
        pass

    @abstractmethod
    def _retrieve_user_data(
        self, request: RecommendationRequest
    ) -> Dict[str, Any]:
        """Retrieve user's reading history and preferences.

        Args:
            request: RecommendationRequest object

        Returns:
            Dictionary containing user data
        """
        pass

    @abstractmethod
    def _retrieve_context_data(
        self, request: RecommendationRequest
    ) -> Dict[str, Any]:
        """Retrieve contextual data for recommendations.

        Args:
            request: RecommendationRequest object

        Returns:
            Dictionary containing context data
        """
        pass

    @abstractmethod
    def _generate_recommendations(
        self,
        request: RecommendationRequest,
        user_data: Dict[str, Any],
        context_data: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Generate book recommendations based on user data and context.

        Args:
            request: RecommendationRequest object
            user_data: User's reading history and preferences
            context_data: Contextual information for recommendations

        Returns:
            List of recommended books with metadata
        """
        pass

    def _validate_request(self, request: RecommendationRequest) -> None:
        """Validate the recommendation request.

        Args:
            request: RecommendationRequest object to validate

        Raises:
            ValueError: If request is invalid
        """
        if not request.user_id:
            raise ValueError("User ID is required")

        if not request.query:
            raise ValueError("Query is required")

        if request.max_recommendations <= 0:
            raise ValueError("Max recommendations must be positive")

    def _get_current_time(self) -> float:
        """Get current timestamp for performance tracking.

        Returns:
            Current timestamp in seconds
        """
        import time

        return time.time()

    def _get_recommendation_reasoning(self) -> str:
        """Get reasoning for the recommendations.

        This method should be overridden by subclasses to provide
        specific reasoning based on their recommendation logic.

        Returns:
            String explaining the recommendation reasoning
        """
        return (
            "Recommendations generated based on user preferences and context."
        )

    def _calculate_confidence_score(
        self, recommendations: List[Dict[str, Any]]
    ) -> float:
        """Calculate confidence score for recommendations.

        Args:
            recommendations: List of recommended books

        Returns:
            Confidence score between 0.0 and 1.0
        """
        # Default implementation - should be overridden by subclasses
        return 0.8

    def _collect_metadata(
        self,
        request: RecommendationRequest,
        user_data: Dict[str, Any],
        context_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Collect metadata for the recommendation result.

        Args:
            request: RecommendationRequest object
            user_data: User's reading history and preferences
            context_data: Contextual information

        Returns:
            Dictionary containing metadata
        """
        return {
            "user_type": request.user_type.value if request.user_type else None,
            "query": request.query,
            "data_sources": ["user_history", "external_apis"],
            "filters_applied": request.filters,
            "timestamp": self._get_current_time(),
        }

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the agent and its components.

        Returns:
            Dictionary containing health status information
        """
        return {
            "status": "healthy",
            "agent_type": self.__class__.__name__,
            "config_loaded": bool(self.config),
            "components": self._get_components_status(),
        }

    @abstractmethod
    def _get_components_status(self) -> Dict[str, Any]:
        """Get status of all components.

        Returns:
            Dictionary containing component status information
        """
        pass
