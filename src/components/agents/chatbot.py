"""ChatBot interface for the book recommendation system.

This module provides a conversational interface for users to interact
with the book recommendation system through natural language.
"""

from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import logging
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from .base_agent import BaseAgent
from .base_agent import RecommendationRequest
from .base_agent import RecommendationResult
from .base_agent import UserType

logger = logging.getLogger(__name__)


class ConversationState(Enum):
    """대화 상태 유형."""

    GREETING = "greeting"
    COLLECTING_PREFERENCES = "collecting_preferences"
    PROCESSING_REQUEST = "processing_request"
    PROVIDING_RECOMMENDATIONS = "providing_recommendations"
    CLARIFYING = "clarifying"
    ENDING = "ending"


@dataclass
class ChatMessage:
    """Chat message data structure."""

    id: str
    user_id: str
    content: str
    timestamp: datetime
    message_type: str  # "user" or "bot"
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ConversationContext:
    """Conversation context data structure."""

    user_id: str
    session_id: str
    state: ConversationState
    user_type: Optional[UserType] = None
    preferences: Dict[str, Any] = None
    current_query: Optional[str] = None
    recommendations_history: List[RecommendationResult] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.preferences is None:
            self.preferences = {}
        if self.recommendations_history is None:
            self.recommendations_history = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ChatResponse:
    """Chat response data structure."""

    message: str
    suggestions: Optional[List[str]] = None
    recommendations: Optional[List[Dict[str, Any]]] = None
    state: Optional[ConversationState] = None
    metadata: Optional[Dict[str, Any]] = None


class BaseChatBot(ABC):
    """Base class for chatbot implementations.

    This class provides a framework for implementing conversational interfaces
    for the book recommendation system.

    The BaseChatBot class serves as the foundation for all chatbot
    implementations in the system. It defines the core interfaces
    for conversation management, intent recognition, and response generation.

    Core Responsibilities:
        - Conversation state management
        - Natural language understanding
        - Intent recognition and classification
        - Response generation and formatting
        - Context tracking and memory management
        - Integration with recommendation agents
    """

    def __init__(self, config: Dict[str, Any], agent: BaseAgent):
        """Initialize the chatbot with configuration and agent.

        Args:
            config: Configuration dictionary containing chatbot settings
            agent: BaseAgent instance for generating recommendations
        """
        self.config = config
        self.agent = agent
        self.logger = logging.getLogger(self.__class__.__name__)
        self.conversations: Dict[str, ConversationContext] = {}
        self._setup_chatbot()

    @abstractmethod
    def _setup_chatbot(self) -> None:
        """Setup chatbot components.

        This method should be implemented by subclasses to initialize
        their specific NLU models, response templates, etc.
        """
        pass

    async def process_message(
        self, user_id: str, message: str, session_id: Optional[str] = None
    ) -> ChatResponse:
        """Process user message and generate response.

        Args:
            user_id: User identifier
            message: User message content
            session_id: Optional session identifier

        Returns:
            ChatResponse object containing bot response
        """
        try:
            # Get or create conversation context
            context = self._get_or_create_context(user_id, session_id)

            # Update context with user message
            context = await self._update_context_with_message(context, message)

            # Process message based on current state
            response = await self._process_message_by_state(context, message)

            # Update conversation state
            if response.state:
                context.state = response.state

            # Store context
            self.conversations[f"{user_id}_{context.session_id}"] = context

            return response

        except Exception as e:
            self.logger.error(f"Error processing message: {e!s}")
            return ChatResponse(
                message="죄송합니다. 처리 중 오류가 발생했습니다. 다시 시도해주세요.",
                state=ConversationState.GREETING,
            )

    def _get_or_create_context(
        self, user_id: str, session_id: Optional[str] = None
    ) -> ConversationContext:
        """Get existing or create new conversation context.

        Args:
            user_id: User identifier
            session_id: Optional session identifier

        Returns:
            ConversationContext object
        """
        if not session_id:
            session_id = self._generate_session_id()

        context_key = f"{user_id}_{session_id}"

        if context_key in self.conversations:
            return self.conversations[context_key]

        return ConversationContext(
            user_id=user_id,
            session_id=session_id,
            state=ConversationState.GREETING,
        )

    def _generate_session_id(self) -> str:
        """Generate unique session ID.

        Returns:
            Unique session identifier
        """
        import uuid

        return str(uuid.uuid4())

    async def _update_context_with_message(
        self, context: ConversationContext, message: str
    ) -> ConversationContext:
        """Update conversation context with user message.

        Args:
            context: Current conversation context
            message: User message content

        Returns:
            Updated ConversationContext
        """
        # Extract intent and entities from message
        intent = await self._extract_intent(message)
        entities = await self._extract_entities(message)

        # Update context based on intent and entities
        if intent == "book_recommendation":
            context.current_query = message
        elif intent == "preference_setting":
            context.preferences.update(entities)

        # Update metadata
        context.metadata["last_message"] = message
        context.metadata["last_intent"] = intent
        context.metadata["last_entities"] = entities
        context.metadata["last_update"] = datetime.now().isoformat()

        return context

    async def _process_message_by_state(
        self, context: ConversationContext, message: str
    ) -> ChatResponse:
        """Process message based on current conversation state.

        Args:
            context: Current conversation context
            message: User message content

        Returns:
            ChatResponse object
        """
        if context.state == ConversationState.GREETING:
            return await self._handle_greeting(context, message)
        elif context.state == ConversationState.COLLECTING_PREFERENCES:
            return await self._handle_preference_collection(context, message)
        elif context.state == ConversationState.PROCESSING_REQUEST:
            return await self._handle_request_processing(context, message)
        elif context.state == ConversationState.PROVIDING_RECOMMENDATIONS:
            return await self._handle_recommendations(context, message)
        elif context.state == ConversationState.CLARIFYING:
            return await self._handle_clarification(context, message)
        else:
            return await self._handle_greeting(context, message)

    async def _handle_greeting(
        self, context: ConversationContext, message: str
    ) -> ChatResponse:
        """Handle greeting state.

        Args:
            context: Conversation context
            message: User message

        Returns:
            ChatResponse for greeting
        """
        # 사용자 타입 확인
        user_type = await self._determine_user_type(context.user_id)
        context.user_type = user_type

        if user_type == UserType.COLD_START:
            # 신규 사용자 온보딩
            response_message = (
                "안녕하세요! 📚 도서 추천 시스템에 오신 것을 환영합니다.\n"
                "저는 당신의 취향에 맞는 완벽한 책을 찾아드리는 AI 큐레이터입니다.\n\n"
                "먼저 몇 가지 질문을 통해 당신의 독서 취향을 알려주세요!"
            )

            suggestions = [
                "선호하는 장르를 알려주세요",
                "최근에 읽고 좋았던 책이 있나요?",
                "어떤 기분의 책을 찾고 있나요?",
            ]

            return ChatResponse(
                message=response_message,
                suggestions=suggestions,
                state=ConversationState.COLLECTING_PREFERENCES,
            )
        else:
            # 기존 사용자
            response_message = (
                "다시 만나서 반가워요! 😊\n"
                "오늘은 어떤 책을 찾고 계신가요? 구체적으로 말씀해주시면 "
                "더 정확한 추천을 드릴 수 있습니다."
            )

            suggestions = [
                "우울할 때 읽기 좋은 책",
                "새로운 장르에 도전하고 싶어요",
                "최근 인기 있는 책 추천해주세요",
            ]

            return ChatResponse(
                message=response_message,
                suggestions=suggestions,
                state=ConversationState.PROCESSING_REQUEST,
            )

    async def _handle_preference_collection(
        self, context: ConversationContext, message: str
    ) -> ChatResponse:
        """Handle preference collection state.

        Args:
            context: Conversation context
            message: User message

        Returns:
            ChatResponse for preference collection
        """
        # Extract preferences from message
        preferences = await self._extract_preferences(message)
        context.preferences.update(preferences)

        # Check if we have enough preferences
        required_prefs = ["genres", "mood", "reading_purpose"]
        collected_prefs = [
            pref for pref in required_prefs if pref in context.preferences
        ]

        if len(collected_prefs) >= 2:
            # Enough preferences collected
            response_message = (
                "좋습니다! 취향을 파악했어요. 📝\n"
                f"선호 장르: {context.preferences.get('genres', '미지정')}\n"
                f"원하는 분위기: {context.preferences.get('mood', '미지정')}\n\n"
                "이제 구체적으로 어떤 책을 찾고 계신지 말씀해주세요!"
            )

            suggestions = [
                "스트레스 해소용 가벼운 책",
                "새로운 관점을 제시하는 책",
                "감동적인 스토리의 책",
            ]

            return ChatResponse(
                message=response_message,
                suggestions=suggestions,
                state=ConversationState.PROCESSING_REQUEST,
            )
        else:
            # Need more preferences
            missing_prefs = [
                pref
                for pref in required_prefs
                if pref not in context.preferences
            ]

            if "genres" in missing_prefs:
                question = "어떤 장르를 선호하시나요? (예: 로맨스, 판타지, 미스터리, 자기계발 등)"
            elif "mood" in missing_prefs:
                question = "어떤 분위기의 책을 원하시나요? (예: 재미있는, 감동적인, 생각하게 하는 등)"
            else:
                question = "독서 목적이 무엇인가요? (예: 재미, 학습, 힐링 등)"

            return ChatResponse(
                message=question, state=ConversationState.COLLECTING_PREFERENCES
            )

    async def _handle_request_processing(
        self, context: ConversationContext, message: str
    ) -> ChatResponse:
        """Handle request processing state.

        Args:
            context: Conversation context
            message: User message

        Returns:
            ChatResponse for request processing
        """
        try:
            # Create recommendation request
            request = RecommendationRequest(
                user_id=context.user_id,
                query=message,
                context=str(context.preferences),
                max_recommendations=3,
                user_type=context.user_type,
            )

            # Generate recommendations
            result = self.agent.process_request(request)
            context.recommendations_history.append(result)

            # Format response
            response_message = self._format_recommendations_response(result)

            suggestions = [
                "다른 장르 추천해주세요",
                "더 자세한 설명을 들려주세요",
                "비슷한 책 더 추천해주세요",
            ]

            return ChatResponse(
                message=response_message,
                recommendations=list(result.recommendations),
                suggestions=suggestions,
                state=ConversationState.PROVIDING_RECOMMENDATIONS,
            )

        except Exception as e:
            self.logger.error(f"Error processing recommendation request: {e!s}")

            return ChatResponse(
                message="죄송합니다. 추천을 생성하는 중 오류가 발생했습니다. 다시 시도해주세요.",
                state=ConversationState.PROCESSING_REQUEST,
            )

    async def _handle_recommendations(
        self, context: ConversationContext, message: str
    ) -> ChatResponse:
        """Handle recommendations state.

        Args:
            context: Conversation context
            message: User message

        Returns:
            ChatResponse for recommendations
        """
        intent = await self._extract_intent(message)

        if intent == "more_recommendations":
            # User wants more recommendations
            return ChatResponse(
                message="더 많은 추천을 원하시는군요! 어떤 스타일의 책을 더 보고 싶으신가요?",
                suggestions=[
                    "같은 장르의 다른 책들",
                    "완전히 다른 스타일의 책",
                    "최신 출간 도서들",
                ],
                state=ConversationState.PROCESSING_REQUEST,
            )
        elif intent == "book_details":
            # User wants more details about a specific book
            return ChatResponse(
                message="어떤 책에 대해 더 자세히 알고 싶으신가요? 책 제목을 말씀해주세요.",
                state=ConversationState.CLARIFYING,
            )
        elif intent == "new_search":
            # User wants to start a new search
            return ChatResponse(
                message="새로운 검색을 시작하시는군요! 어떤 책을 찾고 계신가요?",
                state=ConversationState.PROCESSING_REQUEST,
            )
        else:
            # Default response
            return ChatResponse(
                message="추천 결과가 도움이 되었나요? 다른 도움이 필요하시면 언제든 말씀해주세요!",
                suggestions=[
                    "다른 책 추천해주세요",
                    "책에 대해 더 알고 싶어요",
                    "새로운 검색을 시작할게요",
                ],
                state=ConversationState.PROVIDING_RECOMMENDATIONS,
            )

    async def _handle_clarification(
        self, context: ConversationContext, message: str
    ) -> ChatResponse:
        """Handle clarification state.

        Args:
            context: Conversation context
            message: User message

        Returns:
            ChatResponse for clarification
        """
        # Extract book title from message
        book_title = await self._extract_book_title(message)

        if book_title:
            # Find book details from previous recommendations
            book_details = self._find_book_details(
                book_title, context.recommendations_history
            )

            if book_details:
                response_message = self._format_book_details(book_details)
            else:
                response_message = f"'{book_title}'에 대한 추가 정보를 찾을 수 없습니다. 다른 책에 대해 물어보시겠어요?"
        else:
            response_message = (
                "어떤 책에 대해 알고 싶으신지 책 제목을 명확히 말씀해주세요."
            )

        return ChatResponse(
            message=response_message,
            suggestions=[
                "다른 책에 대해 알고 싶어요",
                "새로운 추천을 받고 싶어요",
                "처음부터 다시 시작할게요",
            ],
            state=ConversationState.PROVIDING_RECOMMENDATIONS,
        )

    @abstractmethod
    async def _extract_intent(self, message: str) -> str:
        """Extract intent from user message.

        Args:
            message: User message content

        Returns:
            Extracted intent string
        """
        pass

    @abstractmethod
    async def _extract_entities(self, message: str) -> Dict[str, Any]:
        """Extract entities from user message.

        Args:
            message: User message content

        Returns:
            Dictionary of extracted entities
        """
        pass

    async def _extract_preferences(self, message: str) -> Dict[str, Any]:
        """Extract user preferences from message.

        Args:
            message: User message content

        Returns:
            Dictionary of extracted preferences
        """
        preferences = {}
        message_lower = message.lower()

        # Genre extraction
        genres = [
            "로맨스",
            "판타지",
            "미스터리",
            "스릴러",
            "자기계발",
            "역사",
            "과학",
            "문학",
        ]
        for genre in genres:
            if genre in message_lower:
                preferences["genres"] = [*preferences.get("genres", []), genre]

        # Mood extraction
        if any(word in message_lower for word in ["재미", "웃긴", "유쾌"]):
            preferences["mood"] = "재미있는"
        elif any(word in message_lower for word in ["감동", "눈물", "슬픈"]):
            preferences["mood"] = "감동적인"
        elif any(word in message_lower for word in ["생각", "철학", "깊은"]):
            preferences["mood"] = "생각하게 하는"

        return preferences

    async def _extract_book_title(self, message: str) -> Optional[str]:
        """Extract book title from message.

        Args:
            message: User message content

        Returns:
            Extracted book title or None
        """
        # Simple implementation - can be enhanced with NER
        import re

        # Look for quoted text
        quoted_match = re.search(r'"([^"]*)"', message)
        if quoted_match:
            return quoted_match.group(1)

        # Look for common patterns
        title_patterns = [
            r'책\s*[\'"]([^\'"]*)[\'"]',
            r"([가-힣\s]+)\s*(?:책|도서)",
            r'제목\s*[\'"]?([^\'"]*)[\'"]?',
        ]

        for pattern in title_patterns:
            match = re.search(pattern, message)
            if match:
                return match.group(1).strip()

        return None

    async def _determine_user_type(self, user_id: str) -> UserType:
        """Determine user type based on reading history.

        Args:
            user_id: User identifier

        Returns:
            UserType enum value
        """
        # This would typically query the data manager
        # For now, return COLD_START as default
        return UserType.COLD_START

    def _format_recommendations_response(
        self, result: RecommendationResult
    ) -> str:
        """Format recommendation result into chat response.

        Args:
            result: RecommendationResult object

        Returns:
            Formatted response string
        """
        if not result.recommendations:
            return "죄송합니다. 조건에 맞는 책을 찾지 못했습니다. 다른 조건으로 다시 시도해보세요."

        response_parts = ["추천 도서를 찾았습니다! 📚\n"]

        for i, book in enumerate(result.recommendations[:3], 1):
            title = book.get("title", "제목 없음")
            author = book.get("author", "작가 미상")
            reason = book.get("reason", "추천 이유 없음")

            response_parts.append(f"{i}. **{title}** - {author}")
            response_parts.append(f"   💡 {reason}\n")

        if result.reasoning:
            response_parts.append(f"📝 {result.reasoning}")

        return "\n".join(response_parts)

    def _find_book_details(
        self,
        book_title: str,
        recommendations_history: List[RecommendationResult],
    ) -> Optional[Dict[str, Any]]:
        """Find book details from recommendations history.

        Args:
            book_title: Book title to search for
            recommendations_history: List of previous recommendations

        Returns:
            Book details dictionary or None
        """
        book_title_lower = book_title.lower()

        for result in recommendations_history:
            for book in result.recommendations:
                if book_title_lower in book.get("title", "").lower():
                    return book

        return None

    def _format_book_details(self, book_details: Dict[str, Any]) -> str:
        """Format book details into response.

        Args:
            book_details: Book details dictionary

        Returns:
            Formatted book details string
        """
        title = book_details.get("title", "제목 없음")
        author = book_details.get("author", "작가 미상")
        genre = book_details.get("genre", "장르 미지정")
        rating = book_details.get("rating", "평점 없음")
        reason = book_details.get("reason", "추천 이유 없음")

        return f"""📖 **{title}**
👤 저자: {author}
🎭 장르: {genre}
⭐ 평점: {rating}
💭 추천 이유: {reason}"""

    def get_conversation_history(
        self, user_id: str, session_id: str
    ) -> List[ChatMessage]:
        """Get conversation history for a session.

        Args:
            user_id: User identifier
            session_id: Session identifier

        Returns:
            List of ChatMessage objects
        """
        # This would typically be stored in a database
        # For now, return empty list
        return []

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the chatbot.

        Returns:
            Dictionary containing health status information
        """
        return {
            "status": "healthy",
            "chatbot_type": self.__class__.__name__,
            "config_loaded": bool(self.config),
            "agent_connected": self.agent is not None,
            "active_conversations": len(self.conversations),
            "components_status": self._get_components_status(),
        }

    @abstractmethod
    def _get_components_status(self) -> Dict[str, Any]:
        """Get status of chatbot components.

        Returns:
            Dictionary containing component status information
        """
        pass


class SimpleChatBot(BaseChatBot):
    """Simple chatbot implementation.

    This class provides a basic implementation of the chatbot using
    simple rule-based intent recognition and response generation.
    """

    def _setup_chatbot(self) -> None:
        """Setup simple chatbot components."""
        # Intent keywords mapping
        self.intent_keywords = {
            "book_recommendation": ["추천", "책", "도서", "읽고 싶", "찾고 있"],
            "preference_setting": ["좋아", "선호", "취향", "장르", "작가"],
            "more_recommendations": ["더", "추가", "다른", "또", "많이"],
            "book_details": ["자세히", "설명", "정보", "알고 싶", "어떤 책"],
            "new_search": ["새로", "다시", "처음", "시작"],
        }

    async def _extract_intent(self, message: str) -> str:
        """Extract intent using keyword matching.

        Args:
            message: User message content

        Returns:
            Extracted intent string
        """
        message_lower = message.lower()

        for intent, keywords in self.intent_keywords.items():
            if any(keyword in message_lower for keyword in keywords):
                return intent

        return "general"

    async def _extract_entities(self, message: str) -> Dict[str, Any]:
        """Extract entities using simple pattern matching.

        Args:
            message: User message content

        Returns:
            Dictionary of extracted entities
        """
        entities = {}
        message_lower = message.lower()

        # Extract genres
        genres = ["로맨스", "판타지", "미스터리", "스릴러", "자기계발"]
        found_genres = [genre for genre in genres if genre in message_lower]
        if found_genres:
            entities["genres"] = found_genres

        # Extract authors (simplified)
        import re

        author_pattern = r"([가-힣]{2,4})\s*(?:작가|저자)"
        author_matches = re.findall(author_pattern, message)
        if author_matches:
            entities["authors"] = author_matches

        return entities

    def _get_components_status(self) -> Dict[str, Any]:
        """Get status of simple chatbot components.

        Returns:
            Dictionary containing component status information
        """
        return {
            "intent_keywords_loaded": bool(self.intent_keywords),
            "total_intent_types": len(self.intent_keywords),
            "nlu_model": "rule_based",
        }
