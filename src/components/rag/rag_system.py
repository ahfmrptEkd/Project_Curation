"""RAG (Retrieval-Augmented Generation) system for book recommendations.

This module implements the RAG system that combines vector search with LLM generation
to provide contextually relevant and personalized book recommendations.
"""

from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import pickle
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import numpy as np

from ..data.data_manager import BookData
from ..data.data_manager import UserProfile

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Search result data structure."""

    book: BookData
    score: float
    query: str
    metadata: Dict[str, Any]


@dataclass
class RAGContext:
    """RAG context data structure."""

    user_profile: Optional[UserProfile]
    query: str
    search_results: List[SearchResult]
    user_type: str
    filters: Optional[Dict[str, Any]] = None
    max_results: int = 10


@dataclass
class GenerationResult:
    """Generation result data structure."""

    recommendations: List[Dict[str, Any]]
    reasoning: str
    confidence_score: float
    context_used: Dict[str, Any]
    metadata: Dict[str, Any]


class BaseVectorStore(ABC):
    """Base class for vector storage and retrieval.

    This class provides a framework for storing and retrieving vectorized
    book data for semantic search operations.

    The BaseVectorStore class serves as the foundation for all vector
    storage implementations in the RAG system. It defines the core
    interfaces for vectorization, storage, and retrieval operations.

    Core Responsibilities:
        - Document vectorization and embedding
        - Vector storage and indexing
        - Similarity search and retrieval
        - Index management and optimization
        - Metadata storage and filtering
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the vector store with configuration.

        Args:
            config: Configuration dictionary containing vector store settings
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.embedding_model = None
        self.index = None
        self._setup_vector_store()

    @abstractmethod
    def _setup_vector_store(self) -> None:
        """Setup vector store components.

        This method should be implemented by subclasses to initialize
        their specific vector storage and embedding components.
        """
        pass

    @abstractmethod
    def add_documents(self, books: List[BookData]) -> bool:
        """Add documents to the vector store.

        Args:
            books: List of BookData objects to add

        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Search for similar documents.

        Args:
            query: Search query string
            top_k: Number of top results to return
            filters: Optional filters to apply

        Returns:
            List of SearchResult objects
        """
        pass

    @abstractmethod
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as numpy array
        """
        pass

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the vector store.

        Returns:
            Dictionary containing health status information
        """
        return {
            "status": "healthy",
            "vector_store_type": self.__class__.__name__,
            "config_loaded": bool(self.config),
            "embedding_model_loaded": self.embedding_model is not None,
            "index_loaded": self.index is not None,
            "document_count": self._get_document_count(),
        }

    @abstractmethod
    def _get_document_count(self) -> int:
        """Get total number of documents in the store.

        Returns:
            Number of documents
        """
        pass


class SimpleVectorStore(BaseVectorStore):
    """Simple vector store implementation using sentence transformers.

    This implementation uses sentence-transformers for embedding generation
    and performs brute-force similarity search for simplicity.
    """

    def _setup_vector_store(self) -> None:
        """Setup simple vector store components."""
        try:
            # sentence-transformers 모델 로드
            from sentence_transformers import SentenceTransformer

            model_name = self.config.get("embedding_model", "all-MiniLM-L6-v2")
            self.embedding_model = SentenceTransformer(model_name)

            # 인덱스 초기화
            self.documents = []
            self.embeddings = []
            self.metadata = []

            # 저장된 인덱스 로드 시도
            self._load_index()

        except ImportError:
            self.logger.error(
                "sentence-transformers not installed. Please install it using: pip install sentence-transformers"
            )
            raise
        except Exception as e:
            self.logger.error(f"Error setting up vector store: {e!s}")
            raise

    def add_documents(self, books: List[BookData]) -> bool:
        """Add books to the vector store.

        Args:
            books: List of BookData objects to add

        Returns:
            True if successful, False otherwise
        """
        try:
            for book in books:
                # 책 정보를 텍스트로 변환
                doc_text = self._book_to_text(book)

                # 임베딩 생성
                embedding = self.get_embedding(doc_text)

                # 인덱스에 추가
                self.documents.append(book)
                self.embeddings.append(embedding)
                self.metadata.append(
                    {
                        "title": book.title,
                        "author": book.author,
                        "genre": book.genre,
                        "trope": book.trope,
                        "rating": book.rating,
                    }
                )

            # 임베딩을 numpy 배열로 변환
            if self.embeddings:
                self.embeddings = np.array(self.embeddings)

            # 인덱스 저장
            self._save_index()

            self.logger.info(f"Added {len(books)} books to vector store")
            return True

        except Exception as e:
            self.logger.error(f"Error adding documents: {e!s}")
            return False

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Search for similar books.

        Args:
            query: Search query string
            top_k: Number of top results to return
            filters: Optional filters to apply

        Returns:
            List of SearchResult objects
        """
        try:
            if not self.embeddings or len(self.embeddings) == 0:
                return []

            # 쿼리 임베딩 생성
            query_embedding = self.get_embedding(query)

            # 유사도 계산
            similarities = np.dot(self.embeddings, query_embedding)

            # 상위 k개 결과 추출
            top_indices = np.argsort(similarities)[::-1][:top_k]

            results = []
            for idx in top_indices:
                if idx < len(self.documents):
                    book = self.documents[idx]
                    score = float(similarities[idx])

                    # 필터 적용
                    if filters and not self._apply_search_filters(
                        book, filters
                    ):
                        continue

                    result = SearchResult(
                        book=book,
                        score=score,
                        query=query,
                        metadata=self.metadata[idx].copy(),
                    )
                    results.append(result)

            return results

        except Exception as e:
            self.logger.error(f"Error searching: {e!s}")
            return []

    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as numpy array
        """
        try:
            embedding = self.embedding_model.encode(text)
            return embedding
        except Exception as e:
            self.logger.error(f"Error generating embedding: {e!s}")
            return np.zeros(384)  # Default embedding size

    def _book_to_text(self, book: BookData) -> str:
        """Convert book data to text for embedding.

        Args:
            book: BookData object

        Returns:
            Text representation of the book
        """
        parts = []

        # 제목과 저자 (필수)
        parts.append(f"Title: {book.title}")
        parts.append(f"Author: {book.author}")

        # 장르 (있는 경우)
        if book.genre:
            parts.append(f"Genre: {book.genre}")

        # 트로프 (있는 경우)
        if book.trope:
            parts.append(f"Trope: {book.trope}")

        # 설명 (있는 경우)
        if book.description:
            parts.append(f"Description: {book.description}")

        # 노트 (있는 경우)
        if book.notes:
            parts.append(f"Notes: {book.notes}")

        # 태그 (있는 경우)
        if book.tags:
            parts.append(f"Tags: {', '.join(book.tags)}")

        return " ".join(parts)

    def _apply_search_filters(
        self, book: BookData, filters: Dict[str, Any]
    ) -> bool:
        """Apply filters to search results.

        Args:
            book: BookData object to filter
            filters: Dictionary of filters

        Returns:
            True if book passes filters, False otherwise
        """
        # 장르 필터
        if "genres" in filters:
            if not book.genre or book.genre not in filters["genres"]:
                return False

        # 작가 필터
        if "authors" in filters:
            if not book.author or book.author not in filters["authors"]:
                return False

        # 평점 필터
        if "min_rating" in filters:
            if not book.rating or book.rating < filters["min_rating"]:
                return False

        return True

    def _save_index(self) -> None:
        """Save vector index to disk."""
        try:
            index_path = Path(
                self.config.get("index_path", "data/vector_index")
            )
            index_path.mkdir(parents=True, exist_ok=True)

            # 임베딩 저장
            if len(self.embeddings) > 0:
                np.save(index_path / "embeddings.npy", self.embeddings)

            # 메타데이터 저장
            with open(index_path / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)

            # 문서 저장
            with open(index_path / "documents.pkl", "wb") as f:
                pickle.dump(self.documents, f)

        except Exception as e:
            self.logger.error(f"Error saving index: {e!s}")

    def _load_index(self) -> None:
        """Load vector index from disk."""
        try:
            index_path = Path(
                self.config.get("index_path", "data/vector_index")
            )

            if not index_path.exists():
                return

            # 임베딩 로드
            embeddings_file = index_path / "embeddings.npy"
            if embeddings_file.exists():
                self.embeddings = np.load(embeddings_file)

            # 메타데이터 로드
            metadata_file = index_path / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file, encoding="utf-8") as f:
                    self.metadata = json.load(f)

            # 문서 로드
            documents_file = index_path / "documents.pkl"
            if documents_file.exists():
                with open(documents_file, "rb") as f:
                    # Note: pickle is used for internal data storage only (not user input)
                    self.documents = pickle.load(f)  # noqa: S301

            self.logger.info(
                f"Loaded {len(self.documents)} documents from index"
            )

        except Exception as e:
            self.logger.error(f"Error loading index: {e!s}")

    def _get_document_count(self) -> int:
        """Get total number of documents in the store.

        Returns:
            Number of documents
        """
        return len(self.documents)


class BaseRAGSystem(ABC):
    """Base class for RAG (Retrieval-Augmented Generation) systems.

    This class provides a framework for implementing RAG systems that
    combine vector search with LLM generation for book recommendations.

    Google Style Docstring:
        The BaseRAGSystem class serves as the foundation for all RAG
        implementations in the system. It defines the core interfaces
        for retrieval, context preparation, and generation processes.

    Core Responsibilities:
        - Document retrieval and ranking
        - Context preparation and augmentation
        - LLM prompt engineering and generation
        - Result post-processing and formatting
        - Performance optimization and caching
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the RAG system with configuration.

        Args:
            config: Configuration dictionary containing RAG settings
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.vector_store = None
        self.llm_client = None
        self._setup_rag_system()

    @abstractmethod
    def _setup_rag_system(self) -> None:
        """Setup RAG system components.

        This method should be implemented by subclasses to initialize
        their specific vector store and LLM components.
        """
        pass

    def generate_recommendations(self, context: RAGContext) -> GenerationResult:
        """Generate recommendations using RAG approach.

        Args:
            context: RAGContext containing user and query information

        Returns:
            GenerationResult containing recommendations and reasoning
        """
        try:
            self.logger.info(
                f"Generating RAG recommendations for query: {context.query}"
            )

            # 1. Retrieve relevant documents
            search_results = self._retrieve_documents(context)

            # 2. Prepare context for generation
            generation_context = self._prepare_generation_context(
                context, search_results
            )

            # 3. Generate recommendations
            recommendations = self._generate_with_llm(generation_context)

            # 4. Post-process results
            processed_recommendations = self._post_process_recommendations(
                recommendations, context, search_results
            )

            # 5. Calculate confidence score
            confidence_score = self._calculate_confidence_score(
                processed_recommendations, search_results
            )

            result = GenerationResult(
                recommendations=processed_recommendations,
                reasoning=self._generate_reasoning(context, search_results),
                confidence_score=confidence_score,
                context_used=generation_context,
                metadata={
                    "search_results_count": len(search_results),
                    "generation_method": "rag",
                    "user_type": context.user_type,
                },
            )

            self.logger.info(
                f"Generated {len(processed_recommendations)} recommendations"
            )
            return result

        except Exception as e:
            self.logger.error(f"Error generating RAG recommendations: {e!s}")
            raise

    def _retrieve_documents(self, context: RAGContext) -> List[SearchResult]:
        """Retrieve relevant documents for the query.

        Args:
            context: RAGContext

        Returns:
            List of SearchResult objects
        """
        if not self.vector_store:
            return []

        # 검색 쿼리 확장
        expanded_query = self._expand_query(context)

        # 벡터 검색 수행
        search_results = self.vector_store.search(
            query=expanded_query,
            top_k=context.max_results,
            filters=context.filters,
        )

        # 사용자 프로필 기반 재랭킹
        if context.user_profile:
            search_results = self._rerank_with_user_profile(
                search_results, context.user_profile
            )

        return search_results

    def _expand_query(self, context: RAGContext) -> str:
        """Expand query with user profile information.

        Args:
            context: RAGContext

        Returns:
            Expanded query string
        """
        expanded_parts = [context.query]

        if context.user_profile:
            # 선호 장르 추가
            if context.user_profile.favorite_genres:
                expanded_parts.append(
                    f"genres: {', '.join(context.user_profile.favorite_genres)}"
                )

            # 선호 작가 추가
            if context.user_profile.favorite_authors:
                expanded_parts.append(
                    f"authors: {', '.join(context.user_profile.favorite_authors)}"
                )

        return " ".join(expanded_parts)

    def _rerank_with_user_profile(
        self, search_results: List[SearchResult], user_profile: UserProfile
    ) -> List[SearchResult]:
        """Rerank search results based on user profile.

        Args:
            search_results: List of SearchResult objects
            user_profile: UserProfile object

        Returns:
            Reranked list of SearchResult objects
        """
        for result in search_results:
            book = result.book

            # 장르 매칭 보너스
            if book.genre and book.genre in user_profile.favorite_genres:
                result.score += 0.1

            # 작가 매칭 보너스
            if book.author and book.author in user_profile.favorite_authors:
                result.score += 0.15

            # 평점 선호도 고려
            if book.rating and user_profile.average_rating:
                rating_diff = abs(book.rating - user_profile.average_rating)
                if rating_diff < 0.5:
                    result.score += 0.05

        # 점수 순으로 재정렬
        return sorted(search_results, key=lambda x: x.score, reverse=True)

    @abstractmethod
    def _prepare_generation_context(
        self, context: RAGContext, search_results: List[SearchResult]
    ) -> Dict[str, Any]:
        """Prepare context for LLM generation.

        Args:
            context: RAGContext
            search_results: List of SearchResult objects

        Returns:
            Dictionary containing generation context
        """
        pass

    @abstractmethod
    def _generate_with_llm(
        self, generation_context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate recommendations using LLM.

        Args:
            generation_context: Dictionary containing generation context

        Returns:
            List of recommendation dictionaries
        """
        pass

    def _post_process_recommendations(
        self,
        recommendations: List[Dict[str, Any]],
        context: RAGContext,
        search_results: List[SearchResult],
    ) -> List[Dict[str, Any]]:
        """Post-process generated recommendations.

        Args:
            recommendations: List of recommendation dictionaries
            context: RAGContext
            search_results: List of SearchResult objects

        Returns:
            List of processed recommendation dictionaries
        """
        processed = []

        for rec in recommendations:
            # 기본 정보 검증
            if not rec.get("title") or not rec.get("author"):
                continue

            # 검색 결과와 매칭
            matching_result = self._find_matching_search_result(
                rec, search_results
            )
            if matching_result:
                rec["search_score"] = matching_result.score
                rec["book_data"] = matching_result.book.__dict__

            # 메타데이터 추가
            rec["generated_by"] = "rag_system"
            rec["timestamp"] = self._get_current_timestamp()

            processed.append(rec)

        return processed

    def _find_matching_search_result(
        self, recommendation: Dict[str, Any], search_results: List[SearchResult]
    ) -> Optional[SearchResult]:
        """Find matching search result for a recommendation.

        Args:
            recommendation: Recommendation dictionary
            search_results: List of SearchResult objects

        Returns:
            Matching SearchResult or None
        """
        rec_title = recommendation.get("title", "").lower()
        rec_author = recommendation.get("author", "").lower()

        for result in search_results:
            if (
                rec_title in result.book.title.lower()
                and rec_author in result.book.author.lower()
            ):
                return result

        return None

    def _calculate_confidence_score(
        self,
        recommendations: List[Dict[str, Any]],
        search_results: List[SearchResult],
    ) -> float:
        """Calculate confidence score for recommendations.

        Args:
            recommendations: List of recommendation dictionaries
            search_results: List of SearchResult objects

        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not recommendations or not search_results:
            return 0.0

        # 검색 결과 품질 기반 점수
        avg_search_score = np.mean([r.score for r in search_results])

        # 추천 결과 매칭 비율
        matched_count = sum(
            1 for rec in recommendations if rec.get("search_score")
        )
        match_ratio = matched_count / len(recommendations)

        # 최종 신뢰도 점수
        confidence = avg_search_score * 0.6 + match_ratio * 0.4

        return min(1.0, confidence)

    def _generate_reasoning(
        self, context: RAGContext, search_results: List[SearchResult]
    ) -> str:
        """Generate reasoning for recommendations.

        Args:
            context: RAGContext
            search_results: List of SearchResult objects

        Returns:
            Reasoning string
        """
        parts = []

        # 검색 결과 기반 설명
        if search_results:
            parts.append(
                f"'{context.query}' 검색 결과를 바탕으로 {len(search_results)}권의 관련 도서를 분석했습니다."
            )

        # 사용자 프로필 기반 설명
        if context.user_profile:
            parts.append(
                f"사용자의 독서 이력({context.user_profile.reading_count}권)과 선호도를 고려했습니다."
            )

            if context.user_profile.favorite_genres:
                parts.append(
                    f"선호 장르: {', '.join(context.user_profile.favorite_genres[:3])}"
                )

        # 사용자 타입 기반 설명
        if context.user_type == "cold_start":
            parts.append(
                "신규 사용자를 위한 안전하고 인기 있는 도서들을 우선적으로 추천했습니다."
            )
        elif context.user_type == "personalized":
            parts.append(
                "개인화된 추천 알고리즘을 적용하여 맞춤형 도서를 선별했습니다."
            )

        return " ".join(parts)

    def _get_current_timestamp(self) -> str:
        """Get current timestamp.

        Returns:
            Current timestamp as string
        """
        from datetime import datetime

        return datetime.now().isoformat()

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the RAG system.

        Returns:
            Dictionary containing health status information
        """
        return {
            "status": "healthy",
            "rag_system_type": self.__class__.__name__,
            "config_loaded": bool(self.config),
            "vector_store_status": self.vector_store.get_health_status()
            if self.vector_store
            else None,
            "llm_client_status": self._get_llm_client_status(),
        }

    @abstractmethod
    def _get_llm_client_status(self) -> Dict[str, Any]:
        """Get LLM client status.

        Returns:
            Dictionary containing LLM client status
        """
        pass


class OpenAIRAGSystem(BaseRAGSystem):
    """OpenAI-based RAG system implementation.

    This class implements the RAG system using OpenAI's API for generation
    and a simple vector store for retrieval.
    """

    def _setup_rag_system(self) -> None:
        """Setup OpenAI RAG system components."""
        try:
            import openai

            # OpenAI 클라이언트 설정
            api_key = self.config.get("openai_api_key")
            if not api_key:
                raise ValueError("OpenAI API key not provided")

            self.llm_client = openai.OpenAI(api_key=api_key)

            # 벡터 스토어 설정
            vector_store_config = self.config.get("vector_store", {})
            self.vector_store = SimpleVectorStore(vector_store_config)

        except ImportError:
            self.logger.error(
                "OpenAI package not installed. Please install it using: pip install openai"
            )
            raise
        except Exception as e:
            self.logger.error(f"Error setting up OpenAI RAG system: {e!s}")
            raise

    def _prepare_generation_context(
        self, context: RAGContext, search_results: List[SearchResult]
    ) -> Dict[str, Any]:
        """Prepare context for OpenAI generation.

        Args:
            context: RAGContext
            search_results: List of SearchResult objects

        Returns:
            Dictionary containing generation context
        """
        # 검색 결과를 텍스트로 변환
        books_context = []
        for _i, result in enumerate(search_results[:10]):  # 상위 10개만 사용
            book = result.book
            book_info = {
                "title": book.title,
                "author": book.author,
                "genre": book.genre,
                "trope": book.trope,
                "rating": book.rating,
                "notes": book.notes,
                "relevance_score": result.score,
            }
            books_context.append(book_info)

        # 사용자 프로필 정보
        user_context = {}
        if context.user_profile:
            user_context = {
                "reading_count": context.user_profile.reading_count,
                "favorite_genres": context.user_profile.favorite_genres,
                "favorite_authors": context.user_profile.favorite_authors,
                "average_rating": context.user_profile.average_rating,
                "user_type": context.user_type,
            }

        return {
            "query": context.query,
            "user_profile": user_context,
            "books_context": books_context,
            "max_recommendations": min(context.max_results, 3),
            "user_type": context.user_type,
        }

    def _generate_with_llm(
        self, generation_context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate recommendations using OpenAI.

        Args:
            generation_context: Dictionary containing generation context

        Returns:
            List of recommendation dictionaries
        """
        try:
            # 프롬프트 생성
            prompt = self._build_prompt(generation_context)

            # OpenAI API 호출
            response = self.llm_client.chat.completions.create(
                model=self.config.get("model", "gpt-3.5-turbo"),
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=1000,
            )

            # 응답 파싱
            content = response.choices[0].message.content
            recommendations = self._parse_llm_response(content)

            return recommendations

        except Exception as e:
            self.logger.error(f"Error generating with OpenAI: {e!s}")
            return []

    def _build_prompt(self, context: Dict[str, Any]) -> str:
        """Build prompt for OpenAI generation.

        Args:
            context: Generation context dictionary

        Returns:
            Generated prompt string
        """
        prompt_parts = []

        # 사용자 요청
        prompt_parts.append(f"사용자 요청: '{context['query']}'")

        # 사용자 프로필 정보
        if context.get("user_profile"):
            profile = context["user_profile"]
            prompt_parts.append("\n사용자 프로필:")
            prompt_parts.append(
                f"- 독서 이력: {profile.get('reading_count', 0)}권"
            )
            prompt_parts.append(
                f"- 선호 장르: {', '.join(profile.get('favorite_genres', []))}"
            )
            prompt_parts.append(
                f"- 선호 작가: {', '.join(profile.get('favorite_authors', []))}"
            )
            prompt_parts.append(
                f"- 평균 평점: {profile.get('average_rating', 0):.1f}"
            )
            prompt_parts.append(
                f"- 사용자 타입: {profile.get('user_type', 'unknown')}"
            )

        # 검색된 도서 정보
        if context.get("books_context"):
            prompt_parts.append("\n검색된 관련 도서들:")
            for i, book in enumerate(
                context["books_context"][:5]
            ):  # 상위 5개만 사용
                prompt_parts.append(
                    f"{i+1}. {book['title']} - {book['author']}"
                )
                if book.get("genre"):
                    prompt_parts.append(f"   장르: {book['genre']}")
                if book.get("trope"):
                    prompt_parts.append(f"   트로프: {book['trope']}")
                if book.get("rating"):
                    prompt_parts.append(f"   평점: {book['rating']:.1f}")
                if book.get("notes"):
                    prompt_parts.append(f"   노트: {book['notes'][:100]}...")

        # 추천 요청
        max_recs = context.get("max_recommendations", 3)
        prompt_parts.append(
            f"\n위 정보를 바탕으로 {max_recs}권의 책을 추천하고 각각의 이유를 설명해주세요."
        )

        return "\n".join(prompt_parts)

    def _get_system_prompt(self) -> str:
        """Get system prompt for OpenAI.

        Returns:
            System prompt string
        """
        return """당신은 전문적인 도서 큐레이터입니다. 사용자의 요청과 프로필을 분석하여 맞춤형 도서를 추천해주세요.

다음 형식으로 응답해주세요:

1. **[제목]** - [저자]
   - 장르: [장르]
   - 추천 이유: [구체적인 이유]

2. **[제목]** - [저자]
   - 장르: [장르]
   - 추천 이유: [구체적인 이유]

3. **[제목]** - [저자]
   - 장르: [장르]
   - 추천 이유: [구체적인 이유]

추천 시 고려사항:
- 사용자의 독서 이력과 선호도
- 요청의 구체적인 내용
- 책의 품질과 평점
- 다양성과 새로운 탐색 가능성"""

    def _parse_llm_response(self, content: str) -> List[Dict[str, Any]]:
        """Parse LLM response to extract recommendations.

        Args:
            content: LLM response content

        Returns:
            List of recommendation dictionaries
        """
        recommendations = []

        try:
            # 간단한 파싱 로직
            lines = content.split("\n")
            current_rec = {}

            for line in lines:
                line = line.strip()

                # 책 제목과 저자 파싱
                if line.startswith(("1.", "2.", "3.", "4.", "5.")):
                    if current_rec:
                        recommendations.append(current_rec)

                    # 제목과 저자 추출
                    title_author = (
                        line.split("**")[1]
                        if "**" in line
                        else line[2:].strip()
                    )
                    if " - " in title_author:
                        title, author = title_author.split(" - ", 1)
                        current_rec = {
                            "title": title.strip(),
                            "author": author.strip(),
                            "genre": "",
                            "reason": "",
                        }
                    else:
                        current_rec = {
                            "title": title_author.strip(),
                            "author": "",
                            "genre": "",
                            "reason": "",
                        }

                # 장르 파싱
                elif line.startswith("장르:") or line.startswith("- 장르:"):
                    if current_rec:
                        current_rec["genre"] = line.split(":", 1)[1].strip()

                # 추천 이유 파싱
                elif line.startswith("추천 이유:") or line.startswith(
                    "- 추천 이유:"
                ):
                    if current_rec:
                        current_rec["reason"] = line.split(":", 1)[1].strip()

            # 마지막 추천 추가
            if current_rec:
                recommendations.append(current_rec)

        except Exception as e:
            self.logger.error(f"Error parsing LLM response: {e!s}")

        return recommendations

    def _get_llm_client_status(self) -> Dict[str, Any]:
        """Get OpenAI client status.

        Returns:
            Dictionary containing OpenAI client status
        """
        return {
            "client_type": "openai",
            "model": self.config.get("model", "gpt-3.5-turbo"),
            "client_initialized": self.llm_client is not None,
            "api_key_configured": bool(self.config.get("openai_api_key")),
        }
