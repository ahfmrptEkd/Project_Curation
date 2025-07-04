"""Recommendation engine module for the book recommendation system.

This module implements various recommendation algorithms including
personalized recommendations and cold start solutions.
"""

from abc import ABC
from abc import abstractmethod
from collections import Counter
from dataclasses import dataclass
from enum import Enum
import logging
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import numpy as np

from ..data.data_manager import BookData
from ..data.data_manager import UserProfile

logger = logging.getLogger(__name__)


class RecommendationStrategy(Enum):
    """추천 전략 유형."""

    COLD_START = "cold_start"  # 신규 사용자 대상
    CONTENT_BASED = "content_based"  # 컨텐츠 기반 필터링
    COLLABORATIVE = "collaborative"  # 협업 필터링
    HYBRID = "hybrid"  # 하이브리드 접근
    TRENDING = "trending"  # 인기도 기반


@dataclass
class RecommendationScore:
    """추천 점수 데이터 구조."""

    book: BookData
    score: float
    reasons: List[str]
    strategy: RecommendationStrategy
    metadata: Dict[str, Any]


@dataclass
class RecommendationContext:
    """추천 컨텍스트 데이터 구조."""

    user_profile: Optional[UserProfile]
    user_type: str  # cold_start, warm_up, personalized
    query: str
    filters: Optional[Dict[str, Any]]
    available_books: List[BookData]
    max_recommendations: int
    diversity_factor: float = 0.3  # 다양성 가중치


class BaseRecommender(ABC):
    """Base class for recommendation algorithms.

    This class provides a framework for implementing various recommendation
    strategies including cold start solutions and personalized recommendations.

    The BaseRecommender class serves as the foundation for all recommendation
    algorithms in the system. It defines the core interfaces for generating
    recommendations based on user profiles and contextual information.

    Core Responsibilities:
        - Strategy selection based on user type and data availability
        - Score calculation and ranking
        - Diversity and exploration management
        - Explanation generation for recommendations
        - Performance optimization and caching
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the recommender with configuration.

        Args:
            config: Configuration dictionary containing recommender settings
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self._setup_recommender()

    @abstractmethod
    def _setup_recommender(self) -> None:
        """Setup recommender components and models.

        This method should be implemented by subclasses to initialize
        their specific recommendation models and components.
        """
        pass

    def recommend(
        self, context: RecommendationContext
    ) -> List[RecommendationScore]:
        """Generate recommendations based on context.

        Args:
            context: RecommendationContext containing user and query information

        Returns:
            List of RecommendationScore objects ranked by relevance
        """
        try:
            self.logger.info(
                f"Generating recommendations for user type: {context.user_type}"
            )

            # 1. Strategy selection
            strategy = self._select_strategy(context)

            # 2. Generate candidate recommendations
            candidates = self._generate_candidates(context, strategy)

            # 3. Score and rank candidates
            scored_candidates = self._score_candidates(
                candidates, context, strategy
            )

            # 4. Apply diversity and filtering
            final_recommendations = self._apply_diversity_and_filtering(
                scored_candidates, context
            )

            # 5. Generate explanations
            final_recommendations = self._generate_explanations(
                final_recommendations, context, strategy
            )

            self.logger.info(
                f"Generated {len(final_recommendations)} recommendations"
            )
            return final_recommendations[: context.max_recommendations]

        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e!s}")
            raise

    def _select_strategy(
        self, context: RecommendationContext
    ) -> RecommendationStrategy:
        """Select appropriate recommendation strategy.

        Args:
            context: RecommendationContext

        Returns:
            Selected RecommendationStrategy
        """
        # 사용자 타입에 따른 전략 선택
        if context.user_type == "cold_start":
            return RecommendationStrategy.COLD_START
        elif context.user_type == "warm_up":
            return RecommendationStrategy.HYBRID
        else:  # personalized
            return RecommendationStrategy.COLLABORATIVE

    @abstractmethod
    def _generate_candidates(
        self, context: RecommendationContext, strategy: RecommendationStrategy
    ) -> List[BookData]:
        """Generate candidate books for recommendation.

        Args:
            context: RecommendationContext
            strategy: Selected recommendation strategy

        Returns:
            List of candidate BookData objects
        """
        pass

    @abstractmethod
    def _score_candidates(
        self,
        candidates: List[BookData],
        context: RecommendationContext,
        strategy: RecommendationStrategy,
    ) -> List[RecommendationScore]:
        """Score and rank candidate books.

        Args:
            candidates: List of candidate BookData objects
            context: RecommendationContext
            strategy: Selected recommendation strategy

        Returns:
            List of RecommendationScore objects
        """
        pass

    def _apply_diversity_and_filtering(
        self,
        scored_candidates: List[RecommendationScore],
        context: RecommendationContext,
    ) -> List[RecommendationScore]:
        """Apply diversity and filtering to recommendations.

        Args:
            scored_candidates: List of scored candidates
            context: RecommendationContext

        Returns:
            List of diversified and filtered recommendations
        """
        # 이미 읽은 책 제외
        if context.user_profile:
            read_titles = {
                book.title.lower()
                for book in context.user_profile.reading_history
            }
            scored_candidates = [
                score
                for score in scored_candidates
                if score.book.title.lower() not in read_titles
            ]

        # 필터 적용
        if context.filters:
            scored_candidates = self._apply_filters(
                scored_candidates, context.filters
            )

        # 다양성 적용
        if context.diversity_factor > 0:
            scored_candidates = self._apply_diversity(
                scored_candidates, context.diversity_factor
            )

        return scored_candidates

    def _apply_filters(
        self, candidates: List[RecommendationScore], filters: Dict[str, Any]
    ) -> List[RecommendationScore]:
        """Apply filters to candidates.

        Args:
            candidates: List of RecommendationScore objects
            filters: Dictionary of filters to apply

        Returns:
            List of filtered RecommendationScore objects
        """
        filtered_candidates = []

        for candidate in candidates:
            book = candidate.book

            # 장르 필터
            if "genres" in filters:
                allowed_genres = [g.lower() for g in filters["genres"]]
                if book.genre and book.genre.lower() not in allowed_genres:
                    continue

            # 평점 필터
            if "min_rating" in filters and book.rating:
                if book.rating < filters["min_rating"]:
                    continue

            # 작가 필터
            if "authors" in filters:
                allowed_authors = [a.lower() for a in filters["authors"]]
                if book.author and book.author.lower() not in allowed_authors:
                    continue

            filtered_candidates.append(candidate)

        return filtered_candidates

    def _apply_diversity(
        self, candidates: List[RecommendationScore], diversity_factor: float
    ) -> List[RecommendationScore]:
        """Apply diversity to recommendations.

        Args:
            candidates: List of RecommendationScore objects
            diversity_factor: Diversity weight (0-1)

        Returns:
            List of diversified RecommendationScore objects
        """
        if not candidates or diversity_factor <= 0:
            return candidates

        # 장르별 다양성 확보
        genre_counts = Counter()
        diversified_candidates = []

        for candidate in candidates:
            book = candidate.book
            genre = book.genre if book.genre else "Unknown"

            # 장르별 가중치 적용
            genre_penalty = genre_counts.get(genre, 0) * diversity_factor
            adjusted_score = candidate.score - genre_penalty

            # 새로운 점수로 업데이트
            candidate.score = max(0, adjusted_score)
            candidate.metadata["diversity_penalty"] = genre_penalty

            diversified_candidates.append(candidate)
            genre_counts[genre] += 1

        # 점수 순으로 재정렬
        return sorted(
            diversified_candidates, key=lambda x: x.score, reverse=True
        )

    def _generate_explanations(
        self,
        recommendations: List[RecommendationScore],
        context: RecommendationContext,
        strategy: RecommendationStrategy,
    ) -> List[RecommendationScore]:
        """Generate explanations for recommendations.

        Args:
            recommendations: List of RecommendationScore objects
            context: RecommendationContext
            strategy: Selected recommendation strategy

        Returns:
            List of RecommendationScore objects with explanations
        """
        for rec in recommendations:
            if not rec.reasons:
                rec.reasons = self._generate_default_explanation(
                    rec, context, strategy
                )

        return recommendations

    def _generate_default_explanation(
        self,
        recommendation: RecommendationScore,
        context: RecommendationContext,
        strategy: RecommendationStrategy,
    ) -> List[str]:
        """Generate default explanation for a recommendation.

        Args:
            recommendation: RecommendationScore object
            context: RecommendationContext
            strategy: Selected recommendation strategy

        Returns:
            List of explanation strings
        """
        reasons = []
        book = recommendation.book

        # 전략별 설명
        if strategy == RecommendationStrategy.COLD_START:
            reasons.append("인기 도서이므로 신규 독자에게 추천합니다")
        elif strategy == RecommendationStrategy.CONTENT_BASED:
            reasons.append("선호하는 장르와 일치합니다")
        elif strategy == RecommendationStrategy.COLLABORATIVE:
            reasons.append("유사한 취향의 독자들이 좋아하는 책입니다")

        # 장르 매칭
        if context.user_profile and book.genre:
            if book.genre in context.user_profile.favorite_genres:
                reasons.append(f"선호 장르 '{book.genre}'와 일치합니다")

        # 작가 매칭
        if context.user_profile and book.author:
            if book.author in context.user_profile.favorite_authors:
                reasons.append(f"선호 작가 '{book.author}'의 작품입니다")

        # 평점 기반
        if book.rating and book.rating >= 4.0:
            reasons.append(f"높은 평점 ({book.rating:.1f}/5.0)을 받은 책입니다")

        return reasons if reasons else ["추천 시스템이 선정한 책입니다"]

    def get_recommendation_stats(
        self, recommendations: List[RecommendationScore]
    ) -> Dict[str, Any]:
        """Get statistics about recommendations.

        Args:
            recommendations: List of RecommendationScore objects

        Returns:
            Dictionary containing recommendation statistics
        """
        if not recommendations:
            return {"total": 0}

        genres = [rec.book.genre for rec in recommendations if rec.book.genre]
        authors = [
            rec.book.author for rec in recommendations if rec.book.author
        ]
        ratings = [
            rec.book.rating for rec in recommendations if rec.book.rating
        ]
        strategies = [rec.strategy.value for rec in recommendations]

        return {
            "total": len(recommendations),
            "average_score": np.mean([rec.score for rec in recommendations]),
            "genre_distribution": dict(Counter(genres)),
            "author_distribution": dict(Counter(authors)),
            "average_rating": np.mean(ratings) if ratings else 0.0,
            "strategy_distribution": dict(Counter(strategies)),
        }

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the recommender.

        Returns:
            Dictionary containing health status information
        """
        return {
            "status": "healthy",
            "recommender_type": self.__class__.__name__,
            "config_loaded": bool(self.config),
            "models_status": self._get_models_status(),
        }

    @abstractmethod
    def _get_models_status(self) -> Dict[str, Any]:
        """Get status of recommendation models.

        Returns:
            Dictionary containing model status information
        """
        pass


class HybridRecommender(BaseRecommender):
    """Hybrid recommender implementation.

    This class implements a hybrid recommendation approach that combines
    multiple strategies based on user type and data availability.
    """

    def _setup_recommender(self) -> None:
        """Setup hybrid recommender components."""
        # 전략별 가중치 설정
        self.strategy_weights = {
            RecommendationStrategy.COLD_START: {
                RecommendationStrategy.TRENDING: 0.7,
                RecommendationStrategy.CONTENT_BASED: 0.3,
            },
            RecommendationStrategy.HYBRID: {
                RecommendationStrategy.CONTENT_BASED: 0.5,
                RecommendationStrategy.COLLABORATIVE: 0.3,
                RecommendationStrategy.TRENDING: 0.2,
            },
            RecommendationStrategy.COLLABORATIVE: {
                RecommendationStrategy.COLLABORATIVE: 0.6,
                RecommendationStrategy.CONTENT_BASED: 0.4,
            },
        }

        # 최소 추천 풀 크기
        self.min_candidate_pool_size = self.config.get(
            "min_candidate_pool_size", 50
        )

    def _generate_candidates(
        self, context: RecommendationContext, strategy: RecommendationStrategy
    ) -> List[BookData]:
        """Generate candidate books using multiple strategies.

        Args:
            context: RecommendationContext
            strategy: Selected recommendation strategy

        Returns:
            List of candidate BookData objects
        """
        candidates = []

        # 전략에 따른 후보 생성
        if strategy == RecommendationStrategy.COLD_START:
            # 인기 도서 기반 후보
            candidates.extend(self._get_trending_candidates(context))

            # 장르 다양성 보장
            candidates.extend(self._get_diverse_genre_candidates(context))

        elif strategy == RecommendationStrategy.HYBRID:
            # 컨텐츠 기반 후보
            candidates.extend(self._get_content_based_candidates(context))

            # 협업 필터링 후보
            candidates.extend(self._get_collaborative_candidates(context))

            # 트렌딩 후보
            candidates.extend(self._get_trending_candidates(context))

        else:  # COLLABORATIVE
            # 주로 협업 필터링
            candidates.extend(self._get_collaborative_candidates(context))

            # 컨텐츠 기반 보조
            candidates.extend(self._get_content_based_candidates(context))

        # 중복 제거
        unique_candidates = []
        seen_books = set()

        for book in candidates:
            book_id = f"{book.title}_{book.author}"
            if book_id not in seen_books:
                unique_candidates.append(book)
                seen_books.add(book_id)

        return unique_candidates[: self.min_candidate_pool_size]

    def _get_trending_candidates(
        self, context: RecommendationContext
    ) -> List[BookData]:
        """Get trending/popular book candidates.

        Args:
            context: RecommendationContext

        Returns:
            List of trending BookData objects
        """
        # 평점 기준 정렬
        books_with_ratings = [
            book for book in context.available_books if book.rating is not None
        ]

        # 평점 순으로 정렬
        sorted_books = sorted(
            books_with_ratings, key=lambda x: x.rating, reverse=True
        )

        return sorted_books[:20]

    def _get_content_based_candidates(
        self, context: RecommendationContext
    ) -> List[BookData]:
        """Get content-based candidates.

        Args:
            context: RecommendationContext

        Returns:
            List of content-based BookData objects
        """
        candidates = []

        if context.user_profile:
            # 선호 장르 기반
            for genre in context.user_profile.favorite_genres:
                genre_books = [
                    book
                    for book in context.available_books
                    if book.genre and genre.lower() in book.genre.lower()
                ]
                candidates.extend(genre_books[:10])

            # 선호 작가 기반
            for author in context.user_profile.favorite_authors:
                author_books = [
                    book
                    for book in context.available_books
                    if book.author and author.lower() in book.author.lower()
                ]
                candidates.extend(author_books[:5])

        return candidates

    def _get_collaborative_candidates(
        self, context: RecommendationContext
    ) -> List[BookData]:
        """Get collaborative filtering candidates.

        Args:
            context: RecommendationContext

        Returns:
            List of collaborative BookData objects
        """
        # 간단한 협업 필터링 구현
        # 실제 구현에서는 더 정교한 알고리즘 사용
        candidates = []

        if context.user_profile and context.user_profile.reading_history:
            # 비슷한 책들을 읽은 사용자들이 좋아하는 책 추천
            # 현재는 같은 장르의 높은 평점 책들로 대체
            user_genres = set()
            for book in context.user_profile.reading_history:
                if book.genre:
                    user_genres.add(book.genre)

            for genre in user_genres:
                similar_books = [
                    book
                    for book in context.available_books
                    if book.genre
                    and genre.lower() in book.genre.lower()
                    and book.rating
                    and book.rating >= 4.0
                ]
                candidates.extend(similar_books[:10])

        return candidates

    def _get_diverse_genre_candidates(
        self, context: RecommendationContext
    ) -> List[BookData]:
        """Get diverse genre candidates for cold start.

        Args:
            context: RecommendationContext

        Returns:
            List of diverse BookData objects
        """
        # 장르별로 고르게 분포된 후보 생성
        genre_books = {}

        for book in context.available_books:
            if book.genre:
                genre = book.genre
                if genre not in genre_books:
                    genre_books[genre] = []
                genre_books[genre].append(book)

        # 각 장르에서 상위 책들 선택
        candidates = []
        for _genre, books in genre_books.items():
            # 평점 기준 정렬
            sorted_books = sorted(
                [b for b in books if b.rating],
                key=lambda x: x.rating,
                reverse=True,
            )
            candidates.extend(sorted_books[:3])  # 각 장르에서 3권씩

        return candidates

    def _score_candidates(
        self,
        candidates: List[BookData],
        context: RecommendationContext,
        strategy: RecommendationStrategy,
    ) -> List[RecommendationScore]:
        """Score candidates using hybrid approach.

        Args:
            candidates: List of candidate BookData objects
            context: RecommendationContext
            strategy: Selected recommendation strategy

        Returns:
            List of RecommendationScore objects
        """
        scored_candidates = []

        for book in candidates:
            # 기본 점수 계산
            base_score = self._calculate_base_score(book, context)

            # 전략별 점수 조정
            strategy_score = self._calculate_strategy_score(
                book, context, strategy
            )

            # 최종 점수 계산
            final_score = base_score * 0.6 + strategy_score * 0.4

            # 추천 점수 객체 생성
            recommendation_score = RecommendationScore(
                book=book,
                score=final_score,
                reasons=[],
                strategy=strategy,
                metadata={
                    "base_score": base_score,
                    "strategy_score": strategy_score,
                    "calculation_method": "hybrid",
                },
            )

            scored_candidates.append(recommendation_score)

        # 점수 순으로 정렬
        return sorted(scored_candidates, key=lambda x: x.score, reverse=True)

    def _calculate_base_score(
        self, book: BookData, context: RecommendationContext
    ) -> float:
        """Calculate base score for a book.

        Args:
            book: BookData object
            context: RecommendationContext

        Returns:
            Base score (0-1)
        """
        score = 0.0

        # 평점 기반 점수
        if book.rating:
            score += (book.rating / 5.0) * 0.4

        # 사용자 선호도 매칭
        if context.user_profile:
            # 장르 매칭
            if (
                book.genre
                and book.genre in context.user_profile.favorite_genres
            ):
                score += 0.3

            # 작가 매칭
            if (
                book.author
                and book.author in context.user_profile.favorite_authors
            ):
                score += 0.3

        return min(1.0, score)

    def _calculate_strategy_score(
        self,
        book: BookData,
        context: RecommendationContext,
        strategy: RecommendationStrategy,
    ) -> float:
        """Calculate strategy-specific score.

        Args:
            book: BookData object
            context: RecommendationContext
            strategy: Selected recommendation strategy

        Returns:
            Strategy score (0-1)
        """
        if strategy == RecommendationStrategy.COLD_START:
            # 인기도와 안전성 중시
            return (book.rating / 5.0) if book.rating else 0.5

        elif strategy == RecommendationStrategy.CONTENT_BASED:
            # 사용자 선호도 매칭 중시
            if not context.user_profile:
                return 0.5

            score = 0.0
            if (
                book.genre
                and book.genre in context.user_profile.favorite_genres
            ):
                score += 0.5
            if (
                book.author
                and book.author in context.user_profile.favorite_authors
            ):
                score += 0.5

            return min(1.0, score)

        elif strategy == RecommendationStrategy.COLLABORATIVE:
            # 협업 필터링 점수 (현재는 평점 기반으로 대체)
            return (book.rating / 5.0) if book.rating else 0.5

        else:
            return 0.5

    def _get_models_status(self) -> Dict[str, Any]:
        """Get status of recommendation models.

        Returns:
            Dictionary containing model status information
        """
        return {
            "hybrid_model": "active",
            "strategy_weights": self.strategy_weights,
            "min_candidate_pool_size": self.min_candidate_pool_size,
            "supported_strategies": [s.value for s in RecommendationStrategy],
        }
