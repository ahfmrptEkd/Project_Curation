"""Data management module for the book recommendation system.

This module handles loading, preprocessing, and vectorization of book data
from various sources including CSV files and external APIs.
"""

from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from datetime import datetime
import logging
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class BookData:
    """Book data structure."""

    title: str
    author: str
    genre: Optional[str] = None
    trope: Optional[str] = None
    rating: Optional[float] = None
    notes: Optional[str] = None
    isbn: Optional[str] = None
    publication_year: Optional[int] = None
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    embedding: Optional[np.ndarray] = None


@dataclass
class UserProfile:
    """User profile data structure."""

    user_id: str
    reading_history: List[BookData]
    preferences: Dict[str, Any]
    favorite_genres: List[str]
    favorite_authors: List[str]
    average_rating: float
    reading_count: int
    last_updated: datetime
    user_type: str  # cold_start, warm_up, personalized


class BaseDataManager(ABC):
    """Base class for data management operations.

    This class provides a framework for handling various data sources
    and preprocessing operations for the recommendation system.

    The BaseDataManager class serves as the foundation for all data
    management operations in the system. It defines the core interfaces
    for loading, preprocessing, and managing book data and user profiles.

    Core Responsibilities:
        - Data loading from various sources (CSV, APIs, databases)
        - Data preprocessing and cleaning
        - Feature extraction and vectorization
        - User profile management
        - Data validation and quality checks
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the data manager with configuration.

        Args:
            config: Configuration dictionary containing data settings
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self._books_cache: Optional[List[BookData]] = None
        self._user_profiles: Dict[str, UserProfile] = {}
        self._setup_data_sources()

    @abstractmethod
    def _setup_data_sources(self) -> None:
        """Setup data sources and connections.

        This method should be implemented by subclasses to initialize
        their specific data sources (file paths, API connections, etc.).
        """
        pass

    def load_books_data(self, force_reload: bool = False) -> List[BookData]:
        """Load books data from configured sources.

        Args:
            force_reload: Whether to force reload data even if cached

        Returns:
            List of BookData objects
        """
        if self._books_cache and not force_reload:
            return self._books_cache

        try:
            self.logger.info("Loading books data...")
            books_data = self._load_books_from_sources()

            # Data preprocessing
            books_data = self._preprocess_books_data(books_data)

            # Data validation
            books_data = self._validate_books_data(books_data)

            # Cache the results
            self._books_cache = books_data

            self.logger.info(f"Successfully loaded {len(books_data)} books")
            return books_data

        except Exception as e:
            self.logger.error(f"Error loading books data: {e!s}")
            raise

    @abstractmethod
    def _load_books_from_sources(self) -> List[BookData]:
        """Load books data from configured sources.

        Returns:
            List of BookData objects
        """
        pass

    def load_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Load user profile data.

        Args:
            user_id: User identifier

        Returns:
            UserProfile object or None if not found
        """
        if user_id in self._user_profiles:
            return self._user_profiles[user_id]

        try:
            profile = self._load_user_profile_from_source(user_id)
            if profile:
                self._user_profiles[user_id] = profile
            return profile

        except Exception as e:
            self.logger.error(
                f"Error loading user profile for {user_id}: {e!s}"
            )
            return None

    @abstractmethod
    def _load_user_profile_from_source(
        self, user_id: str
    ) -> Optional[UserProfile]:
        """Load user profile from data source.

        Args:
            user_id: User identifier

        Returns:
            UserProfile object or None if not found
        """
        pass

    def save_user_profile(self, profile: UserProfile) -> bool:
        """Save user profile data.

        Args:
            profile: UserProfile object to save

        Returns:
            True if successful, False otherwise
        """
        try:
            success = self._save_user_profile_to_source(profile)
            if success:
                self._user_profiles[profile.user_id] = profile
            return success

        except Exception as e:
            self.logger.error(f"Error saving user profile: {e!s}")
            return False

    @abstractmethod
    def _save_user_profile_to_source(self, profile: UserProfile) -> bool:
        """Save user profile to data source.

        Args:
            profile: UserProfile object to save

        Returns:
            True if successful, False otherwise
        """
        pass

    def _preprocess_books_data(
        self, books_data: List[BookData]
    ) -> List[BookData]:
        """Preprocess books data for better quality.

        Args:
            books_data: List of raw BookData objects

        Returns:
            List of preprocessed BookData objects
        """
        processed_books = []

        for book in books_data:
            # Clean and normalize text fields
            book.title = self._clean_text(book.title) if book.title else ""
            book.author = self._clean_text(book.author) if book.author else ""
            book.genre = self._clean_text(book.genre) if book.genre else None
            book.trope = self._clean_text(book.trope) if book.trope else None
            book.notes = self._clean_text(book.notes) if book.notes else None

            # Normalize rating
            if book.rating is not None:
                book.rating = max(0, min(5, float(book.rating)))

            # Extract tags from notes if available
            if book.notes and not book.tags:
                book.tags = self._extract_tags_from_notes(book.notes)

            processed_books.append(book)

        return processed_books

    def _validate_books_data(
        self, books_data: List[BookData]
    ) -> List[BookData]:
        """Validate books data quality.

        Args:
            books_data: List of BookData objects to validate

        Returns:
            List of validated BookData objects
        """
        valid_books = []

        for book in books_data:
            # Check required fields
            if not book.title or not book.author:
                self.logger.warning(
                    f"Skipping book with missing title or author: {book}"
                )
                continue

            # Check for duplicate books
            if self._is_duplicate_book(book, valid_books):
                self.logger.warning(f"Skipping duplicate book: {book.title}")
                continue

            valid_books.append(book)

        return valid_books

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text data.

        Args:
            text: Raw text to clean

        Returns:
            Cleaned text
        """
        if not text:
            return ""

        # Remove extra whitespace
        text = " ".join(text.split())

        # Remove special characters that might cause issues
        text = text.replace('"', "'")
        text = text.replace("\n", " ")
        text = text.replace("\r", " ")

        return text.strip()

    def _extract_tags_from_notes(self, notes: str) -> List[str]:
        """Extract tags from book notes.

        Args:
            notes: Book notes text

        Returns:
            List of extracted tags
        """
        # Simple tag extraction - can be enhanced
        tags = []

        # Look for common patterns in notes
        emotions = [
            "funny",
            "sad",
            "romantic",
            "thrilling",
            "mysterious",
            "dark",
        ]
        for emotion in emotions:
            if emotion.lower() in notes.lower():
                tags.append(emotion)

        # Look for genre indicators
        genres = [
            "romance",
            "fantasy",
            "mystery",
            "thriller",
            "horror",
            "sci-fi",
        ]
        for genre in genres:
            if genre.lower() in notes.lower():
                tags.append(genre)

        return list(set(tags))  # Remove duplicates

    def _is_duplicate_book(
        self, book: BookData, existing_books: List[BookData]
    ) -> bool:
        """Check if a book is a duplicate.

        Args:
            book: BookData object to check
            existing_books: List of existing BookData objects

        Returns:
            True if duplicate, False otherwise
        """
        for existing in existing_books:
            if (
                book.title.lower() == existing.title.lower()
                and book.author.lower() == existing.author.lower()
            ):
                return True
        return False

    def get_user_statistics(self, user_id: str) -> Dict[str, Any]:
        """Get user reading statistics.

        Args:
            user_id: User identifier

        Returns:
            Dictionary containing user statistics
        """
        profile = self.load_user_profile(user_id)
        if not profile:
            return {}

        return {
            "reading_count": profile.reading_count,
            "average_rating": profile.average_rating,
            "favorite_genres": profile.favorite_genres,
            "favorite_authors": profile.favorite_authors,
            "user_type": profile.user_type,
            "last_updated": profile.last_updated.isoformat(),
        }

    def get_books_by_genre(self, genre: str) -> List[BookData]:
        """Get books filtered by genre.

        Args:
            genre: Genre to filter by

        Returns:
            List of BookData objects in the specified genre
        """
        books = self.load_books_data()
        return [
            book
            for book in books
            if book.genre and genre.lower() in book.genre.lower()
        ]

    def get_books_by_author(self, author: str) -> List[BookData]:
        """Get books filtered by author.

        Args:
            author: Author to filter by

        Returns:
            List of BookData objects by the specified author
        """
        books = self.load_books_data()
        return [
            book
            for book in books
            if book.author and author.lower() in book.author.lower()
        ]

    def search_books(self, query: str, limit: int = 10) -> List[BookData]:
        """Search books by query.

        Args:
            query: Search query
            limit: Maximum number of results

        Returns:
            List of BookData objects matching the query
        """
        books = self.load_books_data()
        results = []

        query_lower = query.lower()

        for book in books:
            # Search in title, author, genre, and notes
            if (
                query_lower in book.title.lower()
                or query_lower in book.author.lower()
                or (book.genre and query_lower in book.genre.lower())
                or (book.notes and query_lower in book.notes.lower())
            ):
                results.append(book)

        return results[:limit]

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the data manager.

        Returns:
            Dictionary containing health status information
        """
        try:
            books_count = (
                len(self.load_books_data()) if self._books_cache else 0
            )
            users_count = len(self._user_profiles)

            return {
                "status": "healthy",
                "books_loaded": books_count,
                "users_cached": users_count,
                "cache_status": "active" if self._books_cache else "empty",
                "data_sources": self._get_data_sources_status(),
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "books_loaded": 0,
                "users_cached": 0,
            }

    @abstractmethod
    def _get_data_sources_status(self) -> Dict[str, Any]:
        """Get status of all data sources.

        Returns:
            Dictionary containing data source status information
        """
        pass


class CSVDataManager(BaseDataManager):
    """CSV-based data manager implementation.

    This class implements the BaseDataManager interface for CSV data sources.
    It handles loading book data from CSV files and user profiles from various sources.
    """

    def _setup_data_sources(self) -> None:
        """Setup CSV data sources."""
        self.csv_file_path = self.config.get("csv_file_path", "data/books.csv")
        self.user_data_path = self.config.get("user_data_path", "data/users/")

        # Ensure data directories exist
        Path(self.user_data_path).mkdir(parents=True, exist_ok=True)

    def _load_books_from_sources(self) -> List[BookData]:
        """Load books data from CSV file.

        Returns:
            List of BookData objects
        """
        try:
            df = pd.read_csv(self.csv_file_path)
            books = []

            for _, row in df.iterrows():
                book = BookData(
                    title=row.get("Book Title", ""),
                    author=row.get("Author", ""),
                    genre=row.get("Genre"),
                    trope=row.get("Trope"),
                    rating=row.get("Rating"),
                    notes=row.get("Notes"),
                    isbn=row.get("ISBN"),
                    publication_year=row.get("Publication Year"),
                    description=row.get("Description"),
                )
                books.append(book)

            return books

        except Exception as e:
            self.logger.error(f"Error loading CSV data: {e!s}")
            raise

    def _load_user_profile_from_source(
        self, user_id: str
    ) -> Optional[UserProfile]:
        """Load user profile from JSON file.

        Args:
            user_id: User identifier

        Returns:
            UserProfile object or None if not found
        """
        try:
            import json

            profile_path = Path(self.user_data_path) / f"{user_id}.json"

            if not profile_path.exists():
                return None

            with open(profile_path, encoding="utf-8") as f:
                data = json.load(f)

            # Convert book data
            reading_history = []
            for book_data in data.get("reading_history", []):
                book = BookData(**book_data)
                reading_history.append(book)

            profile = UserProfile(
                user_id=user_id,
                reading_history=reading_history,
                preferences=data.get("preferences", {}),
                favorite_genres=data.get("favorite_genres", []),
                favorite_authors=data.get("favorite_authors", []),
                average_rating=data.get("average_rating", 0.0),
                reading_count=data.get("reading_count", 0),
                last_updated=datetime.fromisoformat(
                    data.get("last_updated", datetime.now().isoformat())
                ),
                user_type=data.get("user_type", "cold_start"),
            )

            return profile

        except Exception as e:
            self.logger.error(f"Error loading user profile from JSON: {e!s}")
            return None

    def _save_user_profile_to_source(self, profile: UserProfile) -> bool:
        """Save user profile to JSON file.

        Args:
            profile: UserProfile object to save

        Returns:
            True if successful, False otherwise
        """
        try:
            import json

            profile_path = Path(self.user_data_path) / f"{profile.user_id}.json"

            # Convert profile to dict
            profile_dict = {
                "user_id": profile.user_id,
                "reading_history": [
                    book.__dict__ for book in profile.reading_history
                ],
                "preferences": profile.preferences,
                "favorite_genres": profile.favorite_genres,
                "favorite_authors": profile.favorite_authors,
                "average_rating": profile.average_rating,
                "reading_count": profile.reading_count,
                "last_updated": profile.last_updated.isoformat(),
                "user_type": profile.user_type,
            }

            with open(profile_path, "w", encoding="utf-8") as f:
                json.dump(profile_dict, f, ensure_ascii=False, indent=2)

            return True

        except Exception as e:
            self.logger.error(f"Error saving user profile to JSON: {e!s}")
            return False

    def _get_data_sources_status(self) -> Dict[str, Any]:
        """Get status of CSV data sources.

        Returns:
            Dictionary containing data source status information
        """
        return {
            "csv_file": {
                "path": self.csv_file_path,
                "exists": Path(self.csv_file_path).exists(),
                "size": Path(self.csv_file_path).stat().st_size
                if Path(self.csv_file_path).exists()
                else 0,
            },
            "user_data_dir": {
                "path": self.user_data_path,
                "exists": Path(self.user_data_path).exists(),
                "user_files": len(
                    list(Path(self.user_data_path).glob("*.json"))
                )
                if Path(self.user_data_path).exists()
                else 0,
            },
        }
