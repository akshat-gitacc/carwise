"""
models.py — Data models and schemas for CarWise SQLite database.
Defines entity representations for Cars, Users, Interactions, Preferences, Shortlists, and Recommendation Logs.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional
import json


@dataclass
class CarRecord:
    id: int
    brand: str
    car: str
    variant: str
    price_lakh: Optional[float]
    price_display: str
    fuel_type: str
    transmission: str
    mileage: Optional[float]
    displacement: Optional[float]
    feature_score: float
    active_status: str
    description: str
    features: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


@dataclass
class UserRecord:
    id: str
    session_id: str
    username: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    last_active: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class UserInteractionRecord:
    id: Optional[int]
    user_id: str
    car_id: int
    interaction_type: str  # view, like, dislike, shortlist, remove_shortlist, search, compare, open_recommendation
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class UserPreferenceRecord:
    id: Optional[int]
    user_id: str
    explicit_preferences: Dict[str, Any] = field(default_factory=dict)
    inferred_preferences: Dict[str, Any] = field(default_factory=dict)
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class UserShortlistRecord:
    id: Optional[int]
    user_id: str
    car_id: int
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RecommendationHistoryRecord:
    id: Optional[int]
    user_id: str
    recommended_car_ids: List[int]
    recommendation_source: str  # 'advisor', 'explore_you_may_like', 'explore_discover_more'
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

