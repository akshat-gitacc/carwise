"""
schemas.py — Pydantic Schemas for Favourites and Shortlist Management.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class AddFavouriteRequest(BaseModel):
    user_id: str = Field(..., description="Unique user ID or session ID")
    car_id: int = Field(..., description="Car ID to add to favourites")
    session_id: Optional[str] = None


class RemoveFavouriteRequest(BaseModel):
    user_id: str = Field(..., description="Unique user ID or session ID")
    car_id: int = Field(..., description="Car ID to remove from favourites")


class ShortlistCarItem(BaseModel):
    id: int
    brand: str
    car: str
    variant: str
    price_lakh: Optional[float] = None
    price_display: str
    fuel_type: str
    transmission: str
    mileage: Optional[float] = None
    displacement: Optional[float] = None
    feature_score: float = 0.0
    active_status: str
    description: Optional[str] = None
    features: Dict[str, Any] = {}
    created_at: str


class ShortlistResponse(BaseModel):
    user_id: str
    total_count: int
    items: List[ShortlistCarItem]


class ShortlistComparisonResponse(BaseModel):
    user_id: str
    cars: List[ShortlistCarItem]
    comparison_metrics: List[Dict[str, Any]]

