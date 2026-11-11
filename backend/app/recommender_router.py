"""
recommender_router.py — FastAPI router for Hybrid Recommendation Engine and Explore endpoints.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from ..database import DatabaseManager
from ..recommender import (
    DynamicProfileGenerator,
    HybridRecommendationEngine,
    PersonalizedExploreRecommender,
    PreferenceInferenceEngine,
)

router = APIRouter(tags=["Recommendations"])

# Singleton engine instances
_db = DatabaseManager()
_hybrid_engine = HybridRecommendationEngine(_db)
_explore_engine = PersonalizedExploreRecommender(_db, _hybrid_engine)
_profile_generator = DynamicProfileGenerator(_db)
_inference_engine = PreferenceInferenceEngine(_db)


class RecommendationGenerateRequest(BaseModel):
    user_id: Optional[str] = Field("guest_user", description="User ID or session ID")
    budget: Optional[List[float]] = Field(None, description="[min_lakh, max_lakh]")
    budget_min: Optional[float] = 3.0
    budget_max: Optional[float] = 250.0
    fuel_type: Optional[str] = None
    transmission: Optional[str] = None
    brand: Optional[str] = None
    min_mileage: Optional[float] = 0.0
    priority: Optional[str] = "balanced"
    strictness: Optional[str] = "strict"
    features: Optional[Dict[str, Any]] = None
    top_n: Optional[int] = 6


@router.get("/recommendations/questions", tags=["Advisor Questions"])
@router.get("/api/advisor/questions", tags=["Advisor Questions"])
def get_advisor_questions():
    """
    Returns dynamically generated preference questions based on the actual database schema.
    No hardcoded fields that do not exist in the database.
    """
    questions = _profile_generator.generate_advisor_questions()
    return {"total_questions": len(questions), "questions": questions}


@router.post("/recommendations/generate", tags=["Recommendations"])
@router.post("/api/recommendations/generate", tags=["Recommendations"])
def generate_recommendations(payload: RecommendationGenerateRequest):
    """
    Generates recommendations using the Car Advisor profile through the Hybrid ML Recommendation Engine.
    Combines Content-based, Collaborative, and Knowledge-based filtering.
    """
    # Build answers dict
    answers: Dict[str, Any] = {
        "budget": payload.budget or [payload.budget_min, payload.budget_max],
        "fuel_type": payload.fuel_type,
        "transmission": payload.transmission,
        "brand": payload.brand,
        "min_mileage": payload.min_mileage,
        "priority": payload.priority,
        "strictness": payload.strictness,
        "features": payload.features or {},
    }

    validated_profile = _profile_generator.validate_and_build_profile(answers)

    # Save explicit preferences for user
    user_id = payload.user_id or "guest_user"
    _db.save_explicit_preferences(user_id, validated_profile)

    recommendations = _hybrid_engine.recommend(
        profile=validated_profile,
        user_id=user_id,
        top_n=payload.top_n or 6,
    )

    return {
        "user_id": user_id,
        "total_results": len(recommendations),
        "profile_applied": validated_profile,
        "results": recommendations,
    }


@router.get("/recommendations/explore", tags=["Explore"])
@router.get("/api/recommendations/explore", tags=["Explore"])
def get_explore_recommendations(user_id: str = Query("guest_user")):
    """
    Returns personalized car discovery for the Explore page.
    Generates "You May Like", "Discover More", and "Recently Viewed" sections.
    """
    explore_data = _explore_engine.get_explore_sections(user_id=user_id)
    return {
        "user_id": user_id,
        "data": explore_data,
    }


@router.get("/recommendations/preferences", tags=["Preferences"])
@router.get("/api/recommendations/preferences", tags=["Preferences"])
def get_user_preferences(user_id: str = Query(...)):
    """
    Retrieves stored explicit and inferred preferences for a user.
    """
    pref_rec = _db.get_user_preferences(user_id)
    inferred = _inference_engine.infer_preferences_from_history(user_id)

    return {
        "user_id": user_id,
        "explicit_preferences": pref_rec.explicit_preferences if pref_rec else {},
        "inferred_preferences": inferred or (pref_rec.inferred_preferences if pref_rec else {}),
        "updated_at": pref_rec.updated_at if pref_rec else None,
    }

