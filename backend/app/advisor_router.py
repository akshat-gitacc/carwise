"""
advisor_router.py — FastAPI router for CarWise Car Advisor service.
Exposes endpoints for vehicle recommendations (powered by Hybrid ML Engine),
head-to-head comparison, authorized dealerships locator, and Gemini AI conversational advisory.
"""

from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np
from fastapi import APIRouter, Query
from pydantic import BaseModel

from ..advisor import (
    call_gemini_chat,
    compare_cars,
    get_brands,
    get_dealership_cities,
    get_dealerships,
    get_fuel_types,
    get_models_for_brand,
    get_transmissions,
    get_unique_car_count,
)
from ..database import DatabaseManager
from ..recommender.hybrid_recommender import HybridRecommendationEngine
from ..recommender.profile_generator import DynamicProfileGenerator

router = APIRouter(prefix="/api/advisor", tags=["Advisor"])
_db = DatabaseManager()
_hybrid_engine = HybridRecommendationEngine(_db)
_profile_generator = DynamicProfileGenerator(_db)


class ChatRequest(BaseModel):
    message: str
    history: List[Dict] = []
    context_cars: List[Dict] = []
    city: Optional[str] = None
    gemini_api_key: Optional[str] = None


@router.get("/meta")
def meta():
    return {
        "unique_cars": get_unique_car_count(),
        "brands": ["Any"] + get_brands(),
        "fuel_types": ["Any"] + get_fuel_types(),
        "transmissions": ["Any"] + get_transmissions(),
        "dealership_cities": ["Any"] + get_dealership_cities(),
    }


@router.get("/models")
def models(brand: str = Query(...)):
    return {"models": get_models_for_brand(brand)}


@router.get("/recommend")
def recommend_cars(
    budget_min: float = Query(3.0),
    budget_max: float = Query(250.0),
    fuel_type: Optional[str] = Query(None),
    transmission: Optional[str] = Query(None),
    min_mileage: float = Query(0.0),
    brand: Optional[str] = Query(None),
    priority: str = Query("balanced"),
    top_n: int = Query(6),
    user_id: Optional[str] = Query("guest_user"),
):
    """
    Upgraded: Replaced old non-ML recommender with the new Hybrid ML Recommendation Engine.
    Executes Content-based filtering (Scikit-learn cosine/KNN), Knowledge-based hard/soft filtering,
    and Collaborative filtering (when user history is available).
    """
    profile_input = {
        "budget": [budget_min, budget_max],
        "budget_min": budget_min,
        "budget_max": budget_max,
        "fuel_type": fuel_type if fuel_type != "Any" else None,
        "transmission": transmission if transmission != "Any" else None,
        "brand": brand if brand != "Any" else None,
        "min_mileage": min_mileage,
        "priority": priority,
    }
    validated = _profile_generator.validate_and_build_profile(profile_input)
    if user_id:
        _db.save_explicit_preferences(user_id, validated)
    recs = _hybrid_engine.recommend(profile=validated, user_id=user_id, top_n=top_n)

    if not recs:
        return {"results": []}

    return {"results": recs}



@router.get("/compare")
def compare(
    brand_a: str = Query(...),
    model_a: str = Query(...),
    brand_b: str = Query(...),
    model_b: str = Query(...),
):
    ca, cb = compare_cars(brand_a, model_a, brand_b, model_b)
    return {"car_a": ca, "car_b": cb}


@router.get("/dealerships")
def dealerships(
    city: Optional[str] = Query(None),
    brand: Optional[str] = Query(None),
):
    dl = get_dealerships(
        city=city if city != "Any" else None,
        brand=brand if brand != "Any" else None,
    )
    return {"dealerships": dl}


@router.post("/chat")
def chat(payload: ChatRequest):
    reply = call_gemini_chat(
        api_key=payload.gemini_api_key,
        message=payload.message,
        history=payload.history,
        context_cars=payload.context_cars,
        city=payload.city,
    )
    return {"reply": reply}

