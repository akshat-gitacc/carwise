"""
favourites_router.py — FastAPI router for Favourites / Shortlist endpoints.
"""

from __future__ import annotations

from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from ..favourites.schemas import (
    AddFavouriteRequest,
    RemoveFavouriteRequest,
    ShortlistResponse,
    ShortlistComparisonResponse,
)
from ..favourites.service import FavouritesService

router = APIRouter(tags=["Favourites"])
_service = FavouritesService()


@router.post("/favourites/add", tags=["Favourites"])
@router.post("/api/favourites/add", tags=["Favourites"])
def add_to_favourites(payload: AddFavouriteRequest):
    """
    Adds a car to user's favourites/shortlist.
    Registers a positive interaction signal (+4.0) with the recommendation engine.
    """
    is_new = _service.add_favourite(
        user_id=payload.user_id,
        car_id=payload.car_id,
        session_id=payload.session_id,
    )
    return {
        "status": "success",
        "message": "Car added to favourites" if is_new else "Car is already in favourites",
        "is_new": is_new,
        "car_id": payload.car_id,
        "user_id": payload.user_id,
    }


@router.delete("/favourites/{car_id}", tags=["Favourites"])
@router.delete("/api/favourites/{car_id}", tags=["Favourites"])
def remove_from_favourites(car_id: int, user_id: str = Query(...)):
    """
    Removes a car from user's favourites/shortlist.
    Tracks removal without treating it as a dislike.
    """
    removed = _service.remove_favourite(user_id=user_id, car_id=car_id)
    return {
        "status": "success" if removed else "not_found",
        "message": "Car removed from favourites" if removed else "Car was not in favourites",
        "removed": removed,
        "car_id": car_id,
        "user_id": user_id,
    }


@router.get("/favourites", response_model=ShortlistResponse, tags=["Favourites"])
@router.get("/api/favourites", response_model=ShortlistResponse, tags=["Favourites"])
def get_user_favourites(user_id: str = Query(...)):
    """
    Retrieves the user's shortlisted cars.
    """
    return _service.get_shortlist(user_id=user_id)


@router.get("/favourites/compare", response_model=ShortlistComparisonResponse, tags=["Favourites"])
@router.get("/api/favourites/compare", response_model=ShortlistComparisonResponse, tags=["Favourites"])
def compare_shortlist_cars(
    user_id: str = Query(...),
    car_ids: Optional[List[int]] = Query(None),
):
    """
    Compares shortlisted vehicles side-by-side.
    """
    return _service.compare_shortlist_cars(user_id=user_id, car_ids=car_ids)

