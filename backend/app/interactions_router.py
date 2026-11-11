"""
interactions_router.py — FastAPI router for user interaction tracking endpoints.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from ..recommender.interaction_tracker import InteractionTracker

router = APIRouter(tags=["Interactions"])
_tracker = InteractionTracker()


class TrackInteractionRequest(BaseModel):
    user_id: str = Field(..., description="User ID or persistent session ID")
    car_id: int = Field(..., description="ID of the interacted car")
    interaction_type: str = Field(
        ...,
        description="Type: view, like, dislike, shortlist, remove_shortlist, search, compare, open_recommendation",
    )
    session_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@router.post("/interactions/track", tags=["Interactions"])
@router.post("/api/interactions/track", tags=["Interactions"])
def track_interaction(payload: TrackInteractionRequest):
    """
    Centralized endpoint to record user interactions.
    Applies time decay and frequency capping in preference models.
    """
    try:
        rec_id = _tracker.track(
            user_id=payload.user_id,
            car_id=payload.car_id,
            interaction_type=payload.interaction_type,
            session_id=payload.session_id,
            metadata=payload.metadata,
        )
        return {
            "status": "success",
            "interaction_id": rec_id,
            "interaction_type": payload.interaction_type,
            "car_id": payload.car_id,
            "user_id": payload.user_id,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/interactions/history", tags=["Interactions"])
@router.get("/api/interactions/history", tags=["Interactions"])
def get_interaction_history(
    user_id: str = Query(...),
    limit: int = Query(50),
    interaction_types: Optional[List[str]] = Query(None),
):
    """
    Retrieves chronological user activity history.
    """
    history = _tracker.get_user_history(
        user_id=user_id, limit=limit, interaction_types=interaction_types
    )
    return {"user_id": user_id, "total_events": len(history), "history": history}

