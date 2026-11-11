"""
interaction_tracker.py — Centralized User Interaction Tracking.
Records views, likes, dislikes, shortlists, removals, searches, comparisons, and clicks.
Applies time decay and frequency capping to prevent repeated views from distorting user preference signals.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
import numpy as np

from ..database import DatabaseManager, UserInteractionRecord


class InteractionTracker:
    """
    Centralized event tracker for all user activities across the CarWise application.
    """

    VALID_INTERACTION_TYPES = {
        "view",
        "like",
        "dislike",
        "shortlist",
        "remove_shortlist",
        "search",
        "compare",
        "open_recommendation",
    }

    # Base affinity weights for preference inference and collaborative matrix
    ACTION_WEIGHTS = {
        "shortlist": 4.0,
        "like": 3.5,
        "compare": 2.5,
        "open_recommendation": 2.0,
        "view": 2.0,
        "search": 1.0,
        "remove_shortlist": 0.0,
        "dislike": -3.0,
    }

    # Instantaneous geometric recency decay: Rank 0 = 1.0, Rank 1 = 0.20, Rank 2 = 0.04
    INSTANTANEOUS_RECENCY_DECAY = 0.20
    TIME_DECAY_HALF_LIFE_DAYS = 14.0  # Calendar half-life fallback

    def __init__(self, db: Optional[DatabaseManager] = None):
        self.db = db or DatabaseManager()

    def track(
        self,
        user_id: str,
        car_id: int,
        interaction_type: str,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Records an interaction event.
        Validates interaction type and logs with precise ISO UTC timestamp.
        """
        interaction_type = interaction_type.lower().strip()
        if interaction_type not in self.VALID_INTERACTION_TYPES:
            raise ValueError(
                f"Invalid interaction type '{interaction_type}'. Must be one of {self.VALID_INTERACTION_TYPES}"
            )

        # Check car exists
        car = self.db.get_car_by_id(car_id)
        if not car:
            raise ValueError(f"Car ID {car_id} not found in database.")

        # If shortlist, ensure synchronized with shortlist table
        if interaction_type == "shortlist":
            self.db.add_to_shortlist(user_id, car_id)
        elif interaction_type == "remove_shortlist":
            self.db.remove_from_shortlist(user_id, car_id)

        rec_id = self.db.log_interaction(
            user_id=user_id,
            car_id=car_id,
            interaction_type=interaction_type,
            session_id=session_id or user_id,
            metadata=metadata or {},
        )
        return rec_id

    def get_user_history(
        self,
        user_id: str,
        limit: int = 50,
        interaction_types: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieves raw interaction history records for a user."""
        records = self.db.get_user_interactions(user_id=user_id, limit=limit, interaction_types=interaction_types)
        results = []
        for r in records:
            car = self.db.get_car_by_id(r.car_id)
            d = r.to_dict()
            if car:
                d["car_details"] = {
                    "brand": car.brand,
                    "car": car.car,
                    "variant": car.variant,
                    "price_display": car.price_display,
                    "fuel_type": car.fuel_type,
                    "transmission": car.transmission,
                }
            results.append(d)
        return results

    def compute_effective_interactions(self, user_id: str) -> Dict[int, float]:
        """
        Aggregates all interactions for a user with:
        1. Base action weight.
        2. Instantaneous recency decay: steep rank decay (0.20^rank) so the latest
           cars and features immediately overtake previous brands.
        3. Calendar time decay fallback for long-inactive sessions.
        4. Diminishing returns frequency capping on repeated views.
        Returns a mapping of car_id -> effective_affinity_score.
        """
        interactions = self.db.get_user_interactions(user_id=user_id, limit=200)
        if not interactions:
            return {}

        now = datetime.utcnow()
        car_view_counts: Dict[int, int] = {}
        car_affinities: Dict[int, float] = {}

        # Active shortlist synchronization: genuine favourites take top priority
        active_shortlist = self.db.get_user_shortlist(user_id)
        active_shortlist_ids = {c.id for c in active_shortlist}
        unshortlisted_cars = set()

        # Prioritize currently active shortlisted cars first (newest shortlisted first)
        distinct_cars_by_recency: List[int] = []
        for c in active_shortlist:
            if c.id not in distinct_cars_by_recency:
                distinct_cars_by_recency.append(c.id)

        # Then append remaining cars from interaction history
        for r in interactions:
            if r.car_id not in distinct_cars_by_recency:
                distinct_cars_by_recency.append(r.car_id)

        for record in interactions:
            # Track explicit shortlist removal
            if record.interaction_type == "remove_shortlist":
                unshortlisted_cars.add(record.car_id)
                continue

            # If car is no longer in active shortlist, ignore historical shortlist weights
            if record.interaction_type == "shortlist" and (record.car_id not in active_shortlist_ids or record.car_id in unshortlisted_cars):
                continue

            base_w = self.ACTION_WEIGHTS.get(record.interaction_type, 1.0)
            if base_w <= 0.0:
                continue

            # Instantaneous recency decay: latest distinct car is rank 0 (multiplier 1.0)
            recency_rank = distinct_cars_by_recency.index(record.car_id)
            recency_decay = float(np.power(self.INSTANTANEOUS_RECENCY_DECAY, recency_rank))

            # Calendar decay as an additional safeguard
            try:
                t = datetime.fromisoformat(record.timestamp)
                days_diff = max(0.0, (now - t).total_seconds() / 86400.0)
                calendar_decay = float(np.power(0.5, days_diff / self.TIME_DECAY_HALF_LIFE_DAYS))
            except Exception:
                calendar_decay = 1.0

            car_id = record.car_id
            effective_w = base_w * recency_decay * calendar_decay

            # Apply frequency capping: diminishing returns on views
            if record.interaction_type == "view":
                views_so_far = car_view_counts.get(car_id, 0)
                # Dampen repetitive views logarithmically
                capped_w = effective_w / (1.0 + np.log1p(views_so_far))
                car_view_counts[car_id] = views_so_far + 1
                car_affinities[car_id] = car_affinities.get(car_id, 0.0) + capped_w
            else:
                car_affinities[car_id] = car_affinities.get(car_id, 0.0) + effective_w

        return car_affinities

