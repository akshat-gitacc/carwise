"""
service.py — Favourites / Shortlist Service.
Handles adding, removing, retrieving, and comparing shortlisted vehicles.
Synchronizes user shortlist actions with the recommendation interaction tracker.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from ..database import DatabaseManager, CarRecord
from ..recommender.interaction_tracker import InteractionTracker
from .schemas import ShortlistCarItem, ShortlistResponse, ShortlistComparisonResponse


class FavouritesService:
    """
    Manages user shortlist operations and dispatches interaction events.
    """

    def __init__(self, db: Optional[DatabaseManager] = None, tracker: Optional[InteractionTracker] = None):
        self.db = db or DatabaseManager()
        self.tracker = tracker or InteractionTracker(self.db)

    def add_favourite(self, user_id: str, car_id: int, session_id: Optional[str] = None) -> bool:
        """
        Adds car to user's shortlist and registers a strong positive interaction (+4.0) with recommender.
        Returns True if newly added, False if duplicate.
        """
        is_new = self.db.add_to_shortlist(user_id=user_id, car_id=car_id)
        if is_new:
            # Emit positive interaction signal
            self.tracker.track(
                user_id=user_id,
                car_id=car_id,
                interaction_type="shortlist",
                session_id=session_id or user_id,
                metadata={"action": "add_to_favourites"},
            )
        return is_new

    def remove_favourite(self, user_id: str, car_id: int, session_id: Optional[str] = None) -> bool:
        """
        Removes car from shortlist and tracks as a distinct remove_shortlist event.
        Does NOT treat as dislike.
        """
        removed = self.db.remove_from_shortlist(user_id=user_id, car_id=car_id)
        if removed:
            self.tracker.track(
                user_id=user_id,
                car_id=car_id,
                interaction_type="remove_shortlist",
                session_id=session_id or user_id,
                metadata={"action": "remove_from_favourites"},
            )
        return removed

    def get_shortlist(self, user_id: str) -> ShortlistResponse:
        """Retrieves all shortlisted cars for a user."""
        records: List[CarRecord] = self.db.get_user_shortlist(user_id=user_id)
        items = []
        for r in records:
            items.append(
                ShortlistCarItem(
                    id=r.id,
                    brand=r.brand,
                    car=r.car,
                    variant=r.variant,
                    price_lakh=r.price_lakh,
                    price_display=r.price_display,
                    fuel_type=r.fuel_type,
                    transmission=r.transmission,
                    mileage=r.mileage,
                    displacement=r.displacement,
                    feature_score=r.feature_score,
                    active_status=r.active_status,
                    description=r.description,
                    features=r.features,
                    created_at=r.created_at,
                )
            )
        return ShortlistResponse(user_id=user_id, total_count=len(items), items=items)

    def is_favourite(self, user_id: str, car_id: int) -> bool:
        return self.db.is_car_shortlisted(user_id, car_id)

    def compare_shortlist_cars(self, user_id: str, car_ids: Optional[List[int]] = None) -> ShortlistComparisonResponse:
        """
        Builds side-by-side comparison metrics between shortlisted vehicles.
        If car_ids is omitted, compares up to top 4 cars on user's shortlist.
        """
        shortlist = self.get_shortlist(user_id)
        if not shortlist.items:
            return ShortlistComparisonResponse(user_id=user_id, cars=[], comparison_metrics=[])

        if car_ids:
            selected_cars = [c for c in shortlist.items if c.id in car_ids][:4]
        else:
            selected_cars = shortlist.items[:4]

        if not selected_cars:
            return ShortlistComparisonResponse(user_id=user_id, cars=[], comparison_metrics=[])

        # Build comparison rows (no passive interaction tracking on read/render)
        metrics = [
            {
                "metric": "Ex-Showroom Price",
                "key": "price_display",
                "values": {c.id: c.price_display for c in selected_cars},
            },
            {
                "metric": "Fuel Powertrain",
                "key": "fuel_type",
                "values": {c.id: c.fuel_type for c in selected_cars},
            },
            {
                "metric": "Transmission",
                "key": "transmission",
                "values": {c.id: c.transmission for c in selected_cars},
            },
            {
                "metric": "Fuel Efficiency (kmpl)",
                "key": "mileage",
                "values": {c.id: f"{c.mileage:.1f} kmpl" if c.mileage else "—" for c in selected_cars},
            },
            {
                "metric": "Engine Displacement",
                "key": "displacement",
                "values": {c.id: f"{int(c.displacement)} cc" if c.displacement else "—" for c in selected_cars},
            },
            {
                "metric": "Tech Equipment Score",
                "key": "feature_score",
                "values": {c.id: f"{int(c.feature_score)} / 8 Features" for c in selected_cars},
            },
            {
                "metric": "Sunroof / Moonroof",
                "key": "sunroof",
                "values": {c.id: "Yes" if c.features.get("sunroof___moonroof") else "No" for c in selected_cars},
            },
            {
                "metric": "Ventilated Seats",
                "key": "ventilated_seats",
                "values": {c.id: "Yes" if c.features.get("ventilated_seats") else "No" for c in selected_cars},
            },
            {
                "metric": "Cruise Control",
                "key": "cruise_control",
                "values": {c.id: "Yes" if c.features.get("cruise_control") else "No" for c in selected_cars},
            },
            {
                "metric": "Wireless Charger",
                "key": "wireless_charger",
                "values": {c.id: "Yes" if c.features.get("wireless_charger") else "No" for c in selected_cars},
            },
            {
                "metric": "Rear AC Vents",
                "key": "rear_ac",
                "values": {c.id: "Yes" if c.features.get("rear_ac") else "No" for c in selected_cars},
            },
        ]

        return ShortlistComparisonResponse(
            user_id=user_id, cars=selected_cars, comparison_metrics=metrics
        )

