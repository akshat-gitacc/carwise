"""
preference_inference.py — User Preference Inference Engine.
Infers implicit preferences from browsing history, likes, and shortlists,
while maintaining strict precedence of explicit Advisor preferences.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from ..database import DatabaseManager
from .interaction_tracker import InteractionTracker


class PreferenceInferenceEngine:
    """
    Analyzes historical interactions to synthesize inferred preference profiles,
    and blends them with explicit advisor preferences following hierarchical precedence.
    """

    def __init__(self, db: Optional[DatabaseManager] = None):
        self.db = db or DatabaseManager()
        self.tracker = InteractionTracker(self.db)

    def infer_preferences_from_history(self, user_id: str) -> Dict[str, Any]:
        """
        Synthesizes an inferred profile from weighted interaction history.
        Returns a dictionary of inferred preferences.
        """
        affinities = self.tracker.compute_effective_interactions(user_id)
        if not affinities:
            return {}

        car_records = []
        weights = []
        for car_id, affinity in affinities.items():
            if affinity > 0:
                car = self.db.get_car_by_id(car_id)
                if car:
                    car_records.append(car)
                    weights.append(affinity)

        if not car_records:
            return {}

        weights_arr = np.array(weights, dtype=np.float32)
        total_weight = weights_arr.sum()
        norm_weights = weights_arr / total_weight if total_weight > 0 else weights_arr

        # 1. Inferred Budget: centered tightly around the weighted target price
        prices = [c.price_lakh for c in car_records if c.price_lakh is not None]
        price_weights = [w for c, w in zip(car_records, norm_weights) if c.price_lakh is not None]
        if prices:
            w_mean_price = float(np.average(prices, weights=price_weights))
            p_min = float(max(2.5, w_mean_price * 0.70))
            p_max = float(min(250.0, w_mean_price * 1.35))
        else:
            w_mean_price = 12.0
            p_min, p_max = 5.0, 25.0

        # 2. Inferred Fuel Types
        fuel_scores: Dict[str, float] = {}
        for c, w in zip(car_records, norm_weights):
            f = c.fuel_type
            fuel_scores[f] = fuel_scores.get(f, 0.0) + float(w)
        top_fuel = max(fuel_scores.items(), key=lambda x: x[1])[0] if fuel_scores else "Petrol"

        # 3. Inferred Transmission
        trans_scores: Dict[str, float] = {}
        for c, w in zip(car_records, norm_weights):
            t = c.transmission
            trans_scores[t] = trans_scores.get(t, 0.0) + float(w)
        top_trans = max(trans_scores.items(), key=lambda x: x[1])[0] if trans_scores else "Manual"

        # 4. Inferred Brands
        brand_scores: Dict[str, float] = {}
        for c, w in zip(car_records, norm_weights):
            b = c.brand
            brand_scores[b] = brand_scores.get(b, 0.0) + float(w)
        top_brands = sorted(brand_scores.items(), key=lambda x: x[1], reverse=True)
        preferred_brand = top_brands[0][0] if top_brands else None

        # 5. Inferred Mileage
        mileages = [c.mileage for c in car_records if c.mileage is not None]
        m_weights = [w for c, w in zip(car_records, norm_weights) if c.mileage is not None]
        avg_mileage = float(np.average(mileages, weights=m_weights)) if mileages else 18.0

        # 6. Inferred Features
        feature_scores: Dict[str, float] = {}
        for c, w in zip(car_records, norm_weights):
            for fk, fv in c.features.items():
                if fv:
                    feature_scores[fk] = feature_scores.get(fk, 0.0) + float(w)
        desired_features = {k: 1 for k, v in feature_scores.items() if v >= 0.25}

        inferred = {
            "budget_min": round(p_min, 1),
            "budget_max": round(p_max, 1),
            "budget_target": round(w_mean_price, 1),
            "fuel_type": top_fuel,
            "transmission": top_trans,
            "brand": preferred_brand,
            "preferred_mileage": round(avg_mileage, 1),
            "features": desired_features,
            "confidence": min(1.0, float(total_weight) / 10.0),
        }

        # Persist inferred profile to database
        self.db.save_inferred_preferences(user_id, inferred)
        return inferred

    def get_blended_profile(self, user_id: str) -> Dict[str, Any]:
        """
        Constructs a combined profile for the Explore experience:
        Explicit preferences have strict precedence; inferred preferences fill gaps.
        """
        pref_record = self.db.get_user_preferences(user_id)
        explicit = pref_record.explicit_preferences if pref_record else {}
        inferred = self.infer_preferences_from_history(user_id)

        # Baseline defaults
        profile: Dict[str, Any] = {
            "budget_min": 3.0,
            "budget_max": 250.0,
            "fuel_type": None,
            "transmission": None,
            "brand": None,
            "preferred_brand": None,
            "preferred_fuel": None,
            "preferred_transmission": None,
            "min_mileage": 0.0,
            "priority": "balanced",
            "strictness": "flexible",  # Explore is exploratory
            "features": {},
        }

        # 1. Overlay Inferred preferences as soft priors
        if inferred:
            profile["budget_min"] = inferred.get("budget_min", profile["budget_min"])
            profile["budget_max"] = inferred.get("budget_max", profile["budget_max"])
            profile["budget_target"] = inferred.get("budget_target")
            # Inferred brand is a soft prior (preferred_brand), NOT a hard constraint
            profile["preferred_brand"] = inferred.get("brand")
            profile["preferred_fuel"] = inferred.get("fuel_type")
            profile["preferred_transmission"] = inferred.get("transmission")
            profile["preferred_mileage"] = inferred.get("preferred_mileage", 18.0)
            profile["features"] = inferred.get("features", {})

        # 2. Overlay Explicit preferences with HIGHER priority (hard constraints)
        for k, v in explicit.items():
            if v is not None and v != "Any" and v != "":
                profile[k] = v

        return profile

