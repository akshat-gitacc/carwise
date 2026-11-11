"""
personalized_recommender.py — Explore Page Personalized Recommendation Service.
Powers the 3 distinct Explore sections:
1. "You May Like" (primary personalized hybrid discovery with factual explanations)
2. "Discover More" (diversity-aware variety, alternative brands, category exploration)
3. "Recently Viewed" (actual chronologically ordered browsing history)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set
import numpy as np
import pandas as pd

from ..database import DatabaseManager, CarRecord
from .hybrid_recommender import HybridRecommendationEngine
from .interaction_tracker import InteractionTracker
from .preference_inference import PreferenceInferenceEngine


class PersonalizedExploreRecommender:
    """
    Dedicated engine for continuous, dynamic car discovery on the Explore page.
    Continuously responds to user interactions (views, likes, shortlists, comparisons).
    """

    def __init__(
        self,
        db: Optional[DatabaseManager] = None,
        hybrid_engine: Optional[HybridRecommendationEngine] = None,
    ):
        self.db = db or DatabaseManager()
        self.hybrid_engine = hybrid_engine or HybridRecommendationEngine(self.db)
        self.tracker = InteractionTracker(self.db)
        self.inference = PreferenceInferenceEngine(self.db)

    def get_explore_sections(self, user_id: str) -> Dict[str, Any]:
        """
        Generates the full data payload for the Explore page:
        - you_may_like: List of top personalized vehicle matches
        - discover_more: List of diverse, alternative brand discoveries
        - recently_viewed: List of chronologically viewed vehicles
        - inferred_summary: Summary of inferred interests powering the page
        """
        # 1. Fetch Recently Viewed
        recently_viewed = self._get_recently_viewed(user_id, limit=6)
        viewed_ids: Set[int] = {c["id"] for c in recently_viewed}

        # 2. Get user's blended profile (explicit preferences + inferred browsing signals)
        blended_profile = self.inference.get_blended_profile(user_id)
        inferred = self.inference.infer_preferences_from_history(user_id)

        # 3. Section 1: "You May Like"
        # Run hybrid engine with blended profile
        you_may_like_candidates = self.hybrid_engine.recommend(
            profile=blended_profile,
            user_id=user_id,
            top_n=12,
            apply_diversity=False,
        )

        # Refine recommendations with personalized reasons derived from user actions
        you_may_like: List[Dict[str, Any]] = []
        for rec in you_may_like_candidates:
            # Check shortlist status
            rec["is_shortlisted"] = self.db.is_car_shortlisted(user_id, rec["id"])
            # Generate custom explainability message based on actual interactions
            rec["explore_explanation"] = self._build_explore_explanation(rec, user_id, blended_profile, inferred)
            you_may_like.append(rec)

        # Slice to top 6
        you_may_like_top6 = you_may_like[:6]
        shown_ids: Set[int] = {c["id"] for c in you_may_like_top6}

        # 4. Section 2: "Discover More"
        # Prioritize diverse recommendations, alternative brands, and variety
        discover_profile = blended_profile.copy()
        discover_profile["preferred_brand"] = None
        discover_profile["brand"] = None  # Ensure discovery explores other manufacturers

        discover_more_candidates = self.hybrid_engine.recommend(
            profile=discover_profile,
            user_id=user_id,
            top_n=20,
            apply_diversity=True,
        )

        discover_more: List[Dict[str, Any]] = []
        for rec in discover_more_candidates:
            # Avoid repeating cars already in You May Like
            if rec["id"] in shown_ids:
                continue
            rec["is_shortlisted"] = self.db.is_car_shortlisted(user_id, rec["id"])
            rec["explore_explanation"] = f"Alternative option from {rec['Brand']} matching your criteria."
            discover_more.append(rec)
            if len(discover_more) >= 6:
                break

        pref_record = self.db.get_user_preferences(user_id)
        explicit = pref_record.explicit_preferences if pref_record else {}
        is_calibrated = bool(inferred or explicit)

        # Collect real-time telemetry info bits used for current calibration
        history_all = self.db.get_user_interactions(user_id=user_id, limit=200)
        shortlist_all = self.db.get_user_shortlist(user_id=user_id)

        shortlist_count = len(shortlist_all)
        view_count = sum(1 for r in history_all if r.interaction_type in ("view", "open_recommendation"))
        compare_count = sum(1 for r in history_all if r.interaction_type == "compare")
        total_signals = len(history_all)

        anchor_models: List[str] = []
        for r in history_all:
            car_obj = self.db.get_car_by_id(r.car_id)
            if car_obj:
                name = f"{car_obj.brand} {car_obj.car}"
                if name not in anchor_models:
                    anchor_models.append(name)
            if len(anchor_models) >= 3:
                break
        if not anchor_models:
            for c in shortlist_all[:3]:
                anchor_models.append(f"{c.brand} {c.car}")

        if explicit and total_signals > 0:
            calibration_mode = "Blended Profile (Questionnaire + Live Signals)"
        elif explicit:
            calibration_mode = "Car Advisor Questionnaire Profile"
        elif total_signals > 0:
            calibration_mode = "Real-Time Interaction Signals"
        else:
            calibration_mode = "Catalog Baseline (Cold Start)"

        fuel_display = explicit.get("fuel_type") or inferred.get("fuel_type") or "All Fuels"
        trans_display = explicit.get("transmission") or inferred.get("transmission") or "All Transmissions"
        brand_display = explicit.get("brand") or inferred.get("brand") or "Multiple Brands"
        b_min = blended_profile.get("budget_min", 3.0)
        b_max = blended_profile.get("budget_max", 250.0)
        if b_min > 3.0 or b_max < 250.0:
            budget_display = f"₹{b_min:.1f}L – ₹{b_max:.1f}L"
        else:
            budget_display = "All Budgets"

        b_target = blended_profile.get("budget_target")
        target_display = f"₹{b_target:.1f}L" if b_target else "Flexible"

        summary_payload = {
            "is_calibrated": is_calibrated,
            "mode": calibration_mode,
            "preferred_fuel": fuel_display,
            "preferred_transmission": trans_display,
            "preferred_brand": brand_display,
            "budget_range": budget_display,
            "budget_target": target_display,
            "shortlist_count": shortlist_count,
            "view_count": view_count,
            "compare_count": compare_count,
            "total_signals": total_signals,
            "anchor_models": anchor_models,
            "decay_status": "Instantaneous Recency Shift (Latest Car Prioritized)" if total_signals > 0 else "Baseline (Cold Start)",
        }

        return {
            "you_may_like": you_may_like_top6,
            "discover_more": discover_more,
            "recently_viewed": recently_viewed,
            "inferred_summary": summary_payload,
        }

    def _get_recently_viewed(self, user_id: str, limit: int = 6) -> List[Dict[str, Any]]:
        """Retrieves user's recently viewed vehicles using actual interaction records."""
        history = self.db.get_user_interactions(
            user_id=user_id, limit=30, interaction_types=["view", "open_recommendation"]
        )
        seen_cars: Set[int] = set()
        recently_viewed: List[Dict[str, Any]] = []

        for record in history:
            if record.car_id in seen_cars:
                continue
            car = self.db.get_car_by_id(record.car_id)
            if car:
                seen_cars.add(record.car_id)
                p_lakh = car.price_lakh or 0.0
                emi_disp = "—"
                if p_lakh > 0:
                    loan = p_lakh * 100_000 * 0.8
                    r = 0.09 / 12
                    emi = int(loan * r * ((1 + r) ** 60) / (((1 + r) ** 60) - 1))
                    emi_disp = f"Rs. {emi:,}/mo"

                recently_viewed.append({
                    "id": car.id,
                    "Brand": car.brand,
                    "Car": car.car,
                    "Variant": car.variant,
                    "Price_Lakh": car.price_lakh,
                    "Price_Display": car.price_display,
                    "Mileage_Clean": car.mileage,
                    "Displacement_Clean": car.displacement,
                    "Fuel_Type_Simple": car.fuel_type,
                    "Transmission_Simple": car.transmission,
                    "Feature_Score": car.feature_score,
                    "estimated_emi": emi_disp,
                    "viewed_at": record.timestamp,
                    "is_shortlisted": self.db.is_car_shortlisted(user_id, car.id),
                })
                if len(recently_viewed) >= limit:
                    break

        return recently_viewed

    def _build_explore_explanation(
        self,
        car_rec: Dict[str, Any],
        user_id: str,
        profile: Dict[str, Any],
        inferred: Dict[str, Any],
    ) -> str:
        """
        Creates an evidence-grounded explanation based on user's actual past actions.
        Examples:
        - "Similar to cars you previously shortlisted"
        - "Matches your interest in Automatic Petrol vehicles"
        - "Aligned with your Rs. 10 - 20 Lakh budget history"
        - "Popular among users with similar preferences"
        """
        # 1. Check if user liked/shortlisted something from same brand
        shortlist = self.db.get_user_shortlist(user_id)
        shortlist_brands = {c.brand.lower() for c in shortlist if c.brand}
        car_brand = str(car_rec.get("Brand", ""))
        if car_brand.lower() in shortlist_brands:
            return f"From {car_brand}, similar to models on your shortlist."

        # 2. Check if user previously interacted with this powertrain
        car_fuel = str(car_rec.get("Fuel_Type_Simple", "")).lower()
        car_trans = str(car_rec.get("Transmission_Simple", "")).lower()
        inf_fuel = str(inferred.get("fuel_type", "")).lower()
        inf_trans = str(inferred.get("transmission", "")).lower()

        if inf_fuel and car_fuel == inf_fuel:
            if inf_trans and car_trans == inf_trans:
                return f"Matches your affinity for {inferred.get('transmission')} {inferred.get('fuel_type')} vehicles."
            return f"Matches your preferred {inferred.get('fuel_type')} powertrain."

        # 3. Collaborative explanation
        if car_rec.get("collab_available") and car_rec.get("collab_score", 0) > 0.5:
            return "Popular choice among buyers with similar preferences."

        # 4. Budget alignment
        b_target = profile.get("budget_target")
        p_lakh = car_rec.get("Price_Lakh")
        if b_target and p_lakh and abs(p_lakh - b_target) <= 3.5:
            return f"Fits closely with your preferred budget of around ₹{b_target:.1f} Lakh."

        # 5. Mileage
        m = car_rec.get("Mileage_Clean")
        if m and m >= 19.0:
            return f"Recommended for its standout fuel efficiency of {m:.1f} kmpl."

        return f"High compatibility match with your recent CarWise activity."

