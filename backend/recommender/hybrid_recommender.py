"""
hybrid_recommender.py — Master Hybrid AI/ML Recommendation Engine.
Integrates Content-Based, Collaborative Filtering, and Knowledge-Based filtering
into a unified modular recommendation pipeline.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from ..database import DatabaseManager, CarRecord
from .collaborative_filtering import CollaborativeFilteringRecommender
from .content_based import ContentBasedRecommender
from .knowledge_based import KnowledgeBasedFilter
from .preprocessing import CarPreprocessor
from .profile_generator import DynamicProfileGenerator
from .ranking import HybridRanker


class HybridRecommendationEngine:
    """
    Main orchestration class for the CarWise recommendation system.
    Supports both Car Advisor (explicit profile matching) and Explore (historical activity matching).
    """

    def __init__(self, db: Optional[DatabaseManager] = None):
        self.db = db or DatabaseManager()
        self.preprocessor = CarPreprocessor(self.db)
        self.profile_generator = DynamicProfileGenerator(self.db)
        self.content_recommender = ContentBasedRecommender(self.preprocessor)
        self.collab_recommender = CollaborativeFilteringRecommender(self.db)
        self.knowledge_filter = KnowledgeBasedFilter()
        self.ranker = HybridRanker()

        # Fit models
        self._initialize_models()

    def _initialize_models(self) -> None:
        """Initializes and trains content and collaborative components."""
        self.content_recommender.fit()
        self.collab_recommender.fit()

    def recommend(
        self,
        profile: Dict[str, Any],
        user_id: Optional[str] = None,
        top_n: int = 6,
        apply_diversity: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Generates hybrid recommendations for a given user profile.
        1. Hard constraint evaluation via KnowledgeBasedFilter.
        2. Content-based similarity scoring via ContentBasedRecommender (KNN/Cosine).
        3. Collaborative filtering scoring via CollaborativeFilteringRecommender (if available).
        4. Soft knowledge-based compatibility scoring.
        5. Score fusion, dynamic weight redistribution for cold-start, deduplication, and ranking.
        """
        # Ensure fresh data if needed
        all_cars = self.content_recommender.df
        if all_cars.empty:
            self._initialize_models()
            all_cars = self.content_recommender.df

        # 1. Hard constraints pruning
        candidates = self.knowledge_filter.apply_hard_constraints(all_cars, profile)
        if candidates.empty:
            # Fallback to broader budget or active cars if constraints eliminated everything
            candidates = all_cars[all_cars["price_lakh"].notna()]
            if candidates.empty:
                return []

        # 2. Content-based scores
        content_scores = self.content_recommender.compute_similarity_scores(profile)

        # 3. Collaborative filtering scores
        collab_scores: Dict[int, float] = {}
        collab_available = False
        if user_id and self.collab_recommender.is_available:
            collab_scores = self.collab_recommender.predict_user_scores(
                user_id=user_id, candidate_car_ids=candidates["id"].tolist()
            )
            collab_available = bool(collab_scores)

        # 4. Knowledge-based compatibility scores
        knowledge_scores = self.knowledge_filter.compute_compatibility_scores(candidates, profile)

        # 5. Hybrid ranking with dynamic weight redistribution
        ranked_df = self.ranker.compute_hybrid_scores(
            candidate_df=candidates,
            content_scores=content_scores,
            knowledge_scores=knowledge_scores,
            collab_scores=collab_scores,
            collab_available=collab_available,
        )

        # Deduplicate: Keep highest scoring trim variant per unique Car model
        ranked_df = ranked_df.drop_duplicates(subset=["brand", "car"], keep="first")

        # 6. Optional diversity reranking
        if apply_diversity:
            ranked_df = self.ranker.apply_diversity_reranking(ranked_df, top_n=top_n)

        results_df = ranked_df.head(top_n).copy()

        # Format output records
        results: List[Dict[str, Any]] = []
        for rank, (_, row) in enumerate(results_df.iterrows(), start=1):
            car_id = int(row["id"])
            p_lakh = float(row["price_lakh"]) if pd.notna(row["price_lakh"]) else 0.0

            # Calculate estimated EMI (80% loan, 5 years @ 9% interest)
            if p_lakh > 0:
                loan = p_lakh * 100_000 * 0.8
                r = 0.09 / 12
                emi = int(loan * r * ((1 + r) ** 60) / (((1 + r) ** 60) - 1))
                emi_disp = f"Rs. {emi:,}/mo"
            else:
                emi_disp = "—"

            # Build clear, factual match explanations based on actual attributes
            reasons = self._generate_explanation(row, profile, collab_available, user_id)

            rec = {
                "id": car_id,
                "rank": rank,
                "Brand": row.get("brand"),
                "Car": row.get("car"),
                "Variant": row.get("variant", "Standard"),
                "Price_Lakh": p_lakh,
                "Price_Display": row.get("price_display", "Price on request"),
                "Mileage_Clean": float(row["mileage"]) if pd.notna(row.get("mileage")) else None,
                "Displacement_Clean": float(row["displacement"]) if pd.notna(row.get("displacement")) else None,
                "Fuel_Type_Simple": row.get("fuel_type_clean", row.get("fuel_type")),
                "Transmission_Simple": row.get("transmission_clean", row.get("transmission")),
                "Feature_Score": float(row.get("feature_score", 0.0)),
                "Active_Status": row.get("active_status", "Active"),
                "Score": float(row.get("hybrid_score", 0.0)),
                "content_score": float(row.get("score_content", 0.0)),
                "knowledge_score": float(row.get("score_knowledge", 0.0)),
                "collab_score": float(row.get("score_collab", 0.0)) if pd.notna(row.get("score_collab")) else None,
                "collab_available": collab_available,
                "estimated_emi": emi_disp,
                "description": row.get("description", ""),
                "recommendation_reason": reasons[0] if reasons else "High overall match with your criteria.",
                "all_reasons": reasons,
            }
            results.append(rec)

        # Log recommendation history if user provided
        if user_id:
            try:
                rec_ids = [r["id"] for r in results]
                self.db.log_recommendation_history(user_id, rec_ids, source="hybrid_advisor")
            except Exception:
                pass

        return results

    def _generate_explanation(
        self, car_row: pd.Series, profile: Dict[str, Any], collab_available: bool, user_id: Optional[str]
    ) -> List[str]:
        """Generates explainable, evidence-backed reasons for recommending this car."""
        reasons = []

        # Budget reason
        p_lakh = car_row.get("price_lakh")
        b_min = profile.get("budget_min", 3.0)
        b_max = profile.get("budget_max", 250.0)
        if pd.notna(p_lakh):
            if b_min <= p_lakh <= b_max:
                reasons.append(f"Fits within your target budget of Rs. {b_min:.1f} - {b_max:.1f} Lakh.")

        # Mileage reason
        m = car_row.get("mileage")
        if pd.notna(m) and m >= 18.0:
            reasons.append(f"Delivers excellent fuel efficiency of {m:.1f} kmpl.")

        # Fuel & Transmission reason
        fuel = car_row.get("fuel_type_clean")
        trans = car_row.get("transmission_clean")
        if profile.get("fuel_type") and profile.get("fuel_type") != "Any":
            reasons.append(f"Matches your preferred {fuel} powertrain.")
        if profile.get("transmission") and profile.get("transmission") != "Any":
            reasons.append(f"Equipped with your chosen {trans} gearbox.")

        # Tech equipment
        f_score = car_row.get("feature_score", 0.0)
        if f_score >= 4.0:
            reasons.append(f"Richly equipped with {int(f_score)} premium comfort & safety amenities.")

        # Collaborative explanation
        if collab_available and pd.notna(car_row.get("score_collab")) and car_row.get("score_collab", 0) > 0.5:
            reasons.append("Popular among drivers with preferences similar to yours.")

        if not reasons:
            reasons.append(f"Strong overall specification alignment for {car_row.get('brand')} {car_row.get('car')}.")

        return reasons

