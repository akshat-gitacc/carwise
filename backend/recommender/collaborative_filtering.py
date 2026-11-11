"""
collaborative_filtering.py — Collaborative Filtering Engine using Implicit Feedback and Matrix Factorization.
Learns patterns from real user interactions (likes, shortlists, views, searches, comparisons, dislikes).
Gracefully handles cold starts without fabricating any synthetic data.
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*encountered in matmul.*")

from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

from ..database import DatabaseManager, UserInteractionRecord


class CollaborativeFilteringRecommender:
    """
    Item-based collaborative filtering and matrix factorization over real user interactions.
    Detects cold start and reports availability status cleanly.
    """

    # Interaction weights for implicit feedback
    INTERACTION_WEIGHTS: Dict[str, float] = {
        "shortlist": 4.0,
        "like": 3.0,
        "compare": 2.0,
        "open_recommendation": 1.5,
        "search": 1.0,
        "view": 1.0,
        "remove_shortlist": 0.0,  # Neutral signal
        "dislike": -3.0,
    }

    # Minimum threshold of users and interactions to activate collaborative filtering
    MIN_USERS_FOR_COLLAB = 2
    MIN_INTERACTIONS_FOR_COLLAB = 3

    # Half-life for time decay in days
    TIME_DECAY_HALF_LIFE_DAYS = 30.0

    def __init__(self, db: Optional[DatabaseManager] = None):
        self.db = db or DatabaseManager()
        self.user_item_matrix: Optional[pd.DataFrame] = None
        self.item_similarity_matrix: Optional[pd.DataFrame] = None
        self.svd_model: Optional[TruncatedSVD] = None
        self.item_factors: Optional[np.ndarray] = None
        self.is_available: bool = False
        self.unique_users_count: int = 0
        self.total_interactions_count: int = 0

    def fit(self) -> "CollaborativeFilteringRecommender":
        """
        Extracts real user interactions from database, applies time decay,
        constructs interaction matrix, and computes item-item similarities.
        """
        interactions: List[UserInteractionRecord] = self.db.get_all_interactions()
        self.total_interactions_count = len(interactions)

        if not interactions:
            self.is_available = False
            return self

        # Extract distinct users
        user_ids = {i.user_id for i in interactions}
        self.unique_users_count = len(user_ids)

        # Check cold-start threshold
        if (
            self.unique_users_count < self.MIN_USERS_FOR_COLLAB
            or self.total_interactions_count < self.MIN_INTERACTIONS_FOR_COLLAB
        ):
            self.is_available = False
            return self

        # Process interaction scores with time decay
        now = datetime.utcnow()
        user_car_scores: Dict[Tuple[str, int], float] = {}

        for record in interactions:
            if record.interaction_type == "remove_shortlist":
                # Explicitly zero out score for un-shortlisted vehicle
                user_car_scores[(record.user_id, record.car_id)] = 0.0
                continue

            base_weight = self.INTERACTION_WEIGHTS.get(record.interaction_type, 1.0)
            if base_weight <= 0.0:
                continue

            # Compute time decay: weight * (0.5)^(days / half_life)
            try:
                t = datetime.fromisoformat(record.timestamp)
                days_diff = max(0.0, (now - t).total_seconds() / 86400.0)
                decay = np.power(0.5, days_diff / self.TIME_DECAY_HALF_LIFE_DAYS)
            except Exception:
                decay = 1.0

            key = (record.user_id, record.car_id)
            effective_weight = base_weight * decay

            # Cumulative scoring with diminishing returns for repeated views
            current_score = user_car_scores.get(key, 0.0)
            if record.interaction_type == "view" and current_score > 0:
                user_car_scores[key] = current_score + (effective_weight * 0.25)
            else:
                user_car_scores[key] = current_score + effective_weight

        # Build DataFrame matrix: Rows = Users, Columns = Car IDs
        rows = []
        for (u_id, c_id), score in user_car_scores.items():
            rows.append({"user_id": u_id, "car_id": c_id, "score": score})

        df_interactions = pd.DataFrame(rows)
        if df_interactions.empty:
            self.is_available = False
            return self

        pivot = df_interactions.pivot_table(
            index="user_id", columns="car_id", values="score", fill_value=0.0
        )
        self.user_item_matrix = pivot

        # Compute Item-Item cosine similarity matrix (Columns are items)
        item_vectors = pivot.values.T  # Shape: (N_cars, N_users)
        if item_vectors.shape[0] > 1:
            norms = np.linalg.norm(item_vectors, axis=1, keepdims=True)
            norms[norms == 0.0] = 1.0
            normed_vectors = item_vectors / norms
            item_sim = np.dot(normed_vectors, normed_vectors.T)
            item_sim = np.clip(item_sim, 0.0, 1.0)
            np.fill_diagonal(item_sim, 1.0)
            self.item_similarity_matrix = pd.DataFrame(
                item_sim, index=pivot.columns, columns=pivot.columns
            )
        else:
            self.item_similarity_matrix = pd.DataFrame([[1.0]], index=pivot.columns, columns=pivot.columns)

        # Optional Matrix Factorization (TruncatedSVD) if sufficient dimensions exist
        n_components = min(pivot.shape[0] - 1, pivot.shape[1] - 1, 10)
        if pivot.shape[0] >= 5 and pivot.shape[1] >= 5 and n_components >= 2:
            try:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    self.svd_model = TruncatedSVD(n_components=n_components, random_state=42)
                    self.item_factors = self.svd_model.fit_transform(item_vectors)
            except Exception:
                self.svd_model = None

        self.is_available = True
        return self

    def predict_user_scores(self, user_id: str, candidate_car_ids: List[int]) -> Dict[int, float]:
        """
        Generates collaborative filtering scores in [0.0, 1.0] for candidate cars.
        If collaborative model is not available or user is brand-new, returns empty dict.
        """
        if not self.is_available or self.user_item_matrix is None or self.item_similarity_matrix is None:
            return {}

        if user_id not in self.user_item_matrix.index:
            # User has no history yet in matrix (user cold-start)
            return {}

        user_vector = self.user_item_matrix.loc[user_id]
        rated_items = user_vector[user_vector > 0]
        if rated_items.empty:
            return {}

        scores: Dict[int, float] = {}
        for car_id in candidate_car_ids:
            if car_id in self.item_similarity_matrix.columns:
                sim_row = self.item_similarity_matrix.loc[car_id]
                # Compute weighted average similarity to cars user interacted with
                common_items = rated_items.index.intersection(sim_row.index)
                if not common_items.empty:
                    weights = rated_items[common_items]
                    sims = sim_row[common_items]
                    sum_sims = sims.sum()
                    if sum_sims > 0:
                        pred_score = float((weights * sims).sum() / sum_sims)
                        scores[car_id] = pred_score
                    else:
                        scores[car_id] = 0.0
                else:
                    scores[car_id] = 0.0
            else:
                scores[car_id] = 0.0

        # Normalize predicted scores to [0.0, 1.0]
        if scores:
            vals = list(scores.values())
            min_v, max_v = min(vals), max(vals)
            if max_v > min_v:
                return {cid: float((v - min_v) / (max_v - min_v)) for cid, v in scores.items()}
            elif max_v > 0:
                return {cid: 0.5 for cid in scores}
        return {cid: 0.0 for cid in candidate_car_ids}
