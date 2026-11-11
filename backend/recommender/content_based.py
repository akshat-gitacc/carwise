"""
content_based.py — Content-Based Recommendation Engine using Scikit-learn, Pandas, and NumPy.
Computes cosine similarity and K-Nearest Neighbors (KNN) in a shared weighted feature vector space.
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*encountered in matmul.*")

from typing import Any, Dict, List, Optional, Tuple
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

from .feature_engineering import CarFeatureEngineer
from .preprocessing import CarPreprocessor


class ContentBasedRecommender:
    """
    Identifies vehicles matching a user profile or seed vehicle using
    cosine similarity and K-Nearest Neighbors on normalized, weighted spec vectors.
    """

    def __init__(self, preprocessor: Optional[CarPreprocessor] = None, feature_engineer: Optional[CarFeatureEngineer] = None):
        self.preprocessor = preprocessor or CarPreprocessor()
        self.engineer = feature_engineer or CarFeatureEngineer()
        self.df: pd.DataFrame = pd.DataFrame()
        self.car_matrix: np.ndarray = np.empty((0, 0))
        self.knn_model: Optional[NearestNeighbors] = None
        self._fitted = False

    def fit(self, active_only: bool = False) -> "ContentBasedRecommender":
        """Loads cleaned car data, fits feature engineer, and trains KNN index."""
        self.df = self.preprocessor.get_cleaned_dataframe(active_only=active_only)
        if self.df.empty:
            raise ValueError("No car data found to fit ContentBasedRecommender.")

        # Fit feature engineer and transform cars to weighted feature matrix
        self.engineer.fit(self.df)
        self.car_matrix = self.engineer.transform_cars(self.df)

        # Train KNN index using cosine distance
        self.knn_model = NearestNeighbors(n_neighbors=min(50, len(self.df)), metric="cosine", algorithm="brute")
        self.knn_model.fit(self.car_matrix)

        self._fitted = True
        return self

    def compute_similarity_scores(
        self, profile: Dict[str, Any], custom_weights: Optional[Dict[str, float]] = None
    ) -> pd.Series:
        """
        Computes cosine similarity between user profile vector and all candidate cars.
        Returns a Pandas Series of scores in [0.0, 1.0], indexed by DataFrame row index.
        """
        if not self._fitted:
            self.fit()

        user_vec = self.engineer.transform_user_profile(profile, custom_weights=custom_weights)
        if np.linalg.norm(user_vec) == 0.0:
            return pd.Series(0.0, index=self.df.index, name="content_score")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            sim_matrix = cosine_similarity(user_vec, self.car_matrix)  # shape (1, N)
        scores = sim_matrix[0]

        # Clip and normalize safely to [0, 1]
        scores = np.clip(scores, 0.0, 1.0)
        return pd.Series(scores, index=self.df.index, name="content_score")

    def find_nearest_neighbors(
        self, profile: Dict[str, Any], n_neighbors: int = 10, custom_weights: Optional[Dict[str, float]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Finds K-Nearest Neighbors for the user profile using KNN model.
        Returns (distances, indices).
        """
        if not self._fitted:
            self.fit()

        user_vec = self.engineer.transform_user_profile(profile, custom_weights=custom_weights)
        n = min(n_neighbors, len(self.df))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            distances, indices = self.knn_model.kneighbors(user_vec, n_neighbors=n)
        return distances[0], indices[0]

    def find_similar_to_car(self, car_id: int, top_n: int = 6) -> List[Dict[str, Any]]:
        """
        Finds cars most similar in specifications to a given car ID.
        Used by the Explore page to recommend cars similar to user's liked or shortlisted vehicles.
        """
        if not self._fitted:
            self.fit()

        matches = self.df[self.df["id"] == car_id]
        if matches.empty:
            return []

        car_idx = matches.index[0]
        seed_vec = self.car_matrix[car_idx : car_idx + 1]

        if np.linalg.norm(seed_vec) == 0.0:
            sim_scores = np.zeros(len(self.df))
        else:
            sim_scores = cosine_similarity(seed_vec, self.car_matrix)[0]

        # Exclude the seed car itself
        df_copy = self.df.copy()
        df_copy["content_similarity"] = sim_scores
        df_copy = df_copy[df_copy["id"] != car_id]

        top_matches = df_copy.sort_values("content_similarity", ascending=False).head(top_n)
        return top_matches.to_dict(orient="records")

    def get_car_by_id(self, car_id: int) -> Optional[pd.Series]:
        if not self._fitted:
            self.fit()
        matches = self.df[self.df["id"] == car_id]
        return matches.iloc[0] if not matches.empty else None
