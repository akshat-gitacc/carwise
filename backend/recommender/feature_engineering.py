"""
feature_engineering.py — Vector space feature engineering for CarWise Recommender.
Transforms raw specifications into unified, normalized, and weighted vector representations.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


class CarFeatureEngineer:
    """Creates a shared vector space between car specifications and user preference profiles."""

    def __init__(self):
        self.num_scaler = MinMaxScaler(feature_range=(0.0, 1.0))
        self.cat_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.is_fitted = False

        self.num_cols: List[str] = ["price_lakh", "mileage_imputed", "displacement_imputed", "feature_score"]
        self.cat_cols: List[str] = ["fuel_type_clean", "transmission_clean"]
        self.feat_binary_cols: List[str] = []

        self.feature_names: List[str] = []
        self.default_weights: np.ndarray = np.array([])

    def fit(self, df: pd.DataFrame) -> "CarFeatureEngineer":
        """Fit scalers and encoders on car dataset DataFrame."""
        if df.empty:
            raise ValueError("Cannot fit feature engineer on empty DataFrame.")

        # Identify binary feature columns (e.g. feat_sunroof___moonroof)
        self.feat_binary_cols = sorted([c for c in df.columns if c.startswith("feat_")])

        # Prepare numerical data with fallback median for price
        df_num = df[self.num_cols].copy()
        for col in self.num_cols:
            med_val = df_num[col].median() if df_num[col].dropna().shape[0] > 0 else 0.0
            df_num[col] = df_num[col].fillna(med_val)

        self.num_scaler.fit(df_num)

        # Prepare categorical data
        df_cat = df[self.cat_cols].fillna("Unknown")
        self.cat_encoder.fit(df_cat)

        # Build feature names
        cat_feature_names = self.cat_encoder.get_feature_names_out(self.cat_cols).tolist()
        self.feature_names = self.num_cols + cat_feature_names + self.feat_binary_cols

        # Default feature weights for balanced vector distance
        # Gives balanced importance to price, mileage, engine, transmission, fuel, and tech features
        weights = []
        for name in self.feature_names:
            if name == "price_lakh":
                weights.append(1.5)
            elif name == "mileage_imputed":
                weights.append(1.3)
            elif name == "displacement_imputed":
                weights.append(1.0)
            elif name == "feature_score":
                weights.append(1.1)
            elif name.startswith("fuel_type_clean"):
                weights.append(1.2)
            elif name.startswith("transmission_clean"):
                weights.append(1.2)
            elif name.startswith("feat_"):
                weights.append(0.8)
            else:
                weights.append(1.0)

        self.default_weights = np.array(weights, dtype=np.float32)
        self.is_fitted = True
        return self

    def transform_cars(self, df: pd.DataFrame, custom_weights: Optional[Dict[str, float]] = None) -> np.ndarray:
        """Transforms DataFrame of cars into a weighted 2D matrix of shape (N, D)."""
        if not self.is_fitted:
            self.fit(df)

        # 1. Numerical
        df_num = df[self.num_cols].copy()
        for col in self.num_cols:
            med = df_num[col].median() if df_num[col].dropna().shape[0] > 0 else 0.0
            df_num[col] = df_num[col].fillna(med)
        num_vecs = self.num_scaler.transform(df_num)

        # 2. Categorical
        df_cat = df[self.cat_cols].fillna("Unknown")
        cat_vecs = self.cat_encoder.transform(df_cat)

        # 3. Binary feature columns
        if self.feat_binary_cols:
            for c in self.feat_binary_cols:
                if c not in df.columns:
                    df[c] = 0.0
            bin_vecs = df[self.feat_binary_cols].fillna(0.0).values.astype(np.float32)
            full_matrix = np.hstack([num_vecs, cat_vecs, bin_vecs])
        else:
            full_matrix = np.hstack([num_vecs, cat_vecs])

        # Apply weights
        weights = self._build_weight_vector(custom_weights)
        return full_matrix * weights

    def transform_user_profile(
        self, profile: Dict[str, Any], custom_weights: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        """
        Transforms a user profile dictionary into the exact same vector space.
        profile can contain:
          - 'budget_target' or 'budget_max': float
          - 'preferred_mileage': float
          - 'preferred_displacement': float
          - 'fuel_type': str
          - 'transmission': str
          - 'features': list of str or dict
        """
        if not self.is_fitted:
            raise RuntimeError("CarFeatureEngineer must be fitted on cars before transforming a user profile.")

        # Numerical defaults from scaler centers
        scaler_mins = self.num_scaler.data_min_
        scaler_maxs = self.num_scaler.data_max_

        target_price = profile.get("budget_target") or profile.get("budget_max")
        if target_price is None:
            # mid-point default
            target_price = (scaler_mins[0] + scaler_maxs[0]) / 2.0

        target_mileage = profile.get("preferred_mileage") or profile.get("min_mileage")
        if target_mileage is None:
            target_mileage = (scaler_mins[1] + scaler_maxs[1]) / 2.0

        target_disp = profile.get("preferred_displacement")
        if target_disp is None:
            target_disp = (scaler_mins[2] + scaler_maxs[2]) / 2.0

        # Feature score target
        user_features = profile.get("features", {})
        if isinstance(user_features, list):
            user_features = {f: 1.0 for f in user_features}
        target_feat_score = float(sum(1 for v in user_features.values() if v)) if user_features else 3.0

        num_input = pd.DataFrame([{
            "price_lakh": float(target_price),
            "mileage_imputed": float(target_mileage),
            "displacement_imputed": float(target_disp),
            "feature_score": target_feat_score,
        }])
        num_vec = self.num_scaler.transform(num_input)

        # Categorical encoding with neutral representation for unspecified categories
        fuel = profile.get("fuel_type") or profile.get("preferred_fuel")
        if fuel in ("Any", "", None):
            fuel_spec = None
        else:
            fuel_spec = fuel

        trans = profile.get("transmission") or profile.get("preferred_transmission")
        if trans in ("Any", "", None):
            trans_spec = None
        else:
            trans_spec = trans

        # Start with standard encoding using default placeholders
        cat_input = pd.DataFrame([{
            "fuel_type_clean": fuel_spec or "Petrol",
            "transmission_clean": trans_spec or "Manual",
        }])
        cat_vec = self.cat_encoder.transform(cat_input).astype(np.float32)

        # If fuel was unspecified, set all fuel one-hot columns to equal neutral affinity (1/N)
        cat_names = self.cat_encoder.get_feature_names_out(self.cat_cols)
        fuel_indices = [i for i, name in enumerate(cat_names) if name.startswith("fuel_type_clean")]
        if fuel_spec is None and fuel_indices:
            cat_vec[0, fuel_indices] = 1.0 / len(fuel_indices)

        # If transmission was unspecified, set all transmission columns to equal neutral affinity
        trans_indices = [i for i, name in enumerate(cat_names) if name.startswith("transmission_clean")]
        if trans_spec is None and trans_indices:
            cat_vec[0, trans_indices] = 1.0 / len(trans_indices)

        # Binary feature columns
        bin_values = []
        for col in self.feat_binary_cols:
            clean_k = col.replace("feat_", "")
            val = 1.0 if user_features.get(clean_k) or user_features.get(col) else 0.0
            bin_values.append(val)
        bin_vec = np.array([bin_values], dtype=np.float32) if bin_values else np.empty((1, 0))

        full_vec = np.hstack([num_vec, cat_vec, bin_vec]) if bin_values else np.hstack([num_vec, cat_vec])

        # Apply weights
        weights = self._build_weight_vector(custom_weights)
        return full_vec * weights

    def _build_weight_vector(self, custom_weights: Optional[Dict[str, float]] = None) -> np.ndarray:
        if not custom_weights:
            return self.default_weights.copy()

        w = self.default_weights.copy()
        for i, name in enumerate(self.feature_names):
            if name in custom_weights:
                w[i] = custom_weights[name]
            elif "mileage" in name and "mileage" in custom_weights:
                w[i] = custom_weights["mileage"]
            elif "price" in name and "price" in custom_weights:
                w[i] = custom_weights["price"]
            elif "displacement" in name and "engine" in custom_weights:
                w[i] = custom_weights["engine"]
            elif "feature" in name and "features" in custom_weights:
                w[i] = custom_weights["features"]
        return w

