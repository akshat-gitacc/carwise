"""
knowledge_based.py — Knowledge-Based Filtering and Constraint Evaluation.
Enforces hard mandatory constraints (budget, fuel, transmission, must-have features)
and calculates soft preference compatibility scores based on automotive domain logic.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


class KnowledgeBasedFilter:
    """
    Applies explicit domain rules and constraints collected from user profiles.
    Separates hard pruning from soft preference scoring.
    """

    def apply_hard_constraints(self, df: pd.DataFrame, profile: Dict[str, Any]) -> pd.DataFrame:
        """
        Filters candidates based on mandatory requirements.
        Candidates violating hard constraints are eliminated.
        """
        if df.empty:
            return df

        candidates = df.copy()

        # Exclude vehicles with no price if budget is constrained
        b_min = profile.get("budget_min", 3.0)
        b_max = profile.get("budget_max", 250.0)
        strictness = profile.get("strictness", "strict")

        if "price_lakh" in candidates.columns:
            candidates = candidates[candidates["price_lakh"].notna()]
            if strictness == "flexible":
                # Allow 15% budget margin
                effective_min = max(1.0, b_min * 0.85)
                effective_max = b_max * 1.15 if b_max < 250.0 else 9999.0
            else:
                effective_min = b_min
                effective_max = b_max if b_max < 250.0 else 9999.0

            candidates = candidates[
                (candidates["price_lakh"] >= effective_min) & (candidates["price_lakh"] <= effective_max)
            ]

        # Fuel type constraint
        fuel = profile.get("fuel_type")
        if fuel and fuel != "Any" and "fuel_type_clean" in candidates.columns:
            candidates = candidates[candidates["fuel_type_clean"].str.lower() == fuel.lower()]

        # Transmission constraint
        trans = profile.get("transmission")
        if trans and trans != "Any" and "transmission_clean" in candidates.columns:
            candidates = candidates[candidates["transmission_clean"].str.lower() == trans.lower()]

        # Brand constraint
        brand = profile.get("brand")
        if brand and brand != "Any" and "brand" in candidates.columns:
            b_norm = brand.lower().replace("-", " ").strip()
            cand_brands = candidates["brand"].str.lower().str.replace("-", " ")
            candidates = candidates[
                (cand_brands == b_norm)
                | (cand_brands.str.startswith(b_norm))
                | (cand_brands.str.contains(b_norm, regex=False))
            ]

        # Minimum mileage constraint
        min_m = profile.get("min_mileage", 0.0)
        if min_m > 0 and "mileage" in candidates.columns:
            # Keep cars meeting min_mileage or those with electric powertrains where kmpl does not apply
            is_electric = candidates["fuel_type_clean"].str.lower() == "electric"
            mileage_col = candidates["mileage_imputed"] if "mileage_imputed" in candidates.columns else candidates["mileage"].fillna(0.0)
            meets_m = mileage_col >= min_m
            candidates = candidates[meets_m | is_electric]

        # Must-have features constraint
        must_have_features = profile.get("must_have_features", [])
        if must_have_features:
            for feat in must_have_features:
                clean_feat_col = "feat_" + feat.lower().replace(" / ", "___").replace(" ", "_").replace("/", "_")
                if clean_feat_col in candidates.columns:
                    candidates = candidates[candidates[clean_feat_col] == 1.0]

        # Prefer active vehicles unless very few match
        if "active_status" in candidates.columns:
            active_subset = candidates[candidates["active_status"] == "Active"]
            if len(active_subset) >= 10:
                candidates = active_subset

        # Graceful fallback: if constraints eliminated all cars, relax budget gracefully
        if candidates.empty:
            fallback = df[df["price_lakh"].notna()].copy()
            # Try relaxing budget bounds by 25%
            rel_min = max(1.0, b_min * 0.75)
            rel_max = b_max * 1.25 if b_max < 250.0 else 9999.0
            fallback_budget = fallback[(fallback["price_lakh"] >= rel_min) & (fallback["price_lakh"] <= rel_max)]
            if not fallback_budget.empty:
                candidates = fallback_budget
            else:
                candidates = fallback

        return candidates

    def compute_compatibility_scores(self, df: pd.DataFrame, profile: Dict[str, Any]) -> pd.Series:
        """
        Calculates soft knowledge-based compatibility score in [0.0, 1.0].
        Considers distance to target budget, mileage bonus, engine suitability, and tech features.
        """
        if df.empty:
            return pd.Series(dtype=float)

        scores = pd.Series(0.0, index=df.index)
        priority = profile.get("priority", "balanced")

        # Configurable component weights depending on user priority
        weights = {
            "balanced":     {"price": 0.25, "mileage": 0.30, "features": 0.25, "engine": 0.20},
            "mileage":      {"price": 0.15, "mileage": 0.50, "features": 0.20, "engine": 0.15},
            "price":        {"price": 0.50, "mileage": 0.20, "features": 0.15, "engine": 0.15},
            "feature_rich": {"price": 0.15, "mileage": 0.15, "features": 0.50, "engine": 0.20},
            "performance":  {"price": 0.15, "mileage": 0.10, "features": 0.25, "engine": 0.50},
        }.get(priority, {"price": 0.25, "mileage": 0.30, "features": 0.25, "engine": 0.20})

        # 1. Price compatibility (Centered around user's target price / budget window)
        if "price_lakh" in df.columns:
            prices = df["price_lakh"].fillna(df["price_lakh"].median())
            b_min = float(profile.get("budget_min", 3.0))
            b_max = float(profile.get("budget_max", 250.0))
            target_price = profile.get("budget_target")
            if target_price is None:
                target_price = (b_min + b_max) / 2.0 if b_max < 200.0 else b_min * 1.5

            target_p = float(target_price)
            tolerance = max(4.0, (b_max - b_min) / 2.0) if b_max < 250.0 else max(10.0, target_p * 0.4)
            price_diff = (prices - target_p).abs()
            price_score = (1.0 - (price_diff / tolerance)).clip(0.0, 1.0)
        else:
            price_score = pd.Series(0.5, index=df.index)

        # 2. Mileage score (Higher is better, normalized)
        if "mileage_imputed" in df.columns:
            mileage = df["mileage_imputed"]
            m_min, m_max = mileage.min(), mileage.max()
            if m_max > m_min:
                mileage_score = (mileage - m_min) / (m_max - m_min)
            else:
                mileage_score = pd.Series(0.5, index=df.index)
        else:
            mileage_score = pd.Series(0.5, index=df.index)

        # 3. Feature compatibility
        desired_features = profile.get("features", {})
        if isinstance(desired_features, list):
            desired_features = {f.lower().replace(" / ", "___").replace(" ", "_"): 1 for f in desired_features}

        if desired_features and isinstance(desired_features, dict):
            feat_matches = pd.Series(0.0, index=df.index)
            num_desired = max(1, sum(1 for v in desired_features.values() if v))
            for f_key, val in desired_features.items():
                if val:
                    col = f"feat_{f_key}" if not f_key.startswith("feat_") else f_key
                    if col in df.columns:
                        feat_matches += df[col].fillna(0.0)
            feature_score = (feat_matches / num_desired).clip(0.0, 1.0)
        elif "feature_score" in df.columns:
            f_vals = df["feature_score"]
            f_max = f_vals.max() if f_vals.max() > 0 else 6.0
            feature_score = (f_vals / f_max).clip(0.0, 1.0)
        else:
            feature_score = pd.Series(0.5, index=df.index)

        # 4. Engine displacement score
        if "displacement_imputed" in df.columns:
            disp = df["displacement_imputed"]
            pref_disp = profile.get("preferred_displacement", 1498.0)
            # Distance from preferred displacement
            disp_diff = (disp - pref_disp).abs()
            max_diff = max(1.0, disp_diff.max())
            engine_score = 1.0 - (disp_diff / max_diff)
        else:
            engine_score = pd.Series(0.5, index=df.index)

        # 5. Preferred Brand Affinity Bonus (soft boost for user's favorite brand)
        pref_brand = profile.get("preferred_brand")
        if pref_brand and "brand" in df.columns:
            b_norm = pref_brand.lower().replace("-", " ").strip()
            cand_b = df["brand"].str.lower().str.replace("-", " ")
            brand_bonus = ((cand_b == b_norm) | (cand_b.str.startswith(b_norm))).astype(float) * 0.15
        else:
            brand_bonus = pd.Series(0.0, index=df.index)

        # 6. Preferred Fuel Affinity Bonus (soft boost for user's dominant fuel preference)
        pref_fuel = profile.get("preferred_fuel")
        if pref_fuel and "fuel_type_clean" in df.columns:
            fuel_bonus = (df["fuel_type_clean"].str.lower() == pref_fuel.lower()).astype(float) * 0.10
        else:
            fuel_bonus = pd.Series(0.0, index=df.index)

        # 7. Preferred Transmission Affinity Bonus
        pref_trans = profile.get("preferred_transmission")
        if pref_trans and "transmission_clean" in df.columns:
            trans_bonus = (df["transmission_clean"].str.lower() == pref_trans.lower()).astype(float) * 0.08
        else:
            trans_bonus = pd.Series(0.0, index=df.index)

        # Combine weighted sub-scores + affinity bonuses
        combined = (
            weights["price"] * price_score
            + weights["mileage"] * mileage_score
            + weights["features"] * feature_score
            + weights["engine"] * engine_score
            + brand_bonus
            + fuel_bonus
            + trans_bonus
        )

        return combined.clip(0.0, 1.0)

