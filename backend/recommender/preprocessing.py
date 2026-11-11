"""
preprocessing.py — Data preprocessing and cleaning pipeline for CarWise Recommender.
Cleans raw attributes, handles missing values with median imputation, and standardizes categories.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from ..database import DatabaseManager, CarRecord


def clean_transmission(val: Any) -> str:
    if val is None or pd.isna(val):
        return "Manual"
    s = str(val).lower()
    if any(k in s for k in ["automatic", "cvt", "dct", "amt", "sport mode"]):
        return "Automatic"
    return "Manual"


def clean_fuel(val: Any) -> str:
    if val is None or pd.isna(val):
        return "Petrol"
    s = str(val).lower()
    if "electric" in s and "hybrid" not in s:
        return "Electric"
    if "hybrid" in s:
        return "Hybrid"
    if "cng" in s:
        return "CNG"
    if "diesel" in s:
        return "Diesel"
    if "lpg" in s:
        return "LPG"
    return "Petrol"


class CarPreprocessor:
    """Extracts, cleans, and prepares car records into structured tabular formats for ML models."""

    def __init__(self, db: Optional[DatabaseManager] = None):
        self.db = db or DatabaseManager()

    def get_cleaned_dataframe(self, active_only: bool = False) -> pd.DataFrame:
        """Fetches all cars from the database and returns a sanitized pandas DataFrame."""
        cars: List[CarRecord] = self.db.get_all_cars(active_only=active_only)
        if not cars:
            return pd.DataFrame()

        rows = []
        for c in cars:
            d = c.to_dict()
            # Flatten features dictionary into columns
            feats = d.pop("features", {}) or {}
            for fk, fv in feats.items():
                d[f"feat_{fk}"] = float(fv)
            rows.append(d)

        df = pd.DataFrame(rows)

        # Standardize and fill numeric columns
        if "price_lakh" in df.columns:
            df["price_lakh"] = pd.to_numeric(df["price_lakh"], errors="coerce")
        else:
            df["price_lakh"] = np.nan

        # Median imputation for specs
        med_mileage = df["mileage"].median() if df["mileage"].dropna().shape[0] > 0 else 18.0
        med_disp = df["displacement"].median() if df["displacement"].dropna().shape[0] > 0 else 1498.0
        df["mileage_imputed"] = df["mileage"].fillna(med_mileage)
        df["displacement_imputed"] = df["displacement"].fillna(med_disp)

        # Categorical cleaning
        df["fuel_type_clean"] = df["fuel_type"].apply(clean_fuel)
        df["transmission_clean"] = df["transmission"].apply(clean_transmission)

        return df

    def get_summary_statistics(self) -> Dict[str, Any]:
        df = self.get_cleaned_dataframe()
        valid_prices = df["price_lakh"].dropna()
        valid_mileage = df["mileage"].dropna()
        valid_disp = df["displacement"].dropna()

        return {
            "total_cars": len(df),
            "unique_models": df["car"].nunique() if "car" in df.columns else 0,
            "brands_count": df["brand"].nunique() if "brand" in df.columns else 0,
            "min_price_lakh": float(valid_prices.min()) if not valid_prices.empty else 2.0,
            "max_price_lakh": float(valid_prices.max()) if not valid_prices.empty else 250.0,
            "median_price_lakh": float(valid_prices.median()) if not valid_prices.empty else 12.0,
            "min_mileage": float(valid_mileage.min()) if not valid_mileage.empty else 10.0,
            "max_mileage": float(valid_mileage.max()) if not valid_mileage.empty else 35.0,
            "median_mileage": float(valid_mileage.median()) if not valid_mileage.empty else 18.0,
            "min_displacement": float(valid_disp.min()) if not valid_disp.empty else 796.0,
            "max_displacement": float(valid_disp.max()) if not valid_disp.empty else 5000.0,
            "median_displacement": float(valid_disp.median()) if not valid_disp.empty else 1498.0,
            "fuel_types": sorted(df["fuel_type_clean"].unique().tolist()),
            "transmissions": sorted(df["transmission_clean"].unique().tolist()),
        }

