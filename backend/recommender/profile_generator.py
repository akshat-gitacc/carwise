"""
profile_generator.py — Dynamic Preference Generation and Schema Inspection.
Inspects database attributes to dynamically generate preference questions, options,
and valid profile schemas without hardcoding or inventing nonexistent fields.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set
import numpy as np
import pandas as pd

from ..database import DatabaseManager
from .preprocessing import CarPreprocessor


class SchemaInspector:
    """Analyzes the car dataset schema to classify candidate preference attributes."""

    EXCLUDED_PATTERNS = {
        "id", "created_at", "description", "features_json", "price_display",
        "combined_description", "cluster_type", "cluster_size", "tachometer",
        "valve_train", "valves_per_cylinder", "cylinder_layout", "cylinders",
        "instrument_cluster", "display", "unnamed"
    }

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.available_columns = set(df.columns)

    def is_attribute_available(self, col_name: str, min_non_null_pct: float = 0.1) -> bool:
        """Checks if a column exists and has sufficient non-null data."""
        if col_name not in self.df.columns:
            return False
        non_null_count = self.df[col_name].dropna().shape[0]
        return (non_null_count / max(1, len(self.df))) >= min_non_null_pct

    def get_numerical_preference_attributes(self) -> List[Dict[str, Any]]:
        """Identifies available numerical preference attributes with their ranges."""
        candidates = []

        # 1. Price / Budget
        if self.is_attribute_available("price_lakh"):
            s = self.df["price_lakh"].dropna()
            candidates.append({
                "attribute": "budget",
                "column": "price_lakh",
                "label": "Budget Range (in Lakhs INR)",
                "min": float(np.floor(s.quantile(0.01))) if not s.empty else 2.0,
                "max": float(np.ceil(s.quantile(0.99))) if not s.empty else 250.0,
                "median": float(s.median()) if not s.empty else 12.0,
                "unit": "Lakh",
            })

        # 2. Mileage / Fuel Efficiency
        if self.is_attribute_available("mileage"):
            s = self.df["mileage"].dropna()
            candidates.append({
                "attribute": "mileage",
                "column": "mileage",
                "label": "Fuel Efficiency (kmpl)",
                "min": float(np.floor(s.min())) if not s.empty else 8.0,
                "max": float(np.ceil(s.max())) if not s.empty else 35.0,
                "median": float(s.median()) if not s.empty else 18.0,
                "unit": "kmpl",
            })

        # 3. Displacement / Engine Size
        if self.is_attribute_available("displacement"):
            s = self.df["displacement"].dropna()
            candidates.append({
                "attribute": "displacement",
                "column": "displacement",
                "label": "Engine Displacement (cc)",
                "min": float(np.floor(s.min())) if not s.empty else 796.0,
                "max": float(np.ceil(s.max())) if not s.empty else 5000.0,
                "median": float(s.median()) if not s.empty else 1498.0,
                "unit": "cc",
            })

        return candidates

    def get_categorical_preference_attributes(self) -> List[Dict[str, Any]]:
        """Identifies available categorical preference attributes with selectable options."""
        candidates = []

        # 1. Fuel Type
        fuel_col = "fuel_type_clean" if "fuel_type_clean" in self.df.columns else "fuel_type"
        if self.is_attribute_available(fuel_col):
            unique_fuels = sorted([str(f) for f in self.df[fuel_col].dropna().unique() if str(f) != "nan"])
            candidates.append({
                "attribute": "fuel_type",
                "column": fuel_col,
                "label": "Fuel Type Preference",
                "options": ["Any"] + unique_fuels,
            })

        # 2. Transmission
        trans_col = "transmission_clean" if "transmission_clean" in self.df.columns else "transmission"
        if self.is_attribute_available(trans_col):
            unique_trans = sorted([str(t) for t in self.df[trans_col].dropna().unique() if str(t) != "nan"])
            candidates.append({
                "attribute": "transmission",
                "column": trans_col,
                "label": "Transmission Preference",
                "options": ["Any"] + unique_trans,
            })

        # 3. Brand
        if self.is_attribute_available("brand"):
            unique_brands = sorted([str(b) for b in self.df["brand"].dropna().unique() if str(b) != "nan"])
            candidates.append({
                "attribute": "brand",
                "column": "brand",
                "label": "Preferred Manufacturer",
                "options": ["Any"] + unique_brands,
            })

        return candidates

    def get_feature_attributes(self) -> List[Dict[str, str]]:
        """Identifies available equipment/feature flags present in the database."""
        feature_cols = [c for c in self.df.columns if c.startswith("feat_")]
        results = []
        for col in sorted(feature_cols):
            clean_name = col.replace("feat_", "").replace("___", " / ").replace("_", " ").title()
            results.append({
                "key": col.replace("feat_", ""),
                "column": col,
                "label": clean_name,
            })
        return results


class DynamicProfileGenerator:
    """Generates preference questions and builds validated user profiles from actual database attributes."""

    def __init__(self, db: Optional[DatabaseManager] = None):
        self.db = db or DatabaseManager()
        self.preprocessor = CarPreprocessor(self.db)
        self.df = self.preprocessor.get_cleaned_dataframe()
        self.inspector = SchemaInspector(self.df)

    def generate_advisor_questions(self) -> List[Dict[str, Any]]:
        """
        Dynamically generates ordered questions for the Car Advisor based on verified available attributes.
        Omits any attribute not present in the database.
        """
        questions = []
        step = 1

        # Question 1: Budget Range (if price is available)
        num_attrs = {a["attribute"]: a for a in self.inspector.get_numerical_preference_attributes()}
        if "budget" in num_attrs:
            b_info = num_attrs["budget"]
            questions.append({
                "step": step,
                "id": "budget",
                "type": "range_slider",
                "title": "What is your budget range?",
                "description": "Select the minimum and maximum price you are comfortable spending.",
                "min": 3.0,
                "max": 250.0,
                "default": [5.0, 25.0],
                "step_size": 1.0,
                "unit": "Lakh",
                "is_optional": False,
            })
            step += 1

        # Question 2: Transmission (if transmission is available)
        cat_attrs = {a["attribute"]: a for a in self.inspector.get_categorical_preference_attributes()}
        if "transmission" in cat_attrs:
            t_info = cat_attrs["transmission"]
            questions.append({
                "step": step,
                "id": "transmission",
                "type": "select",
                "title": "Which transmission do you prefer?",
                "description": "Choose between Manual for full control or Automatic for effortless city driving.",
                "options": t_info["options"],
                "default": "Any",
                "is_optional": True,
            })
            step += 1

        # Question 3: Fuel Type (if fuel is available)
        if "fuel_type" in cat_attrs:
            f_info = cat_attrs["fuel_type"]
            questions.append({
                "step": step,
                "id": "fuel_type",
                "type": "select",
                "title": "What is your preferred powertrain or fuel type?",
                "description": "Select Petrol, Diesel, CNG, Electric, or Hybrid based on your running requirements.",
                "options": f_info["options"],
                "default": "Any",
                "is_optional": True,
            })
            step += 1

        # Question 4: Minimum Mileage (if mileage is available)
        if "mileage" in num_attrs:
            questions.append({
                "step": step,
                "id": "min_mileage",
                "type": "slider",
                "title": "Do you have a minimum fuel economy requirement?",
                "description": "Cars below this threshold will be filtered or deprioritized.",
                "min": 0.0,
                "max": 30.0,
                "default": 0.0,
                "unit": "kmpl",
                "is_optional": True,
            })
            step += 1

        # Question 5: Manufacturer Preference (if brand is available)
        if "brand" in cat_attrs:
            b_info = cat_attrs["brand"]
            questions.append({
                "step": step,
                "id": "brand",
                "type": "select",
                "title": "Do you have a preferred car manufacturer?",
                "description": "Filter for a specific brand or keep as 'Any' to explore all options.",
                "options": b_info["options"],
                "default": "Any",
                "is_optional": True,
            })
            step += 1

        # Question 6: Engine Displacement (if displacement is available)
        if "displacement" in num_attrs:
            questions.append({
                "step": step,
                "id": "displacement_preference",
                "type": "choice",
                "title": "What engine performance class do you prefer?",
                "description": "Choose the engine displacement range that matches your driving style.",
                "options": [
                    "Any",
                    "Compact City Driving (< 1200 cc)",
                    "Balanced Everyday (1200 - 1800 cc)",
                    "High Performance / Highway (1800+ cc)",
                ],
                "default": "Any",
                "is_optional": True,
            })
            step += 1

        # Question 7: Must-Have / Preferred Equipment Features (if feature flags exist)
        features = self.inspector.get_feature_attributes()
        if features:
            questions.append({
                "step": step,
                "id": "features",
                "type": "multiselect",
                "title": "Which premium equipment features are important to you?",
                "description": "Select tech and comfort amenities you desire in your next vehicle.",
                "options": [f["label"] for f in features],
                "feature_keys": {f["label"]: f["key"] for f in features},
                "default": [],
                "is_optional": True,
            })
            step += 1

        # Question 8: Optimization Priority
        questions.append({
            "step": step,
            "id": "priority",
            "type": "radio",
            "title": "What matters most for your final ranking?",
            "description": "Select your core priority to calibrate the hybrid recommendation scoring algorithm.",
            "options": [
                "Balanced (Equal weighting)",
                "Fuel Economy (Best kmpl first)",
                "Budget Affordability (Best value for money)",
                "Feature Rich (Maximum tech & luxury)",
                "Engine Performance (Displacement & power)",
            ],
            "priority_map": {
                "Balanced (Equal weighting)": "balanced",
                "Fuel Economy (Best kmpl first)": "mileage",
                "Budget Affordability (Best value for money)": "price",
                "Feature Rich (Maximum tech & luxury)": "feature_rich",
                "Engine Performance (Displacement & power)": "performance",
            },
            "default": "Balanced (Equal weighting)",
            "is_optional": False,
        })
        step += 1

        # Question 9: Mandatory Constraint Strictness
        questions.append({
            "step": step,
            "id": "strictness",
            "type": "radio",
            "title": "How strictly should the budget constraints be enforced?",
            "description": "Strict adheres strictly to your budget range; Flexible permits vehicles up to 15% outside.",
            "options": [
                "Strict (Hard filter strictly within budget)",
                "Flexible (Allow up to 15% budget stretch for exceptional matches)",
            ],
            "strictness_map": {
                "Strict (Hard filter strictly within budget)": "strict",
                "Flexible (Allow up to 15% budget stretch for exceptional matches)": "flexible",
            },
            "default": "Strict (Hard filter strictly within budget)",
            "is_optional": False,
        })

        return questions

    def validate_and_build_profile(self, answers: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validates raw question responses, handles skipped fields, and formats
        a standardized profile for the Hybrid Recommendation Engine.
        """
        profile: Dict[str, Any] = {}

        # Budget
        if "budget" in answers and isinstance(answers["budget"], (list, tuple)) and len(answers["budget"]) == 2:
            profile["budget_min"] = float(answers["budget"][0])
            profile["budget_max"] = float(answers["budget"][1])
        elif "budget_min" in answers and "budget_max" in answers and answers["budget_min"] is not None and answers["budget_max"] is not None:
            profile["budget_min"] = float(answers["budget_min"])
            profile["budget_max"] = float(answers["budget_max"])
        elif "budget" in answers and isinstance(answers["budget"], (int, float)):
            profile["budget_min"] = 3.0
            profile["budget_max"] = float(answers["budget"])
        else:
            profile["budget_min"] = 3.0
            profile["budget_max"] = 250.0

        if profile["budget_max"] < 250.0:
            profile["budget_target"] = (profile["budget_min"] + profile["budget_max"]) / 2.0
        else:
            profile["budget_target"] = (profile["budget_min"] + min(profile["budget_max"], 50.0)) / 2.0

        # Transmission
        trans = answers.get("transmission", "Any")
        profile["transmission"] = trans if trans and trans != "Any" else None

        # Fuel
        fuel = answers.get("fuel_type", "Any")
        profile["fuel_type"] = fuel if fuel and fuel != "Any" else None

        # Mileage
        min_m = answers.get("min_mileage", 0.0)
        profile["min_mileage"] = float(min_m) if min_m else 0.0
        profile["preferred_mileage"] = max(profile["min_mileage"], 18.0)

        # Brand
        brand = answers.get("brand", "Any")
        profile["brand"] = brand if brand and brand != "Any" else None

        # Displacement
        disp_choice = answers.get("displacement_preference", "Any")
        if "< 1200" in disp_choice:
            profile["preferred_displacement"] = 1000.0
            profile["min_displacement"] = 700.0
            profile["max_displacement"] = 1200.0
        elif "1200 - 1800" in disp_choice:
            profile["preferred_displacement"] = 1498.0
            profile["min_displacement"] = 1200.0
            profile["max_displacement"] = 1800.0
        elif "1800+" in disp_choice:
            profile["preferred_displacement"] = 2200.0
            profile["min_displacement"] = 1800.0
            profile["max_displacement"] = 5000.0
        else:
            profile["preferred_displacement"] = 1498.0
            profile["min_displacement"] = None
            profile["max_displacement"] = None

        # Features
        raw_feats = answers.get("features", [])
        clean_feats: Dict[str, int] = {}
        if isinstance(raw_feats, list):
            for f in raw_feats:
                key = f.lower().replace(" / ", "___").replace(" ", "_")
                clean_feats[key] = 1
        elif isinstance(raw_feats, dict):
            clean_feats = {k: int(v) for k, v in raw_feats.items()}
        profile["features"] = clean_feats

        # Priority
        pri = answers.get("priority", "balanced")
        priority_map = {
            "Balanced (Equal weighting)": "balanced",
            "Fuel Economy (Best kmpl first)": "mileage",
            "Budget Affordability (Best value for money)": "price",
            "Feature Rich (Maximum tech & luxury)": "feature_rich",
            "Engine Performance (Displacement & power)": "performance",
        }
        profile["priority"] = priority_map.get(pri, pri)

        # Strictness
        strict = answers.get("strictness", "strict")
        strictness_map = {
            "Strict (Hard filter strictly within budget)": "strict",
            "Flexible (Allow up to 15% budget stretch for exceptional matches)": "flexible",
        }
        profile["strictness"] = strictness_map.get(strict, strict)

        return profile

