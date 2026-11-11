"""
init_db.py — Database initialization and car dataset ingestion script.
Loads dataset records into SQLite cars table with proper cleaning, feature flags, and indexing.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np

from .db import DatabaseManager, init_db, DB_PATH
from .models import CarRecord

DATASET_CANDIDATES = [
    Path(__file__).resolve().parents[2] / "data" / "indian_cars_dataset.csv",
    Path("/Users/kunaljaju/.cache/kagglehub/datasets/atharvanilawar/indian-cars-dataset/versions/6/indian_cars_dataset.csv"),
    Path(__file__).resolve().parents[2] / "data" / "cardekho_dataset.csv",
]


def _clean_transmission(val) -> str:
    if pd.isna(val):
        return "Manual"
    s = str(val).lower()
    return "Automatic" if any(k in s for k in ["automatic", "cvt", "dct", "amt", "sport mode"]) else "Manual"


def _clean_fuel(val) -> str:
    if pd.isna(val):
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


def _format_price_display(price_lakh: Optional[float]) -> str:
    if price_lakh is None or pd.isna(price_lakh) or price_lakh <= 0:
        return "Price on request"
    if price_lakh >= 100.0:
        cr = price_lakh / 100.0
        return f"Rs. {cr:.2f} Cr" if cr % 1 != 0 else f"Rs. {int(cr)} Cr"
    return f"Rs. {price_lakh:.2f} L"


def find_dataset_path() -> Path:
    for p in DATASET_CANDIDATES:
        if p.exists() and p.stat().st_size > 0:
            return p
    raise FileNotFoundError("Could not find a valid car dataset in candidate paths.")


def populate_cars_table(db_path: Optional[Path] = None, force_reload: bool = False) -> int:
    """Populates the SQLite cars table from dataset CSV if not already populated."""
    db = DatabaseManager(db_path)
    count = db.count_cars()
    if count > 0 and not force_reload:
        return count

    dataset_path = find_dataset_path()
    df = pd.read_csv(dataset_path)

    # Standardize Brand names
    brand_map = {
        "Mg": "MG",
        "Byd": "BYD",
        "Bmw": "BMW",
        "Tata": "Tata Motors",
    }
    if "Brand" in df.columns:
        df["Brand"] = df["Brand"].replace(brand_map)
        brand_col = "Brand"
        car_col = "Car"
        variant_col = "Variant" if "Variant" in df.columns else "Car"
        price_col = "Price"
        mileage_col = "Mileage"
        fuel_col = "Fuel Type"
        trans_col = "Transmission"
        disp_col = "Displacement (cc)"
    else:
        # Fallback cardekho dataset
        brand_col = "brand"
        car_col = "model"
        variant_col = "car_name"
        price_col = "selling_price"
        mileage_col = "mileage"
        fuel_col = "fuel_type"
        trans_col = "transmission_type"
        disp_col = "engine"

    feature_cols = [
        "Sunroof / Moonroof",
        "Cruise Control",
        "Ventilated Seats",
        "Wireless Charger",
        "Rear AC",
        "Heads Up Display",
        "Central Locking",
        "Hill Hold Control",
    ]

    inserted = 0
    for _, row in df.iterrows():
        b = str(row.get(brand_col, "Unknown")).strip()
        c = str(row.get(car_col, "Unknown")).strip()
        v = str(row.get(variant_col, "Standard")).strip() if variant_col in row else "Standard"

        # Price
        raw_price = row.get(price_col)
        price_val = pd.to_numeric(raw_price, errors="coerce")
        if pd.notna(price_val) and price_val > 0:
            # If price > 100_000, it's in INR, convert to Lakhs
            price_lakh = round(float(price_val) / 100_000.0, 2) if price_val > 1000 else round(float(price_val), 2)
        else:
            price_lakh = None

        price_disp = _format_price_display(price_lakh)

        # Specs
        raw_m = row.get(mileage_col)
        mileage_val = pd.to_numeric(raw_m, errors="coerce")
        mileage = round(float(mileage_val), 2) if pd.notna(mileage_val) else None

        raw_d = row.get(disp_col)
        disp_val = pd.to_numeric(raw_d, errors="coerce")
        displacement = round(float(disp_val), 1) if pd.notna(disp_val) else None

        fuel_type = _clean_fuel(row.get(fuel_col))
        transmission = _clean_transmission(row.get(trans_col))

        # Features
        features_dict = {}
        feat_score = 0.0
        for fcol in feature_cols:
            key = fcol.lower().replace(" ", "_").replace("/", "_")
            if fcol in row:
                val_str = str(row[fcol]).lower().strip()
                has_feat = 1 if "yes" in val_str or val_str == "1" or val_str == "true" else 0
            else:
                has_feat = 0
            features_dict[key] = has_feat
            feat_score += float(has_feat)

        # Active status
        disc = str(row.get("Discontinued", "no")).lower().strip()
        active_status = "Discontinued" if "yes" in disc else "Active"

        desc = str(row.get("Description", ""))
        if not desc or desc == "nan":
            desc = f"{b} {c} ({v}) with {fuel_type} engine and {transmission} gearbox."

        car_rec = CarRecord(
            id=0,
            brand=b,
            car=c,
            variant=v,
            price_lakh=price_lakh,
            price_display=price_disp,
            fuel_type=fuel_type,
            transmission=transmission,
            mileage=mileage,
            displacement=displacement,
            feature_score=feat_score,
            active_status=active_status,
            description=desc,
            features=features_dict,
        )
        db.insert_car(car_rec)
        inserted += 1

    return inserted


if __name__ == "__main__":
    count = populate_cars_table(force_reload=True)
    print(f"Successfully populated {count} cars into SQLite database at {DB_PATH}")

