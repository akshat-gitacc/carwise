"""
advisor.py — Car Advisor backend module for CarWise.
Uses the Indian cars dataset (2,633 variants, 741 unique cars across 40 brands)
for data-driven recommendations, comparisons, dealership locator,
and Gemini-powered conversational advisory.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Dataset loading & cleaning
# ---------------------------------------------------------------------------

_local_dataset = Path(__file__).resolve().parent.parent / "data" / "indian_cars_dataset.csv"
DATASET_PATH = _local_dataset if _local_dataset.exists() else Path(
    "/Users/kunaljaju/.cache/kagglehub/datasets/"
    "atharvanilawar/indian-cars-dataset/versions/6/indian_cars_dataset.csv"
)

_df_cache: Optional[pd.DataFrame] = None


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


def _format_price_display(price_lakh: float) -> str:
    if pd.isna(price_lakh) or price_lakh <= 0:
        return "Price on request"
    if price_lakh >= 100.0:
        cr = price_lakh / 100.0
        return f"Rs. {cr:.2f} Cr" if cr % 1 != 0 else f"Rs. {int(cr)} Cr"
    return f"Rs. {price_lakh:.2f} L"


def load_dataset() -> pd.DataFrame:
    global _df_cache
    if _df_cache is not None:
        return _df_cache

    df = pd.read_csv(DATASET_PATH)

    # Standardize Brand names
    brand_map = {
        "Mg": "MG",
        "Byd": "BYD",
        "Bmw": "BMW",
        "Tata": "Tata Motors",
    }
    df["Brand"] = df["Brand"].replace(brand_map)

    # Numeric clean columns
    df["Price_Clean"] = pd.to_numeric(df["Price"], errors="coerce")
    df["Price_Lakh"] = (df["Price_Clean"] / 100_000).round(2)
    df["Price_Display"] = df["Price_Lakh"].apply(_format_price_display)

    df["Mileage_Clean"] = pd.to_numeric(df["Mileage"], errors="coerce")
    df["Displacement_Clean"] = pd.to_numeric(df["Displacement (cc)"], errors="coerce")

    # Simplified categories
    df["Fuel_Type_Simple"] = df["Fuel Type"].apply(_clean_fuel)
    df["Transmission_Simple"] = df["Transmission"].apply(_clean_transmission)

    # Feature indicators (0 or 1)
    feature_cols = [
        "Sunroof / Moonroof",
        "Cruise Control",
        "Ventilated Seats",
        "Wireless Charger",
        "Rear AC",
        "Heads Up Display",
    ]
    df["Feature_Score"] = 0.0
    for col in feature_cols:
        clean_col = "feat_" + col.lower().replace(" ", "_").replace("/", "_")
        df[clean_col] = df[col].astype(str).str.lower().apply(lambda x: 1.0 if "yes" in x else 0.0)
        df["Feature_Score"] += df[clean_col]

    # Active vs discontinued score (active get 1.0, discontinued get 0.3)
    df["Active_Status"] = df["Discontinued"].astype(str).str.lower().apply(
        lambda x: "Discontinued" if "yes" in x else "Active"
    )
    df["Active_Score"] = df["Active_Status"].apply(lambda x: 1.0 if x == "Active" else 0.3)

    _df_cache = df
    return df


def get_unique_car_count() -> int:
    """Return total unique car models in India from the dataset."""
    df = load_dataset()
    return int(df["Car"].nunique())


def get_brands() -> List[str]:
    df = load_dataset()
    return sorted(df[df["Price_Lakh"].notnull()]["Brand"].unique().tolist())


def get_models_for_brand(brand: str) -> List[str]:
    df = load_dataset()
    b_lower = brand.lower().strip()
    match = (df["Brand"].str.lower() == b_lower) | (df["Brand"].str.lower().str.startswith(b_lower))
    return sorted(df[match & (df["Price_Lakh"].notnull())]["Car"].unique().tolist())


def get_fuel_types() -> List[str]:
    return ["Petrol", "Diesel", "CNG", "Electric", "Hybrid"]


def get_transmissions() -> List[str]:
    return ["Manual", "Automatic"]


# ---------------------------------------------------------------------------
# Scoring & Recommendation Engine
# ---------------------------------------------------------------------------

def _normalise_series(s: pd.Series, higher_is_better: bool = True) -> pd.Series:
    mn, mx = s.min(), s.max()
    if mx == mn or pd.isna(mn) or pd.isna(mx):
        return pd.Series(0.5, index=s.index)
    normed = (s - mn) / (mx - mn)
    return normed if higher_is_better else (1 - normed)


def recommend(
    budget_min_lakh: float = 3.0,
    budget_max_lakh: float = 250.0,
    fuel_type: Optional[str] = None,
    transmission: Optional[str] = None,
    min_mileage: float = 0.0,
    brand: Optional[str] = None,
    priority: str = "balanced",
    top_n: int = 6,
) -> pd.DataFrame:
    """
    Filter and rank vehicles from the 741 unique car database.
    budget_max_lakh >= 250.0 represents the '2.5cr+' tier (unbounded upper limit).
    """
    df = load_dataset().copy()

    # Exclude vehicles with missing price
    df = df[df["Price_Lakh"].notnull()]

    # Prefer active cars unless none match
    active_df = df[df["Active_Status"] == "Active"]
    if len(active_df) > 50:
        df = active_df

    # Hard filter: Budget
    if budget_max_lakh >= 250.0:
        df = df[df["Price_Lakh"] >= budget_min_lakh]
    else:
        df = df[df["Price_Lakh"].between(budget_min_lakh, budget_max_lakh)]

    # Hard filter: Fuel Type
    if fuel_type and fuel_type != "Any":
        df = df[df["Fuel_Type_Simple"].str.lower() == fuel_type.lower()]

    # Hard filter: Transmission
    if transmission and transmission != "Any":
        df = df[df["Transmission_Simple"].str.lower() == transmission.lower()]

    # Hard filter: Minimum Mileage
    if min_mileage > 0:
        df = df[df["Mileage_Clean"] >= min_mileage]

    # Hard filter: Brand
    if brand and brand != "Any":
        b_lower = brand.lower().strip()
        df = df[(df["Brand"].str.lower() == b_lower) | (df["Brand"].str.lower().str.startswith(b_lower))]

    if df.empty:
        return df

    # Priority-based weight vector
    weight_map = {
        "balanced":     {"mileage": 0.30, "price": 0.25, "features": 0.20, "engine": 0.15, "active": 0.10},
        "mileage":      {"mileage": 0.55, "price": 0.15, "features": 0.15, "engine": 0.10, "active": 0.05},
        "price":        {"mileage": 0.15, "price": 0.55, "features": 0.15, "engine": 0.10, "active": 0.05},
        "feature_rich": {"mileage": 0.15, "price": 0.20, "features": 0.45, "engine": 0.15, "active": 0.05},
        "performance":  {"mileage": 0.10, "price": 0.20, "features": 0.15, "engine": 0.50, "active": 0.05},
    }
    w = weight_map.get(priority, weight_map["balanced"])

    # Normalization with median imputation for missing specs
    mileage_filled = df["Mileage_Clean"].fillna(df["Mileage_Clean"].median() if not df["Mileage_Clean"].dropna().empty else 18.0)
    engine_filled = df["Displacement_Clean"].fillna(df["Displacement_Clean"].median() if not df["Displacement_Clean"].dropna().empty else 1498.0)

    scores = pd.Series(0.0, index=df.index)
    scores += w["mileage"]  * _normalise_series(mileage_filled, higher_is_better=True)
    scores += w["price"]    * _normalise_series(df["Price_Lakh"], higher_is_better=False)
    scores += w["features"] * _normalise_series(df["Feature_Score"], higher_is_better=True)
    scores += w["engine"]   * _normalise_series(engine_filled, higher_is_better=True)
    scores += w["active"]   * df["Active_Score"]

    df = df.copy()
    df["Score"] = (scores * 100).round(1)
    df = df.sort_values("Score", ascending=False)

    # Deduplicate: return highest-scoring trim variant per unique Car model
    df = df.drop_duplicates(subset=["Brand", "Car"], keep="first")

    return df.head(top_n).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Head-to-Head Comparison
# ---------------------------------------------------------------------------

def compare_cars(brand_a: str, car_a: str, brand_b: str, car_b: str) -> Tuple[Dict, Dict]:
    df = load_dataset()

    def _stats(brand, car):
        sub = df[(df["Brand"].str.lower() == brand.lower()) & (df["Car"].str.lower() == car.lower())]
        if sub.empty:
            return {}
        top_trim = sub.sort_values("Price_Lakh", ascending=True).iloc[0]
        return {
            "Brand": brand,
            "Car": car,
            "Variant": top_trim.get("Variant", "Standard"),
            "Price_Lakh": round(float(sub["Price_Lakh"].median()), 2) if sub["Price_Lakh"].notnull().any() else 0.0,
            "Price_Display": _format_price_display(sub["Price_Lakh"].median()) if sub["Price_Lakh"].notnull().any() else "Price on request",
            "Mileage": round(float(sub["Mileage_Clean"].median()), 1) if sub["Mileage_Clean"].notnull().any() else "—",
            "Displacement": int(sub["Displacement_Clean"].median()) if sub["Displacement_Clean"].notnull().any() else "—",
            "Fuel_Types": ", ".join(sorted(sub["Fuel_Type_Simple"].unique())),
            "Transmissions": ", ".join(sorted(sub["Transmission_Simple"].unique())),
            "Sunroof": "Yes" if (sub["feat_sunroof___moonroof"] == 1.0).any() else "No",
            "Cruise_Control": "Yes" if (sub["feat_cruise_control"] == 1.0).any() else "No",
            "Ventilated_Seats": "Yes" if (sub["feat_ventilated_seats"] == 1.0).any() else "No",
            "Wireless_Charger": "Yes" if (sub["feat_wireless_charger"] == 1.0).any() else "No",
            "Rear_AC": "Yes" if (sub["feat_rear_ac"] == 1.0).any() else "No",
            "HUD": "Yes" if (sub["feat_heads_up_display"] == 1.0).any() else "No",
        }

    return _stats(brand_a, car_a), _stats(brand_b, car_b)


# ---------------------------------------------------------------------------
# Dealership Directory
# ---------------------------------------------------------------------------

DEALERSHIPS = [
    # Mumbai
    {"brand": "Maruti Suzuki", "name": "Navnit Motors", "city": "Mumbai", "area": "Worli", "address": "Dr. Annie Besant Road, Worli", "phone": "022-61400000"},
    {"brand": "Maruti Suzuki", "name": "Sai Service", "city": "Mumbai", "area": "Andheri East", "address": "Western Express Highway, Andheri East", "phone": "022-28392000"},
    {"brand": "Hyundai", "name": "Modi Hyundai", "city": "Mumbai", "area": "Malad West", "address": "Link Road, Malad West", "phone": "022-67891000"},
    {"brand": "Tata Motors", "name": "Wasan Motors", "city": "Mumbai", "area": "Chembur", "address": "Eastern Express Highway, Chembur", "phone": "022-25273000"},
    {"brand": "Mahindra", "name": "Salasar Mahindra", "city": "Mumbai", "area": "Thane", "address": "Eastern Express Highway, Thane", "phone": "022-25873000"},
    {"brand": "Honda", "name": "Apex Honda", "city": "Mumbai", "area": "Andheri East", "address": "JVLR, Andheri East", "phone": "022-26741000"},
    {"brand": "Toyota", "name": "Lakozy Toyota", "city": "Mumbai", "area": "Malad West", "address": "New Link Road, Malad West", "phone": "022-28825000"},
    {"brand": "Kia", "name": "Auto Bahn Kia", "city": "Mumbai", "area": "Kurla", "address": "LBS Marg, Kurla West", "phone": "022-61200000"},
    {"brand": "Volkswagen", "name": "Volkswagen Downtown", "city": "Mumbai", "area": "Prabhadevi", "address": "Appasaheb Marathe Marg, Prabhadevi", "phone": "022-26550000"},
    {"brand": "Skoda", "name": "Zuma Skoda", "city": "Mumbai", "area": "Andheri West", "address": "Link Road, Andheri West", "phone": "022-40112200"},
    {"brand": "BMW", "name": "Infinity Cars", "city": "Mumbai", "area": "Worli", "address": "Worli Sea Face", "phone": "022-68490000"},
    {"brand": "Mercedes Benz", "name": "Auto Hangar", "city": "Mumbai", "area": "Prabhadevi", "address": "Mathuradas Mills Compound", "phone": "022-66123800"},

    # Delhi NCR
    {"brand": "Maruti Suzuki", "name": "Rohan Motors", "city": "Delhi NCR", "area": "Noida Sector 1", "address": "D-2, Sector 1, Noida", "phone": "0120-4360000"},
    {"brand": "Hyundai", "name": "Capital Hyundai", "city": "Delhi NCR", "area": "Okhla", "address": "Mathura Road, Okhla Phase 1", "phone": "011-26382000"},
    {"brand": "Tata Motors", "name": "Aryaveer Motors", "city": "Delhi NCR", "area": "Dwarka", "address": "Sector 7, Dwarka", "phone": "011-45610000"},
    {"brand": "Mahindra", "name": "Koncept Mahindra", "city": "Delhi NCR", "area": "Lajpat Nagar", "address": "Ring Road, Lajpat Nagar", "phone": "011-29871000"},
    {"brand": "Honda", "name": "Prime Honda", "city": "Delhi NCR", "area": "Patparganj", "address": "Patparganj Industrial Area", "phone": "011-43100000"},
    {"brand": "Toyota", "name": "Galaxy Toyota", "city": "Delhi NCR", "area": "Moti Nagar", "address": "Najafgarh Road, Moti Nagar", "phone": "011-45120000"},
    {"brand": "Kia", "name": "Frontier Kia", "city": "Delhi NCR", "area": "Gurugram MG Road", "address": "MG Road, Gurugram", "phone": "0124-4080000"},
    {"brand": "Volkswagen", "name": "Volkswagen Gurugram", "city": "Delhi NCR", "area": "Sector 14 Gurugram", "address": "Old Delhi Road, Sector 14, Gurugram", "phone": "0124-4556000"},
    {"brand": "BMW", "name": "Deutsche Motoren", "city": "Delhi NCR", "area": "Mathura Road", "address": "Mohan Cooperative, Mathura Road", "phone": "011-43090000"},

    # Bangalore
    {"brand": "Maruti Suzuki", "name": "Kalyani Motors", "city": "Bangalore", "area": "Koramangala", "address": "80 Feet Road, Koramangala", "phone": "080-25530000"},
    {"brand": "Hyundai", "name": "Trident Hyundai", "city": "Bangalore", "area": "Whitefield", "address": "ITPL Main Road, Whitefield", "phone": "080-28450000"},
    {"brand": "Tata Motors", "name": "Prerana Motors", "city": "Bangalore", "area": "Lalbagh Road", "address": "Lalbagh Road, Mavalli", "phone": "080-22223000"},
    {"brand": "Mahindra", "name": "Sireesh Auto", "city": "Bangalore", "area": "JP Nagar", "address": "24th Main, JP Nagar 5th Phase", "phone": "080-26492000"},
    {"brand": "Toyota", "name": "Nandi Toyota", "city": "Bangalore", "area": "Hosur Road", "address": "Kudlu Gate, Hosur Road", "phone": "080-25732000"},
    {"brand": "Kia", "name": "Epitome Kia", "city": "Bangalore", "area": "Electronic City", "address": "Hosur Road, Electronic City", "phone": "080-29150000"},
    {"brand": "Honda", "name": "Magnum Honda", "city": "Bangalore", "area": "Kanakapura Road", "address": "Outer Ring Road, Kanakapura Road", "phone": "080-26792000"},
    {"brand": "Volkswagen", "name": "Volkswagen Bangalore Central", "city": "Bangalore", "area": "Cunningham Road", "address": "Cunningham Road, Vasanth Nagar", "phone": "080-22283000"},

    # Hyderabad
    {"brand": "Maruti Suzuki", "name": "Varun Motors", "city": "Hyderabad", "area": "Secunderabad", "address": "MG Road, Secunderabad", "phone": "040-27802000"},
    {"brand": "Hyundai", "name": "Talwar Hyundai", "city": "Hyderabad", "area": "Banjara Hills", "address": "Road No 12, Banjara Hills", "phone": "040-23543000"},
    {"brand": "Tata Motors", "name": "Tejaswi Motors", "city": "Hyderabad", "area": "Madhapur", "address": "Hitech City Road, Madhapur", "phone": "040-23198000"},
    {"brand": "Mahindra", "name": "Automotive Mahindra", "city": "Hyderabad", "area": "Kompally", "address": "NH 44, Kompally", "phone": "040-27643000"},
    {"brand": "Toyota", "name": "Harsha Toyota", "city": "Hyderabad", "area": "Kondapur", "address": "Hitec City Road, Kondapur", "phone": "040-23121000"},

    # Pune
    {"brand": "Maruti Suzuki", "name": "Chowgule Industries", "city": "Pune", "area": "Viman Nagar", "address": "Nagar Road, Viman Nagar", "phone": "020-26620000"},
    {"brand": "Hyundai", "name": "Kundal Hyundai", "city": "Pune", "area": "Kothrud", "address": "Paud Road, Kothrud", "phone": "020-25432000"},
    {"brand": "Tata Motors", "name": "B.U. Bhandari Tata", "city": "Pune", "area": "Wakad", "address": "Hinjewadi Road, Wakad", "phone": "020-27660000"},
    {"brand": "Mahindra", "name": "Silver Jubilee Mahindra", "city": "Pune", "area": "Hadapsar", "address": "Solapur Road, Hadapsar", "phone": "020-26971000"},
    {"brand": "Kia", "name": "Aman Kia", "city": "Pune", "area": "Hinjewadi", "address": "Hinjewadi Phase 1", "phone": "020-67291000"},
]


def get_dealership_cities() -> List[str]:
    return sorted(list({d["city"] for d in DEALERSHIPS}))


def get_dealerships(city: Optional[str] = None, brand: Optional[str] = None) -> List[Dict]:
    results = DEALERSHIPS
    if city and city != "Any":
        results = [d for d in results if d["city"].lower() == city.lower()]
    if brand and brand != "Any":
        results = [d for d in results if brand.lower() in d["brand"].lower()]
    return results


# ---------------------------------------------------------------------------
# Gemini AI Chat Engine (REST API) & Knowledge Retrieval
# ---------------------------------------------------------------------------

def _load_dotenv_if_present() -> None:
    """Optionally loads environment variables from a local .env file if not already set."""
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if env_path.is_file():
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k, v = k.strip(), v.strip().strip("\"'")
                        if k and k not in os.environ:
                            os.environ[k] = v
        except Exception:
            pass


_load_dotenv_if_present()

GEMINI_SYSTEM_PROMPT = """You are CarWise Advisor, an elite automotive consultant specialized in the Indian passenger car market (covering all 741 unique car models and 40 brands in India).
Your mission is to understand each buyer's unique needs, guide them directly to the best car for them, and answer comparisons, technical queries, ownership questions, and showroom inquiries with authority.

Core Guidelines:
- Respond in clear, professional English. Absolutely NO emojis.
- Quote all pricing in Indian Rupees using Lakh (L) and Crore (Cr) notation (e.g. Rs. 8.50 Lakh, Rs. 1.25 Crore).
- Strictly ground your recommendations in the REAL CAR DATABASE RECORDS and SHOWROOM DATA provided below.
- Highlight concrete reasons why a car suits the user (fuel efficiency, cabin room, ground clearance, safety rating, engine displacement, transmission smoothness, maintenance ease, and resale value).
- When a price or budget is discussed, provide an estimated monthly EMI calculation (assuming standard 80% on-road financing over 5 years at ~9% p.a.).
- When the user asks about where to buy or mentions a city, provide matching authorized dealerships with their area and contact phone.
- Keep responses well-structured and concise (under 250 words). Use clean bullet points or short paragraphs."""


def find_relevant_cars_for_query(message: str, top_n: int = 6) -> List[Dict]:
    """
    Dynamically retrieve relevant cars from the 741 unique cars database
    based on keywords, budgets, brands, body types, fuel, and transmission in the user query.
    """
    df = load_dataset()
    q = message.lower()
    matched_models = []

    # 1. Check for specific car model names
    all_models = df["Car"].dropna().unique()
    for m in all_models:
        m_str = str(m)
        if len(m_str) >= 4 and m_str.lower() in q:
            sub = df[df["Car"].str.lower() == m_str.lower()]
            if not sub.empty:
                # pick lowest-price active variant or first
                active_sub = sub[sub["Active_Status"] == "Active"]
                pick = (active_sub if not active_sub.empty else sub).sort_values("Price_Lakh", ascending=True).iloc[0]
                matched_models.append(pick)

    # 2. Extract budget bounds
    b_min, b_max = 3.0, 250.0
    rng = re.search(r'(?:between\s+)?(\d+(?:\.\d+)?)\s*(?:to|-|and)\s*(\d+(?:\.\d+)?)\s*(lakh|lac|cr|crore|l)?', q)
    if rng:
        v1, v2 = float(rng.group(1)), float(rng.group(2))
        u = (rng.group(3) or "lakh").lower()
        mul = 100.0 if "cr" in u else 1.0
        b_min, b_max = min(v1, v2) * mul, max(v1, v2) * mul
    else:
        b_match = re.search(r'(?:under|below|budget of|within|upto|less than|around|approx)\s*(?:rs\.?|inr)?\s*(\d+(?:\.\d+)?)\s*(lakh|lac|cr|crore|l)?', q)
        if b_match:
            val = float(b_match.group(1))
            u = (b_match.group(2) or "lakh").lower()
            b_max = val * 100.0 if "cr" in u else val

    # 3. Fuel
    fuel = None
    if "electric" in q or "ev" in q:
        fuel = "Electric"
    elif "diesel" in q:
        fuel = "Diesel"
    elif "cng" in q:
        fuel = "CNG"
    elif "hybrid" in q:
        fuel = "Hybrid"
    elif "petrol" in q:
        fuel = "Petrol"

    # 4. Transmission
    trans = None
    if any(k in q for k in ["automatic", "auto", "cvt", "dct", "amt"]):
        trans = "Automatic"
    elif "manual" in q:
        trans = "Manual"

    # 5. Brand
    brand = None
    for b in get_brands():
        b_l = b.lower()
        if b_l in q or (len(b.split()) > 1 and b.split()[0].lower() in q):
            brand = b
            break

    # 6. Priority mapping
    priority = "balanced"
    if any(k in q for k in ["mileage", "fuel efficient", "average", "economy"]):
        priority = "mileage"
    elif any(k in q for k in ["cheap", "lowest price", "affordable", "budget"]):
        priority = "price"
    elif any(k in q for k in ["feature", "sunroof", "ventilated", "luxury", "tech"]):
        priority = "feature_rich"
    elif any(k in q for k in ["power", "fast", "performance", "engine", "pickup"]):
        priority = "performance"

    # Query recommendation engine
    recs = recommend(
        budget_min_lakh=b_min,
        budget_max_lakh=b_max,
        fuel_type=fuel,
        transmission=trans,
        brand=brand,
        priority=priority,
        top_n=top_n,
    )

    # Filter for sunroof if explicitly requested
    if "sunroof" in q and not recs.empty and "feat_sunroof___moonroof" in recs.columns:
        s_recs = recs[recs["feat_sunroof___moonroof"] == 1.0]
        if not s_recs.empty:
            recs = s_recs

    # Combine direct model matches and recommendation matches
    combined: List[Dict] = []
    seen = set()
    for row in matched_models:
        key = (row["Brand"], row["Car"])
        if key not in seen:
            combined.append(row.to_dict() if hasattr(row, "to_dict") else dict(row))
            seen.add(key)
    for _, row in recs.iterrows():
        key = (row["Brand"], row["Car"])
        if key not in seen:
            combined.append(row.to_dict() if hasattr(row, "to_dict") else dict(row))
            seen.add(key)

    return combined[:top_n]


def call_gemini_chat(
    api_key: Optional[str] = None,
    message: str = "",
    history: Optional[List[Dict]] = None,
    context_cars: Optional[List[Dict]] = None,
    city: Optional[str] = None,
) -> str:
    key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    history = history or []
    context_cars = context_cars or []

    # 1. Retrieve matching cars from dataset dynamically
    retrieved = find_relevant_cars_for_query(message, top_n=6)
    all_context = context_cars[:3] + [c for c in retrieved if c not in context_cars[:3]]

    ctx_lines = []
    for c in all_context[:6]:
        b = c.get("Brand", "")
        car = c.get("Car", "")
        var = c.get("Variant", "Standard")
        p_disp = c.get("Price_Display", "Price on request")
        mileage = c.get("Mileage_Clean")
        m_disp = f"{mileage:.1f} kmpl" if mileage and str(mileage) != "nan" else "N/A"
        eng = c.get("Displacement_Clean")
        eng_disp = f"{int(eng)} cc" if eng and str(eng) != "nan" else "Electric/Unknown"
        fuel = c.get("Fuel_Type_Simple", "Petrol")
        trans = c.get("Transmission_Simple", "Manual")
        feat_cnt = int(c.get("Feature_Score", 0))

        # Check features
        feats = []
        if c.get("feat_sunroof___moonroof") == 1.0 or c.get("Sunroof") == "Yes":
            feats.append("Sunroof")
        if c.get("feat_ventilated_seats") == 1.0 or c.get("Ventilated_Seats") == "Yes":
            feats.append("Ventilated Seats")
        if c.get("feat_cruise_control") == 1.0 or c.get("Cruise_Control") == "Yes":
            feats.append("Cruise Control")
        if c.get("feat_wireless_charger") == 1.0 or c.get("Wireless_Charger") == "Yes":
            feats.append("Wireless Charger")
        feats_str = ", ".join(feats) if feats else f"{feat_cnt}/6 equipment score"

        ctx_lines.append(
            f"- {b} {car} ({var}): Ex-showroom {p_disp} | {m_disp} | {eng_disp} | {fuel} | {trans} | Features: {feats_str}"
        )

    # 2. Match city & showrooms
    msg_l = message.lower()
    city_matched = city if city and city != "Any" else None
    if not city_matched:
        for c_name in get_dealership_cities():
            if c_name.lower() in msg_l:
                city_matched = c_name
                break

    # If no Gemini API key is configured in environment, fall back to local rule-based advisor
    if not key:
        return _generate_local_rule_advice(message, retrieved, city_matched)

    dealer_lines = []
    if any(k in msg_l for k in ["dealer", "showroom", "near", "buy", "store", "location", "address", "phone", "test drive", "visit"]) or city_matched:
        dealers = get_dealerships(city=city_matched)
        for d in dealers[:6]:
            dealer_lines.append(f"- {d['brand']} in {d['city']} ({d['area']}): {d['name']} | Address: {d['address']} | Phone: {d['phone']}")

    system_instruction = GEMINI_SYSTEM_PROMPT
    if ctx_lines:
        system_instruction += "\n\nDATABASE CAR SPECIFICATIONS (FROM 741 CARS DATASET):\n" + "\n".join(ctx_lines)
    if dealer_lines:
        system_instruction += "\n\nVERIFIED AUTHORIZED SHOWROOMS:\n" + "\n".join(dealer_lines)

    # 3. Format contents for Gemini REST API
    contents = []
    for m in history[-8:]:
        role = "user" if m.get("role") == "user" else "model"
        contents.append({"role": role, "parts": [{"text": str(m.get("content", ""))}]})
    contents.append({"role": "user", "parts": [{"text": message}]})

    payload = {
        "contents": contents,
        "system_instruction": {"parts": [{"text": system_instruction}]},
        "generationConfig": {
            "temperature": 0.35,
            "maxOutputTokens": 2048,
            "thinkingConfig": {"thinkingBudget": 150},
        },
    }

    # Primary model: gemini-2.5-flash
    models = ["gemini-2.5-flash", "gemini-2.5-pro"]
    for model_name in models:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={key}"
        try:
            resp = requests.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30,
            )
            if resp.status_code == 200:
                data = resp.json()
                candidates = data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts:
                        return parts[0].get("text", "").strip()
            elif resp.status_code == 404:
                continue
            else:
                # Fall back to local intelligent advice if API key is invalid or rate limited
                return _generate_local_rule_advice(message, retrieved, city_matched)
        except Exception as e:
            # Fallback to local intelligent advice if network call is blocked
            return _generate_local_rule_advice(message, retrieved, city_matched)

    return _generate_local_rule_advice(message, retrieved, city_matched)


def _generate_local_rule_advice(message: str, cars: List[Dict], city: Optional[str] = None) -> str:
    """Deterministic local advisory response if outbound API connectivity is unavailable."""
    if not cars:
        return "I could not find exact cars matching your criteria. Try adjusting your budget range or fuel preference."

    lines = ["Based on the Indian car market database, here are the top recommended models:"]
    for i, c in enumerate(cars[:4], 1):
        brand = c.get("Brand", "")
        car = c.get("Car", "")
        var = c.get("Variant", "Standard")
        price = c.get("Price_Display", "Price on request")
        fuel = c.get("Fuel_Type_Simple", "Petrol")
        trans = c.get("Transmission_Simple", "Manual")
        mileage = c.get("Mileage_Clean")
        m_str = f" offering approximately {mileage:.1f} kmpl" if mileage and str(mileage) != "nan" else ""
        lines.append(f"{i}. {brand} {car} ({var}) — {price} ({fuel}, {trans}){m_str}.")

    top_car = cars[0]
    p_lakh = top_car.get("Price_Lakh", 0)
    if p_lakh and p_lakh > 0:
        loan = p_lakh * 100_000 * 0.8
        r = 0.09 / 12
        emi = int(loan * r * ((1 + r) ** 60) / (((1 + r) ** 60) - 1))
        lines.append(f"Estimated monthly EMI for {top_car.get('Car')} is approximately Rs. {emi:,}/month (80% loan, 5 years at 9%).")

    if city:
        dealers = get_dealerships(city=city)
        if dealers:
            d = dealers[0]
            lines.append(f"Nearest showroom in {city}: {d['name']} ({d['brand']}) in {d['area']}. Phone: {d['phone']}.")

    return "\n\n".join(lines)
