"""
car_data.py — Loads and indexes the India car database for retrieval.
Part of backend.chatbot service.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Multi-path discovery to find india_cars.json robustly
_possible_paths = [
    Path(__file__).resolve().parent / "data" / "india_cars.json",
    Path(__file__).resolve().parent.parent.parent / "data" / "india_cars.json",
    Path("data/india_cars.json"),
    Path("car-chatbot/data/india_cars.json"),
]
DATA_PATH = next((p for p in _possible_paths if p.exists()), _possible_paths[0])


def _load() -> dict:
    with open(DATA_PATH, encoding="utf-8") as f:
        return json.load(f)


_DB: Dict = _load()
CARS: List[Dict] = _DB["cars"]
DEALERSHIPS: Dict = _DB["dealerships"]
BUYING_TIPS: Dict = _DB["buying_tips"]


# ---------------------------------------------------------------------------
# Budget parsing helpers
# ---------------------------------------------------------------------------

def parse_budget_lakh(text: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Extract a budget range (min, max) in lakhs from free text.
    Handles patterns like:
      "under 10 lakh", "10-15 lakhs", "around 12 lakh",
      "budget of 15 lakh", "upto 20 lakhs", "50000" (assumed rupees -> convert)
    """
    text_l = text.lower()

    # Match explicit range like "10 to 15 lakh" or "10-15 lakh"
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:to|-)\s*(\d+(?:\.\d+)?)\s*(?:lakh|l\b|lac)", text_l)
    if m:
        return float(m.group(1)), float(m.group(2))

    # Match single value with qualifier
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:lakh|l\b|lac)", text_l)
    if m:
        val = float(m.group(1))
        if re.search(r"\b(under|below|within|upto|up to|less than|max|maximum)\b", text_l):
            return 0.0, val
        if re.search(r"\b(above|over|more than|atleast|at least|min|minimum|starting)\b", text_l):
            return val, val * 2.5
        # Treat as max budget by default
        return 0.0, val

    # Crore
    m = re.search(r"(\d+(?:\.\d+)?)\s*crore", text_l)
    if m:
        val = float(m.group(1)) * 100
        return 0.0, val

    return None, None


# ---------------------------------------------------------------------------
# Recommendation engine
# ---------------------------------------------------------------------------

PREFERENCE_KEYWORDS: Dict[str, List[str]] = {
    "fuel_efficient":    ["mileage", "fuel efficient", "economy", "kmpl", "low fuel", "fuel cost"],
    "safety":            ["safe", "safety", "crash", "ncap", "5 star", "airbag"],
    "family":            ["family", "kids", "children", "spacious", "7 seat", "7-seat", "seven seat"],
    "city_driving":      ["city", "urban", "traffic", "daily commute", "office"],
    "highway":           ["highway", "long drive", "road trip", "expressway", "touring"],
    "premium":           ["premium", "luxury", "feature", "loaded", "high end", "rich"],
    "budget_buyer":      ["budget", "affordable", "cheap", "economical", "value"],
    "suv":               ["suv", "crossover", "ground clearance", "off road"],
    "electric_option":   ["electric", "ev", "battery", "zero emission", "plug"],
    "low_maintenance":   ["maintenance", "service cost", "repair cost", "reliable", "reliability"],
    "tech_lover":        ["technology", "tech", "screen", "connected", "adas", "sunroof", "features"],
    "first_car":         ["first car", "new driver", "beginner", "learner"],
    "sporty":            ["sporty", "fast", "performance", "turbocharged", "turbo", "sport"],
}

SEGMENT_KEYWORDS: Dict[str, List[str]] = {
    "hatchback":        ["hatchback", "small car", "compact"],
    "premium_hatchback": ["premium hatchback"],
    "sedan":            ["sedan", "saloon"],
    "compact_suv":      ["compact suv", "sub4m suv"],
    "midsize_suv":      ["midsize suv", "mid suv", "large suv"],
    "mpv":              ["mpv", "van", "people mover", "7 seater", "7-seater"],
    "premium_mpv":      ["premium mpv", "luxury mpv"],
    "micro_suv":        ["micro suv", "mini suv"],
}

BRAND_KEYWORDS: Dict[str, str] = {
    "maruti":        "Maruti Suzuki",
    "suzuki":        "Maruti Suzuki",
    "hyundai":       "Hyundai",
    "tata":          "Tata",
    "mahindra":      "Mahindra",
    "honda":         "Honda",
    "kia":           "Kia",
    "toyota":        "Toyota",
    "volkswagen":    "Volkswagen",
    "vw":            "Volkswagen",
    "mg":            "MG",
    "skoda":         "Skoda",
}

FUEL_KEYWORDS: Dict[str, List[str]] = {
    "petrol": ["petrol", "gasoline"],
    "diesel": ["diesel"],
    "cng":    ["cng", "compressed natural gas"],
    "electric": ["electric", "ev", "battery", "plug-in"],
    "hybrid": ["hybrid", "mhev", "shev"],
}


def extract_preferences(user_text: str) -> dict[str, Any]:
    """
    Parse a user message and extract structured preferences.
    Returns a dict with keys: budget_min, budget_max, segment, fuel,
    brand, seating, preferences (list of tags), raw_text.
    """
    tl = user_text.lower()
    prefs = {
        "budget_min": None,
        "budget_max": None,
        "segment": None,
        "fuel": None,
        "brand": None,
        "seating": None,
        "preference_tags": [],
        "raw_text": user_text,
    }

    # Budget
    bmin, bmax = parse_budget_lakh(tl)
    prefs["budget_min"] = bmin
    prefs["budget_max"] = bmax

    # Segment
    for seg, kws in SEGMENT_KEYWORDS.items():
        if any(kw in tl for kw in kws):
            prefs["segment"] = seg
            break

    # Fuel
    for fuel, kws in FUEL_KEYWORDS.items():
        if any(kw in tl for kw in kws):
            prefs["fuel"] = fuel
            break

    # Brand preference
    for kw, brand in BRAND_KEYWORDS.items():
        if kw in tl:
            prefs["brand"] = brand
            break

    # Seating
    m = re.search(r"(?:(\d+)|(seven|six|five|eight))\s*[- ]?\s*(?:seat|seater|passenger|person)", tl)
    if m:
        val_str = m.group(1) or m.group(2)
        word_map = {"five": 5, "six": 6, "seven": 7, "eight": 8}
        prefs["seating"] = int(val_str) if val_str and val_str.isdigit() else word_map.get(val_str)
    elif "7-seater" in tl or "7 seater" in tl or "seven seater" in tl:
        prefs["seating"] = 7
    elif "6-seater" in tl or "6 seater" in tl or "six seater" in tl:
        prefs["seating"] = 6

    # Preference tags
    for tag, kws in PREFERENCE_KEYWORDS.items():
        if any(kw in tl for kw in kws):
            prefs["preference_tags"].append(tag)

    return prefs


def _score_car(car: dict, prefs: dict) -> float:
    """Score a car 0-100 against extracted preferences."""
    score = 50.0

    # Budget match
    if prefs["budget_max"] is not None:
        bmax = prefs["budget_max"]
        bmin = prefs["budget_min"] or 0.0
        car_min = car["price_lakh"]["min"]
        car_max = car["price_lakh"]["max"]
        if car_min > bmax:
            score -= 60  # Way over budget
        elif car_max < bmin and bmin > 0:
            score -= 20  # Probably below what they want
        else:
            score += 20  # Budget match

    # Segment match
    if prefs["segment"] and car["segment"] == prefs["segment"]:
        score += 25

    # Fuel match
    if prefs["fuel"] and prefs["fuel"] in car["fuel_types"]:
        score += 15

    # Brand match
    if prefs["brand"] and car["brand"] == prefs["brand"]:
        score += 10

    # Seating
    if prefs["seating"] is not None:
        if car["seating"] >= prefs["seating"]:
            score += 25
        else:
            score -= 40  # Inadequate seating capacity penalty

    # SUV preference
    if "suv" in prefs["preference_tags"]:
        if "suv" in car["segment"]:
            score += 20

    # Preference tags
    car_tags = set(car.get("category_tags", []) + car.get("ideal_for", []))
    matched_tags = [t for t in prefs["preference_tags"] if t in car_tags]
    score += len(matched_tags) * 10

    # Safety bonus
    if "safety" in prefs["preference_tags"] or "family" in prefs["preference_tags"]:
        score += car["safety_rating"] * 2

    return score


def recommend_cars(prefs: dict, top_n: int = 3) -> List[Dict]:
    """Return top_n cars scored against preferences."""
    scored = [(car, _score_car(car, prefs)) for car in CARS]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [car for car, _ in scored[:top_n]]


def get_dealerships_for_city(city: str) -> List[Dict]:
    """Return dealerships for the closest matching city."""
    city_l = city.lower().strip()
    for key, dealers in DEALERSHIPS.items():
        if key in city_l or city_l in key:
            return dealers
    # Fuzzy fallback
    for key, dealers in DEALERSHIPS.items():
        if any(tok in city_l for tok in key.split()):
            return dealers
    return []


def get_dealerships_for_brand_city(brand: str, city: str) -> List[Dict]:
    """Filter dealerships by brand and city."""
    all_dealers = get_dealerships_for_city(city)
    if not all_dealers:
        return []
    brand_l = brand.lower()
    return [d for d in all_dealers if brand_l in d["brand"].lower()]


def get_car_by_name(name: str) -> Optional[Dict]:
    """Find a car by exact ID, partial name match, or token overlap."""
    name_l = name.lower().strip()
    if not name_l:
        return None

    # 1. Exact ID or name match
    for car in CARS:
        if name_l == car["id"].lower() or name_l == car["name"].lower():
            return car

    # 2. Bidirectional substring match
    for car in CARS:
        cid = car["id"].lower()
        cname = car["name"].lower()
        if name_l in cname or name_l in cid:
            return car
        if cid in name_l or cname in name_l:
            return car

    # 3. Model token matching (e.g. 'Swift VXi' -> 'Swift', 'Thar 4x4' -> 'Thar')
    tokens = set(re.findall(r"\w+", name_l))
    for car in CARS:
        cid = car["id"].lower()
        cname_parts = set(re.findall(r"\w+", car["name"].lower()))
        if cid in tokens or any(part in tokens for part in cname_parts if len(part) >= 4):
            return car

    return None


def format_car_summary(car: dict, include_dealership_city: Optional[str] = None) -> str:
    """Format a car's details into a readable string for the LLM context."""
    price = f"₹{car['price_lakh']['min']}–{car['price_lakh']['max']} lakh"
    fuels = ", ".join(car["fuel_types"]).title()
    mileage_parts = [f"{v} kmpl ({k})" for k, v in car["mileage_kmpl"].items()]
    mileage = " | ".join(mileage_parts)
    pros = "; ".join(car["pros"][:3])
    cons = "; ".join(car["cons"][:2])

    text = (
        f"**{car['name']}** | Segment: {car['segment'].replace('_', ' ').title()} | "
        f"Price: {price} | Fuel: {fuels} | Mileage: {mileage} | "
        f"Seating: {car['seating']} | Safety: {'⭐' * car['safety_rating']}/5 | "
        f"Pros: {pros} | Cons: {cons}"
    )

    if include_dealership_city:
        dealers = get_dealerships_for_brand_city(car["brand"], include_dealership_city)
        if dealers:
            d = dealers[0]
            text += f" | Nearest dealer in {include_dealership_city.title()}: {d['name']} ({d['area']}) — {d['phone']}"

    return text

