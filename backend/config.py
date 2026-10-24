from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
SEED = 42
N_ROWS = 8000

CLASSES = ["Poor", "Fair", "Good", "Excellent"]

NUMERIC = [
    "age_years", "mileage_km", "previous_owners", "service_completeness",
    "months_since_service", "accident_count", "engine_cc", "safety_rating",
    "seats", "doors", "airbags", "warranty_remaining_months",
]
CATEGORICAL = ["fuel_type", "body_material", "country_of_manufacture", "transmission", "drive_type"]

TARGETS = ["exterior_damage_score", "paint_condition_score", "engine_condition_score"]
DERIVED = TARGETS