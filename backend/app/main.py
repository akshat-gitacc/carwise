from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from .explain import PriceExplainer
from .advisor_router import router as advisor_router
from .recommender_router import router as recommender_router
from .favourites_router import router as favourites_router
from .interactions_router import router as interactions_router
from .chatbot_router import router as chatbot_router

import joblib
import numpy as np
import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "models" / "saved_models" / "xgboost_regression.joblib"
if not MODEL_PATH.exists():
    MODEL_PATH = BASE_DIR / "saved_models" / "xgboost_regression.joblib"

pipeline = joblib.load(MODEL_PATH)
preprocessor = pipeline.named_steps["preprocessor"]
FEATURE_COLUMNS = list(preprocessor.feature_names_in_)
explainer = PriceExplainer(pipeline)

LUXURY_BRANDS = [
    "Bmw", "Mercedes-Benz", "Audi", "Jaguar", "Land Rover",
    "Porsche", "Volvo", "Mini", "Lexus", "Bentley", "Maserati",
    "Ferrari", "Rolls-Royce"
]


class CarInput(BaseModel):
    brand: str
    model: str
    vehicle_age: int
    km_driven: int
    seller_type: str
    fuel_type: str
    transmission_type: str
    mileage: float
    engine: int
    max_power: float
    seats: int


class Contribution(BaseModel):
    feature: str
    effect_pct: float
    direction: str


class PredictionResponse(BaseModel):
    predicted_price: float
    formatted_price: str
    base_price: float
    explanation: list[Contribution]


app = FastAPI(
    title="CarWise API",
    description="CarWise unified backend hosting two services: Used Car Price Prediction and Indian Car Advisor.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Routers
app.include_router(advisor_router)
app.include_router(recommender_router)
app.include_router(favourites_router)
app.include_router(interactions_router)
app.include_router(chatbot_router)


def build_features(data: dict) -> pd.DataFrame:
    df = pd.DataFrame([data])

    df["brand"] = df["brand"].astype(str).str.strip().str.title()
    df.loc[df["brand"] == "Mercedes-Amg", "brand"] = "Mercedes-Benz"
    df.loc[df["brand"] == "Isuzu", "brand"] = "Isuzu"

    df["log_km"] = np.log1p(df["km_driven"])
    df["log_engine"] = np.log1p(df["engine"])
    df["log_power"] = np.log1p(df["max_power"])

    df["power_per_litre"] = (df["max_power"] / (df["engine"] / 1000)).replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0)

    df["km_per_year"] = (df["km_driven"] / (df["vehicle_age"] + 1)).astype(int)
    df["log_km_per_year"] = np.log1p(df["km_per_year"])

    df["mileage_x_age"] = df["mileage"] * df["vehicle_age"]
    df["engine_power_ratio"] = df["engine"] / (df["max_power"] + 1)
    df["power_x_engine"] = df["max_power"] * df["engine"]
    df["age_squared"] = df["vehicle_age"] ** 2
    df["km_squared"] = df["km_driven"] ** 2
    df["seats_x_power"] = df["seats"] * df["max_power"]
    df["mileage_per_power"] = df["mileage"] / (df["max_power"] + 1)

    df["brand_model"] = df["brand"] + "_" + df["model"]

    df["brand_price_rank"] = 0
    df["model_price_rank"] = 0
    df["brand_model_rank"] = 0
    df["brand_median_price"] = 0.0
    df["brand_count"] = 0
    df["brand_target_enc"] = 0.0
    df["model_target_enc"] = 0.0
    df["brand_model_target_enc"] = 0.0

    df["is_luxury"] = df["brand"].isin(LUXURY_BRANDS).astype(int)
    df["is_automatic"] = (df["transmission_type"] == "Automatic").astype(int)
    df["is_diesel"] = (df["fuel_type"] == "Diesel").astype(int)
    df["luxury_auto"] = df["is_luxury"] * df["is_automatic"]

    df.drop(columns=["brand_model"], inplace=True)

    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    for c in missing:
        df[c] = 0

    return df[FEATURE_COLUMNS]


def format_price(price: float) -> str:
    if price >= 1_00_00_000:
        return f"₹{price / 1_00_00_000:.2f} Cr"
    elif price >= 1_00_000:
        return f"₹{price / 1_00_000:.2f} Lakh"
    else:
        return f"₹{price:,.0f}"


@app.get("/", tags=["General"])
def root():
    return {
        "status": "online",
        "service": "CarWise AI/ML Automotive Intelligence Platform",
        "version": "1.0.0",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "predict": "/predict",
            "advisor": "/api/advisor",
            "recommendations": "/recommendations",
            "favourites": "/favourites",
        },
    }


@app.get("/health", tags=["General"])
def health():
    return {"status": "ok", "services": ["price_prediction", "car_advisor", "hybrid_recommender", "favourites"]}


@app.post("/predict", response_model=PredictionResponse, tags=["Price Prediction"])
def predict(car: CarInput):
    features = build_features(car.model_dump())
    prediction_log = pipeline.predict(features)[0]
    prediction = float(np.expm1(prediction_log))
    prediction = round(max(0, prediction), 2)

    return PredictionResponse(
        predicted_price=prediction,
        formatted_price=format_price(prediction),
        base_price=round(explainer.base_price, 2),
        explanation=explainer.explain(features),
    )
