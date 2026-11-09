import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("../data/processed_dataset.csv")

df["brand"] = (
    df["brand"]
    .str.strip()
    .str.title()
    .replace({"Mercedes-Amg": "Mercedes-Benz"})
)

df["log_km"] = np.log1p(df["km_driven"])
df["log_engine"] = np.log1p(df["engine"])
df["log_power"] = np.log1p(df["max_power"])

df["engine_power_ratio"] = df["engine"] / (df["max_power"] + 1)
df["power_x_engine"] = df["max_power"] * df["engine"]
df["age_squared"] = df["vehicle_age"] ** 2
df["km_squared"] = df["km_driven"] ** 2
df["mileage_per_power"] = df["mileage"] / (df["max_power"] + 1)

df["is_luxury"] = df["brand"].isin([
    "Bmw", "Mercedes-Benz", "Audi", "Jaguar", "Land Rover",
    "Porsche", "Volvo", "Mini", "Lexus", "Bentley", "Maserati",
    "Ferrari", "Rolls-Royce"
]).astype(int)

x = df.drop(columns=["selling_price"])
y = df["selling_price"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=42
)

numerical_features = x.select_dtypes(include=np.number).columns.tolist()
categorical_features = x.select_dtypes(include=["object", "category"]).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nLinear Regression Results")
print(f"Training R2: {model.score(x_train, y_train):.4f}")
print(f"Test R2:     {r2:.4f}")
print(f"MAE:         {mae:.2f}")
print(f"RMSE:        {rmse:.2f}")

os.makedirs("../saved_models", exist_ok=True)

joblib.dump(
    model,
    "../saved_models/linear_regression.joblib"
)