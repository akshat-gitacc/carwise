
import numpy as np
import pandas as pd
# pyrefly: ignore[missing-import]
import optuna
import joblib

from pathlib import Path
# pyrefly: ignore[missing-import]
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

optuna.logging.set_verbosity(optuna.logging.WARNING)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed_dataset.csv"
MODEL_DIR = BASE_DIR / "saved_models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["selling_price"])
y = np.log1p(df["selling_price"])

cat_features = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()

for col in cat_features:
    X[col] = X[col].fillna("__MISSING__").astype(str)

feature_columns = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

def objective(trial):
    params = {
        "iterations": 3000,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 15),
        "random_strength": trial.suggest_float("random_strength", 0.1, 5),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 5),
        "border_count": trial.suggest_int("border_count", 64, 254),
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "random_seed": 42,
        "verbose": False,
        "allow_writing_files": False
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    best_iterations = []

    for train_idx, val_idx in kf.split(X_train):
        X_fold_train = X_train.iloc[train_idx]
        y_fold_train = y_train.iloc[train_idx]

        X_fold_val = X_train.iloc[val_idx]
        y_fold_val = y_train.iloc[val_idx]

        X_fit, X_early, y_fit, y_early = train_test_split(
            X_fold_train,
            y_fold_train,
            test_size=0.15,
            random_state=42
        )

        model = CatBoostRegressor(**params)

        model.fit(
            X_fit,
            y_fit,
            cat_features=cat_features,
            eval_set=(X_early, y_early),
            early_stopping_rounds=100,
            verbose=False
        )

        predictions = np.expm1(model.predict(X_fold_val))
        actual = np.expm1(y_fold_val)

        scores.append(r2_score(actual, predictions))
        best_iterations.append(model.best_iteration_ + 1)

    trial.set_user_attr("best_iterations", best_iterations)

    return np.mean(scores)


print("Starting CatBoost hyperparameter tuning...")

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50, show_progress_bar=True)

print("\nHyperparameter tuning completed.")
print("Best parameters:", study.best_params)
print(f"Best cross-validation R2: {study.best_value:.4f}")

best_iterations = [
    iteration
    for trial in study.trials
    if trial.value is not None
    for iteration in trial.user_attrs.get("best_iterations", [])
]

final_iterations = max(1, int(np.median(best_iterations)))

best_params = {
    **study.best_params,
    "iterations": final_iterations,
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "random_seed": 42,
    "verbose": False,
    "allow_writing_files": False
}

print(f"\nTraining final CatBoost model with {final_iterations} iterations...")

model = CatBoostRegressor(**best_params)

model.fit(
    X_train,
    y_train,
    cat_features=cat_features,
    verbose=False
)

train_predictions = np.expm1(model.predict(X_train))
test_predictions = np.expm1(model.predict(X_test))

y_train_actual = np.expm1(y_train)
y_test_actual = np.expm1(y_test)

metrics = {
    "MAE": mean_absolute_error(y_test_actual, test_predictions),
    "RMSE": np.sqrt(mean_squared_error(y_test_actual, test_predictions)),
    "R2": r2_score(y_test_actual, test_predictions),
    "Training R2": r2_score(y_train_actual, train_predictions)
}

model.save_model(str(MODEL_DIR / "catboost_model.cbm"))

joblib.dump(
    {
        "feature_columns": feature_columns,
        "categorical_features": cat_features,
        "best_params": best_params,
        "metrics": metrics
    },
    MODEL_DIR / "catboost_metadata.joblib"
)

print("\nCatBoost Regression Results")
print("---------------------------")
print(f"Training R2: {metrics['Training R2']:.4f}")
print(f"Test R2:     {metrics['R2']:.4f}")
print(f"MAE:         {metrics['MAE']:.2f}")
print(f"RMSE:        {metrics['RMSE']:.2f}")
print(f"Iterations:  {final_iterations}")
print(f"Model saved: {MODEL_DIR / 'catboost_model.cbm'}")
print(f"Metadata saved: {MODEL_DIR / 'catboost_metadata.joblib'}")
