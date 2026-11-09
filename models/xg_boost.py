import pandas as pd
import numpy as np
import joblib
import os
import warnings
from sklearn.base import clone
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor

warnings.filterwarnings('ignore')


df = pd.read_csv("../data/processed_dataset.csv")

df['brand'] = df['brand'].str.strip().str.title()
df.loc[df['brand'] == 'Mercedes-Amg', 'brand'] = 'Mercedes-Benz'
df.loc[df['brand'] == 'Isuzu', 'brand'] = 'Isuzu'


df['log_km'] = np.log1p(df['km_driven'])
df['log_engine'] = np.log1p(df['engine'])
df['log_power'] = np.log1p(df['max_power'])


df['engine_power_ratio'] = df['engine'] / (df['max_power'] + 1)
df['power_x_engine'] = df['max_power'] * df['engine']
df['age_squared'] = df['vehicle_age'] ** 2
df['km_squared'] = df['km_driven'] ** 2
df['mileage_per_power'] = df['mileage'] / (df['max_power'] + 1)


df['is_luxury'] = df['brand'].isin([
    'Bmw', 'Mercedes-Benz', 'Audi', 'Jaguar', 'Land Rover',
    'Porsche', 'Volvo', 'Mini', 'Lexus', 'Bentley', 'Maserati',
    'Ferrari', 'Rolls-Royce'
]).astype(int)


x = df.drop(columns=['selling_price'])
y_raw = df['selling_price']
y = np.log1p(y_raw)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=42
)

y_test_raw = np.expm1(y_test)

numerical_features = x.select_dtypes(include=np.number).columns.tolist()
categorical_features = x.select_dtypes(include=['object', 'str', 'category']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_features)
    ]
)

x_train_proc = preprocessor.fit_transform(x_train)
x_test_proc = preprocessor.transform(x_test)


try:
    # pyrefly: ignore[missing-import]
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

if HAS_OPTUNA:
    print("Running Optuna hyperparameter search (50 trials)...")

    def objective(trial):
        params = {
            'n_estimators': 3000,
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
            'max_depth': trial.suggest_int('max_depth', 5, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 3, 15),
            'subsample': trial.suggest_float('subsample', 0.6, 0.95),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 0.9),
            'gamma': trial.suggest_float('gamma', 0.0, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 5.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 10.0, log=True),
            'objective': 'reg:squarederror',
            'random_state': 42,
            'n_jobs': -1,
            'tree_method': 'hist',
            'early_stopping_rounds': 80,
        }

        cv_inner = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = []

        for train_idx, val_idx in cv_inner.split(x_train_proc):
            xt, xv = x_train_proc[train_idx], x_train_proc[val_idx]
            yt, yv = y_train.iloc[train_idx], y_train.iloc[val_idx]

            m = XGBRegressor(**params)
            m.fit(xt, yt, eval_set=[(xv, yv)], verbose=False)
            pred = m.predict(xv)
            scores.append(r2_score(yv, pred))

        return np.mean(scores)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50, show_progress_bar=True)

    print(f"Best CV R2: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    best = study.best_params
    best_xgb = XGBRegressor(
        n_estimators=5000,
        learning_rate=best['learning_rate'],
        max_depth=best['max_depth'],
        min_child_weight=best['min_child_weight'],
        subsample=best['subsample'],
        colsample_bytree=best['colsample_bytree'],
        colsample_bylevel=best['colsample_bylevel'],
        gamma=best['gamma'],
        reg_alpha=best['reg_alpha'],
        reg_lambda=best['reg_lambda'],
        objective='reg:squarederror',
        eval_metric='rmse',
        early_stopping_rounds=100,
        random_state=42,
        n_jobs=-1,
        tree_method='hist'
    )

else:
    print("Optuna not found, using refined defaults.")
    best_xgb = XGBRegressor(
        n_estimators=3000,
        learning_rate=0.015,
        max_depth=8,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.7,
        colsample_bylevel=0.7,
        gamma=0.1,
        reg_alpha=0.5,
        reg_lambda=2.0,
        objective='reg:squarederror',
        eval_metric='rmse',
        early_stopping_rounds=100,
        random_state=42,
        n_jobs=-1,
        tree_method='hist'
    )


cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []
best_iterations = []

for train_idx, val_idx in cv.split(x_train_proc):
    xt, xv = x_train_proc[train_idx], x_train_proc[val_idx]
    yt, yv = y_train.iloc[train_idx], y_train.iloc[val_idx]

    m = XGBRegressor(**best_xgb.get_params())
    m.set_params(n_estimators=2000, early_stopping_rounds=80)
    m.fit(xt, yt, eval_set=[(xv, yv)], verbose=False)

    pred = m.predict(xv)
    cv_scores.append(r2_score(yv, pred))
    best_iterations.append(m.best_iteration + 1)

optimal_rounds = int(np.median(best_iterations))

print(f"\nFinal Cross-Validation (log-space)")
print(f"-----------------------------------")
print(f"Mean Validation R2: {np.mean(cv_scores):.4f}")
print(f"Validation R2 Std: {np.std(cv_scores):.4f}")
print(f"Selected Boosting Rounds: {optimal_rounds}")

best_xgb.set_params(
    n_estimators=optimal_rounds,
    early_stopping_rounds=None
)

best_xgb.fit(
    x_train_proc,
    y_train
)

y_train_pred_log = best_xgb.predict(x_train_proc)
y_pred_log = best_xgb.predict(x_test_proc)

log_train_r2 = r2_score(y_train, y_train_pred_log)
log_test_r2 = r2_score(y_test, y_pred_log)

y_train_pred = np.expm1(y_train_pred_log)
y_pred = np.expm1(y_pred_log)
y_train_raw = np.expm1(y_train)

train_r2 = r2_score(y_train_raw, y_train_pred)
test_r2 = r2_score(y_test_raw, y_pred)
mae = mean_absolute_error(y_test_raw, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_raw, y_pred))

print("\nXGBoost Optimized Results (log-space)")
print("--------------------------------------")
print(f"Training R2: {log_train_r2:.4f}")
print(f"Test R2:     {log_test_r2:.4f}")

print("\nXGBoost Optimized Results (original-space)")
print("--------------------------------------------")
print(f"Training R2: {train_r2:.4f}")
print(f"Test R2:     {test_r2:.4f}")
print(f"MAE:         {mae:.2f}")
print(f"RMSE:        {rmse:.2f}")


final_preprocessor = clone(preprocessor)
x_all_proc = final_preprocessor.fit_transform(x)

final_xgb = XGBRegressor(**best_xgb.get_params())
final_xgb.set_params(
    n_estimators=optimal_rounds,
    early_stopping_rounds=None
)

final_xgb.fit(x_all_proc, y)

final_pipeline = Pipeline(steps=[
    ('preprocessor', final_preprocessor),
    ('regressor', final_xgb)
])

os.makedirs("../saved_models", exist_ok=True)

joblib.dump(
    final_pipeline,
    "../saved_models/xgboost_regression.joblib"
)

print("\nExisting XGBoost model successfully updated using 100% of the data.")