import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from backend.config import DATA, MODELS, CLASSES, SEED
from backend.features import build_matrix


def train_xgb():
    df = pd.read_csv(DATA / "synthetic_cars.csv")
    y = df["quality"].map({c: i for i, c in enumerate(CLASSES)})
    X = build_matrix(df)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    model = XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        objective="multi:softprob", eval_metric="mlogloss", random_state=SEED,
    )
    model.fit(Xtr, ytr)

    pred = model.predict(Xte)
    print(f"XGBoost  accuracy={accuracy_score(yte, pred):.3f}  macro_f1={f1_score(yte, pred, average='macro'):.3f}")
    print(classification_report(yte, pred, target_names=CLASSES, zero_division=0))
    print("Accuracy on synthetic data does not reflect real-world performance.\n")

    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "columns": list(X.columns)}, MODELS / "xgb.joblib")
    print("Saved models/xgb.joblib")


if __name__ == "__main__":
    train_xgb()