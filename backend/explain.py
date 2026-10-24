import json
import os

import joblib
import numpy as np
import pandas as pd
import shap

from backend.config import MODELS, CLASSES
from backend.features import build_matrix

_art = None


def _load():
    global _art
    if _art is None:
        _art = joblib.load(MODELS / "xgb.joblib")
    return _art


def _pretty(name):
    return name.replace("_", " ")


def _pred_contribs(explainer, X, pred):
    sv = explainer.shap_values(X)
    if isinstance(sv, list):                 # list of (n, features), one per class
        return np.asarray(sv)[pred][0]
    sv = np.asarray(sv)
    if sv.shape[0] == 1:                     # (n, features, classes)
        return sv[0, :, pred]
    return sv[pred, 0, :]                    # (classes, n, features)


def assess(car, top_k=4):
    art = _load()
    model, cols = art["model"], art["columns"]

    X = build_matrix(pd.DataFrame([car]), cols)
    proba = model.predict_proba(X)[0]
    pred = int(np.argmax(proba))

    explainer = shap.TreeExplainer(model)
    c = _pred_contribs(explainer, X, pred)
    order = np.argsort(c)

    def item(i):
        v = X.iloc[0, i]
        return {
            "feature": cols[i],
            "value": None if pd.isna(v) else round(float(v), 3),
            "contribution": round(float(c[i]), 3),
        }

    result = {
        "label": CLASSES[pred],
        "probability": round(float(proba[pred]), 3),
        "probabilities": {cl: round(float(p), 3) for cl, p in zip(CLASSES, proba)},
        "positives": [item(i) for i in order[::-1] if c[i] > 0][:top_k],
        "negatives": [item(i) for i in order if c[i] < 0][:top_k],
    }
    result["explanation"] = _llm(result) or _template(result)
    return result


SYSTEM = (
    "You explain a used-car quality prediction. Use ONLY the JSON facts provided. "
    "Do not invent vehicle history or causes. Name the predicted class, its probability, "
    "and the strongest positive and negative factors. State that the model was trained on "
    "synthetic data and that SHAP shows model behaviour, not real-world causation. "
    "Recommend a professional inspection. Keep it under 120 words, plain English."
)


def _llm(r):
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        return None
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=key)
        msg = client.messages.create(
            model=os.getenv("CARWISE_LLM_MODEL", "claude-haiku-5-5"),
            max_tokens=350,
            system=SYSTEM,
            messages=[{"role": "user", "content": json.dumps(r)}],
        )
        return msg.content[0].text.strip()
    except Exception as e:
        print("LLM explanation failed, using template:", e)
        return None


def _template(r):
    parts = [f"The model rates this car as {r['label']} with {r['probability']:.0%} probability."]
    if r["positives"]:
        parts.append("Factors helping this rating: " + ", ".join(
            f"{_pretty(d['feature'])} ({d['value']})" for d in r["positives"]) + ".")
    if r["negatives"]:
        parts.append("Factors holding it back: " + ", ".join(
            f"{_pretty(d['feature'])} ({d['value']})" for d in r["negatives"]) + ".")
    parts.append("This is a model estimate trained on synthetic demonstration data, not a verdict "
                 "on this vehicle. Get a professional inspection before buying.")
    return "\n\n".join(parts)