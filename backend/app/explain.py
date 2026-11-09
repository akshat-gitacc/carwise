import numpy as np
import shap

LABELS = {
    "log_km": "Kilometres driven",
    "km_squared": "Kilometres driven (squared)",
    "km_per_year": "Yearly usage",
    "log_km_per_year": "Yearly usage (log)",
    "vehicle_age": "Vehicle age",
    "age_squared": "Vehicle age (squared)",
    "mileage_x_age": "Mileage x age",
    "mileage": "Fuel mileage",
    "mileage_per_power": "Mileage per power",
    "engine": "Engine size",
    "log_engine": "Engine size (log)",
    "max_power": "Max power",
    "log_power": "Max power (log)",
    "power_x_engine": "Power x engine",
    "power_per_litre": "Power per litre",
    "engine_power_ratio": "Engine-to-power ratio",
    "seats": "Seats",
    "seats_x_power": "Seats x power",
    "is_luxury": "Luxury brand",
    "is_automatic": "Automatic transmission",
    "is_diesel": "Diesel engine",
    "luxury_auto": "Luxury automatic",
    "brand": "Brand",
    "model": "Model",
    "fuel_type": "Fuel type",
    "seller_type": "Seller type",
    "transmission_type": "Transmission",
}


class PriceExplainer:
    def __init__(self, pipeline):
        self.preprocessor = pipeline.named_steps["preprocessor"]
        self.model = pipeline.named_steps["regressor"]

        self.feature_names = [
            n.split("__", 1)[-1] for n in self.preprocessor.get_feature_names_out()
        ]

        self.explainer = shap.TreeExplainer(self.model)

        base_log = float(np.ravel(self.explainer.expected_value)[0])
        self.base_price = float(np.expm1(base_log))

    def _label(self, name: str) -> str:
        return LABELS.get(name, name.replace("_", " ").capitalize())

    def explain(self, features_df, top_k: int = 8) -> list[dict]:
        X = self.preprocessor.transform(features_df)

     
        phi = self.explainer.shap_values(X)[0]

        order = np.argsort(-np.abs(phi))[:top_k]
        contributions = []
        for i in order:

            pct = (np.exp(phi[i]) - 1) * 100
            name = self.feature_names[i]
            contributions.append({
                "feature": self._label(name),
                "effect_pct": round(float(pct), 2),
                "direction": "up" if phi[i] >= 0 else "down",
            })
        return contributions