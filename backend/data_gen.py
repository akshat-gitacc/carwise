import numpy as np
import pandas as pd
from backend.config import SEED, N_ROWS, CLASSES, DATA


def _z(x):
    x = np.asarray(x, dtype=float)
    return (x - x.mean()) / x.std()


def generate(n=N_ROWS, seed=SEED):
    rng = np.random.default_rng(seed)

    age = rng.integers(0, 20, n)
    mileage = np.clip(age * 12000 + rng.normal(0, 25000, n), 0, None).round()
    owners = np.clip(rng.poisson(1 + age / 8), 1, 6)
    service = np.clip(rng.beta(4, 2, n) - 0.03 * owners, 0, 1)
    months_since = np.clip(rng.exponential(8, n) + age * 0.5, 0, None).round()
    accidents = rng.poisson(0.2 + 0.05 * age + 0.3 * (owners > 3))
    safety = rng.integers(1, 6, n)
    seats = rng.choice([4, 5, 7], n, p=[.1, .8, .1])
    doors = rng.choice([2, 4, 5], n, p=[.1, .5, .4])
    airbags = rng.choice([2, 4, 6, 8], n)
    warranty = np.clip(rng.integers(0, 36, n) - 3 * age, 0, None)

    fuel = rng.choice(["petrol", "diesel", "electric", "hybrid"], n, p=[.55, .25, .10, .10])
    engine_cc = np.where(fuel == "electric", 0, rng.choice([999, 1198, 1498, 1995, 2494], n))
    body = rng.choice(["steel", "aluminium", "composite"], n, p=[.80, .15, .05])
    country = rng.choice(
        ["Japan", "Germany", "India", "USA", "Korea", "China", "Other"],
        n, p=[.20, .15, .20, .10, .15, .10, .10],
    )
    transmission = rng.choice(["manual", "automatic"], n, p=[.6, .4])
    drive = rng.choice(["FWD", "RWD", "AWD"], n, p=[.7, .2, .1])

    latent = _z(-0.8 * _z(age) - 0.8 * _z(mileage) - 0.4 * _z(owners)
                + 0.6 * _z(service) - 0.4 * _z(months_since)
                - 0.5 * _z(accidents) + 0.3 * _z(safety))

    score = latent + rng.normal(0, 0.6, n)
    edges = np.quantile(score, [0.25, 0.5, 0.75])
    quality = np.array(CLASSES)[np.digitize(score, edges)]

    def visual(noise):
        return np.clip(60 + 15 * latent + rng.normal(0, noise, n), 0, 100).round(1)

    df = pd.DataFrame({
        "age_years": age,
        "mileage_km": mileage,
        "previous_owners": owners,
        "service_completeness": service,
        "months_since_service": months_since,
        "accident_count": accidents,
        "engine_cc": engine_cc,
        "safety_rating": safety,
        "seats": seats,
        "doors": doors,
        "airbags": airbags,
        "warranty_remaining_months": warranty,
        "fuel_type": fuel,
        "body_material": body,
        "country_of_manufacture": country,
        "transmission": transmission,
        "drive_type": drive,
        "exterior_damage_score": visual(12),
        "paint_condition_score": visual(12),
        "engine_condition_score": visual(15),
    })

    df.loc[rng.random(n) < 0.05, "service_completeness"] = np.nan
    df.loc[rng.random(n) < 0.02, "mileage_km"] = np.nan

    df["quality"] = quality
    return df


def save():
    df = generate()
    DATA.mkdir(parents=True, exist_ok=True)
    path = DATA / "synthetic_cars.csv"
    df.to_csv(path, index=False)
    print(f"Saved {len(df)} rows to {path}")
    print(df["quality"].value_counts())


if __name__ == "__main__":
    save()