"""
evaluation.py — Recommendation Evaluation Metrics and Offline Benchmark Suite.
Implements Precision@K, Recall@K, NDCG@K, Recommendation Diversity, Catalog Coverage,
and Constraint Satisfaction Rate.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def precision_at_k(recommended_ids: List[int], relevant_ids: Set[int], k: int = 5) -> float:
    """Calculates Precision@K = (relevant in top K) / K."""
    if k <= 0 or not recommended_ids:
        return 0.0
    top_k = recommended_ids[:k]
    hits = sum(1 for cid in top_k if cid in relevant_ids)
    return float(hits / k)


def recall_at_k(recommended_ids: List[int], relevant_ids: Set[int], k: int = 5) -> float:
    """Calculates Recall@K = (relevant in top K) / (total relevant)."""
    if not relevant_ids or not recommended_ids or k <= 0:
        return 0.0
    top_k = recommended_ids[:k]
    hits = sum(1 for cid in top_k if cid in relevant_ids)
    return float(hits / len(relevant_ids))


def ndcg_at_k(recommended_ids: List[int], relevant_ids: Set[int], k: int = 5) -> float:
    """
    Calculates Normalized Discounted Cumulative Gain at K (NDCG@K)
    with binary relevance judgments.
    """
    if k <= 0 or not recommended_ids or not relevant_ids:
        return 0.0

    top_k = recommended_ids[:k]
    dcg = 0.0
    for i, cid in enumerate(top_k):
        rel = 1.0 if cid in relevant_ids else 0.0
        dcg += rel / np.log2(i + 2)  # index starts at 0, so i+2 gives log2(rank + 1)

    # Ideal DCG (all top hits are relevant up to min(k, len(relevant_ids)))
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(relevant_ids))))
    if idcg == 0.0:
        return 0.0
    return float(dcg / idcg)


def recommendation_diversity(feature_vectors: np.ndarray) -> float:
    """
    Calculates intra-list diversity as the average pairwise cosine distance (1 - cosine_similarity)
    between recommended item feature vectors.
    """
    if feature_vectors is None or len(feature_vectors) <= 1:
        return 0.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sim_matrix = cosine_similarity(feature_vectors)
    n = len(feature_vectors)
    # Upper triangle without diagonal
    triu_indices = np.triu_indices(n, k=1)
    if len(triu_indices[0]) == 0:
        return 0.0

    pairwise_distances = 1.0 - sim_matrix[triu_indices]
    return float(np.mean(pairwise_distances))


def catalog_coverage(all_recommended_ids: Set[int], total_catalog_size: int) -> float:
    """Calculates the proportion of items in the catalog recommended at least once."""
    if total_catalog_size <= 0:
        return 0.0
    return float(len(all_recommended_ids) / total_catalog_size)


def constraint_satisfaction_rate(
    recommended_cars: List[Dict[str, Any]], constraints: Dict[str, Any]
) -> float:
    """
    Measures the percentage of recommended cars that satisfy all explicit hard constraints
    (budget range, fuel type, transmission, min mileage).
    """
    if not recommended_cars:
        return 0.0

    satisfied_count = 0
    b_min = constraints.get("budget_min", 0.0)
    b_max = constraints.get("budget_max", 9999.0)
    fuel = constraints.get("fuel_type")
    trans = constraints.get("transmission")
    min_m = constraints.get("min_mileage", 0.0)
    strictness = constraints.get("strictness", "strict")

    # If flexible, allow 15% budget margin
    margin = 0.15 if strictness == "flexible" else 0.0
    eff_min = b_min * (1.0 - margin)
    eff_max = b_max * (1.0 + margin) if b_max < 250.0 else 9999.0

    for car in recommended_cars:
        p_lakh = car.get("Price_Lakh")
        car_fuel = str(car.get("Fuel_Type_Simple", "")).lower()
        car_trans = str(car.get("Transmission_Simple", "")).lower()
        car_m = car.get("Mileage_Clean")

        # Check price
        if p_lakh is not None:
            if p_lakh < eff_min or p_lakh > eff_max:
                continue

        # Check fuel
        if fuel and fuel != "Any" and car_fuel != fuel.lower():
            continue

        # Check transmission
        if trans and trans != "Any" and car_trans != trans.lower():
            continue

        # Check mileage
        if min_m > 0 and car_m is not None:
            if car_m < min_m and car_fuel != "electric":
                continue

        satisfied_count += 1

    return float(satisfied_count / len(recommended_cars))


def run_offline_evaluation(hybrid_engine, test_scenarios: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """
    Runs benchmark evaluation across multiple realistic buyer profiles.
    Returns quantitative metric report.
    """
    if test_scenarios is None:
        test_scenarios = [
            {
                "name": "Budget City Commuter",
                "profile": {
                    "budget_min": 4.0, "budget_max": 10.0,
                    "fuel_type": "Petrol", "transmission": "Manual",
                    "min_mileage": 18.0, "priority": "mileage", "strictness": "strict"
                },
            },
            {
                "name": "Family Diesel Automatic",
                "profile": {
                    "budget_min": 12.0, "budget_max": 25.0,
                    "fuel_type": "Diesel", "transmission": "Automatic",
                    "min_mileage": 14.0, "priority": "balanced", "strictness": "strict"
                },
            },
            {
                "name": "Luxury Feature Rich",
                "profile": {
                    "budget_min": 35.0, "budget_max": 80.0,
                    "fuel_type": "Petrol", "transmission": "Automatic",
                    "min_mileage": 0.0, "priority": "feature_rich", "strictness": "flexible"
                },
            },
            {
                "name": "Modern Hybrid Powertrain",
                "profile": {
                    "budget_min": 15.0, "budget_max": 65.0,
                    "fuel_type": "Hybrid", "transmission": "Automatic",
                    "min_mileage": 16.0, "priority": "mileage", "strictness": "strict"
                },
            },
        ]

    total_coverage_ids: Set[int] = set()
    constraint_scores: List[float] = []
    diversity_scores: List[float] = []
    scenario_reports = []

    for sc in test_scenarios:
        recs = hybrid_engine.recommend(sc["profile"], top_n=5)
        rec_ids = [r["id"] for r in recs]
        total_coverage_ids.update(rec_ids)

        c_rate = constraint_satisfaction_rate(recs, sc["profile"])
        constraint_scores.append(c_rate)

        # Get vectors for diversity
        if recs:
            vec_indices = [
                hybrid_engine.content_recommender.df[hybrid_engine.content_recommender.df["id"] == cid].index[0]
                for cid in rec_ids
                if not hybrid_engine.content_recommender.df[hybrid_engine.content_recommender.df["id"] == cid].empty
            ]
            if len(vec_indices) > 1:
                vecs = hybrid_engine.content_recommender.car_matrix[vec_indices]
                div = recommendation_diversity(vecs)
            else:
                div = 0.0
        else:
            div = 0.0
        diversity_scores.append(div)

        scenario_reports.append({
            "scenario": sc["name"],
            "recommended_count": len(recs),
            "constraint_satisfaction": c_rate,
            "diversity": div,
            "top_car": f"{recs[0]['Brand']} {recs[0]['Car']}" if recs else "None",
        })

    total_catalog = len(hybrid_engine.content_recommender.df)
    cov = catalog_coverage(total_coverage_ids, total_catalog)

    return {
        "mean_constraint_satisfaction_rate": float(np.mean(constraint_scores)) if constraint_scores else 0.0,
        "mean_intra_list_diversity": float(np.mean(diversity_scores)) if diversity_scores else 0.0,
        "catalog_coverage_rate": cov,
        "unique_cars_recommended": len(total_coverage_ids),
        "total_catalog_size": total_catalog,
        "scenarios": scenario_reports,
    }
