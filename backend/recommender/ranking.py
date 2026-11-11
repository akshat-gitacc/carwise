"""
ranking.py — Hybrid Ranking, Score Fusion, and Diversity-Aware Reranking.
Combines Content-based (50%), Collaborative (30%), and Knowledge-based (20%) scores,
with dynamic weight redistribution for cold start and MMR diversity reranking.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


class HybridRanker:
    """
    Fuses multi-modal recommendation scores and performs diversity-aware reranking.
    """

    DEFAULT_WEIGHTS = {
        "content": 0.50,
        "collaborative": 0.30,
        "knowledge": 0.20,
    }

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()

    def compute_hybrid_scores(
        self,
        candidate_df: pd.DataFrame,
        content_scores: pd.Series,
        knowledge_scores: pd.Series,
        collab_scores: Optional[Dict[int, float]] = None,
        collab_available: bool = False,
    ) -> pd.DataFrame:
        """
        Merges component scores into a final hybrid score.
        Normalizes all components to [0, 1].
        Dynamically redistributes collaborative weight if unavailable, preventing zero-penalty distortion.
        """
        if candidate_df.empty:
            return candidate_df

        df = candidate_df.copy()

        # 1. Align Content Score
        df["score_content"] = content_scores.reindex(df.index).fillna(0.5).clip(0.0, 1.0)

        # 2. Align Knowledge Score
        df["score_knowledge"] = knowledge_scores.reindex(df.index).fillna(0.5).clip(0.0, 1.0)

        # 3. Align Collaborative Score
        has_collab = collab_available and bool(collab_scores)
        if has_collab and collab_scores is not None:
            # Map by car id
            df["score_collab"] = df["id"].map(lambda cid: collab_scores.get(cid, 0.0)).fillna(0.0).clip(0.0, 1.0)
            # Use standard weights
            w_c = self.weights["content"]
            w_collab = self.weights["collaborative"]
            w_k = self.weights["knowledge"]
            hybrid_raw = (w_c * df["score_content"]) + (w_collab * df["score_collab"]) + (w_k * df["score_knowledge"])
        else:
            df["score_collab"] = np.nan
            # Dynamically redistribute collaborative weight across content and knowledge
            total_active_w = self.weights["content"] + self.weights["knowledge"]
            w_c = self.weights["content"] / total_active_w
            w_k = self.weights["knowledge"] / total_active_w
            hybrid_raw = (w_c * df["score_content"]) + (w_k * df["score_knowledge"])

        # Final scaled score [0, 100]
        df["hybrid_score"] = (hybrid_raw * 100.0).round(1)
        df["Score"] = df["hybrid_score"]  # For backwards compatibility with UI

        return df.sort_values("hybrid_score", ascending=False)

    def apply_diversity_reranking(
        self,
        df: pd.DataFrame,
        top_n: int = 10,
        brand_diversity_penalty: float = 0.15,
        max_per_brand: int = 2,
    ) -> pd.DataFrame:
        """
        Applies diversity-aware reranking to prevent a single brand or identical vehicle type
        from monopolizing recommendations.
        """
        if df.empty:
            return df

        selected_indices: List[int] = []
        brand_counts: Dict[str, int] = {}
        remaining_indices = list(df.index)

        while remaining_indices and len(selected_indices) < top_n:
            best_idx = None
            best_adjusted_score = -float("inf")

            for idx in remaining_indices:
                row = df.loc[idx]
                brand = str(row.get("brand", "")).lower()
                current_cnt = brand_counts.get(brand, 0)

                # If brand already reached max allowed in top N, apply heavier penalty
                if current_cnt >= max_per_brand:
                    penalty = brand_diversity_penalty * (current_cnt + 2)
                else:
                    penalty = brand_diversity_penalty * current_cnt

                adj_score = row["hybrid_score"] - (penalty * 100.0)

                if adj_score > best_adjusted_score:
                    best_adjusted_score = adj_score
                    best_idx = idx

            if best_idx is not None:
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)
                b = str(df.loc[best_idx].get("brand", "")).lower()
                brand_counts[b] = brand_counts.get(b, 0) + 1
            else:
                break

        return df.loc[selected_indices].reset_index(drop=True)

