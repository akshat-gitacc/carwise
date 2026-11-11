"""
backend.recommender package
"""

from .preprocessing import CarPreprocessor, clean_fuel, clean_transmission
from .feature_engineering import CarFeatureEngineer
from .profile_generator import DynamicProfileGenerator, SchemaInspector
from .content_based import ContentBasedRecommender
from .collaborative_filtering import CollaborativeFilteringRecommender
from .knowledge_based import KnowledgeBasedFilter
from .ranking import HybridRanker
from .hybrid_recommender import HybridRecommendationEngine
from .interaction_tracker import InteractionTracker
from .preference_inference import PreferenceInferenceEngine
from .personalized_recommender import PersonalizedExploreRecommender
from .evaluation import (
    precision_at_k,
    recall_at_k,
    ndcg_at_k,
    recommendation_diversity,
    catalog_coverage,
    constraint_satisfaction_rate,
    run_offline_evaluation,
)

__all__ = [
    "CarPreprocessor",
    "clean_fuel",
    "clean_transmission",
    "CarFeatureEngineer",
    "DynamicProfileGenerator",
    "SchemaInspector",
    "ContentBasedRecommender",
    "CollaborativeFilteringRecommender",
    "KnowledgeBasedFilter",
    "HybridRanker",
    "HybridRecommendationEngine",
    "InteractionTracker",
    "PreferenceInferenceEngine",
    "PersonalizedExploreRecommender",
    "precision_at_k",
    "recall_at_k",
    "ndcg_at_k",
    "recommendation_diversity",
    "catalog_coverage",
    "constraint_satisfaction_rate",
    "run_offline_evaluation",
]

