"""
backend.chatbot package — CarBot Conversational Advisory & RAG Engine.
Provides grounded local automotive advisory, specifications retrieval,
dealership lookups, EMI calculations, and multi-factor recommendation synthesis.
"""

from .car_data import (
    BUYING_TIPS,
    CARS,
    DEALERSHIPS,
    extract_preferences,
    format_car_summary,
    get_car_by_name,
    get_dealerships_for_brand_city,
    get_dealerships_for_city,
    parse_budget_lakh,
    recommend_cars,
)
from .chat_engine import (
    chat,
    chat_simple,
    estimate_emi,
)

__all__ = [
    "CARS",
    "DEALERSHIPS",
    "BUYING_TIPS",
    "extract_preferences",
    "format_car_summary",
    "get_car_by_name",
    "get_dealerships_for_brand_city",
    "get_dealerships_for_city",
    "parse_budget_lakh",
    "recommend_cars",
    "chat",
    "chat_simple",
    "estimate_emi",
]

