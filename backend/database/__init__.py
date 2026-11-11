"""
backend.database package
"""
from .models import (
    CarRecord,
    UserRecord,
    UserInteractionRecord,
    UserPreferenceRecord,
    UserShortlistRecord,
    RecommendationHistoryRecord,
)
from .db import (
    DatabaseManager,
    init_db,
    get_db_connection,
    get_db_path,
    DB_PATH,
)
from .init_db import populate_cars_table

__all__ = [
    "CarRecord",
    "UserRecord",
    "UserInteractionRecord",
    "UserPreferenceRecord",
    "UserShortlistRecord",
    "RecommendationHistoryRecord",
    "DatabaseManager",
    "init_db",
    "get_db_connection",
    "get_db_path",
    "DB_PATH",
    "populate_cars_table",
]

