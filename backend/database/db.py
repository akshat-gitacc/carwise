"""
db.py — SQLite Database access and session management for CarWise.
Handles persistent storage for users, cars, interactions, preferences, shortlists, and recommendations.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

from .models import (
    CarRecord,
    RecommendationHistoryRecord,
    UserInteractionRecord,
    UserPreferenceRecord,
    UserRecord,
    UserShortlistRecord,
)

DB_PATH = Path(__file__).resolve().parent / "carwise.db"
_local = threading.local()


def get_db_path() -> Path:
    return DB_PATH


@contextmanager
def get_db_connection(db_path: Optional[Path] = None) -> Generator[sqlite3.Connection, None, None]:
    path = db_path or DB_PATH
    conn = sqlite3.connect(str(path), timeout=30.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: Optional[Path] = None) -> None:
    """Initialize database tables and indexes."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()

        # 1. Users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                username TEXT,
                created_at TEXT NOT NULL,
                last_active TEXT NOT NULL
            );
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_session ON users(session_id);")

        # 2. Cars table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cars (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                brand TEXT NOT NULL,
                car TEXT NOT NULL,
                variant TEXT NOT NULL,
                price_lakh REAL,
                price_display TEXT NOT NULL,
                fuel_type TEXT NOT NULL,
                transmission TEXT NOT NULL,
                mileage REAL,
                displacement REAL,
                feature_score REAL DEFAULT 0.0,
                active_status TEXT NOT NULL,
                description TEXT,
                features_json TEXT,
                created_at TEXT NOT NULL
            );
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_cars_brand_car ON cars(brand, car);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_cars_price ON cars(price_lakh);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_cars_fuel ON cars(fuel_type);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_cars_trans ON cars(transmission);")

        # 3. User Interactions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                car_id INTEGER NOT NULL,
                interaction_type TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                session_id TEXT,
                metadata_json TEXT,
                FOREIGN KEY (car_id) REFERENCES cars(id) ON DELETE CASCADE
            );
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_inter_user ON user_interactions(user_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_inter_car ON user_interactions(car_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_inter_type ON user_interactions(interaction_type);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_inter_time ON user_interactions(timestamp);")

        # 4. User Preferences table (explicit and inferred)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_preferences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL UNIQUE,
                explicit_preferences_json TEXT,
                inferred_preferences_json TEXT,
                updated_at TEXT NOT NULL
            );
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_pref_user ON user_preferences(user_id);")

        # 5. User Shortlist / Favourites table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_shortlist (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                car_id INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(user_id, car_id),
                FOREIGN KEY (car_id) REFERENCES cars(id) ON DELETE CASCADE
            );
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_shortlist_user ON user_shortlist(user_id);")

        # 6. Recommendation History table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS recommendation_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                recommended_car_ids_json TEXT NOT NULL,
                recommendation_source TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_recs_user ON recommendation_history(user_id);")


# ===========================================================================
# Repository functions
# ===========================================================================

class DatabaseManager:
    """High-level abstraction for CarWise SQLite operations."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DB_PATH
        init_db(self.db_path)

    # ------------------ USERS ------------------
    def get_or_create_user(self, user_id: str, session_id: Optional[str] = None, username: Optional[str] = None) -> UserRecord:
        session_id = session_id or user_id
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
            row = cursor.fetchone()
            if row:
                return UserRecord(
                    id=row["id"],
                    session_id=row["session_id"],
                    username=row["username"],
                    created_at=row["created_at"],
                    last_active=row["last_active"],
                )
            user = UserRecord(id=user_id, session_id=session_id, username=username)
            cursor.execute(
                "INSERT INTO users (id, session_id, username, created_at, last_active) VALUES (?, ?, ?, ?, ?)",
                (user.id, user.session_id, user.username, user.created_at, user.last_active),
            )
            return user

    def update_user_activity(self, user_id: str) -> None:
        with get_db_connection(self.db_path) as conn:
            from datetime import datetime
            now = datetime.utcnow().isoformat()
            conn.execute("UPDATE users SET last_active = ? WHERE id = ?", (now, user_id))

    # ------------------ CARS ------------------
    def insert_car(self, car: CarRecord) -> int:
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO cars (
                    brand, car, variant, price_lakh, price_display,
                    fuel_type, transmission, mileage, displacement,
                    feature_score, active_status, description, features_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    car.brand, car.car, car.variant, car.price_lakh, car.price_display,
                    car.fuel_type, car.transmission, car.mileage, car.displacement,
                    car.feature_score, car.active_status, car.description,
                    json.dumps(car.features or {}), car.created_at
                )
            )
            return cursor.lastrowid

    def get_car_by_id(self, car_id: int) -> Optional[CarRecord]:
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM cars WHERE id = ?", (car_id,))
            row = cursor.fetchone()
            if not row:
                return None
            return self._row_to_car(row)

    def get_car_by_model(self, brand: str, car: str) -> Optional[CarRecord]:
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM cars WHERE LOWER(brand) = ? AND LOWER(car) = ? ORDER BY price_lakh ASC LIMIT 1",
                (brand.lower(), car.lower())
            )
            row = cursor.fetchone()
            if not row:
                return None
            return self._row_to_car(row)

    def get_all_cars(self, active_only: bool = False) -> List[CarRecord]:
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            if active_only:
                cursor.execute("SELECT * FROM cars WHERE LOWER(active_status) = 'active' ORDER BY id ASC")
            else:
                cursor.execute("SELECT * FROM cars ORDER BY id ASC")
            rows = cursor.fetchall()
            return [self._row_to_car(r) for r in rows]

    def count_cars(self) -> int:
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) as cnt FROM cars")
            row = cursor.fetchone()
            return int(row["cnt"]) if row else 0

    def _row_to_car(self, row: sqlite3.Row) -> CarRecord:
        feats = json.loads(row["features_json"]) if row["features_json"] else {}
        return CarRecord(
            id=row["id"],
            brand=row["brand"],
            car=row["car"],
            variant=row["variant"],
            price_lakh=row["price_lakh"],
            price_display=row["price_display"],
            fuel_type=row["fuel_type"],
            transmission=row["transmission"],
            mileage=row["mileage"],
            displacement=row["displacement"],
            feature_score=row["feature_score"],
            active_status=row["active_status"],
            description=row["description"],
            features=feats,
            created_at=row["created_at"],
        )

    # ------------------ INTERACTIONS ------------------
    def log_interaction(
        self,
        user_id: str,
        car_id: int,
        interaction_type: str,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        self.get_or_create_user(user_id, session_id=session_id)
        from datetime import datetime
        now = datetime.utcnow().isoformat()
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO user_interactions (
                    user_id, car_id, interaction_type, timestamp, session_id, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    car_id,
                    interaction_type,
                    now,
                    session_id or user_id,
                    json.dumps(metadata or {}),
                )
            )
            return cursor.lastrowid

    def get_user_interactions(
        self,
        user_id: str,
        limit: int = 100,
        interaction_types: Optional[List[str]] = None,
    ) -> List[UserInteractionRecord]:
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            if interaction_types:
                placeholders = ",".join("?" for _ in interaction_types)
                query = f"""
                    SELECT * FROM user_interactions
                    WHERE user_id = ? AND interaction_type IN ({placeholders})
                    ORDER BY timestamp DESC LIMIT ?
                """
                params = [user_id] + list(interaction_types) + [limit]
                cursor.execute(query, params)
            else:
                cursor.execute(
                    "SELECT * FROM user_interactions WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?",
                    (user_id, limit),
                )
            rows = cursor.fetchall()
            results = []
            for r in rows:
                meta = json.loads(r["metadata_json"]) if r["metadata_json"] else {}
                results.append(
                    UserInteractionRecord(
                        id=r["id"],
                        user_id=r["user_id"],
                        car_id=r["car_id"],
                        interaction_type=r["interaction_type"],
                        timestamp=r["timestamp"],
                        session_id=r["session_id"],
                        metadata=meta,
                    )
                )
            return results

    def get_all_interactions(self) -> List[UserInteractionRecord]:
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM user_interactions ORDER BY timestamp ASC")
            rows = cursor.fetchall()
            results = []
            for r in rows:
                meta = json.loads(r["metadata_json"]) if r["metadata_json"] else {}
                results.append(
                    UserInteractionRecord(
                        id=r["id"],
                        user_id=r["user_id"],
                        car_id=r["car_id"],
                        interaction_type=r["interaction_type"],
                        timestamp=r["timestamp"],
                        session_id=r["session_id"],
                        metadata=meta,
                    )
                )
            return results

    # ------------------ PREFERENCES ------------------
    def get_user_preferences(self, user_id: str) -> Optional[UserPreferenceRecord]:
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM user_preferences WHERE user_id = ?", (user_id,))
            row = cursor.fetchone()
            if not row:
                return None
            return UserPreferenceRecord(
                id=row["id"],
                user_id=row["user_id"],
                explicit_preferences=json.loads(row["explicit_preferences_json"] or "{}"),
                inferred_preferences=json.loads(row["inferred_preferences_json"] or "{}"),
                updated_at=row["updated_at"],
            )

    def save_explicit_preferences(self, user_id: str, prefs: Dict[str, Any]) -> None:
        self.get_or_create_user(user_id)
        existing = self.get_user_preferences(user_id)
        from datetime import datetime
        now = datetime.utcnow().isoformat()
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            if existing:
                cursor.execute(
                    """
                    UPDATE user_preferences
                    SET explicit_preferences_json = ?, updated_at = ?
                    WHERE user_id = ?
                    """,
                    (json.dumps(prefs), now, user_id),
                )
            else:
                cursor.execute(
                    """
                    INSERT INTO user_preferences (user_id, explicit_preferences_json, inferred_preferences_json, updated_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (user_id, json.dumps(prefs), json.dumps({}), now),
                )

    def save_inferred_preferences(self, user_id: str, inferred: Dict[str, Any]) -> None:
        self.get_or_create_user(user_id)
        existing = self.get_user_preferences(user_id)
        from datetime import datetime
        now = datetime.utcnow().isoformat()
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            if existing:
                cursor.execute(
                    """
                    UPDATE user_preferences
                    SET inferred_preferences_json = ?, updated_at = ?
                    WHERE user_id = ?
                    """,
                    (json.dumps(inferred), now, user_id),
                )
            else:
                cursor.execute(
                    """
                    INSERT INTO user_preferences (user_id, explicit_preferences_json, inferred_preferences_json, updated_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (user_id, json.dumps({}), json.dumps(inferred), now),
                )

    # ------------------ SHORTLIST / FAVOURITES ------------------
    def add_to_shortlist(self, user_id: str, car_id: int) -> bool:
        """Add a car to user's favourites/shortlist. Returns True if newly added, False if duplicate."""
        self.get_or_create_user(user_id)
        from datetime import datetime
        now = datetime.utcnow().isoformat()
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id FROM user_shortlist WHERE user_id = ? AND car_id = ?",
                (user_id, car_id),
            )
            if cursor.fetchone():
                return False
            cursor.execute(
                "INSERT INTO user_shortlist (user_id, car_id, created_at) VALUES (?, ?, ?)",
                (user_id, car_id, now),
            )
            return True

    def remove_from_shortlist(self, user_id: str, car_id: int) -> bool:
        """Remove a car from user's favourites/shortlist. Returns True if removed."""
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM user_shortlist WHERE user_id = ? AND car_id = ?",
                (user_id, car_id),
            )
            return cursor.rowcount > 0

    def get_user_shortlist(self, user_id: str) -> List[CarRecord]:
        """Get all shortlisted cars for a user."""
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT c.* FROM cars c
                INNER JOIN user_shortlist s ON c.id = s.car_id
                WHERE s.user_id = ?
                ORDER BY s.created_at DESC
                """,
                (user_id,),
            )
            rows = cursor.fetchall()
            return [self._row_to_car(r) for r in rows]

    def is_car_shortlisted(self, user_id: str, car_id: int) -> bool:
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id FROM user_shortlist WHERE user_id = ? AND car_id = ?",
                (user_id, car_id),
            )
            return cursor.fetchone() is not None

    # ------------------ RECOMMENDATION HISTORY ------------------
    def log_recommendation_history(
        self, user_id: str, recommended_car_ids: List[int], source: str
    ) -> int:
        self.get_or_create_user(user_id)
        from datetime import datetime
        now = datetime.utcnow().isoformat()
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO recommendation_history (
                    user_id, recommended_car_ids_json, recommendation_source, created_at
                ) VALUES (?, ?, ?, ?)
                """,
                (user_id, json.dumps(recommended_car_ids), source, now),
            )
            return cursor.lastrowid

    def save_advisor_session(
        self, user_id: str, profile: Dict[str, Any], results: List[Dict[str, Any]]
    ) -> None:
        """Saves latest advisor form profile and recommended results for seamless reload on refresh."""
        self.get_or_create_user(user_id)
        self.save_explicit_preferences(user_id, profile)
        car_ids = [r.get("id") for r in results if r.get("id")]
        self.log_recommendation_history(user_id, car_ids, source="advisor_session")
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS advisor_saved_results (
                    user_id TEXT PRIMARY KEY,
                    profile_json TEXT NOT NULL,
                    results_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
            """)
            from datetime import datetime
            now = datetime.utcnow().isoformat()
            cursor.execute("""
                INSERT INTO advisor_saved_results (user_id, profile_json, results_json, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    profile_json = excluded.profile_json,
                    results_json = excluded.results_json,
                    updated_at = excluded.updated_at;
            """, (user_id, json.dumps(profile), json.dumps(results), now))

    def get_advisor_session(
        self, user_id: str
    ) -> Optional[Tuple[Dict[str, Any], List[Dict[str, Any]]]]:
        """Retrieves user's last saved advisor profile and recommendation results."""
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS advisor_saved_results (
                    user_id TEXT PRIMARY KEY,
                    profile_json TEXT NOT NULL,
                    results_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
            """)
            cursor.execute(
                "SELECT profile_json, results_json FROM advisor_saved_results WHERE user_id = ?",
                (user_id,)
            )
            row = cursor.fetchone()
            if not row:
                return None
            try:
                prof = json.loads(row["profile_json"])
                res = json.loads(row["results_json"])
                return prof, res
            except Exception:
                return None

