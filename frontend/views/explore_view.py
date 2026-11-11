"""
explore_view.py — Personalized Car Discovery view for CarWise.
Contains 3 dedicated sections:
1. "You May Like" (primary personalized hybrid discovery with clear factual reasons)
2. "Discover More" (diversity-aware variety, alternative brands)
3. "Recently Viewed" (actual browsing history with quick revisit actions)

Card interactions:
- Top-right heart button: Turns red when liked and adds to favourites.
- Single-click on card: Expands full technical specifications drawer.
- Double-click on card: Floating opaque red heart animation + likes the car.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_root = str(Path(__file__).resolve().parent.parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

import requests
import streamlit as st

from .card_component import (
    get_user_liked_cars,
    inject_double_click_heart_script,
    render_car_card,
    toggle_car_like,
)

API_BASE = os.environ.get("CARWISE_API_URL", "http://localhost:8000")


def _fetch_explore_data(user_id: str) -> Dict[str, Any]:
    try:
        r = requests.get(f"{API_BASE}/api/recommendations/explore", params={"user_id": user_id}, timeout=8)
        if r.status_code == 200:
            return r.json().get("data", {})
    except Exception:
        pass

    try:
        from backend.recommender.personalized_recommender import PersonalizedExploreRecommender
        engine = PersonalizedExploreRecommender()
        return engine.get_explore_sections(user_id=user_id)
    except Exception:
        return {"you_may_like": [], "discover_more": [], "recently_viewed": []}


def render_explore_view():
    user_id = st.session_state.get("user_id", "guest_user")

    st.markdown('<div class="hero-title">Explore</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-sub">Personalized vehicle discovery continuously calibrated by your activity, preferences, and interests.</div>',
        unsafe_allow_html=True,
    )

    data = _fetch_explore_data(user_id)
    you_may_like = data.get("you_may_like", [])
    discover_more = data.get("discover_more", [])
    recently_viewed = data.get("recently_viewed", [])

    # ── SECTION 1: YOU MAY LIKE ───────────────────────────────────────────────
    st.markdown('<div class="section-label">You May Like</div>', unsafe_allow_html=True)
    st.markdown(" "
    )

    if you_may_like:
        for i, car in enumerate(you_may_like, start=1):
            render_car_card(
                car=car,
                user_id=user_id,
                key_prefix="exp_yml",
                rank=i,
                show_match_score=True,
            )
    else:
        st.markdown(
            '<div class="result-card" style="text-align:center;padding:1.5rem;color:var(--text-muted);">'
            'Interact with cars in Car Advisor or start searching to populate personalized recommendations.'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── SECTION 2: DISCOVER MORE ──────────────────────────────────────────────
    st.markdown('<div class="section-label">Discover More</div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:0.85rem;color:var(--text-muted);margin-bottom:1rem;">'
        'Explore alternative brands and diverse categories curated using diversity-aware ranking.'
        '</div>',
        unsafe_allow_html=True,
    )

    if discover_more:
        for i, car in enumerate(discover_more, start=1):
            render_car_card(
                car=car,
                user_id=user_id,
                key_prefix="exp_dm",
                rank=None,
                show_match_score=False,
            )
    else:
        st.markdown(
            '<div class="result-card" style="text-align:center;padding:1.5rem;color:var(--text-muted);">'
            'Discover more vehicles as your browsing history grows.'
            '</div>',
            unsafe_allow_html=True,
        )

    # ── SECTION 3: RECENTLY VIEWED ────────────────────────────────────────────
    if recently_viewed:
        st.markdown("---")
        st.markdown('<div class="section-label">Recently Viewed</div>', unsafe_allow_html=True)
        st.markdown(
            '<div style="font-size:0.85rem;color:var(--text-muted);margin-bottom:1rem;">'
            'Your recent browsing history across CarWise.'
            '</div>',
            unsafe_allow_html=True,
        )

        for i, car in enumerate(recently_viewed, start=1):
            render_car_card(
                car=car,
                user_id=user_id,
                key_prefix="exp_rv",
                rank=None,
                show_match_score=False,
            )

    # Inject double-click watcher script for floating opaque red heart animation
    inject_double_click_heart_script()
