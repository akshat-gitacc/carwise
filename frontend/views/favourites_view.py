"""
favourites_view.py — Favourites & Shortlist view for CarWise Streamlit frontend.
Allows users to view saved vehicles, toggle favourites via the top-right heart button,
inspect full specifications on single click, trigger red heart animation on double click,
and compare vehicles side-by-side. Adheres strictly to the dark automotive theme.
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


def _get_favourites(user_id: str) -> List[Dict[str, Any]]:
    try:
        r = requests.get(f"{API_BASE}/api/favourites", params={"user_id": user_id}, timeout=5)
        if r.status_code == 200:
            return r.json().get("items", [])
    except Exception:
        pass

    try:
        from backend.favourites.service import FavouritesService
        service = FavouritesService()
        res = service.get_shortlist(user_id)
        return [c.model_dump() for c in res.items]
    except Exception:
        return []


def _get_comparison(user_id: str, car_ids: Optional[List[int]] = None) -> Dict[str, Any]:
    try:
        params = {"user_id": user_id}
        if car_ids:
            params["car_ids"] = car_ids
        r = requests.get(f"{API_BASE}/api/favourites/compare", params=params, timeout=5)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass

    try:
        from backend.favourites.service import FavouritesService
        service = FavouritesService()
        res = service.compare_shortlist_cars(user_id, car_ids)
        return res.model_dump()
    except Exception:
        return {"cars": [], "comparison_metrics": []}


def render_favourites_view():
    user_id = st.session_state.get("user_id", "guest_user")

    st.markdown('<div class="hero-title">Saved Favourites</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-sub">Your personal shortlist of candidate vehicles. Compare specifications and review details anytime.</div>',
        unsafe_allow_html=True,
    )

    favourites = _get_favourites(user_id)

    if not favourites:
        st.markdown(
            '<div class="result-card" style="text-align:center;padding:2.5rem 1.5rem;">'
            '<div style="font-size:1.1rem;font-weight:600;color:var(--text);margin-bottom:0.5rem;">Your shortlist is currently empty</div>'
            '<div style="font-size:0.875rem;color:var(--text-muted);margin-bottom:1.5rem;">'
            'Save vehicles to your shortlist by clicking the heart button or double-clicking any car card in the Car Advisor or Explore discovery page.'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    tab_list, tab_compare = st.tabs(["Shortlisted Cars", "Compare Shortlist"])

    # ── TAB 1: SHORTLISTED CARS ───────────────────────────────────────────────
    with tab_list:
        st.markdown(
            f'<div class="section-label">{len(favourites)} Vehicles in Your Shortlist</div>',
            unsafe_allow_html=True,
        )

        for i, car in enumerate(favourites, start=1):
            render_car_card(
                car=car,
                user_id=user_id,
                key_prefix="fav",
                rank=i,
                show_match_score=False,
            )

        inject_double_click_heart_script()

    # ── TAB 2: COMPARE SHORTLIST ──────────────────────────────────────────────
    with tab_compare:
        st.markdown(
            '<div class="section-label">Side-by-Side Comparison of Shortlisted Vehicles</div>',
            unsafe_allow_html=True,
        )

        cmp_data = _get_comparison(user_id)
        cmp_cars = cmp_data.get("cars", [])
        cmp_metrics = cmp_data.get("comparison_metrics", [])

        if len(cmp_cars) < 2:
            st.info("You need at least 2 shortlisted vehicles to compare them side-by-side.")
        else:
            table_header = "<th>Specification</th>" + "".join(
                f"<th>{c['brand']} {c['car']}</th>" for c in cmp_cars
            )

            table_rows = ""
            for m in cmp_metrics:
                metric_name = m["metric"]
                vals = m["values"]
                row_cols = f'<td class="metric">{metric_name}</td>'
                for c in cmp_cars:
                    v = vals.get(str(c["id"]), vals.get(c["id"], "—"))
                    row_cols += f"<td>{v}</td>"
                table_rows += f"<tr>{row_cols}</tr>"

            st.markdown(
                f'<div class="card">'
                f'<table class="compare-table">'
                f'<thead><tr>{table_header}</tr></thead>'
                f'<tbody>{table_rows}</tbody>'
                f'</table>'
                f'</div>',
                unsafe_allow_html=True,
            )
