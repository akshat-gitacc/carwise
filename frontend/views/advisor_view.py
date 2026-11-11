"""
advisor_view.py — Car Advisor view for the CarWise Streamlit frontend.
Powered by the Hybrid AI/ML Recommendation Engine.

Features:
- All preference questions displayed at once (no pagination, no step toggles).
- Dynamic preference options derived from actual database attributes.
- Interactive car cards with:
  - Small heart-shaped button in top right corner (turns red when liked and adds to favourites).
  - Single click on card expands/toggles full specifications.
  - Double click on card triggers an opaque red heart animation and likes the car.
- Preserved head-to-head comparison, authorized dealership directory, and Gemini advisory chat.
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


@st.cache_data(ttl=300)
def _fetch_meta() -> Optional[Dict]:
    try:
        r = requests.get(f"{API_BASE}/api/advisor/meta", timeout=5)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass

    try:
        from backend.advisor import (
            get_brands,
            get_dealership_cities,
            get_fuel_types,
            get_transmissions,
            get_unique_car_count,
        )
        return {
            "unique_cars": get_unique_car_count(),
            "brands": ["Any"] + get_brands(),
            "fuel_types": ["Any"] + get_fuel_types(),
            "transmissions": ["Any"] + get_transmissions(),
            "dealership_cities": ["Any"] + get_dealership_cities(),
        }
    except Exception:
        return None


def _fetch_models(brand: str) -> List[str]:
    try:
        r = requests.get(f"{API_BASE}/api/advisor/models", params={"brand": brand}, timeout=5)
        if r.status_code == 200:
            return r.json().get("models", [])
    except Exception:
        pass

    try:
        from backend.advisor import get_models_for_brand
        return get_models_for_brand(brand)
    except Exception:
        return []


def _generate_hybrid_recommendations(profile_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    # In-process execution first for 100% reliability and speed
    try:
        from backend.database import DatabaseManager
        from backend.recommender.hybrid_recommender import HybridRecommendationEngine
        from backend.recommender.profile_generator import DynamicProfileGenerator
        db = DatabaseManager()
        gen = DynamicProfileGenerator(db)
        validated = gen.validate_and_build_profile(profile_payload)
        user_id = profile_payload.get("user_id", "guest_user")
        db.save_explicit_preferences(user_id, validated)
        engine = HybridRecommendationEngine(db)
        return engine.recommend(profile=validated, user_id=user_id, top_n=profile_payload.get("top_n", 6))
    except Exception:
        pass

    try:
        r = requests.post(f"{API_BASE}/recommendations/generate", json=profile_payload, timeout=8)
        if r.status_code == 200:
            return r.json().get("results", [])
    except Exception:
        pass

    return []


def _fetch_compare(brand_a: str, model_a: str, brand_b: str, model_b: str) -> Dict:
    try:
        from backend.advisor import compare_cars
        a, b = compare_cars(brand_a, model_a, brand_b, model_b)
        return {"car_a": a, "car_b": b}
    except Exception:
        try:
            r = requests.get(
                f"{API_BASE}/api/advisor/compare",
                params={"brand_a": brand_a, "model_a": model_a, "brand_b": brand_b, "model_b": model_b},
                timeout=5,
            )
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        return {"car_a": {}, "car_b": {}}


def _fetch_dealerships(city: Optional[str] = None, brand: Optional[str] = None) -> List[Dict]:
    try:
        from backend.advisor import get_dealerships
        return get_dealerships(city=city, brand=brand)
    except Exception:
        try:
            params = {}
            if city and city != "Any":
                params["city"] = city
            if brand and brand != "Any":
                params["brand"] = brand
            r = requests.get(f"{API_BASE}/api/advisor/dealerships", params=params, timeout=5)
            if r.status_code == 200:
                return r.json().get("dealerships", [])
        except Exception:
            pass
        return []


def _call_gemini_advisor(
    message: str,
    history: List[Dict],
    context_cars: List[Dict],
    city: Optional[str],
    gemini_key: Optional[str],
) -> str:
    try:
        from backend.advisor import call_gemini_chat
        return call_gemini_chat(
            api_key=gemini_key,
            message=message,
            history=history,
            context_cars=context_cars,
            city=city,
        )
    except Exception as e:
        return f"Could not reach advisory service: {e}"


def render_advisor_view():
    user_id = st.session_state.get("user_id", "guest_user")
    meta = _fetch_meta()
    unique_cars = meta.get("unique_cars", 741) if meta else 741
    brands = meta["brands"] if meta else ["Any"]
    fuels = meta["fuel_types"] if meta else ["Any", "Petrol", "Diesel", "CNG", "Electric", "Hybrid"]
    transmissions = meta["transmissions"] if meta else ["Any", "Manual", "Automatic"]
    cities = meta.get("dealership_cities", ["Any", "Bangalore", "Chennai", "Delhi NCR", "Hyderabad", "Mumbai", "Pune"]) if meta else ["Any"]

    st.markdown('<div class="hero-title">Car Advisor</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="hero-sub">Matching you to your ideal car.</div>',
        unsafe_allow_html=True,
    )

    tab_recommend, tab_compare, tab_dealers, tab_emi, tab_chat = st.tabs([
        "Recommend", "Compare", "Dealerships", "EMI Calculator", "Advisor Chat"
    ])

    # ── TAB 1: RECOMMEND (ALL QUESTIONS SHOWN AT ONCE) ────────────────────────
    with tab_recommend:
        # Restore saved advisor recommendations and profile from SQLite on refresh
        if "adv_results" not in st.session_state or not st.session_state["adv_results"]:
            st.session_state["adv_results"] = []
            try:
                from backend.database import DatabaseManager
                saved = DatabaseManager().get_advisor_session(user_id)
                if saved:
                    saved_prof, saved_res = saved
                    st.session_state["adv_results"] = saved_res
                    st.session_state["adv_saved_profile"] = saved_prof
            except Exception:
                pass

        saved_profile = st.session_state.get("adv_saved_profile", {})

        # ── 1. Budget Range ───────────────────────────────────────────────────
        st.markdown('<div class="section-label">1. Budget Range</div>', unsafe_allow_html=True)
        price_step_values = [
            3.0, 5.0, 8.0, 10.0, 12.0, 15.0, 18.0, 20.0, 25.0, 30.0, 40.0,
            50.0, 75.0, 100.0, 150.0, 200.0, 250.0
        ]

        def _fmt_slider_price(val: float) -> str:
            if val >= 250.0:
                return "2.5cr+"
            if val >= 100.0:
                cr = val / 100.0
                return f"{cr:.1f} Cr" if cr % 1 != 0 else f"{int(cr)} Cr"
            return f"{int(val)} L"

        def_b_min = float(saved_profile.get("budget_min", 5.0))
        def_b_max = float(saved_profile.get("budget_max", 25.0))
        if def_b_min not in price_step_values:
            def_b_min = 5.0
        if def_b_max not in price_step_values:
            def_b_max = 25.0

        budget_min, budget_max = st.select_slider(
            "Select budget range",
            options=price_step_values,
            value=(def_b_min, def_b_max),
            format_func=_fmt_slider_price,
            label_visibility="collapsed",
            key="adv_budget_scroller",
        )

        st.markdown("---")

        # ── 2. Powertrain & Gearbox ───────────────────────────────────────────
        st.markdown('<div class="section-label">2. Powertrain & Manufacturer</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        def_fuel = saved_profile.get("fuel_type") or "Any"
        fuel_idx = fuels.index(def_fuel) if def_fuel in fuels else 0

        def_trans = saved_profile.get("transmission") or "Any"
        trans_idx = transmissions.index(def_trans) if def_trans in transmissions else 0

        def_brand = saved_profile.get("brand") or "Any"
        brand_idx = brands.index(def_brand) if def_brand in brands else 0

        with c1:
            sel_fuel = st.selectbox("Fuel type", fuels, index=fuel_idx, key="adv_fuel")
        with c2:
            sel_trans = st.selectbox("Transmission", transmissions, index=trans_idx, key="adv_trans")
        with c3:
            sel_brand = st.selectbox("Preferred Manufacturer", brands, index=brand_idx, key="adv_brand")

        st.markdown("---")

        # ── 3. Performance & Fuel Economy ─────────────────────────────────────
        st.markdown('<div class="section-label">3. Fuel Economy & Performance</div>', unsafe_allow_html=True)
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            min_mileage = st.slider(
                "Minimum Fuel Economy (kmpl)",
                min_value=0.0,
                max_value=30.0,
                value=0.0,
                step=1.0,
                key="adv_min_mileage_slider",
            )
        with col_m2:
            disp_options = [
                "Any",
                "Compact City (< 1200 cc)",
                "Balanced Everyday (1200 - 1800 cc)",
                "High Performance (1800+ cc)",
            ]
            sel_disp = st.selectbox(
                "Engine Performance Class",
                disp_options,
                index=0,
                key="adv_disp_class",
            )

        st.markdown("---")

        # ── 4. Premium Equipment & Amenities ──────────────────────────────────
        st.markdown('<div class="section-label">4. Desired Equipment Features</div>', unsafe_allow_html=True)
        all_feature_options = [
            "Sunroof / Moonroof",
            "Ventilated Seats",
            "Cruise Control",
            "Wireless Charger",
            "Rear AC Vents",
            "Hill Hold Control",
            "Heads Up Display",
        ]
        sel_features = st.multiselect(
            "Select desired features",
            options=all_feature_options,
            default=[],
            label_visibility="collapsed",
            key="adv_features_multiselect",
        )

        st.markdown("---")

        # ── 5. Optimization Priority & Budget Strictness ──────────────────────
        st.markdown('<div class="section-label">5. Optimization Priority & Constraint Strictness</div>', unsafe_allow_html=True)
        p_col1, p_col2 = st.columns(2)
        with p_col1:
            priority_options = {
                "Balanced (Equal weighting)": "balanced",
                "Fuel Economy (Best kmpl first)": "mileage",
                "Budget Affordability (Best value for money)": "price",
                "Feature Rich (Maximum tech & luxury)": "feature_rich",
                "Engine Performance (Power & displacement)": "performance",
            }
            chosen_priority_label = st.selectbox(
                "Optimization Priority",
                list(priority_options.keys()),
                index=0,
                key="adv_priority_select",
            )
            chosen_priority = priority_options[chosen_priority_label]

        with p_col2:
            strictness_options = {
                "Strict (Hard filter strictly within budget)": "strict",
                "Flexible (Allow up to 15% budget stretch for exceptional matches)": "flexible",
            }
            chosen_strictness_label = st.selectbox(
                "Budget Strictness",
                list(strictness_options.keys()),
                index=0,
                key="adv_strictness_select",
            )
            chosen_strictness = strictness_options[chosen_strictness_label]

        st.markdown("")

        # ── Action Button ─────────────────────────────────────────────────────
        if st.button("Find Matching Cars", key="adv_btn_recommend", use_container_width=True):
            with st.spinner("Executing Hybrid AI/ML Recommendation Engine..."):
                payload = {
                    "user_id": user_id,
                    "budget": [budget_min, budget_max],
                    "budget_min": budget_min,
                    "budget_max": budget_max,
                    "fuel_type": sel_fuel if sel_fuel != "Any" else None,
                    "transmission": sel_trans if sel_trans != "Any" else None,
                    "brand": sel_brand if sel_brand != "Any" else None,
                    "min_mileage": min_mileage,
                    "displacement_preference": sel_disp,
                    "features": sel_features,
                    "priority": chosen_priority,
                    "strictness": chosen_strictness,
                    "top_n": 6,
                }
                recs = _generate_hybrid_recommendations(payload)
                st.session_state["adv_results"] = recs
                st.session_state["adv_saved_profile"] = payload
                try:
                    from backend.database import DatabaseManager
                    db = DatabaseManager()
                    db.save_advisor_session(user_id, payload, recs)
                    db.save_explicit_preferences(user_id, payload)
                except Exception:
                    pass

        # ── Recommendation Results Display ────────────────────────────────────
        recs = st.session_state.get("adv_results", [])
        if recs:
            st.markdown("---")
            st.markdown(
                f'<div class="section-label">{len(recs)} Top Matches Ranked by Hybrid AI/ML Engine</div>',
                unsafe_allow_html=True,
            )

            # Render each car with interactive card component
            for i, car in enumerate(recs, start=1):
                render_car_card(
                    car=car,
                    user_id=user_id,
                    key_prefix="adv",
                    rank=i,
                    show_match_score=True,
                )

            # Inject double-click heart animation watcher
            inject_double_click_heart_script()

    # ── TAB 2: COMPARE ────────────────────────────────────────────────────────
    with tab_compare:
        st.markdown('<div class="section-label">Select Two Models for Head-to-Head Comparison</div>', unsafe_allow_html=True)
        active_brands = [b for b in brands if b != "Any"]
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown('<div class="section-label">Car A</div>', unsafe_allow_html=True)
            cmp_b_a = st.selectbox("Manufacturer A", active_brands, index=0, key="cmp_ba_sel", label_visibility="collapsed")
            mods_a = _fetch_models(cmp_b_a)
            cmp_m_a = st.selectbox("Model A", mods_a, index=0 if mods_a else 0, key="cmp_ma_sel", label_visibility="collapsed")

        with col_b:
            st.markdown('<div class="section-label">Car B</div>', unsafe_allow_html=True)
            def_b_idx = 1 if len(active_brands) > 1 else 0
            cmp_b_b = st.selectbox("Manufacturer B", active_brands, index=def_b_idx, key="cmp_bb_sel", label_visibility="collapsed")
            mods_b = _fetch_models(cmp_b_b)
            cmp_m_b = st.selectbox("Model B", mods_b, index=0 if mods_b else 0, key="cmp_mb_sel", label_visibility="collapsed")

        st.markdown("")
        if st.button("Compare Specifications", key="adv_btn_compare", use_container_width=True):
            st.session_state["adv_cmp_data"] = _fetch_compare(cmp_b_a, cmp_m_a, cmp_b_b, cmp_m_b)

        cmp_data = st.session_state.get("adv_cmp_data")
        if cmp_data:
            ca, cb = cmp_data.get("car_a", {}), cmp_data.get("car_b", {})
            if ca and cb:
                st.markdown("---")

                def _win_class(val_a, val_b, higher_is_better):
                    if val_a is None or val_b is None or val_a == "—" or val_b == "—":
                        return "", ""
                    try:
                        na, nb = float(val_a), float(val_b)
                        if higher_is_better:
                            return ("compare-win", "compare-lose") if na >= nb else ("compare-lose", "compare-win")
                        else:
                            return ("compare-win", "compare-lose") if na <= nb else ("compare-lose", "compare-win")
                    except Exception:
                        return "", ""

                comp_rows = [
                    ("Median Price", ca.get("Price_Display"), cb.get("Price_Display"), ca.get("Price_Lakh"), cb.get("Price_Lakh"), False),
                    ("Fuel Efficiency", f"{ca.get('Mileage')} kmpl" if ca.get('Mileage') != '—' else '—',
                     f"{cb.get('Mileage')} kmpl" if cb.get('Mileage') != '—' else '—',
                     ca.get("Mileage"), cb.get("Mileage"), True),
                    ("Displacement", f"{ca.get('Displacement')} cc" if ca.get('Displacement') != '—' else '—',
                     f"{cb.get('Displacement')} cc" if cb.get('Displacement') != '—' else '—',
                     ca.get("Displacement"), cb.get("Displacement"), True),
                    ("Fuel Powertrains", ca.get("Fuel_Types", "—"), cb.get("Fuel_Types", "—"), None, None, None),
                    ("Transmissions", ca.get("Transmissions", "—"), cb.get("Transmissions", "—"), None, None, None),
                    ("Sunroof / Moonroof", ca.get("Sunroof", "No"), cb.get("Sunroof", "No"), None, None, None),
                    ("Cruise Control", ca.get("Cruise_Control", "No"), cb.get("Cruise_Control", "No"), None, None, None),
                    ("Ventilated Seats", ca.get("Ventilated_Seats", "No"), cb.get("Ventilated_Seats", "No"), None, None, None),
                    ("Wireless Charger", ca.get("Wireless_Charger", "No"), cb.get("Wireless_Charger", "No"), None, None, None),
                    ("Rear AC", ca.get("Rear_AC", "No"), cb.get("Rear_AC", "No"), None, None, None),
                    ("Heads Up Display", ca.get("HUD", "No"), cb.get("HUD", "No"), None, None, None),
                ]

                table_body = ""
                for label, fmt_a, fmt_b, num_a, num_b, hib in comp_rows:
                    if hib is not None and num_a is not None and num_b is not None:
                        cl_a, cl_b = _win_class(num_a, num_b, hib)
                    else:
                        cl_a, cl_b = "", ""
                    table_body += (
                        f'<tr>'
                        f'<td class="metric">{label}</td>'
                        f'<td class="{cl_a}">{fmt_a}</td>'
                        f'<td class="{cl_b}">{fmt_b}</td>'
                        f'</tr>'
                    )

                st.markdown(
                    f'<div class="card">'
                    f'<table class="compare-table">'
                    f'<thead><tr>'
                    f'<th>Specification</th>'
                    f'<th>{ca.get("Brand")} {ca.get("Car")}</th>'
                    f'<th>{cb.get("Brand")} {cb.get("Car")}</th>'
                    f'</tr></thead>'
                    f'<tbody>{table_body}</tbody>'
                    f'</table>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    # ── TAB 3: DEALERSHIPS ────────────────────────────────────────────────────
    with tab_dealers:
        st.markdown('<div class="section-label">Locate Authorized Dealerships</div>', unsafe_allow_html=True)
        d_c1, d_c2 = st.columns(2)
        with d_c1:
            filter_city = st.selectbox("Select City", cities, key="dealer_city_filter")
        with d_c2:
            filter_brand = st.selectbox("Select Manufacturer", brands, key="dealer_brand_filter")

        dealers_list = _fetch_dealerships(
            city=filter_city if filter_city != "Any" else None,
            brand=filter_brand if filter_brand != "Any" else None,
        )

        st.markdown(
            f'<div class="section-label">{len(dealers_list)} Authorized Dealerships Found</div>',
            unsafe_allow_html=True,
        )

        if dealers_list:
            for d in dealers_list:
                st.markdown(
                    f'<div class="result-card" style="margin-bottom:0.5rem;padding:1rem 1.2rem;">'
                    f'<div class="result-rank">{d.get("city")} &bull; {d.get("area")}</div>'
                    f'<div class="result-name" style="font-size:1rem;margin-bottom:0.25rem;">{d.get("name")}</div>'
                    f'<div style="font-size:0.8rem;color:var(--text-muted);margin-bottom:0.4rem;">{d.get("address")}</div>'
                    f'<div class="stat-row" style="margin-top:0.2rem;">'
                    f'<div class="stat-item"><span class="stat-label">Brand</span><span class="stat-value" style="font-size:0.85rem;">{d.get("brand")}</span></div>'
                    f'<div class="stat-item"><span class="stat-label">Phone</span><span class="stat-value" style="font-size:0.85rem;">{d.get("phone")}</span></div>'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown('<div class="caveat">No dealerships found matching the selected filters.</div>', unsafe_allow_html=True)

    # ── TAB 4: EMI CALCULATOR ────────────────────────────────────────────────
    with tab_emi:
        st.markdown('<div class="section-label">Vehicle Financing & EMI Calculator</div>', unsafe_allow_html=True)

        col_input1, col_input2 = st.columns(2)

        with col_input1:
            st.markdown('<div class="section-label">1. Vehicle Price</div>', unsafe_allow_html=True)
            emi_car_price = st.number_input(
                "Car Ex-Showroom / On-Road Price (₹ Lakh)",
                min_value=1.0,
                max_value=250.0,
                value=12.0,
                step=0.5,
                key="adv_emi_price_input",
                help="Enter total vehicle price in Lakhs (e.g. 12.5 for ₹12,50,000)",
            )

            st.markdown('<div class="section-label">2. Financed Loan Percentage</div>', unsafe_allow_html=True)
            emi_loan_pct = st.slider(
                "Financed Loan Percentage (%)",
                min_value=10,
                max_value=100,
                value=80,
                step=5,
                key="adv_emi_loan_pct_slider",
                help="Select percentage of vehicle price to finance through auto loan (10% to 100%)",
            )

        with col_input2:
            st.markdown('<div class="section-label">3. Loan Tenure</div>', unsafe_allow_html=True)
            tenure_options = [12, 24, 36, 48, 60, 72, 84]

            def _fmt_tenure(months: int) -> str:
                years = months // 12
                return f"{years} Year{'s' if years > 1 else ''} ({months} Months)"

            emi_tenure_months = st.select_slider(
                "Loan Tenure (from 1 Year / 12 Months onwards)",
                options=tenure_options,
                value=60,
                format_func=_fmt_tenure,
                key="adv_emi_tenure_slider",
                help="Tenure from 1 year (12 months) up to 7 years (84 months)",
            )

            st.markdown('<div class="section-label">4. Annual Interest Rate</div>', unsafe_allow_html=True)
            emi_annual_rate = st.slider(
                "Interest Rate (% p.a.)",
                min_value=6.0,
                max_value=16.0,
                value=9.0,
                step=0.25,
                key="adv_emi_rate_slider",
                help="Annual auto loan interest rate offered by banks/NBFCs (typically 8.5% - 10.5%)",
            )

        # ── Calculations ───────────────────────────────────────────────────────
        principal_total = emi_car_price * 100_000
        loan_amount = principal_total * (emi_loan_pct / 100.0)
        down_payment = principal_total - loan_amount

        r = (emi_annual_rate / 100.0) / 12.0
        n = emi_tenure_months
        if r > 0 and n > 0:
            monthly_emi = int(loan_amount * r * ((1 + r) ** n) / (((1 + r) ** n) - 1))
        else:
            monthly_emi = int(loan_amount / n) if n > 0 else 0

        total_repayment = monthly_emi * n
        total_interest = max(0, total_repayment - loan_amount)

        st.markdown("---")

        # ── Summary Metrics ───────────────────────────────────────────────────
        st.markdown('<div class="section-label">Financing Summary</div>', unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Monthly EMI", f"₹{monthly_emi:,}/mo")
        with m2:
            st.metric("Down Payment", f"₹{down_payment/100_000:.2f}L ({100 - emi_loan_pct}%)")
        with m3:
            st.metric("Loan Amount", f"₹{loan_amount/100_000:.2f}L ({emi_loan_pct}%)")
        with m4:
            st.metric("Total Interest", f"₹{total_interest/100_000:.2f}L")

        st.markdown(
            f"""
            <div style="background: rgba(34,197,94,0.08); border-left: 3px solid var(--accent); padding: 0.8rem 1.1rem; border-radius: 6px; margin: 1rem 0;">
                <div style="font-weight: 600; color: var(--accent); font-size: 0.9rem; margin-bottom: 0.25rem;">
                    Total Vehicle Cost (Principal + Interest): ₹{total_repayment/100_000:.2f} Lakh
                </div>
                <div style="color: var(--text-muted); font-size: 0.82rem;">
                    By paying a {100 - emi_loan_pct}% down payment (₹{down_payment:,.0f}), you finance ₹{loan_amount:,.0f} at {emi_annual_rate:.2f}% p.a. over {n // 12} years ({n} months).
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Comparative Tenure Table ───────────────────────────────────────────
        st.markdown('<div class="section-label" style="margin-top: 1.5rem;">Tenure Comparison (From 1 Year Ahead)</div>', unsafe_allow_html=True)
        tenure_rows = []
        for t_m in tenure_options:
            t_r = (emi_annual_rate / 100.0) / 12.0
            t_emi = int(loan_amount * t_r * ((1 + t_r) ** t_m) / (((1 + t_r) ** t_m) - 1)) if t_r > 0 else int(loan_amount / t_m)
            t_total = t_emi * t_m
            t_interest = max(0, t_total - loan_amount)
            tenure_rows.append(
                f"| **{t_m // 12} Year{'s' if t_m > 12 else ''} ({t_m}m)** | ₹{t_emi:,} / mo | ₹{t_interest/100_000:.2f} Lakh | ₹{t_total/100_000:.2f} Lakh |"
            )

        tenure_table_md = "\n".join(tenure_rows)
        st.markdown(
            f"""
| Tenure | Monthly EMI | Total Interest | Total Payout |
|---|---|---|---|
{tenure_table_md}
            """
        )

        # ── 20/4/10 Rule Card ─────────────────────────────────────────────────
        with st.expander("💡 20/4/10 Auto Financing Affordability Rule"):
            st.markdown(
                """
                Financial planners in India recommend the **20/4/10 Rule** when buying a passenger car:
                1. **20% Down Payment**: Put down at least 20% to avoid being underwater (negative equity).
                2. **4 to 5 Years Tenure**: Keep loan tenure to 48–60 months maximum to minimize compounding interest costs.
                3. **10% Gross Income Limit**: Your monthly EMI should never exceed 10% of your gross monthly take-home income.
                """
            )

    # ── TAB 5: ADVISOR CHAT ──────────────────────────────────────────────────
    with tab_chat:
        st.markdown('<div class="section-label">Ask Car Advisor</div>', unsafe_allow_html=True)

        if "adv_chat_history" not in st.session_state:
            st.session_state["adv_chat_history"] = []

        c_space, c_city = st.columns([3, 1])
        with c_city:
            chat_city = st.selectbox("Location", cities, key="adv_chat_city_sel", label_visibility="collapsed")

        chat_html = '<div class="chat-window">'
        if not st.session_state["adv_chat_history"]:
            chat_html += (
                '<div style="color:var(--text-muted);font-size:0.875rem;padding:0.75rem 0;line-height:1.6;">'
                'Ask any question to find the best car for you in India. '
                'Inquire about budget recommendations, mileage, fuel options, model comparisons, or local authorized showrooms.'
                '</div>'
            )
        for msg in st.session_state["adv_chat_history"]:
            if msg["role"] == "user":
                chat_html += (
                    f'<div class="chat-msg-user">'
                    f'<div class="chat-bubble-user">{msg["content"]}</div>'
                    f'</div>'
                )
            else:
                formatted = msg["content"].replace("\n", "<br>")
                chat_html += (
                    f'<div class="chat-msg-bot">'
                    f'<div class="chat-bubble-bot">{formatted}</div>'
                    f'</div>'
                )
        chat_html += "</div>"
        st.markdown(chat_html, unsafe_allow_html=True)

        with st.form("adv_chat_input_form", clear_on_submit=True):
            user_msg = st.text_input(
                "Query",
                placeholder="e.g. Which diesel SUV under 20 lakh has the best mileage and features?",
                label_visibility="collapsed",
                key="adv_chat_input_text",
            )
            chat_submitted = st.form_submit_button("Send Query", use_container_width=True)

        if chat_submitted and user_msg.strip():
            st.session_state["adv_chat_history"].append({"role": "user", "content": user_msg.strip()})
            context_cars = st.session_state.get("adv_results", [])[:5]

            with st.spinner("Finding best match..."):
                reply = _call_gemini_advisor(
                    message=user_msg.strip(),
                    history=st.session_state["adv_chat_history"][:-1],
                    context_cars=context_cars,
                    city=chat_city if chat_city != "Any" else None,
                    gemini_key=None,
                )

            st.session_state["adv_chat_history"].append({"role": "assistant", "content": reply})
            st.rerun()

        if st.session_state["adv_chat_history"]:
            if st.button("Reset Conversation", key="adv_btn_reset_chat"):
                st.session_state["adv_chat_history"] = []
                st.rerun()
