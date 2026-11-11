# pyrefly: ignore[missing-import]
"""
streamlit_app.py — CarWise Web Application.
Hosts two unified services as two distinct pages:
  1. Price Predictor: Used car price estimation with SHAP factor breakdown.
  2. Car Advisor: Indian passenger car matching & recommendations, head-to-head
     comparison, authorized dealerships locator, and Gemini AI advisory chat.

Strictly adheres to:
- Dark automotive theme (#0e0e11 bg, #18181c surface, #2a2a30 border, #22c55e accent)
- Inter font, uppercase section labels
- Top-level segmented navigation switcher
- Zero emojis
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure carwise package root is in sys.path
_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

import requests
import streamlit as st
import uuid

from frontend.views.advisor_view import render_advisor_view
from frontend.views.explore_view import render_explore_view
from frontend.views.favourites_view import render_favourites_view

st.set_page_config(page_title="CarWise", layout="centered")

# Persistent User Session ID (preserved across browser refreshes via query param)
active_user_id = st.query_params.get("user_id")
if not active_user_id:
    active_user_id = f"user_{uuid.uuid4().hex[:8]}"
    st.query_params["user_id"] = active_user_id

st.session_state["user_id"] = active_user_id

API_URL = os.environ.get("CARWISE_API_URL", "http://localhost:8000")

BRANDS = [
    "Audi", "BMW", "Bentley", "Datsun", "Ferrari", "Force", "Ford",
    "Honda", "Hyundai", "Isuzu", "Jaguar", "Jeep", "Kia", "Land Rover",
    "Lexus", "MG", "Mahindra", "Maruti", "Maserati", "Mercedes-Benz",
    "Mini", "Nissan", "Porsche", "Renault", "Rolls-Royce", "Skoda",
    "Tata", "Toyota", "Volkswagen", "Volvo"
]

BRAND_MODELS = {
    "Maruti": ["Alto", "Swift", "Baleno", "Dzire VXI", "Dzire ZXI", "Dzire LXI", "Swift Dzire", "Wagon R", "Celerio", "Ciaz", "Ertiga", "Vitara", "S-Presso", "Ignis", "XL6", "Eeco", "Glanza"],
    "Hyundai": ["i10", "i20", "Grand", "Creta", "Venue", "Verna", "Tucson", "Elantra", "Santro", "Aura"],
    "Honda": ["City", "Amaze", "Jazz", "WR-V", "CR-V", "Civic", "CR"],
    "Toyota": ["Innova", "Fortuner", "Yaris", "Camry", "Glanza"],
    "Ford": ["Ecosport", "Figo", "Endeavour", "Freestyle", "Aspire"],
    "Mahindra": ["XUV500", "XUV300", "Scorpio", "Bolero", "Thar", "Marazzo", "Alturas", "KUV100", "KUV"],
    "Tata": ["Nexon", "Harrier", "Tiago", "Tigor", "Altroz", "Safari", "Hexa"],
    "BMW": ["3", "5", "7", "X1", "X3", "X4", "X5", "6", "Z4"],
    "Mercedes-Benz": ["C-Class", "E-Class", "S-Class", "GLS", "GL-Class", "CLS", "C"],
    "Audi": ["A4", "A6", "A8", "Q7"],
    "Volkswagen": ["Polo", "Vento"],
    "Renault": ["Duster", "KWID", "Triber"],
    "Skoda": ["Rapid", "Octavia", "Superb"],
    "Kia": ["Seltos", "Carnival"],
    "Jaguar": ["XE", "XF", "F-PACE"],
    "Land Rover": ["Rover"],
    "Volvo": ["XC90", "XC60", "XC", "S90"],
    "Porsche": ["Cayenne", "Macan", "Panamera"],
    "Jeep": ["Compass", "Wrangler"],
    "MG": ["Hector"],
    "Datsun": ["GO", "RediGO", "redi-GO"],
    "Mini": ["Cooper"],
    "Nissan": ["Kicks", "X-Trail"],
    "Lexus": ["ES", "NX", "RX"],
    "Bentley": ["Continental"],
    "Maserati": ["Ghibli", "Quattroporte"],
    "Ferrari": ["GTC4Lusso"],
    "Rolls-Royce": ["Ghost"],
    "Force": ["Gurkha"],
    "Isuzu": ["D-Max", "MUX"],
}

FUEL_TYPES = ["Petrol", "Diesel", "CNG", "LPG", "Electric"]
TRANSMISSION_TYPES = ["Manual", "Automatic"]
SELLER_TYPES = ["Individual", "Dealer", "Trustmark Dealer"]

FACTOR_MESSAGES = {
    "Kilometres driven": (
        "Lower usage than most cars of this type, which buyers value.",
        "Higher usage than typical, which buyers usually pay less for.",
    ),
    "Kilometres driven (squared)": (
        "Its overall usage is low, which supports a higher price.",
        "Its overall usage is high, which pulls the price down.",
    ),
    "Yearly usage": (
        "It has been driven gently each year, which helps its value.",
        "It has been driven heavily each year, which lowers its value.",
    ),
    "Yearly usage (log)": (
        "Its yearly usage is on the gentle side, which helps its value.",
        "Its yearly usage is on the heavy side, which lowers its value.",
    ),
    "Vehicle age": (
        "Its age works in its favour compared with similar cars.",
        "Its age counts against it compared with similar cars.",
    ),
    "Vehicle age (squared)": (
        "Its age has a limited effect on price here, which helps a little.",
        "Its age has a growing effect on price, which lowers it.",
    ),
    "Mileage x age": (
        "Its fuel efficiency holds up well for its age.",
        "Its fuel efficiency has dropped more than expected for its age.",
    ),
    "Fuel mileage": (
        "Good fuel efficiency makes it more appealing to buyers.",
        "Lower fuel efficiency makes it less appealing to buyers.",
    ),
    "Mileage per power": (
        "It gets good fuel economy for the power it produces.",
        "It uses more fuel than expected for the power it produces.",
    ),
    "Engine size": (
        "A larger engine than typical for this type of car, which adds value.",
        "A smaller engine than typical for this type of car, which lowers value.",
    ),
    "Engine size (log)": (
        "Its engine size adds to its value compared with similar cars.",
        "Its engine size takes away from its value compared with similar cars.",
    ),
    "Max power": (
        "Strong power output for a car like this, which adds to its price.",
        "Modest power output for a car like this, which lowers its price.",
    ),
    "Max power (log)": (
        "Its power output adds to its value compared with similar cars.",
        "Its power output takes away from its value compared with similar cars.",
    ),
    "Power x engine": (
        "Its power and engine together make it stand out, which helps its price.",
        "Its power and engine together are modest, which lowers its price.",
    ),
    "Power per litre": (
        "It gets a lot of power from each litre of engine, which buyers like.",
        "It gets less power from each litre of engine than similar cars.",
    ),
    "Engine-to-power ratio": (
        "Its engine and power are well matched, which supports its price.",
        "Its engine and power are less well matched, which lowers its price.",
    ),
    "Seats": (
        "Its seating capacity suits many buyers, which helps its price.",
        "Its seating capacity is less in demand, which lowers its price.",
    ),
    "Seats x power": (
        "Its seating and power combination is attractive, which helps its price.",
        "Its seating and power combination is less attractive, which lowers its price.",
    ),
    "Luxury brand": (
        "Being a premium brand raises its price.",
        "Being a non-premium brand keeps its price lower.",
    ),
    "Automatic transmission": (
        "An automatic gearbox is in demand and adds to its value.",
        "A manual gearbox is more affordable and keeps its price lower.",
    ),
    "Transmission": (
        "Its gearbox type is popular with buyers, which helps its price.",
        "Its gearbox type is less popular, which lowers its price.",
    ),
    "Diesel engine": (
        "A diesel engine is in demand for longer and more frequent driving.",
        "A petrol or other engine is less in demand, which lowers its price.",
    ),
    "Luxury automatic": (
        "Being a premium automatic gives it extra appeal.",
        "The premium automatic combination does not add much value here.",
    ),
    "Brand": (
        "The brand has a strong reputation in the used car market.",
        "The brand has a weaker resale reputation, which lowers its price.",
    ),
    "Model": (
        "This model is well regarded and holds its value.",
        "This model holds its value less well than others of its kind.",
    ),
    "Fuel type": (
        "Its fuel type is popular with buyers, which helps its price.",
        "Its fuel type is less popular, which lowers its price.",
    ),
    "Seller type": (
        "Buying from this type of seller adds confidence and value.",
        "Buying from this type of seller carries a little less confidence.",
    ),
}

DEFAULT_MESSAGES = (
    "This factor helps the estimated price.",
    "This factor lowers the estimated price.",
)

# ── Unified Design System ────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    :root {
        --bg: #0e0e11;
        --surface: #18181c;
        --surface-hover: #1f1f24;
        --border: #2a2a30;
        --text: #e8e8ed;
        --text-muted: #8a8a96;
        --accent: #22c55e;
        --accent-soft: rgba(34, 197, 94, 0.12);
        --red: #ef4444;
        --radius: 12px;
    }

    .stApp {
        background: var(--bg) !important;
        font-family: 'Inter', sans-serif !important;
    }

    #MainMenu, footer, header { visibility: hidden; }
    .stDeployButton { display: none !important; }
    div[data-testid="InputInstructions"] { display: none !important; }

    .block-container {
        max-width: 760px !important;
        padding-top: 2rem !important;
        padding-bottom: 4rem !important;
    }

    /* Top Brand Bar */
    .brand-bar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 1.25rem;
    }
    .hero-logo {
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--text);
        letter-spacing: -0.03em;
    }
    .hero-logo span {
        color: var(--accent);
    }

    .hero-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: var(--text);
        letter-spacing: -0.03em;
        margin-bottom: 0.25rem;
    }

    .hero-sub {
        font-size: 0.95rem;
        color: var(--text-muted);
        margin-bottom: 1.75rem;
        line-height: 1.5;
    }

    .section-label {
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--text-muted);
        margin-bottom: 0.75rem;
        margin-top: 0.5rem;
    }

    /* Segmented Navigation Switcher */
    div[data-testid="stSegmentedControl"] {
        display: flex !important;
        justify-content: center !important;
        margin-bottom: 2rem !important;
    }
    div[data-testid="stSegmentedControl"] > div {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        padding: 4px !important;
        display: inline-flex !important;
        gap: 4px !important;
    }
    div[data-testid="stSegmentedControl"] button {
        background: transparent !important;
        color: var(--text-muted) !important;
        border: none !important;
        border-radius: 7px !important;
        padding: 0.45rem 1.75rem !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
        transition: all 0.15s ease !important;
    }
    div[data-testid="stSegmentedControl"] button[aria-checked="true"] {
        background: var(--accent) !important;
        color: #0e0e11 !important;
        font-weight: 600 !important;
    }
    div[data-testid="stSegmentedControl"] button:hover {
        color: var(--text) !important;
    }

    /* Tabs */
    button[data-baseweb="tab"] {
        color: var(--text-muted) !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: var(--accent) !important;
        font-weight: 600 !important;
        border-bottom-color: var(--accent) !important;
    }

    /* Buttons */
    .stButton > button {
        background: var(--accent) !important;
        color: #0e0e11 !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.7rem 2rem !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        width: 100%;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        filter: brightness(1.12);
        transform: translateY(-1px);
        box-shadow: 0 4px 20px rgba(34, 197, 94, 0.35);
    }

    /* Price Result */
    .price-result {
        text-align: center;
        padding: 2rem;
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        margin: 1.5rem 0;
    }

    .price-label {
        font-size: 0.85rem;
        color: var(--text-muted);
        margin-bottom: 0.5rem;
    }

    .price-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: var(--accent);
        letter-spacing: -0.02em;
    }

    .price-exact {
        font-size: 0.85rem;
        color: var(--text-muted);
        margin-top: 0.25rem;
    }

    /* SHAP Explanations */
    .why-title {
        font-size: 0.8rem;
        font-weight: 600;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin: 1.5rem 0 0.5rem;
    }

    .why-intro {
        font-size: 0.9rem;
        color: var(--text-muted);
        line-height: 1.6;
        margin-bottom: 1rem;
    }

    .why-item {
        padding: 0.85rem 1rem;
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 8px;
        margin-bottom: 0.6rem;
    }

    .why-head {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.3rem;
    }

    .why-name {
        font-size: 0.9rem;
        font-weight: 600;
        color: var(--text);
    }

    .why-score {
        font-size: 0.85rem;
        font-weight: 600;
    }

    .why-msg {
        font-size: 0.85rem;
        color: var(--text-muted);
        line-height: 1.5;
    }

    /* Advisor Result Cards */
    .result-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 1.2rem 1.4rem;
        margin-bottom: 0.75rem;
        position: relative;
    }
    .result-card:hover {
        border-color: rgba(34, 197, 94, 0.3);
    }
    .result-rank {
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--text-muted);
        margin-bottom: 0.4rem;
    }
    .result-name {
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--text);
        letter-spacing: -0.01em;
        margin-bottom: 0.6rem;
    }
    .result-score-pill {
        display: inline-block;
        background: var(--accent-soft);
        border: 1px solid rgba(34, 197, 94, 0.3);
        color: var(--accent);
        font-size: 0.75rem;
        font-weight: 600;
        padding: 0.2rem 0.65rem;
        border-radius: 100px;
        margin-bottom: 0.8rem;
    }
    .stat-row {
        display: flex;
        flex-wrap: wrap;
        gap: 1.2rem;
        margin-top: 0.4rem;
    }
    .stat-item {
        display: flex;
        flex-direction: column;
        gap: 0.15rem;
    }
    .stat-label {
        font-size: 0.7rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--text-muted);
    }
    .stat-value {
        font-size: 0.95rem;
        font-weight: 600;
        color: var(--text);
    }

    /* Score Progress Bar */
    .score-bar-wrap {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-top: 0.6rem;
    }
    .score-bar-bg {
        flex: 1;
        height: 4px;
        background: var(--border);
        border-radius: 2px;
        overflow: hidden;
    }
    .score-bar-fill {
        height: 100%;
        background: var(--accent);
        border-radius: 2px;
    }
    .score-num {
        font-size: 0.8rem;
        font-weight: 600;
        color: var(--accent);
        width: 36px;
        text-align: right;
    }

    /* Compare Table */
    .card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 1.4rem 1.5rem;
        margin-bottom: 1rem;
    }
    .compare-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.875rem;
    }
    .compare-table th {
        text-align: left;
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--text-muted);
        padding: 0.5rem 0.75rem;
        border-bottom: 1px solid var(--border);
    }
    .compare-table td {
        padding: 0.6rem 0.75rem;
        color: var(--text);
        border-bottom: 1px solid var(--border);
        font-weight: 400;
    }
    .compare-table tr:last-child td { border-bottom: none; }
    .compare-table td.metric { color: var(--text-muted); font-size: 0.8rem; font-weight: 500; }
    .compare-win { color: var(--accent) !important; font-weight: 600 !important; }
    .compare-lose { color: var(--text-muted) !important; }

    /* Chat Window */
    .chat-window {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
        min-height: 180px;
        max-height: 420px;
        overflow-y: auto;
    }
    .chat-msg-user {
        display: flex;
        justify-content: flex-end;
        margin-bottom: 0.75rem;
    }
    .chat-bubble-user {
        background: var(--surface-hover);
        border: 1px solid var(--border);
        color: var(--text);
        padding: 0.6rem 0.9rem;
        border-radius: 12px 12px 2px 12px;
        font-size: 0.875rem;
        max-width: 82%;
    }
    .chat-msg-bot {
        display: flex;
        justify-content: flex-start;
        margin-bottom: 0.75rem;
    }
    .chat-bubble-bot {
        background: rgba(34, 197, 94, 0.08);
        border: 1px solid rgba(34, 197, 94, 0.22);
        color: var(--text);
        padding: 0.75rem 1rem;
        border-radius: 12px 12px 12px 2px;
        font-size: 0.875rem;
        max-width: 88%;
        line-height: 1.6;
    }

    .caveat {
        font-size: 0.78rem;
        color: var(--text-muted);
        line-height: 1.6;
        padding: 0.75rem 1rem;
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 8px;
        margin-top: 1rem;
    }

    /* ── Interactive Car Card with Single Click Expand & Double Click Heart ── */
    .car-interactive-card {
        position: relative;
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 1.25rem 1.4rem;
        margin-bottom: 0.85rem;
        cursor: pointer;
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
        overflow: hidden;
    }
    .car-interactive-card:hover {
        border-color: rgba(34, 197, 94, 0.45);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
    }
    .card-top-row {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 1rem;
    }
    .card-title-group {
        flex: 1;
    }
    .heart-action-btn {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid var(--border);
        border-radius: 50%;
        width: 38px;
        height: 38px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 1.15rem;
        cursor: pointer;
        transition: transform 0.15s ease, background 0.15s ease, border-color 0.15s ease;
        color: #8a8a96;
        user-select: none;
    }
    .heart-action-btn:hover {
        transform: scale(1.15);
        background: rgba(239, 68, 68, 0.12);
        border-color: rgba(239, 68, 68, 0.4);
    }
    .heart-action-btn.is-active {
        color: #ef4444 !important;
        background: rgba(239, 68, 68, 0.15) !important;
        border-color: rgba(239, 68, 68, 0.5) !important;
    }

    button.heart-icon-btn {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid var(--border) !important;
        border-radius: 50% !important;
        width: 42px !important;
        height: 42px !important;
        min-height: 42px !important;
        max-width: 42px !important;
        padding: 0 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        font-size: 1.25rem !important;
        cursor: pointer !important;
        transition: transform 0.15s ease, background 0.15s ease, border-color 0.15s ease !important;
        margin-top: 0.25rem !important;
    }
    button.heart-icon-btn:hover {
        transform: scale(1.15) !important;
        background: rgba(239, 68, 68, 0.15) !important;
        border-color: rgba(239, 68, 68, 0.5) !important;
    }
    button.heart-icon-btn.heart-liked {
        background: rgba(239, 68, 68, 0.18) !important;
        border-color: rgba(239, 68, 68, 0.6) !important;
        box-shadow: 0 0 12px rgba(239, 68, 68, 0.35) !important;
    }

    /* Double Click Floating Opaque Heart Pop Animation */
    @keyframes heartFloatPop {
        0% {
            opacity: 0;
            transform: translate(-50%, -50%) scale(0.2);
        }
        30% {
            opacity: 0.95;
            transform: translate(-50%, -50%) scale(1.35);
        }
        60% {
            opacity: 0.9;
            transform: translate(-50%, -50%) scale(1.0);
        }
        100% {
            opacity: 0;
            transform: translate(-50%, -70%) scale(1.2);
        }
    }
    .card-heart-popup {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        font-size: 4.5rem;
        color: #ef4444;
        text-shadow: 0 0 30px rgba(239, 68, 68, 0.8), 0 0 10px rgba(239, 68, 68, 0.9);
        pointer-events: none;
        z-index: 999;
        animation: heartFloatPop 0.85s cubic-bezier(0.175, 0.885, 0.32, 1.275) forwards;
    }

    /* Click to Expand Specs Drawer */
    details.card-drawer summary {
        list-style: none !important;
        list-style-type: none !important;
        outline: none !important;
        cursor: pointer !important;
        user-select: none !important;
    }
    details.card-drawer summary::-webkit-details-marker,
    details.card-drawer summary::marker {
        display: none !important;
    }
    .specs-expanded-drawer {
        margin-top: 1rem;
        padding-top: 1rem;
        border-top: 1px dashed var(--border);
        animation: drawerSlide 0.2s ease-out;
    }
    @keyframes drawerSlide {
        from { opacity: 0; transform: translateY(-6px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .specs-expanded-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
        gap: 0.75rem;
        margin-top: 0.75rem;
    }
    .specs-expanded-item {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 0.5rem 0.75rem;
    }
    .specs-expanded-label {
        font-size: 0.68rem;
        color: var(--text-muted);
        text-transform: uppercase;
        font-weight: 500;
        letter-spacing: 0.05em;
    }
    .specs-expanded-value {
        font-size: 0.85rem;
        color: var(--text);
        font-weight: 600;
        margin-top: 0.15rem;
    }
    .features-pill-wrap {
        display: flex;
        flex-wrap: wrap;
        gap: 0.4rem;
        margin-top: 0.6rem;
    }
    .feature-pill {
        background: rgba(34, 197, 94, 0.08);
        border: 1px solid rgba(34, 197, 94, 0.25);
        color: var(--accent);
        font-size: 0.75rem;
        padding: 0.2rem 0.55rem;
        border-radius: 100px;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)


# ── Top Branding & Navigation ────────────────────────────────────────────────
st.markdown(
    '<div class="brand-bar">'
    '<span class="hero-logo">Car<span>Wise</span></span>'
    '</div>',
    unsafe_allow_html=True,
)

tab_param = st.query_params.get("tab")
valid_tabs = ["Car Advisor", "Explore", "Favourites", "Price Predictor"]
default_tab = tab_param if tab_param in valid_tabs else "Car Advisor"

selected_page = st.segmented_control(
    "Service Navigation",
    options=valid_tabs,
    default=default_tab,
    label_visibility="collapsed",
    key="nav_segmented_control",
)
if not selected_page:
    selected_page = default_tab

if st.query_params.get("tab") != selected_page:
    st.query_params["tab"] = selected_page


# ── PAGE 1: PRICE PREDICTOR ──────────────────────────────────────────────────
def render_price_predictor_view():
    st.markdown('<div class="hero-title">Price Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Get an instant price estimate for any used car in India.</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-label">Car Details</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        brand = st.selectbox("Brand", BRANDS, index=BRANDS.index("Maruti"), key="pred_brand")

    with col2:
        models_for_brand = BRAND_MODELS.get(brand, ["Other"])
        model = st.selectbox("Model", models_for_brand, key="pred_model")

    st.markdown("---")
    st.markdown('<div class="section-label">Specifications</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        fuel_type = st.selectbox("Fuel Type", FUEL_TYPES, key="pred_fuel")
        engine = st.number_input("Engine (cc)", min_value=500, max_value=7000, value=1200, step=100, key="pred_engine")

    with col2:
        transmission = st.selectbox("Transmission", TRANSMISSION_TYPES, key="pred_trans")
        max_power = st.number_input("Max Power (bhp)", min_value=30.0, max_value=700.0, value=85.0, step=5.0, key="pred_power")

    with col3:
        seller_type = st.selectbox("Seller Type", SELLER_TYPES, key="pred_seller")
        seats = st.number_input("Seats", min_value=2, max_value=9, value=5, key="pred_seats")

    st.markdown("---")
    st.markdown('<div class="section-label">Usage</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        vehicle_age = st.number_input("Vehicle Age (years)", min_value=0, max_value=30, value=5, key="pred_age")

    with col2:
        km_driven = st.number_input("Km Driven", min_value=0, max_value=500000, value=50000, step=5000, key="pred_km")

    with col3:
        mileage = st.number_input("Mileage (kmpl)", min_value=4.0, max_value=40.0, value=18.0, step=0.5, key="pred_mileage")

    st.markdown("")
    predict_clicked = st.button("Estimate Price", use_container_width=True, key="pred_submit_btn")

    if predict_clicked:
        payload = {
            "brand": brand,
            "model": model,
            "vehicle_age": vehicle_age,
            "km_driven": km_driven,
            "seller_type": seller_type,
            "fuel_type": fuel_type,
            "transmission_type": transmission,
            "mileage": mileage,
            "engine": engine,
            "max_power": max_power,
            "seats": seats,
        }

        with st.spinner("Estimating price with XGBoost regression..."):
            try:
                try:
                    resp = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
                    resp.raise_for_status()
                    data = resp.json()
                except Exception:
                    # In-process fallback to model pipeline and explainer
                    import numpy as np
                    from backend.app.main import build_features, pipeline, explainer, format_price
                    features = build_features(payload)
                    pred_log = pipeline.predict(features)[0]
                    pred_val = float(np.expm1(pred_log))
                    pred_val = round(max(0, pred_val), 2)
                    raw_expl = explainer.explain(features)
                    data = {
                        "predicted_price": pred_val,
                        "formatted_price": format_price(pred_val),
                        "base_price": round(explainer.base_price, 2),
                        "explanation": [c.model_dump() if hasattr(c, "model_dump") else c for c in raw_expl],
                    }

                formatted = data["formatted_price"]
                exact = f"₹{data['predicted_price']:,.0f}"

                st.markdown(
                    f'<div class="price-result">'
                    f'<div class="price-label">Estimated Selling Price</div>'
                    f'<div class="price-value">{formatted}</div>'
                    f'<div class="price-exact">{exact}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

                explanation = data.get("explanation", [])
                base_price = data.get("base_price", 0)

                items = []
                for c in explanation:
                    pct = c["effect_pct"]
                    feature = c["feature"]
                    up_msg, down_msg = FACTOR_MESSAGES.get(feature, DEFAULT_MESSAGES)
                    helps = pct >= 0
                    message = up_msg if helps else down_msg
                    color = "#22c55e" if helps else "#ef4444"
                    sign = "+" if helps else ""
                    items.append(
                        f'<div class="why-item">'
                        f'<div class="why-head">'
                        f'<span class="why-name">{feature}</span>'
                        f'<span class="why-score" style="color:{color}">{sign}{pct:.1f}%</span>'
                        f'</div>'
                        f'<div class="why-msg">{message}</div>'
                        f'</div>'
                    )

                st.markdown(
                    '<div class="why-title">Why this price?</div>'
                    f'<div class="why-intro">Compared with a typical car in our data, '
                    f'which sells for around ₹{base_price:,.0f}, these are the main factors '
                    f'behind this estimate.</div>'
                    + "".join(items),
                    unsafe_allow_html=True,
                )

            except requests.ConnectionError:
                st.error("Cannot connect to backend. Make sure the API is running: `uvicorn backend.app.main:app`")
            except Exception as e:
                st.error(f"Error: {e}")


# ── Render Selected Service Page ─────────────────────────────────────────────
if selected_page == "Car Advisor":
    render_advisor_view()
elif selected_page == "Explore":
    render_explore_view()
elif selected_page == "Favourites":
    render_favourites_view()
else:
    render_price_predictor_view()
