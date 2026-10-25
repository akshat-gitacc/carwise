"""
CarWise — Streamlit frontend.
Talks to the FastAPI backend at /api/assess.
"""

import json
import requests
import streamlit as st
import streamlit.components.v1 as components

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="CarWise",
    layout="centered",
)

API_URL = "http://localhost:8000/api/assess"

# ── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Root variables */
    :root {
        --bg: #0e0e11;
        --surface: #18181c;
        --surface-hover: #1f1f24;
        --border: #2a2a30;
        --text: #e8e8ed;
        --text-muted: #8a8a96;
        --accent: #22c55e;
        --accent-soft: rgba(34, 197, 94, 0.12);
        --green: #34d399;
        --green-soft: rgba(52, 211, 153, 0.12);
        --amber: #fbbf24;
        --amber-soft: rgba(251, 191, 36, 0.12);
        --red: #f87171;
        --red-soft: rgba(248, 113, 113, 0.12);
        --radius: 12px;
    }

    /* Global */
    .stApp {
        background: var(--bg) !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    /* Hide Streamlit chrome */
    #MainMenu, footer, header { visibility: hidden; }
    .stDeployButton { display: none !important; }

    /* Hide "press Enter to apply" helper text */
    .stNumberInput div[data-testid="InputInstructions"],
    .stNumberInput [data-testid="InputInstructions"],
    div[data-testid="InputInstructions"] {
        display: none !important;
    }

    /* Main container */
    .block-container {
        max-width: 680px !important;
        padding-top: 3rem !important;
        padding-bottom: 4rem !important;
    }

    /* Typography */
    h1, h2, h3, h4, p, label, span, div {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    /* Title area */
    .hero-title {
        font-size: 2.25rem;
        font-weight: 700;
        color: var(--text);
        letter-spacing: -0.03em;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        line-height: 1.15;
    }
    .hero-sub {
        font-size: 1rem;
        color: var(--text-muted);
        font-weight: 400;
        margin-bottom: 2.5rem;
        line-height: 1.5;
        animation: fadeSlideUp 0.8s ease-out both;
        animation-delay: 0.3s;
    }

    @keyframes fadeSlideUp {
        from {
            opacity: 0;
            transform: translateY(12px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    /* Card surfaces */
    .card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    .card-title {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--text-muted);
        margin-bottom: 1rem;
    }

    /* Streamlit input overrides */
    .stNumberInput > div > div > input,
    .stSelectbox > div > div,
    .stFileUploader > div {
        background: var(--surface) !important;
        border-color: var(--border) !important;
        color: var(--text) !important;
        border-radius: 8px !important;
        font-family: 'Inter', sans-serif !important;
    }
    .stNumberInput label,
    .stSelectbox label,
    .stFileUploader label {
        color: var(--text) !important;
        font-weight: 400 !important;
        font-size: 0.875rem !important;
    }

    /* Button */
    .stButton > button {
        background: var(--accent) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.7rem 2rem !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        letter-spacing: 0.01em;
        transition: all 0.2s ease;
        font-family: 'Inter', sans-serif !important;
        width: 100%;
    }
    .stButton > button:hover {
        filter: brightness(1.12);
        transform: translateY(-1px);
        box-shadow: 0 4px 20px rgba(34, 197, 94, 0.35);
    }
    .stButton > button:active {
        transform: translateY(0);
    }

    /* Result badge */
    .quality-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.6rem 1.2rem;
        border-radius: 100px;
        font-weight: 600;
        font-size: 1rem;
        letter-spacing: -0.01em;
    }
    .quality-excellent { background: var(--green-soft); color: var(--green); }
    .quality-good { background: var(--green-soft); color: var(--green); }
    .quality-fair { background: var(--amber-soft); color: var(--amber); }
    .quality-poor { background: var(--red-soft); color: var(--red); }

    /* Probability bars */
    .prob-row {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 0.5rem;
    }
    .prob-label {
        width: 70px;
        font-size: 0.8rem;
        color: var(--text-muted);
        font-weight: 500;
        text-align: right;
    }
    .prob-bar-bg {
        flex: 1;
        height: 6px;
        background: var(--border);
        border-radius: 3px;
        overflow: hidden;
    }
    .prob-bar-fill {
        height: 100%;
        border-radius: 3px;
        background: var(--accent);
        transition: width 0.6s ease;
    }
    .prob-pct {
        width: 36px;
        font-size: 0.8rem;
        color: var(--text-muted);
        font-weight: 500;
    }

    /* Factor chips */
    .factor-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.35rem 0.75rem;
        border-radius: 8px;
        font-size: 0.8rem;
        font-weight: 500;
        margin: 0.2rem 0.25rem;
    }
    .factor-positive {
        background: var(--green-soft);
        color: var(--green);
    }
    .factor-negative {
        background: var(--red-soft);
        color: var(--red);
    }

    /* Explanation card */
    .explanation-text {
        font-size: 0.9rem;
        color: var(--text);
        line-height: 1.65;
        border-left: 3px solid var(--accent);
        padding-left: 1rem;
        margin: 0.5rem 0;
    }

    /* Visual score meter */
    .visual-meter {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        margin-bottom: 0.5rem;
    }
    .visual-label {
        width: 140px;
        font-size: 0.8rem;
        color: var(--text-muted);
        font-weight: 500;
    }
    .visual-bar-bg {
        flex: 1;
        height: 6px;
        background: var(--border);
        border-radius: 3px;
        overflow: hidden;
    }
    .visual-bar-fill {
        height: 100%;
        border-radius: 3px;
        transition: width 0.6s ease;
    }
    .visual-score {
        width: 40px;
        font-size: 0.8rem;
        color: var(--text-muted);
        font-weight: 500;
        text-align: right;
    }

    /* Divider */
    hr {
        border: none;
        border-top: 1px solid var(--border);
        margin: 1.25rem 0;
    }

    /* Results scroll anchor */
    #results-anchor {
        scroll-margin-top: 2rem;
    }

    /* Caveat */
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

    /* Section header */
    .section-label {
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--text-muted);
        margin-bottom: 0.75rem;
        margin-top: 0.25rem;
    }

    /* Expander overrides */
    .streamlit-expanderHeader {
        background: var(--surface) !important;
        border-color: var(--border) !important;
        color: var(--text) !important;
        font-family: 'Inter', sans-serif !important;
        border-radius: 8px !important;
    }

    /* Boxed buttons navigation using stTabs */
    div[data-testid="stTabs"] {
        margin-bottom: 1.5rem !important;
    }
    div[data-testid="stTabs"] [role="tablist"],
    div[data-testid="stTabs"] [data-baseweb="tab-list"] {
        display: inline-flex !important;
        flex-direction: row !important;
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        padding: 4px !important;
        gap: 6px !important;
        margin-bottom: 1.5rem !important;
        width: fit-content !important;
    }
    div[data-testid="stTabs"] button[role="tab"] {
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        border-radius: 7px !important;
        padding: 0.55rem 1.35rem !important;
        height: auto !important;
        border: 1px solid transparent !important;
        background: transparent !important;
        color: var(--text-muted) !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        letter-spacing: -0.01em !important;
        cursor: pointer !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        outline: none !important;
        box-shadow: none !important;
    }
    div[data-testid="stTabs"] button[role="tab"]:hover {
        color: var(--text) !important;
        background: rgba(255, 255, 255, 0.04) !important;
        border-color: rgba(255, 255, 255, 0.06) !important;
    }
    div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
        background: rgba(34, 197, 94, 0.12) !important;
        border: 1px solid rgba(34, 197, 94, 0.4) !important;
        color: var(--green) !important;
        font-weight: 600 !important;
        box-shadow: 0 1px 10px rgba(34, 197, 94, 0.15) !important;
    }
    div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] p,
    div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] span,
    div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] div {
        color: var(--green) !important;
        font-weight: 600 !important;
    }
    div[data-testid="stTabs"] button[role="tab"] p {
        color: inherit !important;
        font-weight: inherit !important;
        margin: 0 !important;
    }
    /* Hide default tab line indicators */
    div[data-testid="stTabs"] [data-baseweb="tab-highlight"],
    div[data-testid="stTabs"] [data-baseweb="tab-border"] {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)


# ── Definitions ──────────────────────────────────────────────────────────────

NUMERIC_FIELDS = [
    ("age_years",                 "Age (years)",                          0, 40,     1,    5),
    ("mileage_km",                "Mileage (km)",                         0, 500000, 1000, 50000),
    ("previous_owners",           "Previous owners",                      1, 10,     1,    1),
    ("service_completeness",      "Service history completeness (0–1)",   0.0, 1.0,  0.05, 0.75),
    ("months_since_service",      "Months since last service",            0, 120,    1,    6),
    ("accident_count",            "Accidents reported",                   0, 10,     1,    0),
    ("engine_cc",                 "Engine capacity (cc, 0 if electric)",  0, 6000,   1,    1200),
    ("safety_rating",             "Safety rating",                        1, 5,      1,    4),
    ("seats",                     "Seats",                                2, 9,      1,    5),
    ("doors",                     "Doors",                                2, 5,      1,    4),
    ("airbags",                   "Airbags",                              0, 10,     1,    6),
    ("warranty_remaining_months", "Warranty remaining (months)",          0, 60,     1,    0),
]

SELECT_FIELDS = [
    ("fuel_type",                "Fuel type",       ["petrol", "diesel", "electric", "hybrid"]),
    ("body_material",            "Body material",   ["steel", "aluminium", "composite"]),
    ("country_of_manufacture",   "Country",         ["Japan", "Germany", "India", "USA", "Korea", "China", "Other"]),
    ("transmission",             "Transmission",    ["manual", "automatic"]),
    ("drive_type",               "Drive type",      ["FWD", "RWD", "AWD"]),
]

QUALITY_ICONS = {"Excellent": "✦", "Good": "●", "Fair": "◐", "Poor": "○"}


# ── Helper: pretty feature names ─────────────────────────────────────────────

def pretty(name: str) -> str:
    return name.replace("_", " ").title()


# ── Header ───────────────────────────────────────────────────────────────────

st.markdown('<div class="hero-title">CarWise</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Enter your car\'s details or upload photos.<br>'
    'Get an instant quality estimate.</div>',
    unsafe_allow_html=True,
)


# ── Input form ───────────────────────────────────────────────────────────────

tab_details, tab_photos = st.tabs(["Vehicle Details", "Upload Photos"])

with tab_details:
    st.markdown('<div class="section-label">Basics</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    inputs = {}
    for i, (key, label, mn, mx, step, default) in enumerate(NUMERIC_FIELDS):
        col = col1 if i % 2 == 0 else col2
        with col:
            inputs[key] = st.number_input(
                label,
                min_value=mn,
                max_value=mx,
                step=step,
                value=default,
                key=key,
            )

    st.markdown("---")
    st.markdown('<div class="section-label">Specifications</div>', unsafe_allow_html=True)
    cols = st.columns(len(SELECT_FIELDS))
    for idx, (key, label, opts) in enumerate(SELECT_FIELDS):
        with cols[idx]:
            inputs[key] = st.selectbox(label, opts, key=key)

with tab_photos:
    st.markdown('<div class="section-label">Exterior photos</div>', unsafe_allow_html=True)
    exterior_files = st.file_uploader(
        "Upload exterior photos (optional)",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        key="exterior_upload",
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown('<div class="section-label">Engine bay photos</div>', unsafe_allow_html=True)
    engine_files = st.file_uploader(
        "Upload engine bay photos (optional)",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        key="engine_upload",
        label_visibility="collapsed",
    )

has_photos = bool(exterior_files) or bool(engine_files)


# ── Submit ───────────────────────────────────────────────────────────────────

st.markdown("")
assess_clicked = st.button("Assess this car", use_container_width=True)

if assess_clicked:
    with st.spinner("Analyzing your car — this may take a moment on first run…"):
        # Build multipart form data
        car_payload = {}
        for key, *_ in NUMERIC_FIELDS:
            car_payload[key] = inputs[key]
        for key, *_ in SELECT_FIELDS:
            car_payload[key] = inputs[key]

        files = []
        files.append(("car", (None, json.dumps(car_payload))))

        if exterior_files:
            for f in exterior_files:
                files.append(("exterior", (f.name, f.read(), f.type or "image/jpeg")))
        if engine_files:
            for f in engine_files:
                files.append(("engine", (f.name, f.read(), f.type or "image/jpeg")))

        try:
            resp = requests.post(API_URL, files=files, timeout=120)
            resp.raise_for_status()
            data = resp.json()
        except requests.ConnectionError:
            st.error("Could not connect to the backend. Make sure it's running on port 8000.")
            st.stop()
        except Exception as e:
            st.error(f"Something went wrong: {e}")
            st.stop()

    # ── Results ──────────────────────────────────────────────────────────

    st.markdown('<div id="results-anchor"></div>', unsafe_allow_html=True)
    components.html(
        """
        <script>
            setTimeout(function() {
                try {
                    const target = window.parent.document.getElementById('results-anchor');
                    if (target) {
                        target.scrollIntoView({ behavior: 'smooth', block: 'start' });
                    }
                } catch (e) {
                    console.error("Auto-scroll failed:", e);
                }
            }, 150);
        </script>
        """,
        height=0,
        width=0,
    )

    st.markdown("---")

    # Quality badge
    label = data["label"]
    prob = data["probability"]
    badge_class = f"quality-{label.lower()}"
    icon = QUALITY_ICONS.get(label, "●")

    st.markdown(
        f'<div style="text-align:center; margin: 1.5rem 0;">'
        f'<span class="quality-badge {badge_class}">{icon}  {label}</span>'
        f'<div style="color: var(--text-muted); font-size: 0.85rem; margin-top: 0.5rem;">'
        f'{prob:.0%} confidence</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Probability bars
    st.markdown('<div class="section-label">Class Probabilities</div>', unsafe_allow_html=True)
    probs_html = ""
    for cls, p in data["probabilities"].items():
        width = p * 100
        probs_html += (
            f'<div class="prob-row">'
            f'<span class="prob-label">{cls}</span>'
            f'<div class="prob-bar-bg"><div class="prob-bar-fill" style="width:{width}%"></div></div>'
            f'<span class="prob-pct">{p:.0%}</span>'
            f'</div>'
        )
    st.markdown(probs_html, unsafe_allow_html=True)

    # Visual scores — only show when photos were uploaded
    if has_photos and data.get("visual_scores"):
        st.markdown("---")
        st.markdown('<div class="section-label">Photo Scores</div>', unsafe_allow_html=True)
        visuals_html = ""
        for k, v in data["visual_scores"].items():
            # Color gradient: red → amber → green
            if v >= 66:
                fill_color = "var(--green)"
            elif v >= 33:
                fill_color = "var(--amber)"
            else:
                fill_color = "var(--red)"
            visuals_html += (
                f'<div class="visual-meter">'
                f'<span class="visual-label">{pretty(k)}</span>'
                f'<div class="visual-bar-bg"><div class="visual-bar-fill" '
                f'style="width:{v}%; background:{fill_color}"></div></div>'
                f'<span class="visual-score">{v:.0f}</span>'
                f'</div>'
            )
        st.markdown(visuals_html, unsafe_allow_html=True)

        if data.get("photo_notes"):
            for note in data["photo_notes"]:
                st.caption(f"ℹ️  {note}")

    # Key factors
    st.markdown("---")
    positives = data.get("positives", [])
    negatives = data.get("negatives", [])

    col_pos, col_neg = st.columns(2)
    with col_pos:
        st.markdown('<div class="section-label">↑ Helping the Rating</div>', unsafe_allow_html=True)
        if positives:
            chips = ""
            for d in positives:
                val = d["value"] if d["value"] is not None else "—"
                chips += f'<span class="factor-chip factor-positive">{pretty(d["feature"])} · {val}</span>'
            st.markdown(chips, unsafe_allow_html=True)
        else:
            st.caption("No significant positive factors")

    with col_neg:
        st.markdown('<div class="section-label">↓ Holding it Back</div>', unsafe_allow_html=True)
        if negatives:
            chips = ""
            for d in negatives:
                val = d["value"] if d["value"] is not None else "—"
                chips += f'<span class="factor-chip factor-negative">{pretty(d["feature"])} · {val}</span>'
            st.markdown(chips, unsafe_allow_html=True)
        else:
            st.caption("No significant negative factors")

    # Explanation
    if data.get("explanation"):
        st.markdown("---")
        st.markdown('<div class="section-label">Explanation</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="explanation-text">{data["explanation"]}</div>',
            unsafe_allow_html=True,
        )

    # Caveat
    st.markdown(
        '<div class="caveat">'
        '⚠️  This model was trained on synthetic demonstration data. '
        'Ratings reflect model behaviour, not real-world car condition. '
        'Always get a professional inspection before purchasing a vehicle.'
        '</div>',
        unsafe_allow_html=True,
    )
