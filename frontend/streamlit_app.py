# pyrefly: ignore[missing-import]
import streamlit as st
import requests

st.set_page_config(page_title="CarWise", layout="centered")

API_URL = "http://localhost:8000"

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

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    :root {
        --bg: #0e0e11;
        --surface: #18181c;
        --border: #2a2a30;
        --text: #e8e8ed;
        --text-muted: #8a8a96;
        --accent: #22c55e;
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
        max-width: 700px !important;
        padding-top: 2.5rem !important;
    }

    .hero-title {
        font-size: 2.25rem;
        font-weight: 700;
        color: var(--text);
        letter-spacing: -0.03em;
        margin-bottom: 0.25rem;
    }

    .hero-sub {
        font-size: 1rem;
        color: var(--text-muted);
        margin-bottom: 2rem;
    }

    .section-label {
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--text-muted);
        margin-bottom: 0.75rem;
    }

    .stButton > button {
        background: var(--accent) !important;
        color: #fff !important;
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
</style>
""", unsafe_allow_html=True)


st.markdown('<div class="hero-title">CarWise</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Get an instant price estimate for any used car.</div>', unsafe_allow_html=True)


st.markdown('<div class="section-label">Car Details</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    brand = st.selectbox("Brand", BRANDS, index=BRANDS.index("Maruti"))

with col2:
    models_for_brand = BRAND_MODELS.get(brand, ["Other"])
    model = st.selectbox("Model", models_for_brand)


st.markdown("---")
st.markdown('<div class="section-label">Specifications</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    fuel_type = st.selectbox("Fuel Type", FUEL_TYPES)
    engine = st.number_input("Engine (cc)", min_value=500, max_value=7000, value=1200, step=100)

with col2:
    transmission = st.selectbox("Transmission", TRANSMISSION_TYPES)
    max_power = st.number_input("Max Power (bhp)", min_value=30.0, max_value=700.0, value=85.0, step=5.0)

with col3:
    seller_type = st.selectbox("Seller Type", SELLER_TYPES)
    seats = st.number_input("Seats", min_value=2, max_value=9, value=5)


st.markdown("---")
st.markdown('<div class="section-label">Usage</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    vehicle_age = st.number_input("Vehicle Age (years)", min_value=0, max_value=30, value=5)

with col2:
    km_driven = st.number_input("Km Driven", min_value=0, max_value=500000, value=50000, step=5000)

with col3:
    mileage = st.number_input("Mileage (kmpl)", min_value=4.0, max_value=40.0, value=18.0, step=0.5)


st.markdown("")
predict_clicked = st.button("Estimate Price", use_container_width=True)

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

    with st.spinner("Estimating..."):
        try:
            resp = requests.post(f"{API_URL}/predict", json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()

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

        except requests.ConnectionError:
            st.error("Cannot connect to backend. Make sure the API is running: `uvicorn backend.app.main:app`")
        except Exception as e:
            st.error(f"Error: {e}")

st.markdown(
    '<div class="caveat">'
    '⚠️ This is a prototype. Predictions are based on historical data patterns '
    'and should not be used as the sole basis for pricing decisions.'
    '</div>',
    unsafe_allow_html=True,
)
