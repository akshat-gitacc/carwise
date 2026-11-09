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

