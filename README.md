# CarWise

CarWise is an intelligent automotive platform for the Indian car market, offering unified services across five specialized experiences:

1. **Car Advisor**: Conversational profile-based recommendation engine powered by a **Hybrid AI/ML Recommendation Engine** (combining Content-Based Cosine/KNN, Collaborative Filtering with implicit feedback, and Knowledge-Based constraint evaluation).
2. **CarBot Chat**: Grounded conversational AI automotive advisor with Smart Finder, head-to-head model comparison, verified Indian dealership locator, and EMI calculations.
3. **Explore**: Personalized discovery page continuously calibrated by user browsing activity, saved preferences, and shortlists, featuring **"You May Like"**, **"Discover More"**, and **"Recently Viewed"**.
4. **Favourites / Shortlist**: Persistent vehicle shortlisting with duplicate prevention, positive interaction signal dispatch, and side-by-side technical comparison.
5. **Price Predictor**: Estimates used car selling price with instant TreeSHAP factor attribution explaining value drivers.

---

## Architecture & Services

```
[ User Browser ]
       |
       +---> [ Streamlit Frontend (Port 8501) ]
                   |
                   +---> Segmented Navigation Switcher
                            |-- [ Car Advisor Page ]
                            |-- [ Explore Page ]
                            |-- [ Favourites Page ]
                            `-- [ Price Predictor Page ]
                   |
                   v (REST API with In-Process Fallback)
             [ FastAPI Backend (Port 8000) ]
                   |
                   +-- [ SQLite Storage Layer (`backend/database/carwise.db`) ]
                   |     +-- Cars catalog (2,633 variants, 741 models)
                   |     +-- User profiles & sessions (`users`)
                   |     +-- User interaction logs (views, likes, shortlists, compares)
                   |     +-- User explicit & inferred preferences
                   |     `-- User shortlist
                   |
                   +-- [ Service 1: Hybrid Recommendation Engine (`backend/recommender/`) ]
                   |     +-- Content-Based (MinMaxScaler, OneHotEncoder, Cosine Sim, KNN)
                   |     +-- Collaborative Filtering (User-Item matrix, implicit feedback, SVD, time decay)
                   |     +-- Knowledge-Based (Hard constraints, soft compatibility scoring)
                   |     +-- Dynamic Weight Redistribution (Cold-start safe)
                   |     `-- Diversity-Aware Ranking (MMR)
                   |
                   +-- [ Service 2: CarBot Conversational Advisory (`backend/chatbot/`) ]
                   |     +-- Grounded Local Synthesis & Spec Retrieval (`chat_engine.py`)
                   |     +-- Preference & Intent Entity Extraction (`car_data.py`)
                   |     +-- Dedicated REST Router (`backend/app/chatbot_router.py`)
                   |     `-- Showroom Network & EMI Financing Rules
                   |
                   +-- [ Service 3: Personalized Explore Engine ]
                   |     +-- "You May Like"
                   |     +-- "Discover More"
                   |     `-- "Recently Viewed"
                   |
                   +-- [ Service 4: Favourites & Comparison (`backend/favourites/`) ]
                   |     +-- Add/remove shortlist
                   |     `-- Side-by-side comparative table
                   |
                   `-- [ Service 5: Used Car Price Prediction ]
                         +-- XGBoost Pipeline (`xgboost_regression.joblib`)
                         `-- TreeSHAP Feature Attribution (`advisor.py`)
```

---

## Technical Documentation & Presentation Guide

For an exhaustive technical breakdown of every service, algorithm, mathematical formulation, technology decision, and defense viva questions, consult:
📄 **[WORKING.md](WORKING.md)** — *Technical Master Guide, Mathematical Formulations, Architecture Dossier & Project Defense Cheat Sheet.*

---

## How to Run

### 1. Run Automated Tests
```bash
cd /Users/kunaljaju/ai/carwise/carwise
source .venv/bin/activate
python -m unittest discover tests -v
```

### 2. Start the Primary CarWise Platform
```bash
# Terminal 1: FastAPI Backend
cd /Users/kunaljaju/ai/carwise/carwise
source .venv/bin/activate
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2: Streamlit Frontend
cd /Users/kunaljaju/ai/carwise/carwise
source .venv/bin/activate
streamlit run frontend/streamlit_app.py --server.port 8501
```
Access the application at `http://localhost:8501`.

---

## Documentation Directory

The platform includes comprehensive end-to-end documentation for technical evaluators, reviewers, and viva examiners:

- **[WORKING.md](WORKING.md)**: Master functional platform specification, service architectures (Car Advisor, Explore, Favourites, Price Predictor), plain-English concepts, UI workflows, and mathematical formulations.
- **[BUILDING.md](BUILDING.md)**: End-to-end data cleaning, preprocessing, feature engineering, model training pipelines, and Master Viva Q&A.

---

## Design System

- **Background**: `#0e0e11`
- **Surface**: `#18181c`
- **Border**: `#2a2a30`
- **Accent**: `#22c55e`
- **Typography**: Inter
- **Aesthetic**: Modern automotive dark theme with zero emojis.
