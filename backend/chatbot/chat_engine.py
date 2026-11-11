"""
chat_engine.py — Conversational engine for CarBot India.
Provides grounded automotive advisory using local vehicle specifications,
dealership data, EMI calculations, and intelligent response synthesis.
Optional: Can stream via Anthropic Claude if ANTHROPIC_API_KEY is configured.
Part of backend.chatbot service.
"""

from __future__ import annotations

import os
import re
import time
from typing import Dict, Generator, List, Optional

from .car_data import (
    BUYING_TIPS,
    CARS,
    DEALERSHIPS,
    extract_preferences,
    format_car_summary,
    get_car_by_name,
    get_dealerships_for_brand_city,
    get_dealerships_for_city,
    recommend_cars,
)

# ---------------------------------------------------------------------------
# EMI Calculator
# ---------------------------------------------------------------------------

def estimate_emi(price_lakh: float, loan_pct: float = 0.80, rate_annual: float = 9.0, tenure_months: int = 60) -> int:
    """Return approximate monthly EMI in ₹."""
    principal = price_lakh * 100_000 * loan_pct
    r = rate_annual / 100 / 12
    emi = principal * r * (1 + r) ** tenure_months / ((1 + r) ** tenure_months - 1)
    return int(emi)


# ---------------------------------------------------------------------------
# Intent & City Extraction
# ---------------------------------------------------------------------------

def _detect_intent(text: str) -> Dict[str, bool]:
    """Detect user intent flags from text."""
    tl = text.lower()
    return {
        "is_greeting": any(w in tl for w in ["hi", "hello", "hey", "namaste", "good morning", "good evening"]) and len(tl.split()) <= 4,
        "wants_recommendation": any(w in tl for w in [
            "recommend", "suggest", "best", "which car", "what car", "choose",
            "good car", "ideal car", "should i buy", "help me choose", "what should i", "advise", "under", "budget",
            "seater", "seat", "family car", "top car", "top 3", "top 5", "options", "alternative", "alternatives",
            "suv", "sedan", "hatchback", "electric car", "ev", "cng"
        ]),
        "wants_comparison": any(w in tl for w in [
            "compare", "vs", "versus", "difference between", "better than", "which one", " or "
        ]),
        "wants_dealership": any(w in tl for w in [
            "dealer", "dealership", "showroom", "where to buy", "where can i buy", "where can i test", "near me",
            "test drive", "outlet", "showrooms"
        ]),
        "wants_emi": any(w in tl for w in [
            "emi", "loan", "finance", "monthly", "installment", "down payment", "interest rate", "afford", "downpayment"
        ]),
        "wants_tips": any(w in tl for w in [
            "tip", "guide", "first time", "buying guide", "insurance", "negotiate", "discount", "pdi"
        ]),
    }


def _detect_city(text: str) -> Optional[str]:
    """Extract city name from text."""
    cities = [
        "mumbai", "delhi", "bangalore", "bengaluru", "hyderabad", "chennai", "pune",
        "kolkata", "ahmedabad", "jaipur", "lucknow", "chandigarh", "noida", "gurgaon", "gurugram"
    ]
    tl = text.lower()
    for city in cities:
        if city in tl:
            return "bangalore" if city == "bengaluru" else ("gurugram" if city == "gurgaon" else city)
    return None


def _find_mentioned_cars(text: str) -> List[Dict]:
    """Find cars mentioned in user query using word boundaries."""
    tl = text.lower()
    found = []

    # Priority multi-word or specific checks
    if re.search(r"\bhonda\s+city\b", tl):
        c = get_car_by_name("city")
        if c and c not in found:
            found.append(c)
    elif re.search(r"\bcity\b", tl):
        # Exclude common phrases like "city driving", "city traffic", "which city", "my city"
        if not re.search(r"\bcity\s+(?:driving|commute|commuter|traffic|roads|use|car|limits|mileage)\b|\b(?:in|my|your|which|the|metro)\s+city\b", tl):
            c = get_car_by_name("city")
            if c and c not in found:
                found.append(c)

    if re.search(r"\binnova(?:\s+hycross)?\b", tl):
        c = get_car_by_name("innova_hycross")
        if c and c not in found:
            found.append(c)

    if re.search(r"\bxuv(?:700)?\b", tl):
        c = get_car_by_name("xuv700")
        if c and c not in found:
            found.append(c)

    # Standard car models with word boundaries
    car_keywords = [
        "swift", "creta", "nexon", "baleno", "seltos", "punch",
        "ertiga", "taigun", "hector", "safari", "elevate", "thar"
    ]
    for kw in car_keywords:
        if re.search(rf"\b{re.escape(kw)}\b", tl):
            c = get_car_by_name(kw)
            if c and c not in found:
                found.append(c)

    return found


def _format_mileage(car: Dict) -> str:
    m = car.get("mileage_kmpl", {})
    if isinstance(m, dict):
        return " | ".join(f"{v} kmpl ({k.title()})" for k, v in m.items())
    return str(m)


def _format_safety(car: Dict) -> str:
    r = car.get("safety_rating", 4)
    if isinstance(r, int):
        return f"{'⭐' * r} ({r}/5 GNCAP)"
    return str(r)


# ---------------------------------------------------------------------------
# Grounded Local Response Synthesis (Zero-Dependency & Zero-Latency)
# ---------------------------------------------------------------------------

def _generate_local_reply(user_message: str, history: List[Dict], city: Optional[str] = None) -> str:
    """
    Generates a natural, informative, and grounded response directly from local data.
    Ensures 100% offline functionality without requiring paid LLM tokens or external network.
    """
    intent = _detect_intent(user_message)
    mentioned_cars = _find_mentioned_cars(user_message)
    detected_city = city or _detect_city(user_message)
    tl = user_message.lower()

    # 1. Comparison Intent
    if (intent["wants_comparison"] or len(mentioned_cars) >= 2) and len(mentioned_cars) >= 2:
        car_a, car_b = mentioned_cars[0], mentioned_cars[1]
        emi_a = estimate_emi((car_a["price_lakh"]["min"] + car_a["price_lakh"]["max"]) / 2)
        emi_b = estimate_emi((car_b["price_lakh"]["min"] + car_b["price_lakh"]["max"]) / 2)
        
        reply = (
            f"### ⚖️ Head-to-Head Comparison: **{car_a['name']}** vs **{car_b['name']}**\n\n"
            f"| Specification | {car_a['name']} | {car_b['name']} |\n"
            f"|---|---|---|\n"
            f"| **Price Range** | ₹{car_a['price_lakh']['min']} - {car_a['price_lakh']['max']} Lakh | ₹{car_b['price_lakh']['min']} - {car_b['price_lakh']['max']} Lakh |\n"
            f"| **Segment** | {car_a['segment'].replace('_', ' ').title()} | {car_b['segment'].replace('_', ' ').title()} |\n"
            f"| **Mileage** | {_format_mileage(car_a)} | {_format_mileage(car_b)} |\n"
            f"| **Safety Rating** | {_format_safety(car_a)} | {_format_safety(car_b)} |\n"
            f"| **Est. Monthly EMI** | ~₹{emi_a:,}/mo | ~₹{emi_b:,}/mo |\n\n"
            f"**Key Trade-offs**:\n"
            f"- **{car_a['name']} Strengths**: {', '.join(car_a['pros'][:2])}.\n"
            f"- **{car_b['name']} Strengths**: {', '.join(car_b['pros'][:2])}.\n\n"
            f"💡 **Recommendation**: Choose **{car_a['name']}** if you prioritize {car_a['pros'][0].lower()}; "
            f"opt for **{car_b['name']}** if {car_b['pros'][0].lower()} matters more to your daily driving."
        )
        return reply

    # 2. Dealership Locator Query
    if intent["wants_dealership"]:
        if detected_city:
            if mentioned_cars:
                target_brand = mentioned_cars[0]["brand"]
                dealers = get_dealerships_for_brand_city(target_brand, detected_city)
                brand_label = f"**{target_brand}**"
            else:
                dealers = get_dealerships_for_city(detected_city)
                brand_label = "Authorized"

            if dealers:
                reply = f"### 🏙️ {brand_label} Showrooms & Dealerships in **{detected_city.title()}**:\n\n"
                for d in dealers[:6]:
                    reply += f"- **{d['brand']} — {d['name']}**\n  📍 {d['address']} ({d['area']})\n  📞 `{d['phone']}`\n\n"
                reply += "💡 *Tip: Contact the showroom directly to schedule a free doorstep test drive or inquire about current discounts.*"
                return reply
        return "🏙️ Please tell me which city you are located in (Mumbai, Delhi, Bangalore, Hyderabad, Chennai, or Pune) so I can locate authorized showrooms for you."

    # 3. EMI & Financial Guidance Query
    if intent["wants_emi"]:
        if mentioned_cars:
            car = mentioned_cars[0]
            avg_price = (car["price_lakh"]["min"] + car["price_lakh"]["max"]) / 2
            emi_3yr = estimate_emi(avg_price, tenure_months=36)
            emi_5yr = estimate_emi(avg_price, tenure_months=60)
            emi_7yr = estimate_emi(avg_price, tenure_months=84)
            down_payment = avg_price * 100_000 * 0.20
            loan_amount = avg_price * 100_000 * 0.80

            return (
                f"### 🧮 Loan & EMI Breakdown for **{car['name']}**\n\n"
                f"- **Ex-showroom Price**: ₹{car['price_lakh']['min']} - {car['price_lakh']['max']} Lakh (Average: ₹{avg_price:.2f} Lakh)\n"
                f"- **Recommended Down Payment (20%)**: ~₹{down_payment:,.0f}\n"
                f"- **Financed Loan Amount (80%)**: ~₹{loan_amount:,.0f} @ ~9.0% annual interest\n\n"
                f"| Tenure | Monthly EMI | Total Interest | Total Payout |\n"
                f"|---|---|---|---|\n"
                f"| **3 Years (36 mos)** | ₹{emi_3yr:,} | ₹{(emi_3yr*36 - loan_amount):,.0f} | ₹{(emi_3yr*36):,.0f} |\n"
                f"| **5 Years (60 mos)** | ₹{emi_5yr:,} | ₹{(emi_5yr*60 - loan_amount):,.0f} | ₹{(emi_5yr*60):,.0f} |\n"
                f"| **7 Years (84 mos)** | ₹{emi_7yr:,} | ₹{(emi_7yr*84 - loan_amount):,.0f} | ₹{(emi_7yr*84):,.0f} |\n\n"
                f"💡 **Affordability Advice**: Under the **20/4/10 Rule**, ensure the monthly EMI (₹{emi_5yr:,}) is under 10% of your net household monthly salary."
            )
        else:
            return (
                "### 🧮 Car Loan & EMI Guidance (20/4/10 Rule)\n\n"
                "When financing a car in India, automotive financial planners recommend the **20/4/10 rule**:\n"
                "1. **20% Down Payment**: Pay at least 20% upfront to avoid negative equity.\n"
                "2. **4-5 Years Loan Tenure**: Keep loan tenure to 4-5 years maximum to minimize interest burden.\n"
                "3. **10% Income Limit**: Monthly EMI should not exceed 10% of your gross monthly take-home salary.\n\n"
                "**Quick Monthly EMI Estimates** (at ~9% interest, 5-year tenure, 80% finance):\n"
                "- **₹8 Lakh Car**: ~₹13,280 / month\n"
                "- **₹12 Lakh Car**: ~₹19,920 / month\n"
                "- **₹16 Lakh Car**: ~₹26,570 / month\n"
                "- **₹20 Lakh Car**: ~₹33,210 / month\n\n"
                "💡 *Tell me a specific car model to calculate exact monthly EMIs!*"
            )

    # 4. Buying Tips / Advice Query
    if intent["wants_tips"]:
        return (
            "### 💡 Expert Car Buying Tips for India\n\n"
            "- **Fuel Selection**: Choose Petrol/CNG if monthly usage is under 1,000 km. Opt for Diesel or Strong Hybrid only if driving over 1,500 km/month.\n"
            "- **Insurance Strategy**: Never buy insurance blindly from the showroom; get quotes online and ask the dealer to match. Always insist on **Zero Depreciation (Bumper-to-Bumper)** and **Engine Protection** add-ons.\n"
            "- **Year-End Discounts**: November and December offer maximum corporate discounts and dealer clearance offers on outgoing models.\n"
            "- **Pre-Delivery Inspection (PDI)**: Conduct a thorough PDI using Form 22 and VIN decoder before releasing the final payment or registration."
        )

    # 5. Recommendation Query (Budget, Segment, Needs, or Alternatives)
    is_seeking_recommendation = intent["wants_recommendation"] or any(
        w in tl for w in ["recommend", "suggest", "like", "similar", "alternative", "alternatives", "under", "budget", "best", "options", "better than", "instead of"]
    )
    if is_seeking_recommendation:
        prefs = extract_preferences(user_message)
        if mentioned_cars:
            ref_car = mentioned_cars[0]
            if not prefs["segment"]:
                prefs["segment"] = ref_car.get("segment")
            if not prefs["seating"] and ref_car.get("seating", 5) > 5:
                prefs["seating"] = ref_car.get("seating")

        top_cars = recommend_cars(prefs, top_n=3)
        if top_cars:
            if mentioned_cars:
                reply = f"### 🚗 Top Recommended Cars Similar to **{mentioned_cars[0]['name']}** & Matching Your Criteria:\n\n"
            else:
                reply = "### 🚗 Top Recommended Cars Matching Your Needs:\n\n"

            for i, car in enumerate(top_cars, 1):
                avg_price = (car["price_lakh"]["min"] + car["price_lakh"]["max"]) / 2
                emi = estimate_emi(avg_price)
                reply += (
                    f"**{i}. {car['name']}** ({car['segment'].replace('_', ' ').title()})\n"
                    f"- **Price**: ₹{car['price_lakh']['min']} - {car['price_lakh']['max']} Lakh (Ex-showroom)\n"
                    f"- **Fuel & Trans**: {', '.join(car['fuel_types']).title()} | {', '.join(car['transmissions']).title()}\n"
                    f"- **Mileage**: {_format_mileage(car)} | **Safety**: {_format_safety(car)}\n"
                    f"- **Est. EMI**: ~₹{emi:,}/month\n"
                    f"- **Why it fits**: {car['pros'][0]}.\n\n"
                )
            if detected_city:
                reply += f"🏙️ *Authorized dealerships are available in {detected_city.title()} for test drives.*"
            else:
                reply += "💡 *Tell me your city if you'd like to find nearby authorized dealerships for test drives.*"
            return reply

    # 6. Specific Single Car Deep-Dive (Specs, Pros/Cons, Pricing)
    if len(mentioned_cars) == 1:
        car = mentioned_cars[0]
        avg_price = (car["price_lakh"]["min"] + car["price_lakh"]["max"]) / 2
        emi = estimate_emi(avg_price)
        reply = (
            f"### 🚗 **{car['name']}** ({car['segment'].replace('_', ' ').title()})\n\n"
            f"- **Price**: ₹{car['price_lakh']['min']} - {car['price_lakh']['max']} Lakh (Ex-showroom)\n"
            f"- **Fuel & Transmission**: {', '.join(car['fuel_types']).title()} | {', '.join(car['transmissions']).title()}\n"
            f"- **Mileage**: {_format_mileage(car)}\n"
            f"- **Safety**: {_format_safety(car)}\n"
            f"- **Estimated EMI**: ~₹{emi:,}/month (based on 80% loan over 5 years @ 9% interest)\n\n"
            f"**Key Highlights**:\n"
            f"- ✅ {car['pros'][0]}\n"
            f"- ✅ {car['pros'][1] if len(car['pros']) > 1 else 'Great value for money'}\n"
            f"- ⚠️ Note: {car['cons'][0] if car['cons'] else 'Waiting periods may apply in certain cities'}\n"
        )
        if detected_city:
            dealers = get_dealerships_for_brand_city(car["brand"], detected_city)
            if dealers:
                reply += f"\n🏙️ **Authorized Dealerships in {detected_city.title()}**:\n"
                for d in dealers[:3]:
                    reply += f"- **{d['name']}** ({d['area']}) — Ph: `{d['phone']}`\n"
        return reply

    # 7. Greeting or Default Assistance
    return (
        "Hello! 🚗 I'm **CarBot**, your AI automotive advisor for the Indian car market.\n\n"
        "How can I assist you today?\n"
        "- **Find a Car**: Tell me your budget and driving needs (e.g., *'Best SUV under 15 lakh for Bangalore city'*).\n"
        "- **Compare Models**: Ask me to compare two cars (e.g., *'Creta vs Seltos'* or *'Swift vs Baleno'*).\n"
        "- **Dealerships**: Find showrooms in Mumbai, Delhi, Bangalore, Hyderabad, Chennai, or Pune.\n"
        "- **EMI & Buying Tips**: Ask for loan calculations or insurance negotiation advice."
    )


# ---------------------------------------------------------------------------
# Streaming Chat Function (Local Generator + Optional Claude Support)
# ---------------------------------------------------------------------------

def chat(
    user_message: str,
    history: List[Dict],
    api_key: Optional[str] = None,
) -> Generator[str, None, None]:
    """
    Stream a chat response.
    If an Anthropic API key is provided and anthropic package is installed, streams from Claude.
    Otherwise, instantly generates a rich, grounded response from local Indian automotive data.
    """
    key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")

    # If Anthropic key is present, attempt LLM call
    if key:
        try:
            import anthropic  # Dynamic import so anthropic is not a hard requirement
            client = anthropic.Anthropic(api_key=key)
            full_system = (
                "You are CarBot, an expert automotive advisor for India. "
                "Recommend specific models, price in Lakhs, and provide grounded advice."
            )
            messages = [{"role": m["role"], "content": m["content"]} for m in history]
            messages.append({"role": "user", "content": user_message})

            with client.messages.stream(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1024,
                system=full_system,
                messages=messages,
            ) as stream:
                for chunk in stream.text_stream:
                    yield chunk
            return
        except Exception:
            # Fall back to local grounded response if API key fails or network error occurs
            pass

    # Grounded Local Execution (Zero-API, Zero-Cost, 100% Offline)
    city = _detect_city(user_message)
    if not city:
        for msg in reversed(history[-4:]):
            if msg.get("role") == "user":
                city = _detect_city(msg.get("content", ""))
                if city:
                    break

    reply_text = _generate_local_reply(user_message, history, city=city)

    # Stream text cleanly in small sentences / paragraphs
    lines = reply_text.split("\n")
    for i, line in enumerate(lines):
        yield line + ("\n" if i < len(lines) - 1 else "")
        time.sleep(0.015)


def chat_simple(
    user_message: str,
    history: List[Dict],
    api_key: Optional[str] = None,
) -> str:
    """Non-streaming version of chat() — returns full response string instantly."""
    key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if key:
        return "".join(chat(user_message, history, api_key))

    city = _detect_city(user_message)
    if not city:
        for msg in reversed(history[-4:]):
            if msg.get("role") == "user":
                city = _detect_city(msg.get("content", ""))
                if city:
                    break
    return _generate_local_reply(user_message, history, city=city)

