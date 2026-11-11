"""
card_component.py — Interactive Vehicle Card for CarWise Streamlit Frontend.
Implements:
1. Top-right heart-shaped button (🤍 turns red ❤️ when liked/shortlisted).
2. Single-click on card toggles/expands full technical specifications drawer (no separate button).
3. Double-click on card triggers an opaque red heart animation in the center and likes the car.
4. Clean in-process SQLite and API synchronization.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
import streamlit as st
import streamlit.components.v1 as components


def get_user_liked_cars(user_id: str) -> set:
    """Retrieves the set of car IDs liked/shortlisted by the current user."""
    if st.session_state.get("_liked_cars_user") != user_id or "liked_cars" not in st.session_state:
        st.session_state["_liked_cars_user"] = user_id
        st.session_state["liked_cars"] = set()
        try:
            from backend.database import DatabaseManager
            db = DatabaseManager()
            favs = db.get_user_shortlist(user_id)
            st.session_state["liked_cars"] = {int(c.id) for c in favs}
        except Exception:
            pass
    return st.session_state["liked_cars"]


def toggle_car_like(user_id: str, car_id: int) -> bool:
    """Toggles like state for a car with immediate in-memory and SQLite persistence."""
    liked_set = get_user_liked_cars(user_id)
    is_currently_liked = car_id in liked_set

    try:
        from backend.favourites.service import FavouritesService
        service = FavouritesService()
        if is_currently_liked:
            service.remove_favourite(user_id, car_id)
            liked_set.discard(car_id)
            return False
        else:
            service.add_favourite(user_id, car_id)
            liked_set.add(car_id)
            return True
    except Exception:
        # Fallback in-memory
        if is_currently_liked:
            liked_set.discard(car_id)
            return False
        else:
            liked_set.add(car_id)
            return True


def render_car_card(
    car: Dict[str, Any],
    user_id: str,
    key_prefix: str = "card",
    rank: Optional[int] = None,
    show_match_score: bool = True,
):
    """
    Renders an interactive car card with:
    - Small heart-shaped button in the top right corner (red when liked).
    - Single click on card toggles full specifications drawer.
    - Double click on card triggers opaque red heart animation and likes the car.
    """
    cid = int(car.get("id", 0))
    brand = car.get("Brand") or car.get("brand", "")
    model = car.get("Car") or car.get("car", "")
    variant = car.get("Variant") or car.get("variant", "Standard")
    price_disp = car.get("Price_Display") or car.get("price_display", "Price on request")
    mileage = car.get("Mileage_Clean") or car.get("mileage")
    m_disp = f"{mileage:.1f} kmpl" if mileage else "—"
    disp = car.get("Displacement_Clean") or car.get("displacement")
    d_disp = f"{int(disp)} cc" if disp else "Electric / Unknown"
    fuel = car.get("Fuel_Type_Simple") or car.get("fuel_type", "Petrol")
    trans = car.get("Transmission_Simple") or car.get("transmission", "Manual")
    feat_score = int(car.get("Feature_Score") or car.get("feature_score", 0))

    # Auto-calculate EMI if not already provided
    emi_disp = car.get("estimated_emi")
    if not emi_disp or emi_disp == "—":
        p_lakh = float(car.get("Price_Lakh") or car.get("price_lakh") or 0.0)
        if p_lakh > 0:
            loan = p_lakh * 100_000 * 0.8
            r = 0.09 / 12
            emi = int(loan * r * ((1 + r) ** 60) / (((1 + r) ** 60) - 1))
            emi_disp = f"Rs. {emi:,}/mo"
        else:
            emi_disp = "—"

    # Contextual reason
    if car.get("explore_explanation"):
        reason = car["explore_explanation"]
    elif car.get("recommendation_reason"):
        reason = car["recommendation_reason"]
    elif car.get("viewed_at"):
        reason = "Browsed recently in your history"
    else:
        reason = f"Verified {brand} variant matching your criteria"

    desc = car.get("description")
    if not desc:
        desc = (
            f"{brand} {model} ({variant}) featuring a {int(disp) if disp else 'efficient'} cc "
            f"{fuel} engine paired with a {trans} transmission, offering an ARAI certified {m_disp}."
        )

    features_dict = car.get("features", {}) or {}

    liked_set = get_user_liked_cars(user_id)
    is_liked = cid in liked_set

    # Score
    score = float(car.get("Score") or car.get("hybrid_score", 0.0))

    # Features list
    feature_labels = []
    feat_map = {
        "sunroof___moonroof": "Sunroof / Moonroof",
        "cruise_control": "Cruise Control",
        "ventilated_seats": "Ventilated Seats",
        "wireless_charger": "Wireless Charger",
        "rear_ac": "Rear AC Vents",
        "heads_up_display": "Heads Up Display",
        "central_locking": "Central Locking",
        "hill_hold_control": "Hill Hold Control",
    }
    for fk, flabel in feat_map.items():
        if features_dict.get(fk) or features_dict.get(f"feat_{fk}"):
            feature_labels.append(flabel)
    if not feature_labels and feat_score > 0:
        feature_labels = [f"{feat_score} Premium Amenities Included"]

    # Wrap in Streamlit container: Card content on left (11 cols), Heart button on right (1 col)
    with st.container():
        c_left, c_heart = st.columns([11, 1])

        with c_heart:
            # Anchor tag specifically tagged for this card's button
            st.markdown(f'<div id="heart_anchor_{key_prefix}_{cid}" style="display:none;"></div>', unsafe_allow_html=True)
            heart_icon = "❤️" if is_liked else "🤍"
            btn_key = f"btn_heart_{key_prefix}_{cid}"
            if st.button(heart_icon, key=btn_key, help="Like"):
                toggle_car_like(user_id, cid)
                st.rerun()

        with c_left:
            # Render card with single-click expand and double-click heart animation
            rank_label = f"Rank {rank} &bull; " if rank else ""
            score_pill = f'<span class="result-score-pill">Match: {score:.1f}%</span>' if show_match_score else ""
            features_html = "".join(f'<span class="feature-pill">{f}</span>' for f in feature_labels)

            # Build strictly unindented HTML to prevent CommonMark code block parsing
            card_html = (
                f'<details class="card-drawer">'
                f'<summary style="all: unset; display: block; width: 100%; cursor: pointer;">'
                f'<div class="car-interactive-card" id="card_{key_prefix}_{cid}" data-card-key="{key_prefix}_{cid}">'
                f'<div class="card-top-row">'
                f'<div class="card-title-group">'
                f'<div style="font-size:0.72rem;font-weight:600;text-transform:uppercase;letter-spacing:0.08em;color:var(--text-muted);margin-bottom:0.25rem;">{rank_label}Ex-Showroom {price_disp}</div>'
                f'<div class="result-name" style="margin-bottom:0.4rem;">{brand} {model} <span style="font-weight:400;font-size:0.85rem;color:var(--text-muted);">({variant})</span></div>'
                f'<div style="background:rgba(34,197,94,0.1);border-left:3px solid var(--accent);color:var(--text);font-size:0.82rem;padding:0.35rem 0.65rem;border-radius:4px;margin-bottom:0.6rem;">{reason}</div>'
                f'</div>'
                f'<div style="text-align: right;">{score_pill}</div>'
                f'</div>'
                f'<div class="stat-row">'
                f'<div class="stat-item"><span class="stat-label">Price</span><span class="stat-value">{price_disp}</span></div>'
                f'<div class="stat-item"><span class="stat-label">Mileage</span><span class="stat-value">{m_disp}</span></div>'
                f'<div class="stat-item"><span class="stat-label">Engine</span><span class="stat-value">{d_disp}</span></div>'
                f'<div class="stat-item"><span class="stat-label">Powertrain</span><span class="stat-value">{fuel}</span></div>'
                f'<div class="stat-item"><span class="stat-label">Gearbox</span><span class="stat-value">{trans}</span></div>'
                f'<div class="stat-item"><span class="stat-label">Equipment</span><span class="stat-value">{feat_score}/8 Features</span></div>'
                f'<div class="stat-item"><span class="stat-label">Est. EMI</span><span class="stat-value">{emi_disp}</span></div>'
                f'</div>'
                f'</div>'
                f'</summary>'
                f'<div class="specs-expanded-drawer">'
                f'<div style="font-size:0.75rem;font-weight:600;text-transform:uppercase;letter-spacing:0.08em;color:var(--text-muted);margin-bottom:0.4rem;">Complete Vehicle Specifications &amp; Features</div>'
                f'<div style="font-size:0.85rem;color:var(--text);line-height:1.5;margin-bottom:0.75rem;">{desc}</div>'
                f'<div class="specs-expanded-grid">'
                f'<div class="specs-expanded-item"><div class="specs-expanded-label">Manufacturer</div><div class="specs-expanded-value">{brand}</div></div>'
                f'<div class="specs-expanded-item"><div class="specs-expanded-label">Model &amp; Trim</div><div class="specs-expanded-value">{model} ({variant})</div></div>'
                f'<div class="specs-expanded-item"><div class="specs-expanded-label">Price Bracket</div><div class="specs-expanded-value">{price_disp}</div></div>'
                f'<div class="specs-expanded-item"><div class="specs-expanded-label">Engine CC</div><div class="specs-expanded-value">{d_disp}</div></div>'
                f'<div class="specs-expanded-item"><div class="specs-expanded-label">Fuel Type</div><div class="specs-expanded-value">{fuel}</div></div>'
                f'<div class="specs-expanded-item"><div class="specs-expanded-label">Transmission</div><div class="specs-expanded-value">{trans}</div></div>'
                f'<div class="specs-expanded-item"><div class="specs-expanded-label">Efficiency</div><div class="specs-expanded-value">{m_disp}</div></div>'
                f'<div class="specs-expanded-item"><div class="specs-expanded-label">Financing (60m)</div><div class="specs-expanded-value">{emi_disp}</div></div>'
                f'</div>'
                f'<div style="margin-top:0.75rem;">'
                f'<div style="font-size:0.72rem;font-weight:600;text-transform:uppercase;color:var(--text-muted);margin-bottom:0.35rem;">Equipment &amp; Amenities</div>'
                f'<div class="features-pill-wrap">{features_html}</div>'
                f'</div>'
                f'</div>'
                f'</details>'
            )

            st.markdown(card_html, unsafe_allow_html=True)


def inject_double_click_heart_script():
    """
    Injects a client-side JavaScript watcher that detects double clicks on car cards,
    triggers the floating opaque red heart pop animation, and activates the top-right heart button.
    Uses exact data-card-key matching so every card activates its own heart button independently.
    """
    js_code = """
    <script>
    (function() {
        try {
            const parentDoc = window.parent.document;
            function attachDoubleTapListeners() {
                // 1. Style top-right heart buttons
                const buttons = parentDoc.querySelectorAll('button');
                buttons.forEach(btn => {
                    if (btn.innerText && (btn.innerText.includes('🤍') || btn.innerText.includes('❤️'))) {
                        btn.classList.add('heart-icon-btn');
                        if (btn.innerText.includes('❤️')) {
                            btn.classList.add('heart-liked');
                        } else {
                            btn.classList.remove('heart-liked');
                        }
                    }
                });

                // 2. Attach double-click listener to all cards
                const cards = parentDoc.querySelectorAll('.car-interactive-card');
                cards.forEach(card => {
                    if (card.dataset.heartWatcherAttached) return;
                    card.dataset.heartWatcherAttached = "true";

                    function triggerCardLike(e) {
                        if (e) {
                            e.preventDefault();
                            e.stopPropagation();
                        }

                        // 1. Play floating opaque red heart animation on THIS card
                        const heartElem = parentDoc.createElement('div');
                        heartElem.className = 'card-heart-popup';
                        heartElem.innerHTML = '❤️';
                        card.appendChild(heartElem);
                        setTimeout(() => {
                            if (heartElem && heartElem.parentNode) {
                                heartElem.parentNode.removeChild(heartElem);
                            }
                        }, 850);

                        // 2. Find THIS card's unique key and click its exact matching heart button
                        const cardKey = card.getAttribute('data-card-key') || card.id.replace('card_', '');
                        if (cardKey) {
                            const anchor = parentDoc.getElementById("heart_anchor_" + cardKey);
                            if (anchor) {
                                const colContainer = anchor.closest('[data-testid="column"]') || anchor.parentElement;
                                if (colContainer) {
                                    const targetBtn = colContainer.querySelector('button');
                                    if (targetBtn) {
                                        targetBtn.click();
                                        return;
                                    }
                                }
                            }
                        }

                        // Fallback: search within immediate container block
                        const parentRow = card.closest('[data-testid="stHorizontalBlock"]') || card.parentElement;
                        if (parentRow) {
                            const rowButtons = parentRow.querySelectorAll('button');
                            for (const b of rowButtons) {
                                if (b.innerText && (b.innerText.includes('🤍') || b.innerText.includes('❤️'))) {
                                    b.click();
                                    return;
                                }
                            }
                        }
                    }

                    let lastClickTime = 0;
                    card.addEventListener('dblclick', function(e) {
                        triggerCardLike(e);
                    });

                    card.addEventListener('click', function(e) {
                        const now = Date.now();
                        if (now - lastClickTime < 420) {
                            triggerCardLike(e);
                        }
                        lastClickTime = now;
                    });
                });
            }

            attachDoubleTapListeners();
            setInterval(attachDoubleTapListeners, 650);
        } catch (e) {
            // Sandboxed fallback
        }
    })();
    </script>
    """
    components.html(js_code, height=0, width=0)
