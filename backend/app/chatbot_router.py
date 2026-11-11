"""
chatbot_router.py — FastAPI router for CarBot India Conversational Advisor.
Integrates CarBot RAG service into the primary CarWise FastAPI backend.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..chatbot.car_data import (
    CARS,
    DEALERSHIPS,
    get_car_by_name,
    get_dealerships_for_city,
)
from ..chatbot.chat_engine import chat, chat_simple

router = APIRouter(tags=["CarBot Chatbot"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class Message(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    history: List[Message] = Field(default_factory=list)
    api_key: Optional[str] = Field(default=None, description="Optional Anthropic API key")


class ChatResponse(BaseModel):
    reply: str
    model: str = "carbot-grounded-local"


# ---------------------------------------------------------------------------
# Chat Endpoints
# ---------------------------------------------------------------------------

@router.post("/api/chatbot/chat", response_model=ChatResponse, summary="Chat with CarBot")
@router.post("/chat", response_model=ChatResponse, include_in_schema=False)
async def chat_endpoint(req: ChatRequest):
    """
    Send a natural language prompt and receive CarBot's grounded advice.
    """
    history = [{"role": m.role, "content": m.content} for m in req.history]
    reply = await asyncio.to_thread(chat_simple, req.message, history, req.api_key)
    return ChatResponse(reply=reply)


@router.post("/api/chatbot/chat/stream", summary="SSE Streaming chat with CarBot")
@router.post("/chat/stream", include_in_schema=False)
async def chat_stream_endpoint(req: ChatRequest):
    """
    Stream CarBot's response as Server-Sent Events.
    """
    history = [{"role": m.role, "content": m.content} for m in req.history]

    async def event_generator():
        try:
            gen = chat(req.message, history, req.api_key)
            for chunk in gen:
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Vehicle Catalog & Showroom Endpoints
# ---------------------------------------------------------------------------

@router.get("/api/chatbot/cars", summary="List CarBot vehicles")
@router.get("/cars", include_in_schema=False)
async def list_cars(
    segment: Optional[str] = Query(default=None, description="Filter by segment"),
    brand: Optional[str] = Query(default=None, description="Filter by brand"),
    max_price: Optional[float] = Query(default=None, description="Max price in lakhs"),
    fuel: Optional[str] = Query(default=None, description="Fuel filter"),
):
    results = CARS
    if segment:
        results = [c for c in results if c["segment"].lower() == segment.lower()]
    if brand:
        results = [c for c in results if brand.lower() in c["brand"].lower()]
    if max_price:
        results = [c for c in results if c["price_lakh"]["min"] <= max_price]
    if fuel:
        results = [c for c in results if fuel.lower() in [f.lower() for f in c["fuel_types"]]]
    return {"count": len(results), "cars": results}


@router.get("/api/chatbot/cars/{car_id}", summary="Get specific CarBot vehicle")
@router.get("/cars/{car_id}", include_in_schema=False)
async def get_car(car_id: str):
    for car in CARS:
        if car["id"] == car_id:
            return car
    car = get_car_by_name(car_id)
    if car:
        return car
    raise HTTPException(status_code=404, detail=f"Car '{car_id}' not found")


@router.get("/api/chatbot/dealers", summary="List dealerships in a city")
@router.get("/dealers", include_in_schema=False)
async def list_dealers(
    city: str = Query(..., description="City name"),
    brand: Optional[str] = Query(default=None, description="Filter by brand"),
):
    dealers = get_dealerships_for_city(city)
    if not dealers:
        raise HTTPException(
            status_code=404,
            detail=f"No dealerships found for city '{city}'. Supported: {', '.join(DEALERSHIPS.keys())}",
        )
    if brand:
        dealers = [d for d in dealers if brand.lower() in d["brand"].lower()]
    return {"city": city, "count": len(dealers), "dealers": dealers}


@router.get("/api/chatbot/health", summary="CarBot Health Check")
@router.get("/health", include_in_schema=False)
async def health():
    return {
        "status": "ok",
        "service": "CarBot India",
        "cars_in_db": len(CARS),
        "cities_with_dealerships": list(DEALERSHIPS.keys()),
        "api_key_configured": bool(os.environ.get("ANTHROPIC_API_KEY")),
    }

