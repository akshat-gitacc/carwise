import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor

MODEL_NAME = "openai/clip-vit-base-patch32"
_model = None
_proc = None

PROMPTS = {
    "exterior_damage_score": (
        "a photo of a car with dents, scratches and body damage",
        "a photo of a car with no dents or scratches, undamaged body",
    ),
    "paint_condition_score": (
        "a photo of a car with faded, dull, peeling or discoloured paint",
        "a photo of a car with shiny, clean, well kept paint",
    ),
    "engine_condition_score": (
        "a photo of a dirty engine bay with rust and corrosion",
        "a photo of a clean engine bay with no rust",
    ),
}


def _load():
    global _model, _proc
    if _model is None:
        _model = CLIPModel.from_pretrained(MODEL_NAME).eval()
        _proc = CLIPProcessor.from_pretrained(MODEL_NAME)
    return _model, _proc


def _good_prob(images, bad, good):
    model, proc = _load()
    inputs = proc(text=[bad, good], images=images, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits_per_image      # shape (n_photos, 2)
    probs = logits.softmax(dim=1)[:, 1].numpy()
    return float(np.mean(probs))


def score_images(exterior, engine):
    """
    exterior, engine: lists of PIL images.
    Returns {target: score 0-100 or None}. None means no photos for that target.
    """
    out = {}
    for t, (bad, good) in PROMPTS.items():
        imgs = engine if t == "engine_condition_score" else exterior
        if not imgs:
            out[t] = None
        else:
            out[t] = 100 * _good_prob(imgs, bad, good)
    return out