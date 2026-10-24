import json
from io import BytesIO

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.staticfiles import StaticFiles
from PIL import Image

from backend.config import ROOT, TARGETS
from backend.cnn import score_images
from backend.explain import assess

app = FastAPI(title="CarWise")


async def _read_images(files):
    images = []
    for f in files:
        data = await f.read()
        if not data:
            continue
        try:
            images.append(Image.open(BytesIO(data)).convert("RGB"))
        except Exception:
            pass 
    return images


@app.post("/api/assess")
async def api_assess(
    car: str = Form(...),
    exterior: list[UploadFile] = File(default=[]),
    engine: list[UploadFile] = File(default=[]),
):
    car_data = json.loads(car)

    ext = await _read_images(exterior)
    eng = await _read_images(engine)
    visual = score_images(ext, eng)

    notes = []
    for t in TARGETS:
        if visual[t] is None:
            visual[t] = 50.0
            notes.append(f"No photo score for {t.replace('_', ' ')} (missing photo or untrained CNN head). Neutral 50 used.")
    car_data.update(visual)

    result = assess(car_data)
    result["visual_scores"] = {k: round(v, 1) for k, v in visual.items()}
    result["photo_notes"] = notes
    return result


app.mount("/", StaticFiles(directory=ROOT / "frontend", html=True), name="site")