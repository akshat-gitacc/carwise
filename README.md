# CarWise

CarWise estimates the quality of a used car. You enter the car's details, optionally upload exterior and engine bay photos, and get a quality rating, the probability behind it, and an explanation of which factors pushed the rating up or down.

## What it does

- Predicts one of four quality levels: Poor, Fair, Good, or Excellent
- Uses an XGBoost model trained on synthetic car data
- Scores photos with CLIP, a pretrained image-text model, to produce rough exterior and engine condition signals
- Uses SHAP (TreeSHAP) to show which factors helped or hurt the rating
- Writes a plain-English explanation, either with the Claude API or from a template if no key is set

## Limitations

- The training data is synthetic. The high accuracy reflects how the data was generated, not real-world performance.
- The photo scores come from CLIP's matching against text prompts. They are not validated measurements of paint or damage.
- SHAP shows what the model relies on, not what causes a car to be in good or bad condition.
- This is not a substitute for a professional inspection.

## Project layout

```
carwise/
├── backend/          # Data, model, photo scoring, explanations, API
├── frontend/         # HTML, CSS and JavaScript
├── data/             # Generated dataset (created by the scripts)
├── models/           # Trained model (created by the scripts)
└── requirements.txt
```

## Getting started

You need Python 3.9 or newer.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Generate the dataset and train the model:

```bash
python -m backend.data_gen
python -m backend.train
```

Start the site:

```bash
uvicorn backend.app:app --reload
```

Then open http://127.0.0.1:8000 in your browser.

The first time you upload a photo, CLIP downloads its weights (about 600 MB), so the first request can take a few minutes.

## Optional: Claude-written explanations

To get explanations written by Claude instead of the template, set your API key before starting the server.

```bash
export ANTHROPIC_API_KEY=your_key_here          # Windows PowerShell: $env:ANTHROPIC_API_KEY="your_key_here"
```

Keep your API key out of the repository.

## Credits

Built with XGBoost, SHAP, PyTorch, Hugging Face Transformers (CLIP), FastAPI, and the Anthropic API.