# 💊 DawaRx — AI Prescription Decipherer & Generic Finder

> Upload a photo of your handwritten Indian doctor's prescription. DawaRx uses a **fully offline, local AI model** to extract medicines and find cheaper **Jan Aushadhi (PMBJP) generic alternatives** with price savings — no API key, no internet required after setup.

> 📖 **Want to understand how the project works?** See [Explanation.md](./Explanation.md) for a detailed step-by-step technical walkthrough.

---

## ✨ Features

- 🔒 **100% Offline** — handwriting recognition runs locally using Microsoft's TrOCR model. No cloud API, no data leaves your machine.
- 💊 **Smart Medicine Extraction** — detects medicine names, dosage form, strength, frequency (OD / BD / TDS / 1-0-1 notation), timing (AC/PC), and duration from messy handwritten prescriptions.
- 🔍 **Fuzzy Brand Matching** — maps extracted names to a database of ~191,000 Indian branded medicines with composition and price.
- 🏥 **Jan Aushadhi Generic Finder** — finds cheaper PMBJP generic alternatives for each medicine with price savings calculated.
- 📊 **Confidence & Accuracy Metrics** — per-medicine OCR confidence, brand match scores, and overall coverage rate.

---

## Architecture

```
[Next.js UI] ──multipart image──▶ [FastAPI /api/analyze]
                                         │
                     1. Pillow: resize ≤1600px, fix EXIF, convert JPEG
                     2. OpenCV: adaptive thresholding for contrast enhancement
                     3. EasyOCR: detect and crop individual text lines
                     4. TrOCR (local): handwriting → raw text per line
                     5. Rule-based parser → structured medicines[]
                     6. Per medicine:
                          a. Normalize name (strip form/dosage tokens)
                          b. Fuzzy match → Brand DB (RapidFuzz WRatio)
                             → composition / strength / price
                          c. Fuzzy match composition → Jan Aushadhi list
                             → generic name / MRP
                          d. Fuzzy autocorrect (brand score ≥ 70)
                          e. Compute savings
                     7. Return combined JSON with confidence scores & metrics
```

---

## Folder Structure

```
backend/
  app/
    main.py            # FastAPI app, CORS, startup data loading
    config.py          # Pydantic-settings env config
    schemas.py         # All Pydantic v2 models
    routes/
      analyze.py       # POST /api/analyze  GET /api/search  GET /api/health
      evaluate.py      # POST /api/evaluate (with ground-truth comparison)
    services/
      vision.py        # TrOCR pipeline: OpenCV → EasyOCR → TrOCR → parser
      preprocess.py    # Pillow image cleanup (resize, EXIF fix)
      normalizer.py    # Name/dosage normalization
      matcher.py       # RapidFuzz brand→generic pipeline
      pricing.py       # Savings calculation
      evaluator.py     # CER / WER / Precision / Recall / F1 metrics
    data/
      A_Z_medicines_dataset_of_India.csv              # ~191k brand medicines
      sarkariyojana.com-jan-aushadhi-*.csv            # ~1.6k Jan Aushadhi generics
  tests/
    test_normalizer.py
    test_matcher.py
    test_evaluator.py

frontend/
  app/
    page.tsx           # Single-page app
    layout.tsx         # Root layout + SEO metadata
    globals.css        # Full design system
  components/
    UploadCard.tsx
    MedicineCard.tsx
    SavingsBanner.tsx
    ConfidenceBadge.tsx
  lib/api.ts           # Typed fetch client
  types/index.ts       # Shared TypeScript types
```

---

## Quick Start

### Prerequisites

- Python 3.11+ and `pip`
- Node.js 18+ and `npm`
- ~1.5 GB free disk space (for TrOCR model weights, downloaded once on first run)

**No API key required.**

---

### 1. Backend

```bash
cd backend

# Install Python dependencies
pip install -r requirements.txt

# Start the server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
# → http://localhost:8000
```

> **First run note:** The TrOCR model (~1.2 GB) is automatically downloaded from HuggingFace and cached at `~/.cache/huggingface/`. Subsequent starts are instant.

---

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
# → http://localhost:3000
```

---

## Environment Variables

All configuration lives in `backend/.env`. No secrets or API keys needed.

| Variable | Default | Description |
|---|---|---|
| `TROCR_MODEL` | `microsoft/trocr-large-handwritten` | HuggingFace model for OCR. Use `trocr-base-handwritten` for faster (less accurate) inference. |
| `TROCR_DEVICE` | `cpu` | Inference device. Set to `cuda` if you have an NVIDIA GPU for ~5× speedup. |
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | Backend URL used by the frontend. |

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/health` | Liveness check — reports model status and dataset sizes |
| `POST` | `/api/analyze` | Analyze prescription image (multipart `file`) |
| `GET` | `/api/search?q=` | Fuzzy search both databases for manual correction |
| `POST` | `/api/evaluate` | Full evaluation with optional ground-truth comparison |

---

## Running Tests

```bash
cd backend
python -m pytest tests/ -v
```

---

## Data Sources

| Dataset | Records | Source |
|---|---|---|
| A-Z Medicines India | ~191,637 | Kaggle — Indian medicines with brand name, composition, price |
| Jan Aushadhi (PMBJP) | ~1,629 | sarkariyojana.com / janaushadhi.gov.in product list |

---

## Known Limitations & Future Improvements

- **OCR accuracy** depends on image quality. Very faint, smudged, or rotated prescriptions may produce low-confidence results. Photographing under good lighting significantly helps.
- **Brand prices** in the dataset may be outdated; savings figures are indicative only.
- **No user session / history** — all processing is stateless and images are never stored.
- **Strength matching** is best-effort: if the Jan Aushadhi list has no entry for the exact strength, the closest compositional match is shown with a warning.
- **CPU inference is slow** (~15–30s per prescription). Using `TROCR_DEVICE=cuda` gives ~5× speedup.
- **Future**: PDF prescription support, drug interaction warnings, nearby Jan Aushadhi store locator, and a fine-tuned Indian handwriting OCR model.

---

## Disclaimer

> ⚠️ AI-generated interpretation. **Always confirm with your doctor or pharmacist** before purchasing or substituting any medicine. Generic substitution should only be done under medical supervision.
