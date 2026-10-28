# 🧠 How DawaRx Works — Detailed Explanation

This document is a code-accurate, step-by-step guide explaining exactly what happens from the moment you click **"Analyze"** to the moment the results appear on screen.

---

## Big Picture Flow

```
You upload an image
       │
       ▼
[Frontend — Next.js]
       │  multipart/form-data POST /api/analyze
       ▼
[Backend — FastAPI]
       │
       ├─ Step 1:  Validate & Preprocess Image    (preprocess.py)
       ├─ Step 2:  Enhance Image with OpenCV      (vision.py)
       ├─ Step 3:  Detect Text Lines via EasyOCR  (vision.py)
       ├─ Step 4:  Read Handwriting via TrOCR     (vision.py)
       ├─ Step 5:  Parse Medicines from Raw Text  (vision.py)
       ├─ Step 6:  Normalize Medicine Names       (normalizer.py)
       ├─ Step 7:  Fuzzy-Match to Brand DB        (matcher.py)
       ├─ Step 8:  Fuzzy-Match to Jan Aushadhi    (matcher.py)
       ├─ Step 9:  Calculate Savings              (pricing.py)
       └─ Step 10: Build & Return Response        (analyze.py)
       │
       ▼
[Frontend renders result cards]
```

---

## Step 1 — Image Validation & Preprocessing
**File:** `backend/app/services/preprocess.py`

When your image arrives at the server, the first thing we do is **clean it up** before any AI sees it. This is handled by `preprocess_image()` using the **Pillow** library.

```
Raw upload bytes
  → Fix EXIF rotation  (a phone photo taken sideways gets corrected automatically)
  → Resize             (if larger than 1600×1600 px, shrink proportionally)
  → Convert to JPEG RGB (strips alpha/transparency from PNGs)
  → Output: clean image bytes
```

**Why this matters:** TrOCR and EasyOCR both perform much better on properly oriented, reasonably sized images. A 12 MP photo taken sideways would confuse the line detector entirely.

```python
# preprocess.py
img = ImageOps.exif_transpose(img)            # Fix phone rotation
img.thumbnail((1600, 1600), Image.LANCZOS)    # Resize if needed
img.save(output, format="JPEG", quality=90)
```

---

## Step 2 — Image Enhancement with OpenCV
**File:** `backend/app/services/vision.py` → `run_trocr()`

After Pillow cleanup, the image is passed to **OpenCV** for contrast enhancement. Prescriptions are often photographed under uneven lighting — one corner brighter than another, or the paper has a yellowish tint.

```
Clean JPEG
  → Convert to grayscale         (removes colour, focuses on ink vs. paper)
  → Adaptive Thresholding:
       Divides image into 11×11 pixel blocks
       Each block gets its OWN brightness cutoff (not one global level)
       Result: pure black ink on white paper regardless of lighting variation
  → Convert back to RGB          (required by EasyOCR's input format)
```

**Why adaptive (not global) thresholding?** A global threshold like "pixels darker than 128 = ink" fails when part of the prescription is in shadow. Adaptive thresholding computes the ideal cutoff locally for each region, so shadows are handled correctly.

```python
# vision.py
gray   = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
thresh = cv2.adaptiveThreshold(gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2)
```

---

## Step 3 — Text Line Detection with EasyOCR
**File:** `backend/app/services/vision.py` → `run_trocr()`

The enhanced binary image is handed to **EasyOCR** — but only for **detection** (finding *where* the text lines are), not for reading them. EasyOCR is excellent at locating text regions but is less accurate for messy handwriting than TrOCR.

```
Enhanced binary image
  → EasyOCR.detect()  scans for horizontal text regions
  → Returns bounding boxes: [x_min, x_max, y_min, y_max] per line
  → Boxes sorted top-to-bottom (by y_min) to preserve reading order
  → Each box padded by 5 px on all sides (avoid cutting off edge characters)
  → Each box cropped from the ORIGINAL colour image (not the binary one)
```

**Why this two-model approach?** TrOCR was trained on **single lines** of text. Feeding it a full page of a prescription gives terrible results because it cannot handle multiple lines at once. EasyOCR solves the "where is the text?" problem; TrOCR solves the "what does it say?" problem.

```python
# vision.py
detect_res = _easyocr_reader.detect(thresh_rgb)
bboxes     = sorted(detect_res[0][0], key=lambda b: b[2])  # sort by y_min

for box in bboxes:
    crop = image.crop((x_min, y_min, x_max, y_max))  # crop original image
    # → feed crop to TrOCR (Step 4)
```

---

## Step 4 — Handwriting Recognition with TrOCR
**File:** `backend/app/services/vision.py` → `run_trocr()`

Each cropped line image is individually fed into **Microsoft TrOCR** (`microsoft/trocr-large-handwritten`). TrOCR is a transformer model trained specifically on handwritten text. It is made of two parts:

### 4a. Vision Encoder (ViT — Vision Transformer)
The image crop is split into a grid of 16×16 pixel patches. Each patch becomes a 768-dimensional vector. The encoder attends to relationships between all patches to build a rich understanding of the visual structure of the handwriting.

### 4b. Text Decoder (GPT-2 style)
The decoder takes the encoder's output and generates text **one token at a time** — predicting the most probable next character or word-piece given everything generated so far. It stops when it generates an end-of-sequence token or hits the 128-token limit.

### 4c. Confidence Score
After each token is generated, the model outputs a probability distribution over its entire vocabulary (~50,000 tokens). The probability of the *chosen* token (via softmax) is its confidence. The **mean of all per-token probabilities** is reported as the line's confidence score (0.0–1.0).

```
Line crop image
  → TrOCRProcessor:  resize to 384×384, normalize pixel values
  → ViT Encoder:     image patches → context vectors
  → GPT-2 Decoder:  context → token IDs  (max 128 tokens)
  → batch_decode():  token IDs → string, e.g. "Tab Augmentin 625 BD x7 days PC"
  → confidence:      mean softmax probability across all generated tokens
```

All line texts are joined with `\n` to form the full **raw OCR text** of the prescription.

```python
# vision.py
outputs   = _trocr_model.generate(pixel_values,
                output_scores=True, return_dict_in_generate=True, max_new_tokens=128)
line_text = _trocr_processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0]
line_conf = _compute_confidence(outputs.scores)  # mean softmax probability
```

---

## Step 5 — Parsing Medicines from Raw Text
**File:** `backend/app/services/vision.py` → `parse_prescription_text()`

The raw OCR text (e.g., `"Tab Augmentin 625 BD x7 days PC\nTab Pan 40 OD AC"`) is processed by a **rule-based parser** to extract structured fields.

### Header Detection (first 5 lines)
```
Pattern: "Mr./Mrs./Ms./Dr. <Name>"  →  patient_name / doctor_name
Pattern: "12/10/2024" or "12-10-24"  →  date
```

### Medicine Line Detection
Each line is tested against a relaxed regex that accepts:
- Lines starting with a number: `"1. Dolo 650"`
- Lines starting with Tab / Cap / Syp / Inj: `"Tab Pan 40"`
- Any line starting with 2+ alphanumeric characters (forgiving of OCR typos)

### Per-Medicine Field Extraction (all via regex lookups)

```
Line: "Tab Augmentin 625 BD x7 days PC"
  → name:            "Augmentin 625"            (regex capture group)
  → form:            "tablet"                   (found keyword "Tab")
  → strength:        "625mg"                    (STRENGTH_RE pattern)
  → frequency:       "BD"                       (FREQ_MAP key match)
  → frequency_plain: "Twice a day"              (FREQ_MAP value)
  → timing:          "after food"               (TIMING_MAP: "PC" → "after food")
  → duration:        "7 days"                   (DURATION_RE: \d+ days?)
```

### Indian Abbreviation Reference

| Written on prescription | Meaning |
|---|---|
| OD | Once a day |
| BD | Twice a day |
| TDS | Three times a day |
| QID | Four times a day |
| HS | At bedtime |
| SOS | As needed (when required) |
| 1-0-1 | Twice a day (morning and night) |
| 1-1-1 | Three times a day |
| 1-0-0 | Once a day (morning only) |
| 0-0-1 | Once a day (night only) |
| AC | Before food (Ante Cibum) |
| PC | After food (Post Cibum) |
| Tab | Tablet |
| Cap | Capsule |
| Syp | Syrup |
| Inj | Injection |

---

## Step 6 — Name Normalization
**File:** `backend/app/services/normalizer.py`

Before the extracted medicine name can be matched against our database, it needs to be **normalized** — stripped of form prefixes, dosage numbers, and punctuation that would confuse the fuzzy search.

```
"Tab. Dolo 650mg"          →  "dolo"
"Augmentin 625 Duo Tablet" →  "augmentin duo"
"Pan-40 Cap"               →  "pan-40"
```

Three cleaning steps applied in sequence:

1. **Strip form tokens:** `tab`, `tablet`, `cap`, `capsule`, `syp`, `syrup`, `inj`, `gel`, `cream`, `lotion`, `drops`, etc.
2. **Strip dosage patterns:** `650mg`, `10 ml`, `0.5%`, and bare standalone integers not part of a brand name.
3. **Strip punctuation** — but **hyphens are kept** because `Pan-40` is a real brand name and removing the hyphen would break matching.

```python
# normalizer.py
text = _FORM_TOKENS.sub(" ", text.lower())  # remove "tab", "cap" etc.
text = _DOSAGE_PATTERN.sub(" ", text)       # remove "650mg", "10ml" etc.
text = _PUNCTUATION.sub(" ", text)          # remove . ( ) but NOT -
text = _WHITESPACE.sub(" ", text).strip()
```

---

## Step 7 — Fuzzy Brand Matching
**File:** `backend/app/services/matcher.py` → `match_brand()`

The normalized name is matched against our **~191,637 Indian brand medicines** database using **RapidFuzz**. RapidFuzz implements Levenshtein-based fuzzy matching but is ~100× faster than `fuzzywuzzy` because it is written in C++.

**Scorer used: `fuzz.WRatio`** — a weighted combination of:
- `partial_ratio` — handles substrings
- `token_sort_ratio` — handles word order differences
- `token_set_ratio` — handles extra/missing words

This combination handles:
- Different word orderings: `"Pan 40"` vs `"40 Pan"`
- Partial matches: `"Augmentin"` matching `"Augmentin 625 Duo"`
- OCR character errors: `"Augrnentin"` still matching `"Augmentin"` (score ~90)

```
Normalized query: "augmentin duo"
Top matches from 191k brands:
  → "augmentin 625 duo"   score=95  ✅ best match
  → "augmentin duo forte" score=89
  → "augmentin 375"       score=82
If best score < 80 (brand_match_cutoff) → needs_review = True
```

The brand match gives us back the **composition/salt string** (e.g., `"Amoxycillin + Clavulanic Acid"`) which drives the next step.

**Fuzzy Autocorrect:** Before matching, if the raw OCR name scores ≥ 70 against a known brand, the OCR typo is silently corrected to the real brand name improving downstream matching.

```python
# matcher.py
results = process.extract(normalize_name(query),
    _brands_df["name_normalized"],
    scorer=fuzz.WRatio, limit=3, score_cutoff=0)
best_text, best_score, best_idx = results[0]
```

---

## Step 8 — Jan Aushadhi Generic Matching
**File:** `backend/app/services/matcher.py` → `match_generic()`

The **composition string** from the brand match (e.g., `"Amoxycillin + Clavulanic Acid"`) is matched against our **~1,629 Jan Aushadhi (PMBJP) generic medicines** using a different scorer.

**Scorer used: `fuzz.token_set_ratio`** — compares the *set* of unique words, ignoring order and duplicates. This is ideal for compositions because:

```
"Amoxycillin + Clavulanic Acid 125mg"  ==  "Clavulanic Acid + Amoxycillin"
```

Both have the same chemical ingredients regardless of order, so `token_set_ratio` scores them as 100.

### Strength-Aware Matching
```
composition:         "Amoxycillin + Clavulanic Acid"
preferred_strength:  "625mg"
  → Fetch top 6 candidates by composition score
  → Among those, filter for entries where strength == "625mg"
  → If found:     use that as best match, no warning
  → If NOT found: use best compositional match + set strength_warning = True
```

A `strength_warning = True` means the generic found has a different pack strength than prescribed — the frontend shows a yellow warning badge.

```python
# matcher.py
results = process.extract(normalize_composition(composition),
    _generics_df["name_normalized"],
    scorer=fuzz.token_set_ratio, limit=6)

strength_matches = [(t,s,i) for t,s,i in candidates
    if generics_df.iloc[i]["strength_normalized"] == preferred_strength]
```

---

## Step 9 — Savings Calculation
**File:** `backend/app/services/pricing.py`

With both the brand price and generic MRP known, the savings calculation is simple arithmetic:

```python
# pricing.py
amount  = brand_price - generic_mrp          # e.g., ₹220.50 − ₹85.00 = ₹135.50
percent = (amount / brand_price) * 100       # e.g., 61.5%
```

**`needs_review` flag** is set to `True` if any of these conditions hold:
| Condition | Threshold |
|---|---|
| Per-medicine OCR confidence | < 0.6 |
| Brand fuzzy match score | < 80 (configurable) |
| Generic fuzzy match score | < 75 (configurable) |

A medicine flagged `needs_review` is highlighted in amber in the UI with a "Please verify" tag and a manual search box.

---

## Step 10 — Response Assembly
**File:** `backend/app/routes/analyze.py`

All per-medicine results are assembled into the final JSON response:

```json
{
  "prescription": {
    "patient_name": "Ramesh Kumar",
    "doctor_name": "Dr. Sharma",
    "date": "10/10/2024",
    "overall_legibility": "medium",
    "ocr_confidence": 0.7123
  },
  "medicines": [
    {
      "extracted": {
        "name": "Augmentin 625",
        "form": "tablet",
        "strength": "625mg",
        "frequency": "BD",
        "frequency_plain": "Twice a day",
        "timing": "after food",
        "duration": "7 days",
        "confidence": 0.693
      },
      "brand_match": {
        "name": "Augmentin 625 Duo",
        "composition": "Amoxycillin + Clavulanic Acid",
        "price": 220.5,
        "score": 95.0
      },
      "generic_match": {
        "name": "AMOXYCILLIN + CLAVULANIC ACID",
        "strength": "625mg",
        "mrp": 85.0,
        "score": 88.0
      },
      "savings": { "amount": 135.5, "percent": 61.5 },
      "needs_review": false
    }
  ],
  "total_savings": { "amount": 135.5, "percent": 61.5 },
  "model_metrics": {
    "ocr":        { "avg_confidence": 0.712, "cer": null, "wer": null },
    "extraction": { "medicine_count": 3, "precision": null, "recall": null, "f1": null },
    "matching":   { "coverage_rate": 1.0, "avg_brand_score": 92.3, "avg_generic_score": 85.1, "no_match_rate": 0.0 }
  }
}
```

The frontend renders a card for each medicine showing brand → generic comparison, savings percentage, confidence badge, and a "Please verify" warning for low-confidence entries.

---

## How Accuracy Metrics Are Computed
**File:** `backend/app/services/evaluator.py`

### 1. OCR Metrics (text quality)
| Metric | How it's calculated | When available |
|---|---|---|
| `avg_confidence` | Mean softmax token probability from TrOCR (0–1) | Always (every request) |
| `cer` | Character Error Rate via `jiwer.cer(reference, prediction)` | Only with ground truth |
| `wer` | Word Error Rate via `jiwer.wer(reference, prediction)` | Only with ground truth |

### 2. Extraction Metrics (medicine detection quality)
Uses **token-overlap (Jaccard similarity)** — "Dolo" matches "Dolo 650" if overlap ≥ 60%.

| Metric | Formula |
|---|---|
| Precision | TP / (TP + FP) — what fraction of extracted medicines are correct |
| Recall    | TP / (TP + FN) — what fraction of real medicines were found |
| F1        | 2 × P × R / (P + R) — balanced score |

> Requires ground-truth medicine names — use `POST /api/evaluate`.

### 3. Matching Metrics (pipeline quality — always live)
| Metric | Meaning |
|---|---|
| `coverage_rate` | Fraction of medicines that got a Jan Aushadhi generic match |
| `avg_brand_score` | Average RapidFuzz score across brand matches (0–100) |
| `avg_generic_score` | Average RapidFuzz score across generic matches (0–100) |
| `no_match_rate` | Fraction of medicines with no generic found |

---

## Architecture Diagram (Full)

```
[Next.js UI — localhost:3000]
         │
         │  POST /api/analyze  (multipart image)
         ▼
[FastAPI — localhost:8000]
         │
         ├─► preprocess.py
         │     Pillow: fix EXIF rotation → resize ≤1600px → convert to JPEG
         │
         ├─► vision.py
         │     │
         │     ├─► OpenCV
         │     │     grayscale → adaptive thresholding → binary image
         │     │
         │     ├─► EasyOCR (detect only)
         │     │     binary image → bounding boxes per text line
         │     │     sort boxes top→bottom
         │     │
         │     ├─► TrOCR  (for each line crop)
         │     │     ViT encoder: image patches → context vectors
         │     │     GPT-2 decoder: context → token IDs → text string
         │     │     confidence: mean softmax token probability
         │     │
         │     └─► Rule-based Parser
         │           raw text → patient, doctor, date
         │           regex → medicines[], form, strength, frequency, timing, duration
         │           FREQ_MAP: OD/BD/TDS/1-0-1 → plain English
         │           TIMING_MAP: AC/PC → before/after food
         │
         ├─► normalizer.py
         │     strip form tokens (tab/cap/syp...)
         │     strip dosage patterns (650mg, 10ml...)
         │     strip punctuation (keep hyphens)
         │
         ├─► matcher.py
         │     │
         │     ├─► Brand DB (191,637 rows) via RapidFuzz WRatio
         │     │     → composition string + price
         │     │
         │     └─► Jan Aushadhi (1,629 rows) via RapidFuzz token_set_ratio
         │           strength-aware: prefer matching strength → otherwise warn
         │           → generic name + MRP
         │
         ├─► pricing.py
         │     savings = brand_price − generic_mrp
         │     percent = savings / brand_price × 100
         │
         └─► evaluator.py
               OCR metrics: avg_confidence, CER, WER
               Extraction metrics: precision, recall, F1
               Matching metrics: coverage_rate, avg_brand_score, no_match_rate
```
