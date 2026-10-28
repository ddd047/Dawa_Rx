"""
HuggingFace TrOCR Vision service.

Replaces Gemini Vision API with a fully local, offline OCR pipeline:
  1. TrOCR (microsoft/trocr-base-handwritten) — handwriting → raw text
  2. Rule-based parser — raw text → structured PrescriptionExtraction

No API key required. Model weights (~350 MB) are downloaded on first run
and cached at ~/.cache/huggingface/.

Architecture:
  Image bytes
    └─► preprocess_for_trocr()   (Pillow → RGB PIL image)
    └─► run_trocr()              (TrOCR inference → raw_text + confidence)
    └─► parse_prescription_text() (regex parser → PrescriptionExtraction)
"""
from __future__ import annotations

import logging
import math
import re
from io import BytesIO
from typing import Optional

from PIL import Image

from app.schemas import ExtractedMedicine, PrescriptionExtraction

logger = logging.getLogger(__name__)

# ── Global model cache (loaded once at startup) ────────────────────────────────
_trocr_processor = None
_trocr_model = None
_trocr_loaded = False
_easyocr_reader = None

# ── Indian medicine abbreviation map ─────────────────────────────────────────
FREQ_MAP = {
    r"\bOD\b": "Once a day",
    r"\bBD\b": "Twice a day",
    r"\bTDS\b": "Three times a day",
    r"\bQID\b": "Four times a day",
    r"\bHS\b": "At bedtime",
    r"\bSOS\b": "As needed",
    r"1[-–]0[-–]1": "Twice a day (morning and night)",
    r"1[-–]1[-–]1": "Three times a day",
    r"1[-–]0[-–]0": "Once a day (morning only)",
    r"0[-–]0[-–]1": "Once a day (night only)",
}

TIMING_MAP = {
    r"\bAC\b": "before food",
    r"\bPC\b": "after food",
    r"\bEmpty\s+stomach\b": "on empty stomach",
    r"\bWith\s+food\b": "with food",
}

DURATION_RE = re.compile(
    r"(?:×|x|for)?\s*(\d+)\s*(days?|weeks?|months?)", re.IGNORECASE
)

# Medicine line pattern: line starting with a number or Tab/Cap/Syp/Inj etc.
MED_LINE_RE = re.compile(
    r"(?:^|\n)\s*"
    r"(?:\d+[\.\)]\s*)?"          # optional numbering: "1." or "1)"
    r"(?:Tab\.?|Cap\.?|Syp\.?|Inj\.?|Oint\.?|Drop\.?|Gel\.?)?\s*"
    r"([A-Z][A-Za-z\s\-]+?\d{0,4})"  # medicine name (may include dose like 650)
    r"(?:\s+(\d+(?:\.\d+)?)\s*(mg|mcg|ml|g|IU|units?))?",  # optional strength
    re.MULTILINE,
)

# Strength pattern for inline dosage (e.g. "625mg", "0.25mg")
STRENGTH_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(mg|mcg|ml|g|IU|units?)", re.IGNORECASE)


# ── Model loading ─────────────────────────────────────────────────────────────

def load_trocr_model(model_name: str = "microsoft/trocr-base-handwritten",
                     device: str = "cpu") -> bool:
    """
    Load TrOCR processor and model into global cache.
    Returns True on success, False on failure.
    Called once at startup via main.py lifespan.
    """
    global _trocr_processor, _trocr_model, _trocr_loaded, _easyocr_reader
    if _trocr_loaded:
        return True
    try:
        import easyocr
        logger.info("Loading EasyOCR text detector...")
        _easyocr_reader = easyocr.Reader(['en'], gpu=(device != "cpu"))

        from transformers import TrOCRProcessor, VisionEncoderDecoderModel  # type: ignore
        logger.info("Loading TrOCR model: %s on device: %s", model_name, device)
        _trocr_processor = TrOCRProcessor.from_pretrained(model_name)
        _trocr_model = VisionEncoderDecoderModel.from_pretrained(model_name)
        _trocr_model.to(device)
        _trocr_model.eval()
        _trocr_loaded = True
        logger.info("✅ TrOCR model loaded successfully")
        return True
    except Exception as exc:
        logger.error("Failed to load TrOCR model: %s", exc)
        _trocr_loaded = False
        return False


def is_trocr_loaded() -> bool:
    return _trocr_loaded


# ── OCR inference ─────────────────────────────────────────────────────────────

def run_trocr(image: Image.Image, device: str = "cpu") -> tuple[str, float]:
    """
    Run TrOCR on a PIL image.
    Returns (raw_text, mean_confidence).

    Confidence is computed as the mean of softmax probabilities of generated tokens.
    Range: 0.0 (low confidence) to 1.0 (high confidence).
    """
    import torch  # type: ignore
    import numpy as np

    if not _trocr_loaded or _easyocr_reader is None:
        raise RuntimeError("TrOCR model not loaded. Call load_trocr_model() first.")

    # TrOCR expects RGB
    if image.mode != "RGB":
        image = image.convert("RGB")

    import cv2
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    # Adaptive thresholding to handle uneven lighting and shadows
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # Convert back to RGB for EasyOCR so it doesn't complain about shapes
    thresh_rgb = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)

    # Detect lines using EasyOCR on the enhanced image
    detect_res = _easyocr_reader.detect(thresh_rgb)
    
    # detect_res[0] is horizontal_list. For single image, it's detect_res[0][0]
    bboxes = detect_res[0][0] if detect_res[0] and len(detect_res[0]) > 0 else []
    
    if not bboxes:
        bboxes = [[0, gray.shape[1], 0, gray.shape[0]]]
        
    # Sort boxes top-to-bottom (by y_min)
    bboxes = sorted(bboxes, key=lambda b: b[2])
    
    lines = []
    confidences = []
    
    for box in bboxes:
        x_min, x_max, y_min, y_max = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        # Add some padding
        pad_x, pad_y = 5, 5
        x_min = max(0, x_min - pad_x)
        x_max = min(gray.shape[1], x_max + pad_x)
        y_min = max(0, y_min - pad_y)
        y_max = min(gray.shape[0], y_max + pad_y)
        
        if x_max <= x_min or y_max <= y_min:
            continue
            
        crop = image.crop((x_min, y_min, x_max, y_max))
        pixel_values = _trocr_processor(images=crop, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device)

        with torch.no_grad():
            outputs = _trocr_model.generate(
                pixel_values,
                output_scores=True,
                return_dict_in_generate=True,
                max_new_tokens=128,
            )

        generated_ids = outputs.sequences
        line_text = _trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        line_conf = _compute_confidence(outputs.scores)
        
        if line_text.strip():
            lines.append(line_text.strip())
            confidences.append(line_conf)

    if not lines:
        return "", 0.0

    raw_text = "\\n".join(lines)
    mean_conf = sum(confidences) / len(confidences)
    return raw_text, round(mean_conf, 4)


def _compute_confidence(scores: tuple) -> float:
    """
    Compute mean token probability from TrOCR generation scores.
    scores: tuple of (vocab_size,) tensors, one per generated token.
    """
    import torch

    if not scores:
        return 0.0

    token_probs = []
    for score_tensor in scores:
        # score_tensor shape: (batch, vocab_size) — logits before softmax
        probs = torch.softmax(score_tensor, dim=-1)
        max_prob = probs.max(dim=-1).values.item()
        token_probs.append(max_prob)

    mean_prob = sum(token_probs) / len(token_probs)
    return round(float(mean_prob), 4)


# ── Rule-based prescription parser ───────────────────────────────────────────

def _match_frequency(text: str) -> tuple[Optional[str], Optional[str]]:
    """Return (frequency_code, frequency_plain) from text."""
    for pattern, plain in FREQ_MAP.items():
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return m.group(0).strip(), plain
    return None, None


def _match_timing(text: str) -> Optional[str]:
    for pattern, timing in TIMING_MAP.items():
        if re.search(pattern, text, re.IGNORECASE):
            return timing
    return None


def _match_duration(text: str) -> Optional[str]:
    m = DURATION_RE.search(text)
    if m:
        return f"{m.group(1)} {m.group(2).lower()}"
    return None


def _match_strength(text: str) -> Optional[str]:
    m = STRENGTH_RE.search(text)
    if m:
        return f"{m.group(1)}{m.group(2).lower()}"
    return None


def _classify_legibility(confidence: float) -> str:
    if confidence >= 0.75:
        return "high"
    elif confidence >= 0.45:
        return "medium"
    return "low"


def _classify_form(text: str) -> Optional[str]:
    text_lower = text.lower()
    if re.search(r"\btab\b", text_lower):
        return "tablet"
    if re.search(r"\bcap\b", text_lower):
        return "capsule"
    if re.search(r"\bsyp\b|\bsyrup\b", text_lower):
        return "syrup"
    if re.search(r"\binj\b|\binjection\b", text_lower):
        return "injection"
    if re.search(r"\boint\b|\bointment\b|\bgel\b", text_lower):
        return "ointment"
    if re.search(r"\bdrop\b", text_lower):
        return "drops"
    return None


def parse_prescription_text(raw_text: str, ocr_confidence: float = 0.0) -> PrescriptionExtraction:
    """
    Parse raw OCR text into a structured PrescriptionExtraction.

    Strategy:
    - Split text into lines
    - Detect patient/doctor/date from header lines
    - Detect medicine lines via numbering or Tab/Cap/etc keywords
    - Per medicine line: extract name, form, strength, frequency, timing, duration
    """
    lines = [l.strip() for l in raw_text.split("\n") if l.strip()]

    patient_name: Optional[str] = None
    doctor_name: Optional[str] = None
    date_str: Optional[str] = None
    medicines: list[ExtractedMedicine] = []

    # --- Header extraction ---
    for line in lines[:5]:  # patient/doctor info usually in top 5 lines
        # Date detection
        date_match = re.search(
            r"\b(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})\b", line
        )
        if date_match and not date_str:
            date_str = date_match.group(1)

        # "Mr/Mrs/Ms Name" pattern
        name_match = re.search(r"\b(?:Mr\.?|Mrs\.?|Ms\.?|Dr\.?)\s+([A-Za-z\s]+)", line)
        if name_match:
            full = name_match.group(0).strip()
            if "Dr" in full and not doctor_name:
                doctor_name = full
            elif not patient_name and "Mr" in full or "Mrs" in full or "Ms" in full:
                patient_name = name_match.group(1).strip()

    # --- Medicine extraction ---
    # A line is a medicine candidate if it:
    # (a) starts with a number and a dot/paren, OR
    # (b) starts with Tab/Cap/Syp/Inj/T./C., OR
    # (c) starts with a capital letter and has known dose patterns
    med_line_re = re.compile(
        r"^(?:\d+[\.\)]\s*)?(?:Tab\.?\s*|Cap\.?\s*|Syp\.?\s*|Inj\.?\s*|T\.\s*|C\.\s*)?([A-Za-z0-9\-]{2,}(?:\s+\d+(?:\.\d+)?(?:mg|mcg|ml|g)?)?)",
        re.IGNORECASE,
    )

    i = 0
    while i < len(lines):
        line = lines[i]

        m = med_line_re.match(line)
        if not m:
            i += 1
            continue

        # Gather context: this line + next line (frequency/timing often on next line)
        context = line
        if i + 1 < len(lines):
            next_line = lines[i + 1]
            # Only merge if next line looks like dosage instructions, not a new medicine
            if not med_line_re.match(next_line):
                context = line + " " + next_line
                i += 1  # consume next line too

        # Extract fields
        name = m.group(1).strip()

        # Remove trailing single digits that are clearly not part of the name
        name = re.sub(r"\s+\d+$", "", name).strip()

        # Attempt to autocorrect OCR typos via fuzzy matching against known brands
        from app.services.matcher import match_brand
        b_match = match_brand(name, top_n=1)
        if b_match and b_match.score >= 70 and b_match.name:
            # Good fuzzy match, assume it's the intended medicine
            name = b_match.name

        form = _classify_form(context)
        strength = _match_strength(context)
        if not strength:
            # Try to find inline strength in the name part
            strength_in_name = STRENGTH_RE.search(name)
            if strength_in_name:
                strength = f"{strength_in_name.group(1)}{strength_in_name.group(2).lower()}"

        frequency, frequency_plain = _match_frequency(context)
        timing = _match_timing(context)
        duration = _match_duration(context)

        # Confidence: use OCR confidence degraded slightly per medicine
        # (longer names or complex lines get slightly lower confidence)
        word_count = len(name.split())
        med_confidence = max(0.0, ocr_confidence - (word_count * 0.02))

        medicines.append(
            ExtractedMedicine(
                raw_text=context,
                name=name,
                form=form,
                strength=strength,
                dosage="1 tablet" if form == "tablet" else ("1 capsule" if form == "capsule" else None),
                frequency=frequency,
                frequency_plain=frequency_plain,
                timing=timing,
                duration=duration,
                confidence=round(min(1.0, med_confidence), 3),
            )
        )
        i += 1

    legibility = _classify_legibility(ocr_confidence)

    return PrescriptionExtraction(
        patient_name=patient_name,
        doctor_name=doctor_name,
        date=date_str,
        medicines=medicines,
        overall_legibility=legibility,
        notes=f"Extracted via TrOCR ({len(medicines)} medicine(s) found).",
        raw_ocr_text=raw_text,
        ocr_confidence=ocr_confidence,
    )


# ── Mock fallback ─────────────────────────────────────────────────────────────

MOCK_RESPONSE = PrescriptionExtraction(
    patient_name="Demo Patient",
    doctor_name="Dr. Demo",
    date="01/01/2025",
    medicines=[
        ExtractedMedicine(
            raw_text="Tab Dolo 650 mg 1-0-1 × 5 days PC",
            name="Dolo 650",
            form="tablet",
            strength="650mg",
            dosage="1 tablet",
            frequency="1-0-1",
            frequency_plain="Twice a day (morning and night)",
            timing="after food",
            duration="5 days",
            confidence=0.92,
        ),
        ExtractedMedicine(
            raw_text="Tab Pan 40 OD AC",
            name="Pan 40",
            form="tablet",
            strength="40mg",
            dosage="1 tablet",
            frequency="OD",
            frequency_plain="Once a day",
            timing="before food",
            duration=None,
            confidence=0.88,
        ),
        ExtractedMedicine(
            raw_text="Tab Augmentin 625 BD × 7 days PC",
            name="Augmentin 625",
            form="tablet",
            strength="625mg",
            dosage="1 tablet",
            frequency="BD",
            frequency_plain="Twice a day",
            timing="after food",
            duration="7 days",
            confidence=0.85,
        ),
    ],
    overall_legibility="high",
    notes="Mock response — TrOCR model not loaded.",
    raw_ocr_text="[mock] Tab Dolo 650mg 1-0-1 PC\nTab Pan 40 OD AC\nTab Augmentin 625 BD x7 days PC",
    ocr_confidence=0.90,
)


# ── Main entry point ──────────────────────────────────────────────────────────

def extract_prescription(image_bytes: bytes) -> PrescriptionExtraction:
    """
    Main entry point.
    1. Converts image bytes to PIL image.
    2. Runs TrOCR OCR → raw_text + confidence.
    3. Parses raw_text into PrescriptionExtraction.
    Falls back to MOCK_RESPONSE if TrOCR model not loaded.
    """
    from app.config import get_settings

    settings = get_settings()

    if not _trocr_loaded:
        logger.warning("TrOCR model not loaded — returning mock response")
        return MOCK_RESPONSE

    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise RuntimeError(f"Cannot open image: {exc}") from exc

    raw_text, confidence = run_trocr(image, device=settings.trocr_device)
    logger.info("TrOCR result: %d chars, confidence=%.3f", len(raw_text), confidence)

    return parse_prescription_text(raw_text, ocr_confidence=confidence)
