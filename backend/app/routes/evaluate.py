"""
POST /api/evaluate — evaluate TrOCR + parser quality against ground truth.

This endpoint is for model benchmarking. It accepts:
  - image file (required)
  - ground_truth_text (optional): correct raw transcription of the prescription
  - ground_truth_medicines (optional): comma-separated list of correct medicine names

When ground truth is provided, it returns CER, WER, Precision, Recall, F1.
Without ground truth, it returns confidence-based metrics only.
"""
from __future__ import annotations
import logging
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.schemas import EvaluateResponse, ModelMetrics
from app.services.preprocess import preprocess_image
from app.services.vision import extract_prescription
from app.services.matcher import match_brand, match_generic
from app.services.pricing import compute_savings
from app.schemas import MedicineResult, BrandMatch, GenericMatch, Savings
from app.services.evaluator import build_model_metrics
from app.config import get_settings

logger = logging.getLogger(__name__)
router = APIRouter()

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}


@router.post("/evaluate", response_model=EvaluateResponse)
async def evaluate(
    file: UploadFile = File(..., description="Prescription image to evaluate"),
    ground_truth_text: Optional[str] = Form(
        default=None,
        description="Correct OCR transcription (enables CER and WER metrics)",
    ),
    ground_truth_medicines: Optional[str] = Form(
        default=None,
        description="Comma-separated correct medicine names (enables Precision, Recall, F1)",
    ),
) -> EvaluateResponse:
    """
    Evaluate model performance on a prescription image.

    Returns full model metrics including:
    - OCR confidence (always)
    - CER and WER (when ground_truth_text provided)
    - Precision, Recall, F1 (when ground_truth_medicines provided)
    - Coverage Rate, Avg Brand Score, Avg Generic Score, No-Match Rate (always)
    """
    settings = get_settings()

    # ── Validate upload ────────────────────────────────────────────────────
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{file.content_type}'. Use JPG, PNG or WebP.",
        )

    image_bytes = await file.read()
    if len(image_bytes) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="File too large. Maximum size is 8 MB.")
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # ── Pre-process ────────────────────────────────────────────────────────
    try:
        image_bytes = preprocess_image(image_bytes)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not process image: {exc}") from exc

    # ── TrOCR extraction ───────────────────────────────────────────────────
    try:
        extraction = extract_prescription(image_bytes)
    except Exception as exc:
        logger.exception("TrOCR extraction failed in /evaluate")
        raise HTTPException(status_code=502, detail=f"Vision service error: {exc}") from exc

    # ── Run matching pipeline (needed for matching metrics) ────────────────
    medicine_results: list[MedicineResult] = []
    for med in extraction.medicines:
        brand_match = match_brand(med.name)
        composition = brand_match.composition or med.name
        preferred_strength = med.strength or brand_match.strength
        generic_match = match_generic(composition, preferred_strength)
        savings = compute_savings(brand_match.price, generic_match.mrp)
        needs_review = (
            med.confidence < 0.6
            or brand_match.score < settings.brand_match_cutoff
            or generic_match.score < settings.generic_match_cutoff
        )
        medicine_results.append(
            MedicineResult(
                extracted=med,
                brand_match=brand_match,
                generic_match=generic_match,
                savings=savings,
                needs_review=needs_review,
            )
        )

    # ── Parse ground truth medicines ───────────────────────────────────────
    gt_medicines_list: Optional[list[str]] = None
    if ground_truth_medicines:
        gt_medicines_list = [
            m.strip() for m in ground_truth_medicines.split(",") if m.strip()
        ]

    # ── Build metrics ──────────────────────────────────────────────────────
    predicted_names = [med.name for med in extraction.medicines]
    metrics_dict = build_model_metrics(
        predicted_text=extraction.raw_ocr_text or "",
        avg_confidence=extraction.ocr_confidence or 0.0,
        predicted_names=predicted_names,
        medicine_results=medicine_results,
        ground_truth_text=ground_truth_text,
        ground_truth_medicines=gt_medicines_list,
    )
    model_metrics = ModelMetrics(**metrics_dict)

    return EvaluateResponse(
        raw_ocr_text=extraction.raw_ocr_text or "",
        model_metrics=model_metrics,
        medicines_extracted=predicted_names,
    )
