"""Pydantic v2 schemas for request/response contracts."""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field

# ── TrOCR Vision output ───────────────────────────────────────────────────────


class ExtractedMedicine(BaseModel):
    raw_text: str
    name: str
    form: Optional[str] = None  # tablet|capsule|syrup|injection|ointment|drops|other
    strength: Optional[str] = None
    dosage: Optional[str] = None
    frequency: Optional[str] = None
    frequency_plain: Optional[str] = None
    timing: Optional[str] = None
    duration: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)


class PrescriptionExtraction(BaseModel):
    patient_name: Optional[str] = None
    doctor_name: Optional[str] = None
    date: Optional[str] = None
    medicines: list[ExtractedMedicine] = []
    overall_legibility: str = "medium"  # high|medium|low
    notes: Optional[str] = None
    raw_ocr_text: Optional[str] = None  # raw text from TrOCR before parsing
    ocr_confidence: Optional[float] = None  # mean token confidence from TrOCR


# ── Matching output ───────────────────────────────────────────────────────────


class MatchCandidate(BaseModel):
    name: str
    score: float
    composition: Optional[str] = None
    strength: Optional[str] = None
    price: Optional[float] = None
    manufacturer: Optional[str] = None


class BrandMatch(BaseModel):
    name: Optional[str] = None
    composition: Optional[str] = None
    strength: Optional[str] = None
    price: Optional[float] = None
    score: float = 0.0
    alternatives: list[MatchCandidate] = []


class GenericMatch(BaseModel):
    product_code: Optional[str] = None
    name: Optional[str] = None
    strength: Optional[str] = None
    unit_size: Optional[str] = None
    mrp: Optional[float] = None
    score: float = 0.0
    strength_warning: bool = False
    alternatives: list[MatchCandidate] = []


class Savings(BaseModel):
    amount: Optional[float] = None
    percent: Optional[float] = None


class MedicineResult(BaseModel):
    extracted: ExtractedMedicine
    brand_match: BrandMatch = Field(default_factory=BrandMatch)
    generic_match: GenericMatch = Field(default_factory=GenericMatch)
    savings: Savings = Field(default_factory=Savings)
    needs_review: bool = False


# ── Model Evaluation Metrics ──────────────────────────────────────────────────


class OCRMetrics(BaseModel):
    """OCR quality metrics — CER and WER require ground truth."""

    avg_confidence: float = Field(
        description="Mean token probability from TrOCR (0–1, higher is better)",
        ge=0.0,
        le=1.0,
        default=0.0,
    )
    cer: Optional[float] = Field(
        default=None,
        description="Character Error Rate (0–∞, lower is better). Needs ground truth.",
    )
    wer: Optional[float] = Field(
        default=None,
        description="Word Error Rate (0–∞, lower is better). Needs ground truth.",
    )


class ExtractionMetrics(BaseModel):
    """Medicine extraction quality — requires ground truth medicine list."""

    medicine_count: int = Field(description="Number of medicines extracted")
    precision: Optional[float] = Field(
        default=None,
        description="Precision: fraction of extracted medicines that are correct (0–1).",
    )
    recall: Optional[float] = Field(
        default=None,
        description="Recall: fraction of actual medicines that were found (0–1).",
    )
    f1: Optional[float] = Field(
        default=None,
        description="F1 Score: harmonic mean of Precision and Recall (0–1).",
    )


class MatchingMetrics(BaseModel):
    """Matching pipeline performance metrics (no ground truth needed)."""

    coverage_rate: float = Field(
        description="Fraction of medicines matched to a Jan Aushadhi generic (0–1)",
        ge=0.0,
        le=1.0,
    )
    avg_brand_score: float = Field(
        description="Average fuzzy brand match score (0–100)", ge=0.0, le=100.0
    )
    avg_generic_score: float = Field(
        description="Average fuzzy generic match score (0–100)", ge=0.0, le=100.0
    )
    no_match_rate: float = Field(
        description="Fraction of medicines with no generic alternative found (0–1)",
        ge=0.0,
        le=1.0,
    )


class ModelMetrics(BaseModel):
    """Consolidated model evaluation metrics for a single prescription analysis."""

    ocr: OCRMetrics
    extraction: ExtractionMetrics
    matching: MatchingMetrics


# ── API responses ─────────────────────────────────────────────────────────────


class AnalyzeResponse(BaseModel):
    prescription: dict
    medicines: list[MedicineResult]
    total_savings: Savings = Field(default_factory=Savings)
    model_metrics: Optional[ModelMetrics] = None
    disclaimer: str = (
        "AI-generated interpretation. Always confirm with your doctor or "
        "pharmacist before purchasing or substituting any medicine."
    )


class EvaluateResponse(BaseModel):
    """Response for POST /api/evaluate — full metrics including optional ground truth comparison."""

    raw_ocr_text: str
    model_metrics: ModelMetrics
    medicines_extracted: list[str] = Field(
        description="List of medicine names extracted from the prescription"
    )


class SearchResult(BaseModel):
    brand_candidates: list[MatchCandidate] = []
    generic_candidates: list[MatchCandidate] = []


class HealthResponse(BaseModel):
    status: str = "ok"
    brands_loaded: int = 0
    generics_loaded: int = 0
    trocr_loaded: bool = False
