"""
Model evaluation service.

Computes three categories of metrics:

1. OCR Metrics (compare predicted text vs ground truth):
   - CER  : Character Error Rate  (jiwer)
   - WER  : Word Error Rate       (jiwer)
   - avg_confidence : mean TrOCR token probability

2. Extraction Metrics (compare extracted medicine names vs ground truth list):
   - Precision : TP / (TP + FP)
   - Recall    : TP / (TP + FN)
   - F1        : 2 * P * R / (P + R)
   Token-level matching with optional fuzzy tolerance.

3. Matching Metrics (evaluate the fuzzy matching pipeline output):
   - coverage_rate     : fraction of medicines matched to a generic
   - avg_brand_score   : average brand fuzzy score
   - avg_generic_score : average generic fuzzy score
   - no_match_rate     : fraction with no generic found

CER and WER are only computed when ground_truth_text is provided.
Precision, Recall, F1 are only computed when ground_truth_medicines is provided.
All Matching Metrics are always computed from the pipeline results.
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


# ── OCR Metrics ───────────────────────────────────────────────────────────────

def compute_ocr_metrics(
    predicted_text: str,
    ground_truth_text: Optional[str] = None,
    avg_confidence: float = 0.0,
) -> dict:
    """
    Compute OCR quality metrics.

    Args:
        predicted_text: Raw text output from TrOCR.
        ground_truth_text: Known-correct transcription (optional).
        avg_confidence: Mean token probability from TrOCR.

    Returns:
        dict with keys: avg_confidence, cer (optional), wer (optional).
    """
    result: dict = {"avg_confidence": round(avg_confidence, 4), "cer": None, "wer": None}

    if not ground_truth_text:
        return result

    try:
        import jiwer  # type: ignore

        # Normalize: lowercase, collapse whitespace
        pred = " ".join(predicted_text.lower().split())
        ref  = " ".join(ground_truth_text.lower().split())

        if not ref:
            return result

        wer = jiwer.wer(ref, pred)
        cer = jiwer.cer(ref, pred)

        result["wer"] = round(float(wer), 4)
        result["cer"] = round(float(cer), 4)

    except Exception as exc:
        logger.warning("Could not compute CER/WER: %s", exc)

    return result


# ── Extraction Metrics ────────────────────────────────────────────────────────

def _tokenize_name(name: str) -> set[str]:
    """Lowercase tokens from a medicine name, ignoring short words."""
    return {t.lower() for t in name.split() if len(t) >= 2}


def _names_match(predicted: str, ground_truth: str, threshold: float = 0.6) -> bool:
    """
    Soft token-overlap matching between two medicine names.
    Returns True if Jaccard similarity of tokens >= threshold.
    Falls back to fuzzy ratio if jiwer not available.
    """
    pred_tokens = _tokenize_name(predicted)
    gt_tokens   = _tokenize_name(ground_truth)

    if not pred_tokens or not gt_tokens:
        return False

    intersection = len(pred_tokens & gt_tokens)
    union        = len(pred_tokens | gt_tokens)
    jaccard      = intersection / union if union > 0 else 0.0
    return jaccard >= threshold


def compute_extraction_metrics(
    predicted_names: list[str],
    ground_truth_names: Optional[list[str]] = None,
) -> dict:
    """
    Compute Precision, Recall, F1 for medicine name extraction.

    Matching is token-overlap based (soft matching), so "Dolo 650" matches
    "Dolo" or "650mg Dolo" with configurable tolerance.

    Args:
        predicted_names: Medicine names extracted by TrOCR + parser.
        ground_truth_names: Correct medicine names (optional).

    Returns:
        dict with keys: medicine_count, precision, recall, f1.
    """
    result: dict = {
        "medicine_count": len(predicted_names),
        "precision": None,
        "recall": None,
        "f1": None,
    }

    if not ground_truth_names:
        return result

    if not predicted_names and not ground_truth_names:
        result.update({"precision": 1.0, "recall": 1.0, "f1": 1.0})
        return result

    if not predicted_names:
        result.update({"precision": 0.0, "recall": 0.0, "f1": 0.0})
        return result

    if not ground_truth_names:
        result.update({"precision": 0.0, "recall": 0.0, "f1": 0.0})
        return result

    # True positives: each ground truth medicine matched by at most one predicted
    gt_matched = [False] * len(ground_truth_names)
    true_positives = 0

    for pred in predicted_names:
        for j, gt in enumerate(ground_truth_names):
            if not gt_matched[j] and _names_match(pred, gt):
                true_positives += 1
                gt_matched[j] = True
                break

    precision = true_positives / len(predicted_names) if predicted_names else 0.0
    recall    = true_positives / len(ground_truth_names) if ground_truth_names else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )

    result.update({
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1":        round(f1, 4),
    })
    return result


# ── Matching Metrics ──────────────────────────────────────────────────────────

def compute_matching_metrics(medicine_results: list) -> dict:
    """
    Compute matching pipeline performance metrics from AnalyzeResponse medicines.

    Args:
        medicine_results: list of MedicineResult objects.

    Returns:
        dict with keys: coverage_rate, avg_brand_score, avg_generic_score, no_match_rate.
    """
    if not medicine_results:
        return {
            "coverage_rate":     0.0,
            "avg_brand_score":   0.0,
            "avg_generic_score": 0.0,
            "no_match_rate":     1.0,
        }

    n = len(medicine_results)
    matched_count = 0
    no_match_count = 0
    brand_scores = []
    generic_scores = []

    for result in medicine_results:
        brand_score   = result.brand_match.score if result.brand_match else 0.0
        generic_score = result.generic_match.score if result.generic_match else 0.0
        generic_name  = result.generic_match.name if result.generic_match else None

        brand_scores.append(brand_score)
        generic_scores.append(generic_score)

        if generic_name:
            matched_count += 1
        else:
            no_match_count += 1

    coverage_rate     = matched_count / n
    avg_brand_score   = sum(brand_scores) / n if brand_scores else 0.0
    avg_generic_score = sum(generic_scores) / n if generic_scores else 0.0
    no_match_rate     = no_match_count / n

    return {
        "coverage_rate":     round(coverage_rate, 4),
        "avg_brand_score":   round(avg_brand_score, 2),
        "avg_generic_score": round(avg_generic_score, 2),
        "no_match_rate":     round(no_match_rate, 4),
    }


# ── Convenience: build full ModelMetrics dict ─────────────────────────────────

def build_model_metrics(
    predicted_text: str,
    avg_confidence: float,
    predicted_names: list[str],
    medicine_results: list,
    ground_truth_text: Optional[str] = None,
    ground_truth_medicines: Optional[list[str]] = None,
) -> dict:
    """
    Build the complete ModelMetrics payload as a dict (for schema validation).

    Args:
        predicted_text:         Raw OCR text from TrOCR.
        avg_confidence:         Mean token probability from TrOCR.
        predicted_names:        Medicine names extracted by parser.
        medicine_results:       List of MedicineResult from matching pipeline.
        ground_truth_text:      Optional known-correct transcription.
        ground_truth_medicines: Optional list of correct medicine names.
    """
    ocr_m  = compute_ocr_metrics(predicted_text, ground_truth_text, avg_confidence)
    ext_m  = compute_extraction_metrics(predicted_names, ground_truth_medicines)
    mat_m  = compute_matching_metrics(medicine_results)

    return {
        "ocr": ocr_m,
        "extraction": ext_m,
        "matching": mat_m,
    }
