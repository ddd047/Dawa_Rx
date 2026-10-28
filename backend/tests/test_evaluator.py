"""Tests for the model evaluation metrics service."""
from __future__ import annotations
import pytest
from app.services.evaluator import (
    compute_ocr_metrics,
    compute_extraction_metrics,
    compute_matching_metrics,
    build_model_metrics,
)
from app.schemas import MedicineResult, BrandMatch, GenericMatch, Savings, ExtractedMedicine


# ── Helper: build a MedicineResult ───────────────────────────────────────────

def _make_result(brand_score=90.0, generic_score=85.0, has_generic=True):
    return MedicineResult(
        extracted=ExtractedMedicine(raw_text="Tab Dolo 650", name="Dolo 650", confidence=0.9),
        brand_match=BrandMatch(name="Dolo 650", score=brand_score),
        generic_match=GenericMatch(
            name="Paracetamol" if has_generic else None,
            score=generic_score,
        ),
        savings=Savings(amount=50.0, percent=40.0),
        needs_review=False,
    )


# ── OCR Metrics tests ─────────────────────────────────────────────────────────

class TestOCRMetrics:
    def test_perfect_match_gives_zero_cer_wer(self):
        """When predicted == ground truth, CER and WER should both be 0."""
        result = compute_ocr_metrics(
            predicted_text="Tab Dolo 650 OD",
            ground_truth_text="Tab Dolo 650 OD",
            avg_confidence=0.95,
        )
        assert result["cer"] == 0.0
        assert result["wer"] == 0.0
        assert result["avg_confidence"] == 0.95

    def test_no_ground_truth_returns_none_cer_wer(self):
        """Without ground truth, CER and WER must be None."""
        result = compute_ocr_metrics(
            predicted_text="some text",
            ground_truth_text=None,
            avg_confidence=0.80,
        )
        assert result["cer"] is None
        assert result["wer"] is None
        assert result["avg_confidence"] == 0.80

    def test_partial_wer_is_nonzero(self):
        """When one word differs, WER should be > 0."""
        result = compute_ocr_metrics(
            predicted_text="Tab Dolo 500 OD",
            ground_truth_text="Tab Dolo 650 OD",
            avg_confidence=0.75,
        )
        assert result["wer"] is not None
        assert result["wer"] > 0.0

    def test_completely_wrong_has_high_wer(self):
        """Completely different text should yield high WER."""
        result = compute_ocr_metrics(
            predicted_text="xyz abc qrs",
            ground_truth_text="Tab Dolo 650 OD",
            avg_confidence=0.20,
        )
        assert result["wer"] is not None
        assert result["wer"] > 0.5


# ── Extraction Metrics tests ──────────────────────────────────────────────────

class TestExtractionMetrics:
    def test_perfect_extraction_gives_f1_one(self):
        """When predicted == ground truth names, P=R=F1=1.0."""
        result = compute_extraction_metrics(
            predicted_names=["Dolo 650", "Pan 40"],
            ground_truth_names=["Dolo 650", "Pan 40"],
        )
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1"] == 1.0

    def test_no_ground_truth_returns_none_metrics(self):
        """Without ground truth, P/R/F1 must be None."""
        result = compute_extraction_metrics(
            predicted_names=["Dolo 650"],
            ground_truth_names=None,
        )
        assert result["precision"] is None
        assert result["recall"] is None
        assert result["f1"] is None
        assert result["medicine_count"] == 1

    def test_missed_medicine_lowers_recall(self):
        """Missing a medicine from ground truth should lower recall."""
        result = compute_extraction_metrics(
            predicted_names=["Dolo 650"],
            ground_truth_names=["Dolo 650", "Pan 40", "Augmentin 625"],
        )
        assert result["recall"] is not None
        assert result["recall"] < 1.0
        assert result["precision"] == 1.0  # what we did find is correct

    def test_extra_medicine_lowers_precision(self):
        """Extracting a medicine not in ground truth should lower precision."""
        result = compute_extraction_metrics(
            predicted_names=["Dolo 650", "Random Drug"],
            ground_truth_names=["Dolo 650"],
        )
        assert result["precision"] is not None
        assert result["precision"] < 1.0
        assert result["recall"] == 1.0  # correct medicine was found

    def test_empty_prediction_zero_f1(self):
        """Empty prediction with non-empty ground truth = 0 F1."""
        result = compute_extraction_metrics(
            predicted_names=[],
            ground_truth_names=["Dolo 650"],
        )
        assert result["f1"] == 0.0


# ── Matching Metrics tests ────────────────────────────────────────────────────

class TestMatchingMetrics:
    def test_full_coverage_when_all_matched(self):
        """All medicines with generics → coverage_rate=1.0, no_match_rate=0.0."""
        results = [_make_result(has_generic=True) for _ in range(3)]
        metrics = compute_matching_metrics(results)
        assert metrics["coverage_rate"] == 1.0
        assert metrics["no_match_rate"] == 0.0

    def test_partial_coverage(self):
        """Half matched, half not → coverage_rate=0.5."""
        results = [
            _make_result(has_generic=True),
            _make_result(has_generic=False),
        ]
        metrics = compute_matching_metrics(results)
        assert metrics["coverage_rate"] == 0.5
        assert metrics["no_match_rate"] == 0.5

    def test_empty_results(self):
        """Empty results → coverage_rate=0.0, no_match_rate=1.0."""
        metrics = compute_matching_metrics([])
        assert metrics["coverage_rate"] == 0.0
        assert metrics["no_match_rate"] == 1.0

    def test_avg_scores_computed_correctly(self):
        """Average scores should be mean of all medicine scores."""
        results = [
            _make_result(brand_score=80.0, generic_score=70.0),
            _make_result(brand_score=60.0, generic_score=50.0),
        ]
        metrics = compute_matching_metrics(results)
        assert metrics["avg_brand_score"] == pytest.approx(70.0, abs=0.1)
        assert metrics["avg_generic_score"] == pytest.approx(60.0, abs=0.1)


# ── build_model_metrics integration test ─────────────────────────────────────

class TestBuildModelMetrics:
    def test_returns_all_sections(self):
        """build_model_metrics must return ocr, extraction, matching keys."""
        results = [_make_result()]
        metrics = build_model_metrics(
            predicted_text="Tab Dolo 650 OD PC",
            avg_confidence=0.88,
            predicted_names=["Dolo 650"],
            medicine_results=results,
            ground_truth_text="Tab Dolo 650 OD PC",
            ground_truth_medicines=["Dolo 650"],
        )
        assert "ocr" in metrics
        assert "extraction" in metrics
        assert "matching" in metrics
        # With matching ground truth text, CER should be 0
        assert metrics["ocr"]["cer"] == 0.0
        # With matching medicine names, F1 should be 1.0
        assert metrics["extraction"]["f1"] == 1.0
