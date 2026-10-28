"use client";

import { useState, useCallback } from "react";
import { UploadCard } from "@/components/UploadCard";
import { MedicineCard } from "@/components/MedicineCard";
import { SavingsBanner } from "@/components/SavingsBanner";
import { analyzePrescription } from "@/lib/api";
import type { AnalyzeResponse } from "@/types";

const LOADING_STEPS = [
  "📸 Reading prescription…",
  "🧠 Analysing handwriting with AI…",
  "🔍 Finding generic alternatives…",
  "💊 Calculating savings…",
];

export default function Home() {
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [loadingStep, setLoadingStep] = useState(0);

  const handleFile = useCallback(async (file: File) => {
    setError(null);
    setResult(null);
    setIsLoading(true);
    setLoadingStep(0);

    // Animate loading steps
    const intervals = LOADING_STEPS.map((_, i) =>
      setTimeout(() => setLoadingStep(i), i * 1800)
    );

    try {
      const data = await analyzePrescription(file);
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong. Please try again.");
    } finally {
      intervals.forEach(clearTimeout);
      setIsLoading(false);
    }
  }, []);

  const handleDownloadJSON = () => {
    if (!result) return;
    const blob = new Blob([JSON.stringify(result, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "prescription-analysis.json";
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleCopySummary = async () => {
    if (!result) return;
    const lines: string[] = [
      `DawaRx Prescription Analysis`,
      `Patient: ${result.prescription.patient_name ?? "Unknown"}`,
      `Doctor: ${result.prescription.doctor_name ?? "Unknown"}`,
      `Date: ${result.prescription.date ?? "Unknown"}`,
      "",
      "Medicines:",
      ...result.medicines.map((m, i) => {
        const brand = m.brand_match.name ?? m.extracted.name;
        const generic = m.generic_match.name ?? "No generic found";
        const saving = m.savings.amount ? `₹${m.savings.amount.toFixed(2)} saved` : "";
        return `  ${i + 1}. ${brand} → ${generic} ${saving}`;
      }),
      "",
      `Total savings: ₹${result.total_savings.amount?.toFixed(2) ?? "0"}`,
      "",
      result.disclaimer,
    ];
    await navigator.clipboard.writeText(lines.join("\n"));
    alert("Summary copied to clipboard!");
  };

  return (
    <main className="app-root">
      {/* Nav */}
      <nav className="app-nav">
        <div className="app-nav__inner">
          <div className="app-nav__logo">
            <span className="logo-icon">💊</span>
            <span className="logo-text">DawaRx</span>
          </div>
          <p className="app-nav__tagline">AI Prescription Decipherer &amp; Generic Finder</p>
        </div>
      </nav>

      <div className="app-content">
        {/* Upload section */}
        <section className="upload-section" aria-label="Upload prescription">
          <h1 className="upload-section__heading">
            Upload your prescription
          </h1>
          <p className="upload-section__sub">
            We'll identify your medicines and find affordable Jan Aushadhi generic alternatives.
          </p>
          <UploadCard onFileSelected={handleFile} isLoading={isLoading} />
        </section>

        {/* Loading state */}
        {isLoading && (
          <div className="loading-state" aria-live="polite" aria-atomic="true">
            <div className="loading-state__dots">
              <span /><span /><span />
            </div>
            <p className="loading-state__step">{LOADING_STEPS[loadingStep]}</p>
          </div>
        )}

        {/* Error */}
        {error && (
          <div className="error-card" role="alert">
            <p className="error-card__icon">⚠️</p>
            <p className="error-card__message">{error}</p>
            <p className="error-card__hint">Please check the image and try again.</p>
          </div>
        )}

        {/* Results */}
        {result && !isLoading && (
          <section className="results-section" aria-label="Analysis results">
            {/* Prescription info */}
            <div className="prescription-info">
              <div className="prescription-info__grid">
                {result.prescription.patient_name && (
                  <div className="info-chip">
                    <span className="info-chip__label">Patient</span>
                    <span className="info-chip__value">{result.prescription.patient_name}</span>
                  </div>
                )}
                {result.prescription.doctor_name && (
                  <div className="info-chip">
                    <span className="info-chip__label">Doctor</span>
                    <span className="info-chip__value">{result.prescription.doctor_name}</span>
                  </div>
                )}
                {result.prescription.date && (
                  <div className="info-chip">
                    <span className="info-chip__label">Date</span>
                    <span className="info-chip__value">{result.prescription.date}</span>
                  </div>
                )}
                <div className="info-chip">
                  <span className="info-chip__label">Legibility</span>
                  <span className={`info-chip__value legibility--${result.prescription.overall_legibility}`}>
                    {result.prescription.overall_legibility}
                  </span>
                </div>
              </div>
            </div>

            {/* Total savings banner */}
            <SavingsBanner
              totalSavings={result.total_savings}
              medicineCount={result.medicines.length}
            />

            {/* Action buttons */}
            <div className="results-actions">
              <button id="download-json" className="btn-secondary" onClick={handleDownloadJSON}>
                ⬇ Download JSON
              </button>
              <button id="copy-summary" className="btn-secondary" onClick={handleCopySummary}>
                📋 Copy summary
              </button>
            </div>

            {/* Medicine cards */}
            <div className="medicine-list" role="list">
              {result.medicines.map((med, i) => (
                <MedicineCard key={i} result={med} index={i} />
              ))}
            </div>

            {/* Disclaimer */}
            <div className="disclaimer" role="note">
              <p className="disclaimer__text">⚕️ {result.disclaimer}</p>
            </div>
          </section>
        )}
      </div>
    </main>
  );
}
