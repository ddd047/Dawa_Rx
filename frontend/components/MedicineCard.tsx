"use client";

import { useState } from "react";
import { MedicineResult } from "@/types";
import { ConfidenceBadge } from "./ConfidenceBadge";
import { searchMedicines } from "@/lib/api";
import type { MatchCandidate } from "@/types";

interface Props {
  result: MedicineResult;
  index: number;
}

export function MedicineCard({ result, index }: Props) {
  const { extracted, brand_match, generic_match, savings, needs_review } = result;
  const [searchQuery, setSearchQuery] = useState(extracted.name);
  const [searchResults, setSearchResults] = useState<{ brand_candidates: MatchCandidate[]; generic_candidates: MatchCandidate[] } | null>(null);
  const [isSearching, setIsSearching] = useState(false);
  const [isExpanded, setIsExpanded] = useState(needs_review);

  const handleSearch = async () => {
    if (!searchQuery.trim()) return;
    setIsSearching(true);
    try {
      const res = await searchMedicines(searchQuery);
      setSearchResults(res);
    } catch {
      // ignore
    } finally {
      setIsSearching(false);
    }
  };

  const confidenceColor =
    extracted.confidence >= 0.75 ? "border-emerald-500" :
    extracted.confidence >= 0.5 ? "border-amber-400" : "border-red-500";

  return (
    <div className={`medicine-card ${needs_review ? "medicine-card--review" : ""} border-l-4 ${confidenceColor}`}>
      {/* Header */}
      <div className="medicine-card__header">
        <div className="medicine-card__title-row">
          <h3 className="medicine-card__name">
            <span className="medicine-card__index">#{index + 1}</span>
            {extracted.name}
            {extracted.strength && (
              <span className="medicine-card__strength">{extracted.strength}</span>
            )}
          </h3>
          <ConfidenceBadge confidence={extracted.confidence} />
        </div>

        {needs_review && (
          <div className="review-alert" role="alert">
            ⚠️ <strong>Please verify</strong> — low confidence or weak match
          </div>
        )}

        {/* Dosage row */}
        <div className="medicine-card__meta">
          {extracted.form && <span className="meta-pill">{extracted.form}</span>}
          {extracted.frequency_plain && <span className="meta-pill">{extracted.frequency_plain}</span>}
          {extracted.timing && <span className="meta-pill">{extracted.timing}</span>}
          {extracted.duration && <span className="meta-pill">📅 {extracted.duration}</span>}
        </div>
      </div>

      {/* Matching comparison */}
      <div className="medicine-card__comparison">
        {/* Brand match */}
        <div className="match-box match-box--brand">
          <p className="match-box__label">Brand</p>
          <p className="match-box__name">{brand_match.name ?? "Not found"}</p>
          {brand_match.composition && (
            <p className="match-box__detail">{brand_match.composition}</p>
          )}
          {brand_match.price !== null && brand_match.price !== undefined ? (
            <p className="match-box__price">₹{brand_match.price.toFixed(2)}</p>
          ) : (
            <p className="match-box__price match-box__price--na">Price N/A</p>
          )}
          <p className="match-box__score">Match: {brand_match.score}%</p>
        </div>

        {/* Arrow */}
        <div className="comparison-arrow">→</div>

        {/* Generic match */}
        <div className="match-box match-box--generic">
          <p className="match-box__label">Jan Aushadhi Generic</p>
          <p className="match-box__name">{generic_match.name ?? "Not found"}</p>
          {generic_match.strength && (
            <p className="match-box__detail">{generic_match.strength} · {generic_match.unit_size}</p>
          )}
          {generic_match.mrp !== null && generic_match.mrp !== undefined ? (
            <p className="match-box__price match-box__price--generic">₹{generic_match.mrp.toFixed(2)}</p>
          ) : (
            <p className="match-box__price match-box__price--na">Price N/A</p>
          )}
          {generic_match.strength_warning && (
            <p className="match-box__warning">⚠️ Strength may differ</p>
          )}
          <p className="match-box__score">Match: {generic_match.score}%</p>
        </div>
      </div>

      {/* Savings */}
      {savings.amount !== null && savings.amount !== undefined && savings.amount > 0 && (
        <div className="medicine-card__savings">
          <span className="savings-chip">
            💰 Save ₹{savings.amount.toFixed(2)}
            {savings.percent !== null && savings.percent !== undefined && ` (${savings.percent}%)`}
          </span>
        </div>
      )}

      {/* Manual search section */}
      <div className="medicine-card__search">
        <button
          className="btn-ghost"
          onClick={() => setIsExpanded(!isExpanded)}
          aria-expanded={isExpanded}
        >
          {isExpanded ? "▲ Hide" : "🔍 Search manually"}
        </button>

        {isExpanded && (
          <div className="manual-search">
            <div className="manual-search__row">
              <input
                id={`search-${index}`}
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleSearch()}
                className="manual-search__input"
                placeholder="Type medicine name…"
                aria-label="Search medicine"
              />
              <button
                className="btn-primary btn-sm"
                onClick={handleSearch}
                disabled={isSearching}
                aria-busy={isSearching}
              >
                {isSearching ? "…" : "Search"}
              </button>
            </div>

            {searchResults && (
              <div className="search-results">
                {searchResults.brand_candidates.length > 0 && (
                  <div>
                    <p className="search-results__label">Brands</p>
                    {searchResults.brand_candidates.map((c, i) => (
                      <div key={i} className="search-result-item">
                        <span>{c.name}</span>
                        <span className="search-result-item__score">{c.score}%</span>
                        {c.price && <span className="search-result-item__price">₹{c.price}</span>}
                      </div>
                    ))}
                  </div>
                )}
                {searchResults.generic_candidates.length > 0 && (
                  <div>
                    <p className="search-results__label">Jan Aushadhi Generics</p>
                    {searchResults.generic_candidates.map((c, i) => (
                      <div key={i} className="search-result-item search-result-item--generic">
                        <span>{c.name}</span>
                        <span className="search-result-item__score">{c.score}%</span>
                        {c.price && <span className="search-result-item__price">₹{c.price}</span>}
                      </div>
                    ))}
                  </div>
                )}
                {searchResults.brand_candidates.length === 0 && searchResults.generic_candidates.length === 0 && (
                  <p className="search-results__empty">No results found.</p>
                )}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
