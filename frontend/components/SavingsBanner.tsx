"use client";

import { Savings } from "@/types";

interface Props {
  totalSavings: Savings;
  medicineCount: number;
}

export function SavingsBanner({ totalSavings, medicineCount }: Props) {
  if (totalSavings.amount === null || totalSavings.amount === undefined) return null;

  const hasSavings = totalSavings.amount > 0;

  return (
    <div className={`savings-banner ${hasSavings ? "savings-banner--positive" : "savings-banner--neutral"}`}>
      <div className="savings-banner__icon">{hasSavings ? "💊" : "ℹ️"}</div>
      <div className="savings-banner__content">
        <p className="savings-banner__title">
          {hasSavings
            ? `You could save ₹${totalSavings.amount.toFixed(2)} on ${medicineCount} medicine${medicineCount > 1 ? "s" : ""}`
            : "No price savings found for these medicines"}
        </p>
        {hasSavings && totalSavings.percent !== null && (
          <p className="savings-banner__subtitle">
            That's <strong>{totalSavings.percent}%</strong> cheaper using Jan Aushadhi generics!
          </p>
        )}
      </div>
    </div>
  );
}
