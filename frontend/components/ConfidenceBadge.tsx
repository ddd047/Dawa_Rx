"use client";

import { useMemo } from "react";

interface Props {
  confidence: number;
}

export function ConfidenceBadge({ confidence }: Props) {
  const { label, cls } = useMemo(() => {
    if (confidence >= 0.75) return { label: "High confidence", cls: "badge-green" };
    if (confidence >= 0.5) return { label: "Medium confidence", cls: "badge-amber" };
    return { label: "Low confidence", cls: "badge-red" };
  }, [confidence]);

  return (
    <span className={`confidence-badge ${cls}`}>
      {label} ({Math.round(confidence * 100)}%)
    </span>
  );
}
