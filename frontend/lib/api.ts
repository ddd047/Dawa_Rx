// Typed API client for DawaRx backend
import type { AnalyzeResponse, SearchResponse } from "@/types";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export async function analyzePrescription(file: File): Promise<AnalyzeResponse> {
  const form = new FormData();
  form.append("file", file);

  const res = await fetch(`${API_URL}/api/analyze`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: "Unknown error" }));
    throw new Error(error.detail ?? `Server error ${res.status}`);
  }

  return res.json() as Promise<AnalyzeResponse>;
}

export async function searchMedicines(q: string): Promise<SearchResponse> {
  const res = await fetch(`${API_URL}/api/search?q=${encodeURIComponent(q)}`);
  if (!res.ok) {
    throw new Error(`Search failed: ${res.status}`);
  }
  return res.json() as Promise<SearchResponse>;
}

export async function healthCheck(): Promise<{ status: string }> {
  const res = await fetch(`${API_URL}/api/health`);
  return res.json();
}
