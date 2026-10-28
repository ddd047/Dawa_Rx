// Shared TypeScript types for DawaRx frontend

export interface ExtractedMedicine {
  raw_text: string;
  name: string;
  form: string | null;
  strength: string | null;
  dosage: string | null;
  frequency: string | null;
  frequency_plain: string | null;
  timing: string | null;
  duration: string | null;
  confidence: number;
}

export interface MatchCandidate {
  name: string;
  score: number;
  composition?: string;
  strength?: string;
  price?: number;
  manufacturer?: string;
}

export interface BrandMatch {
  name: string | null;
  composition: string | null;
  strength: string | null;
  price: number | null;
  score: number;
  alternatives: MatchCandidate[];
}

export interface GenericMatch {
  product_code: string | null;
  name: string | null;
  strength: string | null;
  unit_size: string | null;
  mrp: number | null;
  score: number;
  strength_warning: boolean;
  alternatives: MatchCandidate[];
}

export interface Savings {
  amount: number | null;
  percent: number | null;
}

export interface MedicineResult {
  extracted: ExtractedMedicine;
  brand_match: BrandMatch;
  generic_match: GenericMatch;
  savings: Savings;
  needs_review: boolean;
}

export interface PrescriptionInfo {
  patient_name: string | null;
  doctor_name: string | null;
  date: string | null;
  overall_legibility: "high" | "medium" | "low";
  notes: string | null;
}

export interface AnalyzeResponse {
  prescription: PrescriptionInfo;
  medicines: MedicineResult[];
  total_savings: Savings;
  disclaimer: string;
}

export interface SearchResponse {
  brand_candidates: MatchCandidate[];
  generic_candidates: MatchCandidate[];
}
