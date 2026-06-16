export interface ThresholdInfo {
  min: number;
  max: number;
  step: number;
  default: number;
}

export interface RuntimeInfo {
  app_name: string;
  ensemble_display_name: string;
  model_identifier: string;
  threshold: ThresholdInfo;
  segmentation_enabled: boolean;
  gradcam_enabled: boolean;
  classifier_member_count: number;
  disclaimer: string;
}

export interface DiagnosticResult {
  benign_probability: number;
  malignant_probability: number;
  final_label: "benign" | "malignant" | string;
  final_label_text: string;
  confidence_band: "high" | "low" | "borderline" | string;
  confidence_band_text: string;
  recommendation_text: string;
  analysis_timestamp: string;
  model_version: string;
  model_display_name: string;
  auxiliary_use_disclaimer: string;
}

export interface ImagePayload {
  original: string | null;
  lesion: string | null;
  explanation: string | null;
}

export interface DiagnosisPayload {
  status: string;
  status_text: string;
  input_filename: string;
  result: DiagnosticResult | null;
  images: ImagePayload;
  warnings: string[];
  raw_warnings: string[];
  lesion_visualization_missing_reason: string | null;
  explanation_missing_reason: string | null;
  metadata: Record<string, unknown>;
}

export interface PatientCaseInput {
  patient_code: string;
  name: string;
  sex: string;
  age: number | null;
  contact: string;
  visit_date: string;
  department: string;
  primary_complaint: string;
  ultrasound_description: string;
  birads_category: string;
  pathology_status: string;
  risk_level: string;
  notes: string;
}

export interface PatientCase extends PatientCaseInput {
  id: number;
  created_at: string;
  updated_at: string;
}

export interface KnowledgeEntry {
  id: string;
  category: string;
  title: string;
  summary: string;
  details: string[];
  clinical_use: string;
  keywords: string[];
  references: string[];
}

export interface AgentInterpretationRequest {
  question: string;
  patient_case: PatientCase | null;
  diagnosis: DiagnosisPayload | null;
}

export interface AgentInterpretationResponse {
  answer: string;
  model: string;
  provider: string;
  fallback_used: boolean;
  retrieved_knowledge: KnowledgeEntry[];
  warnings: string[];
}
