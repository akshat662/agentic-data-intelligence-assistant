/**
 * TypeScript mirror of adia/api/schemas.py. Keep in sync by hand -- there is no shared schema
 * generation between the FastAPI backend and this frontend.
 */

export const DATASET_ID_PATTERN = /^[a-zA-Z0-9_-]+$/;

export interface ChatRequest {
  dataset_id: string;
  question: string;
}

export interface ChatResponse {
  run_id: string;
  dataset_id: string;
  question: string;
  answer: string | null;
  validation_passed: boolean | null;
  evidence_ids: string[];
  tools_used: string[];
  feasibility_verdict: string | null;
  refused: boolean;
  duration_ms: number;
}

export interface DatasetUploadResponse {
  dataset_id: string;
  description: string;
  file_path: string;
  row_count: number;
  column_count: number;
}

/** adia/evidence/renderer.py::RenderedEvidence -- already bounded/truncated server-side. */
export interface RenderedEvidence {
  evidence_id: string;
  tool: string;
  arguments: Record<string, unknown>;
  generated_at: string;
  key_values: Record<string, unknown>;
  summary: string;
}

// --- POST /chat/stream events ---------------------------------------------------------------

export interface StreamPhaseEvent {
  type: "phase";
  node: string;
  data: Record<string, unknown>;
}

export interface StreamEvidenceEvent {
  type: "evidence";
  evidence: RenderedEvidence;
}

export interface StreamFinalEvent {
  type: "final";
  run_id: string;
  dataset_id: string;
  question: string;
  answer: string | null;
  validation_passed: boolean | null;
  evidence: RenderedEvidence[];
  tools_used: string[];
  feasibility_verdict: string | null;
  refused: boolean;
  duration_ms: number;
}

export interface StreamErrorEvent {
  type: "error";
  detail: string;
}

export type StreamEvent =
  | StreamPhaseEvent
  | StreamEvidenceEvent
  | StreamFinalEvent
  | StreamErrorEvent;

/**
 * A dataset known to this session -- either the shipped default or one uploaded this session.
 * Deliberately the exact shape a future `GET /datasets` response item would need, so swapping
 * the session-local list in `app/page.tsx` for a real fetch later is a one-line change, not a
 * component rework.
 */
export interface DatasetSummary {
  dataset_id: string;
  description: string;
}
