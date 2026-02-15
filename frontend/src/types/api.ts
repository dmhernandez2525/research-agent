export type SessionStatus =
  | "QUEUED"
  | "RUNNING"
  | "COMPLETED"
  | "FAILED"
  | "CANCELLED";

export interface SessionRecord {
  id: string;
  query: string;
  status: SessionStatus;
  progress: number;
  current_step: string;
  cost_usd: number;
  tokens_used: number;
  report_path: string | null;
  duration_seconds: number;
  sources: SourceEntry[];
  error: string | null;
  queued_position: number | null;
  created_at: string;
  updated_at: string;
}

export interface SessionListResponse {
  sessions: SessionRecord[];
}

export interface SessionCreateRequest {
  query: string;
  budget?: number;
  output_format: "md" | "pdf";
}

export interface SessionEvent {
  id: number;
  session_id: string;
  event_type: string;
  timestamp: string;
  payload: Record<string, unknown>;
}

export interface SourceEntry {
  id: string;
  domain: string;
  title: string;
  freshness: number;
  quality_score: number;
  subtopic: string;
  query: string;
  content_preview: string;
}

export interface SessionHistoryEntry {
  session: SessionRecord;
  tags: string[];
  bookmarked: boolean;
}
