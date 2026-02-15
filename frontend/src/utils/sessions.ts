import type { SessionRecord, SessionStatus } from "../types/api";

const STATUS_LABELS: Record<SessionStatus, string> = {
  QUEUED: "Queued",
  RUNNING: "Running",
  COMPLETED: "Completed",
  FAILED: "Failed",
  CANCELLED: "Cancelled",
};

const STATUS_STYLES: Record<SessionStatus, string> = {
  QUEUED: "bg-slate-200 text-slate-700",
  RUNNING: "bg-sky-100 text-sky-700",
  COMPLETED: "bg-emerald-100 text-emerald-700",
  FAILED: "bg-rose-100 text-rose-700",
  CANCELLED: "bg-amber-100 text-amber-700",
};

export function sessionStatusLabel(status: SessionStatus): string {
  return STATUS_LABELS[status];
}

export function sessionStatusClass(status: SessionStatus): string {
  return STATUS_STYLES[status];
}

export function sessionDurationSeconds(session: SessionRecord): number {
  if (session.duration_seconds > 0) {
    return Math.round(session.duration_seconds);
  }
  const start = Date.parse(session.created_at);
  const end = Date.parse(session.updated_at);
  if (Number.isNaN(start) || Number.isNaN(end) || end < start) {
    return 0;
  }
  return Math.round((end - start) / 1000);
}

export function toCurrency(value: number): string {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 4,
    maximumFractionDigits: 4,
  }).format(value);
}

export function toReadableDate(iso: string): string {
  const parsed = new Date(iso);
  if (Number.isNaN(parsed.getTime())) {
    return "Unknown";
  }
  return parsed.toLocaleString();
}
