import { useMemo } from "react";

import type { SessionEvent, SessionRecord } from "../../types/api";
import { toCurrency } from "../../utils/sessions";

interface ProgressDashboardProps {
  session: SessionRecord;
  events: SessionEvent[];
  connected: boolean;
  transport: "websocket" | "sse" | "disconnected";
}

const PIPELINE = ["plan", "search", "scrape", "summarize", "synthesize"];

interface SubtopicProgress {
  name: string;
  progress: number;
}

function toNumber(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === "string") {
    const parsed = Number.parseFloat(value);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}

function progressFromEvents(events: SessionEvent[]): SubtopicProgress[] {
  const map = new Map<string, number>();
  for (const event of events) {
    const subtopic = (event.payload.subtopic as string | undefined) ?? "overall";
    const progress = toNumber(event.payload.progress);
    if (progress !== null) {
      map.set(subtopic, progress);
    }
  }
  return [...map.entries()].map(([name, progress]) => ({ name, progress }));
}

function activeStep(events: SessionEvent[], fallback: string): string {
  for (let index = events.length - 1; index >= 0; index -= 1) {
    const step = events[index].payload.step;
    if (typeof step === "string") {
      return step;
    }
  }
  return fallback;
}

function counters(events: SessionEvent[], session: SessionRecord): { tokens: number; cost: number } {
  let tokens = session.tokens_used;
  let cost = session.cost_usd;

  for (const event of events) {
    const payloadTokens = toNumber(event.payload.tokens);
    const payloadCost = toNumber(event.payload.cost_usd);
    if (payloadTokens !== null) {
      tokens = Math.max(tokens, payloadTokens);
    }
    if (payloadCost !== null) {
      cost = Math.max(cost, payloadCost);
    }
  }

  return { tokens, cost };
}

export function ProgressDashboard({
  session,
  events,
  connected,
  transport,
}: ProgressDashboardProps): JSX.Element {
  const current = useMemo(() => activeStep(events, session.current_step), [events, session.current_step]);
  const subtopic = useMemo(() => progressFromEvents(events), [events]);
  const totals = useMemo(() => counters(events, session), [events, session]);

  return (
    <section className="card p-5">
      <div className="mb-4 flex items-center justify-between">
        <h3 className="text-lg font-semibold">Real-Time Progress</h3>
        <span className={`status-pill ${connected ? "bg-emerald-100 text-emerald-700" : "bg-rose-100 text-rose-700"}`}>
          {connected ? `${transport.toUpperCase()} connected` : "Disconnected"}
        </span>
      </div>

      <div className="mb-4 grid gap-2 md:grid-cols-5">
        {PIPELINE.map((step) => {
          const isDone = PIPELINE.indexOf(step) < PIPELINE.indexOf(current);
          const isActive = step === current;
          return (
            <div
              key={step}
              className={`rounded-xl border px-3 py-2 text-center text-xs uppercase tracking-wide ${
                isDone
                  ? "border-emerald-200 bg-emerald-50 text-emerald-700"
                  : isActive
                    ? "border-accent bg-amber-50 text-accent"
                    : "border-slate-200 bg-white text-slate-500"
              }`}
            >
              {isActive ? <span className="mr-1 inline-block h-2 w-2 animate-pulse rounded-full bg-accent" /> : null}
              {step}
            </div>
          );
        })}
      </div>

      <div className="mb-4 grid gap-2 md:grid-cols-2">
        <div className="rounded-xl bg-slate-100 p-3">
          <p className="text-xs uppercase tracking-wide text-slate-600">Cost</p>
          <p className="text-lg font-semibold">{toCurrency(totals.cost)}</p>
        </div>
        <div className="rounded-xl bg-slate-100 p-3">
          <p className="text-xs uppercase tracking-wide text-slate-600">Tokens</p>
          <p className="text-lg font-semibold">{totals.tokens.toLocaleString()}</p>
        </div>
      </div>

      <div className="space-y-2">
        {(subtopic.length ? subtopic : [{ name: "overall", progress: session.progress }]).map((item) => (
          <div key={item.name}>
            <div className="mb-1 flex items-center justify-between text-xs text-ink/70">
              <span>{item.name}</span>
              <span>{item.progress.toFixed(1)}%</span>
            </div>
            <div className="h-2 rounded-full bg-slate-200">
              <div
                className="h-2 rounded-full bg-calm transition-all"
                style={{ width: `${Math.min(100, Math.max(0, item.progress))}%` }}
              />
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}
