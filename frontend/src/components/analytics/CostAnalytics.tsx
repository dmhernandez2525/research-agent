import { useMemo } from "react";

import type { SessionRecord } from "../../types/api";
import { toCurrency } from "../../utils/sessions";

interface CostAnalyticsProps {
  sessions: SessionRecord[];
}

interface TrendPoint {
  label: string;
  value: number;
}

function linePath(points: TrendPoint[]): string {
  if (points.length === 0) {
    return "";
  }
  const width = 280;
  const height = 80;
  const max = Math.max(...points.map((point) => point.value), 1);

  return points
    .map((point, index) => {
      const x = (index / Math.max(points.length - 1, 1)) * width;
      const y = height - (point.value / max) * height;
      return `${index === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
}

function downloadCsv(rows: string[][]): void {
  const csv = rows.map((row) => row.map((cell) => JSON.stringify(cell)).join(",")).join("\n");
  const blob = new Blob([csv], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = "session-analytics.csv";
  anchor.click();
  URL.revokeObjectURL(url);
}

export function CostAnalytics({ sessions }: CostAnalyticsProps): JSX.Element {
  const totalCost = useMemo(() => sessions.reduce((sum, session) => sum + session.cost_usd, 0), [sessions]);
  const totalTokens = useMemo(
    () => sessions.reduce((sum, session) => sum + session.tokens_used, 0),
    [sessions]
  );

  const trend = useMemo(() => {
    const sorted = [...sessions].sort((a, b) => Date.parse(a.created_at) - Date.parse(b.created_at));
    return sorted.map((session) => ({
      label: new Date(session.created_at).toLocaleDateString(),
      value: session.cost_usd,
    }));
  }, [sessions]);

  const byStatus = useMemo(() => {
    const totals: Record<string, number> = {};
    for (const session of sessions) {
      totals[session.status] = (totals[session.status] ?? 0) + session.cost_usd;
    }
    return Object.entries(totals).sort((a, b) => b[1] - a[1]);
  }, [sessions]);

  const budgetUsed = sessions.reduce((sum, session) => sum + Math.min(1, session.cost_usd / 0.35), 0);
  const avgBudget = sessions.length > 0 ? (budgetUsed / sessions.length) * 100 : 0;
  const tier = avgBudget < 35 ? "FULL" : avgBudget < 60 ? "REDUCED" : avgBudget < 85 ? "CACHED" : "PARTIAL";

  const exportCsv = () => {
    const rows = [["session_id", "status", "cost_usd", "tokens", "created_at"]];
    for (const session of sessions) {
      rows.push([
        session.id,
        session.status,
        session.cost_usd.toFixed(4),
        String(session.tokens_used),
        session.created_at,
      ]);
    }
    downloadCsv(rows);
  };

  return (
    <section className="card p-5">
      <div className="mb-4 flex items-center justify-between">
        <h2 className="text-lg font-semibold">Cost Analytics Dashboard</h2>
        <button type="button" onClick={exportCsv} className="rounded-full border border-slate-300 px-3 py-1 text-xs font-semibold uppercase tracking-wide text-ink/70">
          Export CSV
        </button>
      </div>

      <div className="mb-4 grid gap-3 md:grid-cols-3">
        <div className="rounded-xl bg-slate-100 p-3">
          <p className="text-xs uppercase tracking-wide text-slate-600">Total Cost</p>
          <p className="text-xl font-semibold">{toCurrency(totalCost)}</p>
        </div>
        <div className="rounded-xl bg-slate-100 p-3">
          <p className="text-xs uppercase tracking-wide text-slate-600">Total Tokens</p>
          <p className="text-xl font-semibold">{totalTokens.toLocaleString()}</p>
        </div>
        <div className="rounded-xl bg-slate-100 p-3">
          <p className="text-xs uppercase tracking-wide text-slate-600">Degradation Tier</p>
          <p className="text-xl font-semibold">{tier}</p>
        </div>
      </div>

      <div className="mb-4 grid gap-3 lg:grid-cols-2">
        <div className="rounded-xl border border-slate-200 bg-white p-4">
          <h3 className="mb-2 text-sm font-semibold text-ink/70">Historical Cost Trend</h3>
          <svg viewBox="0 0 280 80" className="h-24 w-full" role="img" aria-label="Cost trend">
            <path d={linePath(trend)} fill="none" stroke="#3d5a80" strokeWidth="3" strokeLinecap="round" />
          </svg>
          <div className="mt-2 flex flex-wrap gap-2 text-xs text-ink/60">
            {trend.map((point) => (
              <span key={`${point.label}-${point.value}`}>{point.label}: {toCurrency(point.value)}</span>
            ))}
          </div>
        </div>

        <div className="rounded-xl border border-slate-200 bg-white p-4">
          <h3 className="mb-2 text-sm font-semibold text-ink/70">Cost by Session Status</h3>
          <ul className="space-y-2 text-sm">
            {byStatus.map(([status, value]) => (
              <li key={status} className="flex items-center justify-between">
                <span>{status}</span>
                <span className="font-semibold">{toCurrency(value)}</span>
              </li>
            ))}
          </ul>
          <div className="mt-3">
            <p className="text-xs text-ink/60">Budget utilization: {avgBudget.toFixed(1)}%</p>
            <div className="mt-1 h-2 rounded-full bg-slate-200">
              <div className="h-2 rounded-full bg-accent" style={{ width: `${Math.min(avgBudget, 100)}%` }} />
            </div>
          </div>
        </div>
      </div>

      <p className="text-xs text-ink/60">Token distribution approximates input/output/cached split using session totals.</p>
    </section>
  );
}
