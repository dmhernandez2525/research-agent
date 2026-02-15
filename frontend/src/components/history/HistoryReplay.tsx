import { useMemo, useState } from "react";

import { useLocalStorage } from "../../hooks/useLocalStorage";
import type { SessionHistoryEntry, SessionRecord } from "../../types/api";
import { toReadableDate } from "../../utils/sessions";

interface HistoryReplayProps {
  sessions: SessionRecord[];
}

const STORAGE_KEY = "ra_history_meta";

interface HistoryMetaRecord {
  tags: string[];
  bookmarked: boolean;
}

function statusTimeline(session: SessionRecord): string[] {
  const base = ["plan", "search", "scrape", "summarize", "synthesize"];
  if (session.status === "FAILED") {
    return [...base, "failed"];
  }
  if (session.status === "CANCELLED") {
    return [...base.slice(0, 3), "cancelled"];
  }
  return [...base, "completed"];
}

function exportArchive(entries: SessionHistoryEntry[]): void {
  const blob = new Blob([JSON.stringify(entries, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = "session-history.json";
  anchor.click();
  URL.revokeObjectURL(url);
}

export function HistoryReplay({ sessions }: HistoryReplayProps): JSX.Element {
  const [search, setSearch] = useState("");
  const [dateFilter, setDateFilter] = useState("");
  const [selectedId, setSelectedId] = useState<string | null>(sessions[0]?.id ?? null);
  const [comparisonId, setComparisonId] = useState<string | null>(sessions[1]?.id ?? null);
  const [timelineIndex, setTimelineIndex] = useState(0);
  const [historyMeta, setHistoryMeta] = useLocalStorage<Record<string, HistoryMetaRecord>>(STORAGE_KEY, {});

  const filtered = useMemo(
    () =>
      sessions.filter((session) => {
        if (search && !session.query.toLowerCase().includes(search.toLowerCase())) {
          return false;
        }
        if (dateFilter && !session.created_at.startsWith(dateFilter)) {
          return false;
        }
        return true;
      }),
    [dateFilter, search, sessions]
  );

  const selected = filtered.find((item) => item.id === selectedId) ?? filtered[0] ?? null;
  const comparison = filtered.find((item) => item.id === comparisonId) ?? null;
  const timeline = selected ? statusTimeline(selected) : [];

  const toggleBookmark = (sessionId: string) => {
    const current = historyMeta[sessionId] ?? { tags: [], bookmarked: false };
    setHistoryMeta({
      ...historyMeta,
      [sessionId]: {
        ...current,
        bookmarked: !current.bookmarked,
      },
    });
  };

  const updateTags = (sessionId: string, raw: string) => {
    const tags = raw
      .split(",")
      .map((tag) => tag.trim())
      .filter(Boolean);
    const current = historyMeta[sessionId] ?? { tags: [], bookmarked: false };
    setHistoryMeta({
      ...historyMeta,
      [sessionId]: {
        ...current,
        tags,
      },
    });
  };

  const entries: SessionHistoryEntry[] = filtered.map((session) => {
    const meta = historyMeta[session.id] ?? { tags: [], bookmarked: false };
    return { session, ...meta };
  });

  return (
    <section className="card p-5">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <h2 className="text-lg font-semibold">Session History & Replay</h2>
        <button type="button" onClick={() => exportArchive(entries)} className="rounded-full border border-slate-300 px-3 py-1 text-xs font-semibold uppercase tracking-wide text-ink/70">
          Export JSON
        </button>
      </div>

      <div className="mb-3 grid gap-2 md:grid-cols-3">
        <input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Search query" className="rounded-xl border border-slate-300 bg-white px-3 py-2 text-sm" />
        <input type="date" value={dateFilter} onChange={(event) => setDateFilter(event.target.value)} className="rounded-xl border border-slate-300 bg-white px-3 py-2 text-sm" />
        <label className="rounded-xl border border-dashed border-slate-300 bg-white px-3 py-2 text-xs text-ink/60">
          Import JSON
          <input
            type="file"
            accept="application/json"
            onChange={(event) => {
              const file = event.target.files?.[0];
              if (!file) {
                return;
              }
              void file.text().then(() => {
                // Import placeholder: preserves UI contract without mutating server state.
              });
            }}
            className="mt-1 block w-full"
          />
        </label>
      </div>

      <div className="grid gap-3 lg:grid-cols-[280px_minmax(0,1fr)]">
        <div className="max-h-[420px] space-y-2 overflow-auto pr-1">
          {filtered.map((session) => {
            const meta = historyMeta[session.id] ?? { tags: [], bookmarked: false };
            return (
              <button
                key={session.id}
                type="button"
                onClick={() => setSelectedId(session.id)}
                className={`w-full rounded-xl border p-3 text-left ${selected?.id === session.id ? "border-calm bg-sky-50" : "border-slate-200 bg-white"}`}
              >
                <p className="text-sm font-semibold">{session.query}</p>
                <p className="text-xs text-ink/60">{toReadableDate(session.created_at)}</p>
                <p className="mt-1 text-xs text-ink/70">{session.status}</p>
                <p className="text-xs text-ink/50">Tags: {meta.tags.join(", ") || "none"}</p>
                {meta.bookmarked ? <span className="text-xs text-accent">Bookmarked</span> : null}
              </button>
            );
          })}
        </div>

        <div className="space-y-3">
          {selected ? (
            <>
              <div className="rounded-xl border border-slate-200 bg-white p-4">
                <div className="mb-2 flex items-center justify-between">
                  <h3 className="font-semibold">Timeline Replay</h3>
                  <span className="text-xs text-ink/60">Step {timelineIndex + 1} / {timeline.length}</span>
                </div>
                <div className="mb-2 flex flex-wrap gap-2">
                  {timeline.map((step, index) => (
                    <button
                      key={`${selected.id}-${step}`}
                      type="button"
                      onClick={() => setTimelineIndex(index)}
                      className={`rounded-full px-3 py-1 text-xs ${timelineIndex === index ? "bg-calm text-white" : "bg-slate-100 text-slate-700"}`}
                    >
                      {step}
                    </button>
                  ))}
                </div>
                <p className="text-sm text-ink/70">Current replay step: {timeline[timelineIndex] ?? "n/a"}</p>
              </div>

              <div className="rounded-xl border border-slate-200 bg-white p-4">
                <h3 className="mb-2 font-semibold">Session Comparison</h3>
                <select value={comparisonId ?? ""} onChange={(event) => setComparisonId(event.target.value)} className="mb-3 w-full rounded-lg border border-slate-300 px-2 py-1 text-sm">
                  <option value="">Select comparison session</option>
                  {filtered
                    .filter((item) => item.id !== selected.id)
                    .map((item) => (
                      <option key={item.id} value={item.id}>
                        {item.query} ({item.status})
                      </option>
                    ))}
                </select>
                {comparison ? (
                  <div className="grid gap-2 text-sm md:grid-cols-2">
                    <div className="rounded-lg bg-slate-50 p-2">
                      <p className="font-semibold">Primary</p>
                      <p>{selected.query}</p>
                      <p>Status: {selected.status}</p>
                    </div>
                    <div className="rounded-lg bg-slate-50 p-2">
                      <p className="font-semibold">Comparison</p>
                      <p>{comparison.query}</p>
                      <p>Status: {comparison.status}</p>
                    </div>
                  </div>
                ) : (
                  <p className="text-sm text-ink/60">Pick another session to view side-by-side diff context.</p>
                )}
              </div>

              <div className="rounded-xl border border-slate-200 bg-white p-4">
                <h3 className="mb-2 font-semibold">Bookmark & Tags</h3>
                <button type="button" onClick={() => toggleBookmark(selected.id)} className="mb-2 rounded-full border border-slate-300 px-3 py-1 text-xs font-semibold uppercase tracking-wide text-ink/70">
                  Toggle Bookmark
                </button>
                <input
                  defaultValue={(historyMeta[selected.id]?.tags ?? []).join(",")}
                  onBlur={(event) => updateTags(selected.id, event.target.value)}
                  placeholder="security, long-run"
                  className="w-full rounded-lg border border-slate-300 px-2 py-1 text-sm"
                />
              </div>
            </>
          ) : (
            <p className="text-sm text-ink/60">No historical sessions available.</p>
          )}
        </div>
      </div>
    </section>
  );
}
