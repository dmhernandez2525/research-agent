import { useMemo, useState } from "react";

import type { SourceEntry } from "../../types/api";

interface SourceBrowserProps {
  sources: SourceEntry[];
}

function scoreTag(value: number): string {
  if (value >= 0.8) {
    return "text-emerald-700";
  }
  if (value >= 0.55) {
    return "text-amber-700";
  }
  return "text-rose-700";
}

function duplicateMap(sources: SourceEntry[]): Set<string> {
  const normalized = new Map<string, string>();
  const duplicates = new Set<string>();

  for (const source of sources) {
    const key = `${source.domain}-${source.title.toLowerCase().replace(/[^a-z0-9]/g, "")}`;
    const prior = normalized.get(key);
    if (prior) {
      duplicates.add(prior);
      duplicates.add(source.id);
    } else {
      normalized.set(key, source.id);
    }
  }

  return duplicates;
}

export function SourceBrowser({ sources }: SourceBrowserProps): JSX.Element {
  const [minQuality, setMinQuality] = useState(0);
  const [minFreshness, setMinFreshness] = useState(0);
  const [domainFilter, setDomainFilter] = useState("all");
  const [selectedId, setSelectedId] = useState<string | null>(null);

  const domains = useMemo(() => [...new Set(sources.map((source) => source.domain))].sort(), [sources]);
  const duplicates = useMemo(() => duplicateMap(sources), [sources]);

  const filtered = useMemo(
    () =>
      sources.filter((source) => {
        if (source.quality_score < minQuality) {
          return false;
        }
        if (source.freshness < minFreshness) {
          return false;
        }
        if (domainFilter !== "all" && source.domain !== domainFilter) {
          return false;
        }
        return true;
      }),
    [domainFilter, minFreshness, minQuality, sources]
  );

  const selected = filtered.find((source) => source.id === selectedId) ?? filtered[0] ?? null;

  return (
    <section className="card p-5">
      <div className="mb-3 flex items-center justify-between">
        <h3 className="text-lg font-semibold">Source Browser</h3>
        <p className="text-xs text-ink/60">{filtered.length} shown / {sources.length} total</p>
      </div>

      <div className="mb-3 grid gap-2 md:grid-cols-3">
        <label className="text-xs text-ink/70">
          Quality >= {minQuality.toFixed(2)}
          <input type="range" min="0" max="1" step="0.05" value={minQuality} onChange={(event) => setMinQuality(Number(event.target.value))} className="w-full" />
        </label>
        <label className="text-xs text-ink/70">
          Freshness >= {minFreshness.toFixed(2)}
          <input type="range" min="0" max="1" step="0.05" value={minFreshness} onChange={(event) => setMinFreshness(Number(event.target.value))} className="w-full" />
        </label>
        <label className="text-xs text-ink/70">
          Domain
          <select value={domainFilter} onChange={(event) => setDomainFilter(event.target.value)} className="mt-1 w-full rounded-lg border border-slate-300 bg-white px-2 py-1 text-sm">
            <option value="all">All domains</option>
            {domains.map((domain) => (
              <option key={domain} value={domain}>
                {domain}
              </option>
            ))}
          </select>
        </label>
      </div>

      <div className="grid gap-3 lg:grid-cols-[320px_minmax(0,1fr)]">
        <div className="max-h-[420px] space-y-2 overflow-auto pr-1">
          {filtered.map((source) => (
            <button
              key={source.id}
              id={`source-${source.id}`}
              type="button"
              onClick={() => setSelectedId(source.id)}
              className={`w-full rounded-xl border p-3 text-left ${selected?.id === source.id ? "border-calm bg-sky-50" : "border-slate-200 bg-white"}`}
            >
              <p className="text-sm font-semibold text-ink">{source.title}</p>
              <p className="mt-1 text-xs text-ink/60">{source.domain}</p>
              <div className="mt-2 flex flex-wrap gap-2 text-xs">
                <span className={scoreTag(source.quality_score)}>Q: {(source.quality_score * 100).toFixed(0)}%</span>
                <span className={scoreTag(source.freshness)}>F: {(source.freshness * 100).toFixed(0)}%</span>
                {duplicates.has(source.id) ? <span className="rounded-full bg-amber-100 px-2 py-0.5 text-amber-700">Near-duplicate</span> : null}
              </div>
            </button>
          ))}
        </div>

        <div className="rounded-xl border border-slate-200 bg-white p-4">
          {selected ? (
            <>
              <h4 className="text-base font-semibold">{selected.title}</h4>
              <p className="mt-2 text-xs text-ink/60">Subtopic: {selected.subtopic}</p>
              <p className="text-xs text-ink/60">Query provenance: {selected.query}</p>
              <p className="mt-3 rounded-lg bg-slate-50 p-3 text-sm leading-relaxed text-ink/80">{selected.content_preview}</p>
            </>
          ) : (
            <p className="text-sm text-ink/60">No sources available for this session yet.</p>
          )}
        </div>
      </div>
    </section>
  );
}
