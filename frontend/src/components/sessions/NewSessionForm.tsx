import { type FormEvent, useState } from "react";

import type { SessionCreateRequest } from "../../types/api";

interface NewSessionFormProps {
  onSubmit: (payload: SessionCreateRequest) => Promise<void>;
  busy: boolean;
}

export function NewSessionForm({ onSubmit, busy }: NewSessionFormProps): JSX.Element {
  const [query, setQuery] = useState("");
  const [budget, setBudget] = useState("0.35");
  const [outputFormat, setOutputFormat] = useState<"md" | "pdf">("md");
  const [error, setError] = useState<string | null>(null);

  const submit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!query.trim()) {
      setError("Query is required.");
      return;
    }

    const parsedBudget = Number.parseFloat(budget);
    const payload: SessionCreateRequest = {
      query: query.trim(),
      output_format: outputFormat,
      budget: Number.isFinite(parsedBudget) && parsedBudget > 0 ? parsedBudget : undefined,
    };

    setError(null);
    await onSubmit(payload);
    setQuery("");
  };

  return (
    <section className="card p-5">
      <h2 className="mb-3 text-lg font-semibold">New Research Session</h2>
      <form className="grid gap-3 md:grid-cols-4" onSubmit={submit}>
        <label className="md:col-span-2">
          <span className="mb-1 block text-sm text-ink/70">Query</span>
          <input
            type="text"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Investigate local-first vector databases..."
            className="w-full rounded-xl border border-slate-300 bg-white px-3 py-2 text-sm outline-none focus:border-accent"
          />
        </label>

        <label>
          <span className="mb-1 block text-sm text-ink/70">Budget (USD)</span>
          <input
            type="number"
            min="0"
            step="0.01"
            value={budget}
            onChange={(event) => setBudget(event.target.value)}
            className="w-full rounded-xl border border-slate-300 bg-white px-3 py-2 text-sm outline-none focus:border-accent"
          />
        </label>

        <label>
          <span className="mb-1 block text-sm text-ink/70">Format</span>
          <select
            value={outputFormat}
            onChange={(event) => setOutputFormat(event.target.value as "md" | "pdf")}
            className="w-full rounded-xl border border-slate-300 bg-white px-3 py-2 text-sm outline-none focus:border-accent"
          >
            <option value="md">Markdown</option>
            <option value="pdf">PDF</option>
          </select>
        </label>

        <div className="md:col-span-4 flex items-center gap-3">
          <button
            type="submit"
            disabled={busy}
            className="rounded-full bg-calm px-4 py-2 text-sm font-semibold text-white transition hover:brightness-95 disabled:opacity-60"
          >
            {busy ? "Submitting..." : "Start Session"}
          </button>
          {error ? <p className="text-sm text-rose-600">{error}</p> : null}
        </div>
      </form>
    </section>
  );
}
