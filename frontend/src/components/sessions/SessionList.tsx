import { Link } from "react-router-dom";

import type { SessionRecord } from "../../types/api";
import {
  sessionDurationSeconds,
  sessionStatusClass,
  sessionStatusLabel,
  toCurrency,
  toReadableDate,
} from "../../utils/sessions";

interface SessionListProps {
  sessions: SessionRecord[];
  onCancelOrDelete: (session: SessionRecord) => Promise<void>;
}

function actionLabel(session: SessionRecord): string {
  if (session.status === "RUNNING" || session.status === "QUEUED") {
    return "Cancel";
  }
  return "Delete";
}

export function SessionList({ sessions, onCancelOrDelete }: SessionListProps): JSX.Element {
  return (
    <section className="card p-5">
      <div className="mb-4 flex items-center justify-between">
        <h2 className="text-lg font-semibold">Sessions</h2>
        <p className="text-sm text-ink/60">{sessions.length} total</p>
      </div>

      {sessions.length === 0 ? (
        <p className="text-sm text-ink/70">No sessions yet.</p>
      ) : (
        <div className="grid gap-3 md:grid-cols-2">
          {sessions.map((session) => (
            <article
              key={session.id}
              className="rounded-2xl border border-slate-200 bg-white/90 p-4 transition hover:-translate-y-0.5"
            >
              <div className="mb-2 flex items-start justify-between gap-2">
                <Link to={`/sessions/${session.id}`} className="font-semibold text-calm hover:underline">
                  {session.query}
                </Link>
                <span className={`status-pill ${sessionStatusClass(session.status)}`}>
                  {sessionStatusLabel(session.status)}
                </span>
              </div>
              <p className="mb-3 text-xs text-ink/60">{session.id}</p>

              <dl className="grid grid-cols-2 gap-2 text-sm">
                <div>
                  <dt className="text-ink/50">Progress</dt>
                  <dd>{session.progress.toFixed(1)}%</dd>
                </div>
                <div>
                  <dt className="text-ink/50">Cost</dt>
                  <dd>{toCurrency(session.cost_usd)}</dd>
                </div>
                <div>
                  <dt className="text-ink/50">Tokens</dt>
                  <dd>{session.tokens_used.toLocaleString()}</dd>
                </div>
                <div>
                  <dt className="text-ink/50">Duration</dt>
                  <dd>{sessionDurationSeconds(session)}s</dd>
                </div>
                <div className="col-span-2">
                  <dt className="text-ink/50">Updated</dt>
                  <dd>{toReadableDate(session.updated_at)}</dd>
                </div>
              </dl>

              <button
                type="button"
                onClick={() => onCancelOrDelete(session)}
                className="mt-3 rounded-full border border-slate-300 px-3 py-1 text-xs font-semibold uppercase tracking-wide text-ink/70 hover:border-rose-300 hover:text-rose-600"
              >
                {actionLabel(session)}
              </button>
            </article>
          ))}
        </div>
      )}
    </section>
  );
}
