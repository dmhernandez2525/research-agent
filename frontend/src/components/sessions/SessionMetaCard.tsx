import type { SessionRecord } from "../../types/api";
import {
  sessionDurationSeconds,
  sessionStatusClass,
  sessionStatusLabel,
  toCurrency,
  toReadableDate,
} from "../../utils/sessions";

interface SessionMetaCardProps {
  session: SessionRecord;
}

export function SessionMetaCard({ session }: SessionMetaCardProps): JSX.Element {
  return (
    <section className="card p-5">
      <div className="mb-3 flex items-center justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold">Session Details</h2>
          <p className="text-xs text-ink/60">{session.id}</p>
        </div>
        <span className={`status-pill ${sessionStatusClass(session.status)}`}>
          {sessionStatusLabel(session.status)}
        </span>
      </div>

      <dl className="grid gap-3 text-sm md:grid-cols-2">
        <div>
          <dt className="text-ink/50">Current Step</dt>
          <dd>{session.current_step}</dd>
        </div>
        <div>
          <dt className="text-ink/50">Progress</dt>
          <dd>{session.progress.toFixed(1)}%</dd>
        </div>
        <div>
          <dt className="text-ink/50">Cost</dt>
          <dd>{toCurrency(session.cost_usd)}</dd>
        </div>
        <div>
          <dt className="text-ink/50">Token Usage</dt>
          <dd>{session.tokens_used.toLocaleString()}</dd>
        </div>
        <div>
          <dt className="text-ink/50">Duration</dt>
          <dd>{sessionDurationSeconds(session)} seconds</dd>
        </div>
        <div>
          <dt className="text-ink/50">Last Updated</dt>
          <dd>{toReadableDate(session.updated_at)}</dd>
        </div>
      </dl>

      {session.error ? (
        <p className="mt-3 rounded-xl bg-rose-50 p-2 text-sm text-rose-700">{session.error}</p>
      ) : null}
    </section>
  );
}
