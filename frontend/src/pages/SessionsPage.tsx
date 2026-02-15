import { useCallback, useEffect, useState } from "react";

import { cancelSession, createSession, listSessions } from "../api/client";
import { NewSessionForm } from "../components/sessions/NewSessionForm";
import { SessionList } from "../components/sessions/SessionList";
import type { SessionCreateRequest, SessionRecord } from "../types/api";

export function SessionsPage(): JSX.Element {
  const [sessions, setSessions] = useState<SessionRecord[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      const data = await listSessions();
      setSessions(data);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load sessions");
    }
  }, []);

  useEffect(() => {
    void load();
    const timer = window.setInterval(() => {
      void load();
    }, 4000);
    return () => window.clearInterval(timer);
  }, [load]);

  const handleCreate = async (payload: SessionCreateRequest) => {
    setLoading(true);
    try {
      const created = await createSession(payload);
      setSessions((prev) => [created, ...prev]);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not create session");
    } finally {
      setLoading(false);
    }
  };

  const handleCancel = async (session: SessionRecord) => {
    const action = session.status === "RUNNING" || session.status === "QUEUED" ? "cancel" : "delete";
    const proceed = window.confirm(`Confirm ${action} for session ${session.id}?`);
    if (!proceed) {
      return;
    }
    try {
      await cancelSession(session.id);
      await load();
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Action failed");
    }
  };

  return (
    <div className="space-y-4">
      <NewSessionForm onSubmit={handleCreate} busy={loading} />
      {error ? <p className="rounded-xl bg-rose-50 p-3 text-sm text-rose-700">{error}</p> : null}
      <SessionList sessions={sessions} onCancelOrDelete={handleCancel} />
    </div>
  );
}
