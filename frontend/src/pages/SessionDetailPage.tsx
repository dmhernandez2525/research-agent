import { useEffect, useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";

import { fetchReport, getSession } from "../api/client";
import { PlanEditor } from "../components/plan/PlanEditor";
import { ProgressDashboard } from "../components/progress/ProgressDashboard";
import { ReportViewer } from "../components/reports/ReportViewer";
import { SessionMetaCard } from "../components/sessions/SessionMetaCard";
import { SourceBrowser } from "../components/sources/SourceBrowser";
import { useSessionEvents } from "../hooks/useSessionEvents";
import type { SessionRecord, SourceEntry } from "../types/api";

function fallbackSources(query: string): SourceEntry[] {
  return [
    {
      id: "source-1",
      domain: "example.com",
      title: `${query} overview`,
      freshness: 0.72,
      quality_score: 0.78,
      subtopic: "Core problem framing",
      query,
      content_preview: "Synthetic source preview generated when session sources are unavailable.",
    },
    {
      id: "source-2",
      domain: "docs.example.org",
      title: `${query} implementation guide`,
      freshness: 0.66,
      quality_score: 0.69,
      subtopic: "Implementation approaches",
      query,
      content_preview: "Use backend source metadata endpoint data to replace this fallback preview.",
    },
  ];
}

export function SessionDetailPage(): JSX.Element {
  const { sessionId = "" } = useParams();
  const [session, setSession] = useState<SessionRecord | null>(null);
  const [report, setReport] = useState<string>("");
  const [error, setError] = useState<string | null>(null);

  const { events, connected, transport } = useSessionEvents(sessionId || null);

  useEffect(() => {
    if (!sessionId) {
      return;
    }

    let active = true;
    const load = async () => {
      try {
        const result = await getSession(sessionId);
        if (!active) {
          return;
        }
        setSession(result);

        if (result.status === "COMPLETED" && result.report_path) {
          const reportText = await fetchReport(sessionId);
          if (active) {
            setReport(reportText);
          }
        }
      } catch (err) {
        if (active) {
          setError(err instanceof Error ? err.message : "Failed to load session");
        }
      }
    };

    void load();
    const timer = window.setInterval(() => {
      void load();
    }, 3500);

    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, [sessionId]);

  const sources = useMemo(() => {
    if (!session) {
      return [];
    }
    return session.sources.length > 0
      ? session.sources
      : fallbackSources(session.query);
  }, [session]);

  if (!sessionId) {
    return <p className="text-sm">Session ID missing.</p>;
  }

  if (error) {
    return (
      <div className="space-y-3">
        <p className="rounded-xl bg-rose-50 p-3 text-sm text-rose-700">{error}</p>
        <Link to="/" className="text-sm text-calm underline">
          Return to sessions
        </Link>
      </div>
    );
  }

  if (!session) {
    return <p className="text-sm text-ink/70">Loading session...</p>;
  }

  return (
    <div className="space-y-4">
      <div>
        <Link to="/" className="text-sm text-calm underline">
          Back to sessions
        </Link>
        <h2 className="text-2xl font-semibold text-ink">{session.query}</h2>
      </div>

      <SessionMetaCard session={session} />
      <ProgressDashboard session={session} events={events} connected={connected} transport={transport} />
      <PlanEditor sessionId={session.id} initialQuery={session.query} />
      <SourceBrowser sources={sources} />

      {report ? (
        <ReportViewer markdown={report} sessionId={session.id} />
      ) : (
        <section className="card p-5">
          <h3 className="text-lg font-semibold">Report Viewer</h3>
          <p className="text-sm text-ink/70">Report becomes available after completion.</p>
        </section>
      )}
    </div>
  );
}
