import { useEffect, useState } from "react";

import { listSessions } from "../api/client";
import { CostAnalytics } from "../components/analytics/CostAnalytics";
import type { SessionRecord } from "../types/api";

export function AnalyticsPage(): JSX.Element {
  const [sessions, setSessions] = useState<SessionRecord[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    const load = async () => {
      try {
        const data = await listSessions();
        if (active) {
          setSessions(data);
          setError(null);
        }
      } catch (err) {
        if (active) {
          setError(err instanceof Error ? err.message : "Failed to load analytics");
        }
      }
    };

    void load();
    return () => {
      active = false;
    };
  }, []);

  if (error) {
    return <p className="rounded-xl bg-rose-50 p-3 text-sm text-rose-700">{error}</p>;
  }

  return <CostAnalytics sessions={sessions} />;
}
