import { useEffect, useMemo, useRef, useState } from "react";

import { openWebSocket, streamEvents } from "../api/client";
import type { SessionEvent } from "../types/api";

interface SessionEventState {
  events: SessionEvent[];
  connected: boolean;
  transport: "websocket" | "sse" | "disconnected";
}

export function useSessionEvents(sessionId: string | null): SessionEventState {
  const [events, setEvents] = useState<SessionEvent[]>([]);
  const [connected, setConnected] = useState(false);
  const [transport, setTransport] = useState<"websocket" | "sse" | "disconnected">(
    "disconnected"
  );
  const reconnectRef = useRef<number>(0);
  const lastEventIdRef = useRef<number | undefined>(undefined);

  useEffect(() => {
    if (!sessionId) return;
    setEvents([]);
    lastEventIdRef.current = undefined;
    let active = true;
    let ws: WebSocket | null = null;

    const connectWs = () => {
      if (!active) return;
      ws = openWebSocket(sessionId);

      ws.addEventListener("open", () => {
        setConnected(true);
        setTransport("websocket");
        reconnectRef.current = 0;
      });

      ws.addEventListener("message", (msg) => {
        try {
          const data = JSON.parse(msg.data) as SessionEvent | { event_type: string };
          if ("id" in data) {
            lastEventIdRef.current = data.id;
            setEvents((prev) => [...prev, data]);
          }
        } catch {
          // ignore malformed events
        }
      });

      ws.addEventListener("close", () => {
        setConnected(false);
        if (!active) return;

        reconnectRef.current += 1;
        const delay = Math.min(1500 * reconnectRef.current, 7000);
        window.setTimeout(() => {
          if (reconnectRef.current > 2) {
            void connectSse();
            return;
          }
          connectWs();
        }, delay);
      });
    };

    const connectSse = async () => {
      setTransport("sse");
      setConnected(true);
      try {
        for await (const event of streamEvents(sessionId, lastEventIdRef.current)) {
          if (!active) return;
          lastEventIdRef.current = event.id;
          setEvents((prev) => [...prev, event]);
        }
      } catch {
        setConnected(false);
      }
    };

    connectWs();

    return () => {
      active = false;
      setConnected(false);
      setTransport("disconnected");
      ws?.close();
    };
  }, [sessionId]);

  return { events, connected, transport };
}
