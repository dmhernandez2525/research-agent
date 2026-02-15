import type {
  SessionCreateRequest,
  SessionEvent,
  SessionRecord,
} from "../types/api";
import type { SessionCreatePost, SessionCreateResponse, SessionListGet } from "./generated";

const API_KEY_STORAGE = "ra_api_key";

function headers(): HeadersInit {
  const key = localStorage.getItem(API_KEY_STORAGE) || "";
  return key ? { "X-API-Key": key } : {};
}

export function setApiKey(value: string): void {
  localStorage.setItem(API_KEY_STORAGE, value);
}

async function json<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Request failed: ${response.status}`);
  }
  return (await response.json()) as T;
}

export async function listSessions(): Promise<SessionRecord[]> {
  const response = await fetch("/api/sessions", { headers: headers() });
  const data = await json<SessionListGet>(response);
  return data.sessions;
}

export async function createSession(
  payload: SessionCreateRequest
): Promise<SessionRecord> {
  const requestPayload: SessionCreatePost = payload;
  const response = await fetch("/api/sessions", {
    method: "POST",
    headers: { "Content-Type": "application/json", ...headers() },
    body: JSON.stringify(requestPayload),
  });
  return json<SessionCreateResponse>(response);
}

export async function getSession(sessionId: string): Promise<SessionRecord> {
  const response = await fetch(`/api/sessions/${sessionId}`, { headers: headers() });
  return json<SessionRecord>(response);
}

export async function cancelSession(sessionId: string): Promise<void> {
  const response = await fetch(`/api/sessions/${sessionId}`, {
    method: "DELETE",
    headers: headers(),
  });
  if (!response.ok) {
    throw new Error(await response.text());
  }
}

export async function fetchReport(sessionId: string): Promise<string> {
  const response = await fetch(`/api/sessions/${sessionId}/report`, {
    headers: headers(),
  });
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return response.text();
}

export function openWebSocket(sessionId: string): WebSocket {
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  const apiKey = localStorage.getItem(API_KEY_STORAGE) || "";
  const query = apiKey ? `?api_key=${encodeURIComponent(apiKey)}` : "";
  return new WebSocket(
    `${protocol}://${window.location.host}/ws/sessions/${sessionId}${query}`
  );
}

export async function* streamEvents(
  sessionId: string,
  lastEventId?: number
): AsyncGenerator<SessionEvent> {
  const response = await fetch(`/api/sessions/${sessionId}/events`, {
    headers: {
      ...headers(),
      ...(lastEventId ? { "Last-Event-ID": String(lastEventId) } : {}),
    },
  });
  if (!response.body) {
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    const chunks = buffer.split("\n\n");
    buffer = chunks.pop() ?? "";

    for (const chunk of chunks) {
      const line = chunk
        .split("\n")
        .find((item) => item.startsWith("data: "));
      if (!line) continue;
      const raw = line.slice(6);
      yield JSON.parse(raw) as SessionEvent;
    }
  }
}
