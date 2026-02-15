"""Progress notification delivery (webhook/slack/email)."""

from __future__ import annotations

import asyncio
import smtplib
from email.message import EmailMessage
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from research_agent.api.models import SessionRecord
    from research_agent.config import APISettings


class NotificationDispatcher:
    """Dispatch filtered notifications with retry/backoff."""

    def __init__(self, settings: APISettings) -> None:
        self._settings = settings

    async def notify(
        self,
        event_type: str,
        session: SessionRecord,
        message: str,
    ) -> None:
        if event_type not in self._settings.notify_on:
            return

        payload = {
            "event": event_type,
            "session_id": session.id,
            "status": session.status.value,
            "query": session.query,
            "message": message,
        }

        tasks: list[asyncio.Task[None]] = []
        if self._settings.webhook_url:
            tasks.append(
                asyncio.create_task(
                    self._post_with_retry(self._settings.webhook_url, payload)
                )
            )
        if self._settings.slack_webhook_url:
            slack_payload = {"text": f"[{event_type}] {session.id}: {message}"}
            tasks.append(
                asyncio.create_task(
                    self._post_with_retry(
                        self._settings.slack_webhook_url, slack_payload
                    )
                )
            )
        if self._settings.smtp_host and self._settings.smtp_username:
            tasks.append(
                asyncio.create_task(
                    asyncio.to_thread(self._send_email, event_type, session, message)
                )
            )

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _post_with_retry(
        self,
        url: str,
        payload: dict[str, str],
        attempts: int = 3,
    ) -> None:
        last_exc: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.post(url, json=payload)
                    response.raise_for_status()
                return
            except Exception as exc:
                last_exc = exc
                if attempt < attempts:
                    await asyncio.sleep(2**attempt)
        if last_exc is not None:
            raise last_exc

    def _send_email(
        self, event_type: str, session: SessionRecord, message: str
    ) -> None:
        smtp_host = self._settings.smtp_host
        if smtp_host is None:
            return

        msg = EmailMessage()
        msg["Subject"] = f"research-agent {event_type}: {session.id}"
        msg["From"] = self._settings.smtp_username or "research-agent@localhost"
        msg["To"] = self._settings.smtp_username or "research-agent@localhost"
        msg.set_content(
            f"Session: {session.id}\n"
            f"Status: {session.status.value}\n"
            f"Query: {session.query}\n\n"
            f"{message}"
        )

        with smtplib.SMTP(smtp_host, self._settings.smtp_port, timeout=10) as smtp:
            if self._settings.smtp_username and self._settings.smtp_password:
                smtp.starttls()
                smtp.login(self._settings.smtp_username, self._settings.smtp_password)
            smtp.send_message(msg)
