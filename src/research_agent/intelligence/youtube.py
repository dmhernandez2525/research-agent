"""YouTube transcript extraction with whisper + caption fallback."""

from __future__ import annotations

import json
import re
import subprocess
from typing import TYPE_CHECKING

from research_agent.intelligence.models import (
    VideoTranscriptChunk,
    VideoTranscriptResult,
)

if TYPE_CHECKING:
    from pathlib import Path

_TIMESTAMP_RE = re.compile(r"\[(\d{2}):(\d{2}):(\d{2})\]\s*(.+)")


class YouTubeTranscriptExtractor:
    """Extract and chunk YouTube transcripts for research ingestion."""

    def __init__(self, work_dir: Path) -> None:
        self._work_dir = work_dir
        self._work_dir.mkdir(parents=True, exist_ok=True)

    def extract(self, video_url: str) -> VideoTranscriptResult | None:
        """Extract transcript via whisper with caption fallback."""
        metadata = self._fetch_metadata(video_url)
        if not self._is_quality_video(metadata):
            return None

        transcript = self._transcribe_with_whisper(video_url)
        if not transcript.strip():
            transcript = self._fallback_captions(video_url)

        chunks = self._chunk_transcript(transcript)
        return VideoTranscriptResult(
            video_id=str(metadata.get("id", "")),
            title=str(metadata.get("title", "")),
            channel=str(metadata.get("channel", "")),
            duration_seconds=int(metadata.get("duration", 0) or 0),
            views=int(metadata.get("view_count", 0) or 0),
            language=str(metadata.get("language", "en")),
            chunks=chunks,
        )

    def _fetch_metadata(self, video_url: str) -> dict[str, str | int]:
        command = ["yt-dlp", "--dump-single-json", video_url]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            return {}
        payload = json.loads(result.stdout)
        return {
            "id": str(payload.get("id", "")),
            "title": str(payload.get("title", "")),
            "channel": str(payload.get("channel", "")),
            "duration": int(payload.get("duration", 0) or 0),
            "view_count": int(payload.get("view_count", 0) or 0),
            "language": str(payload.get("language", "en")),
        }

    def _transcribe_with_whisper(self, video_url: str) -> str:
        audio_path = self._work_dir / "audio.m4a"
        transcript_path = self._work_dir / "transcript.txt"

        download = subprocess.run(
            ["yt-dlp", "-f", "bestaudio", "-o", str(audio_path), video_url],
            capture_output=True,
            text=True,
            check=False,
        )
        if download.returncode != 0:
            return ""

        whisper = subprocess.run(
            [
                "whisper-cli",
                "-f",
                str(audio_path),
                "-otxt",
                "-of",
                str(transcript_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if whisper.returncode != 0 or not transcript_path.exists():
            return ""

        return transcript_path.read_text(encoding="utf-8", errors="ignore")

    def _fallback_captions(self, video_url: str) -> str:
        command = [
            "yt-dlp",
            "--skip-download",
            "--write-auto-subs",
            "--sub-langs",
            "en",
            "--sub-format",
            "vtt",
            "-o",
            str(self._work_dir / "caption.%(ext)s"),
            video_url,
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            return ""

        for path in self._work_dir.glob("caption*.vtt"):
            return self._vtt_to_text(path.read_text(encoding="utf-8", errors="ignore"))
        return ""

    def _vtt_to_text(self, vtt: str) -> str:
        lines: list[str] = []
        for line in vtt.splitlines():
            stripped = line.strip()
            if not stripped or "-->" in stripped or stripped.startswith("WEBVTT"):
                continue
            lines.append(stripped)
        return "\n".join(lines)

    def _chunk_transcript(
        self, transcript: str, chunk_seconds: int = 120
    ) -> list[VideoTranscriptChunk]:
        chunks: list[VideoTranscriptChunk] = []
        for line in transcript.splitlines():
            match = _TIMESTAMP_RE.match(line.strip())
            if match:
                hours, minutes, seconds, text = match.groups()
                start = int(hours) * 3600 + int(minutes) * 60 + int(seconds)
                chunks.append(
                    VideoTranscriptChunk(
                        start_seconds=float(start),
                        end_seconds=float(start + chunk_seconds),
                        text=text.strip(),
                    )
                )
            elif line.strip():
                start_seconds = float(len(chunks) * chunk_seconds)
                chunks.append(
                    VideoTranscriptChunk(
                        start_seconds=start_seconds,
                        end_seconds=start_seconds + chunk_seconds,
                        text=line.strip(),
                    )
                )

        merged: list[VideoTranscriptChunk] = []
        for chunk in chunks:
            if merged and chunk.start_seconds - merged[-1].end_seconds <= 10:
                prior = merged[-1]
                merged[-1] = VideoTranscriptChunk(
                    start_seconds=prior.start_seconds,
                    end_seconds=chunk.end_seconds,
                    text=f"{prior.text} {chunk.text}".strip(),
                )
            else:
                merged.append(chunk)
        return merged

    def _is_quality_video(self, metadata: dict[str, str | int]) -> bool:
        duration = int(metadata.get("duration", 0) or 0)
        language = str(metadata.get("language", "en")).lower()
        title = str(metadata.get("title", "")).lower()

        if duration < 180:
            return False
        if language not in {"en", "en-us", "en-gb"}:
            return False
        return not ("music" in title or "lyrics" in title)
