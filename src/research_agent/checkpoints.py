"""Atomic checkpoint manager with crash-recovery guarantees.

Uses temp file -> fsync -> os.replace pattern for atomic writes,
SHA-256 checksums for integrity verification, corruption recovery,
and automatic rotation of old checkpoints.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import structlog
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from pathlib import Path

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class CheckpointError(Exception):
    """Base exception for checkpoint operations."""


class CheckpointCorruptionError(CheckpointError):
    """Raised when a checkpoint fails integrity verification."""


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class CheckpointMetadata(BaseModel):
    """Metadata stored alongside each checkpoint."""

    checkpoint_id: str = Field(description="Unique checkpoint identifier.")
    created_at: str = Field(
        default_factory=lambda: datetime.now(tz=UTC).isoformat(),
    )
    step_index: int = Field(default=0, description="Graph step index at save time.")
    step_name: str = Field(default="", description="Graph node name at save time.")
    sha256: str = Field(
        default="", description="SHA-256 hex digest of the state payload."
    )
    state_size_bytes: int = Field(default=0, ge=0)


# ---------------------------------------------------------------------------
# Checkpoint Manager
# ---------------------------------------------------------------------------


class CheckpointManager:
    """Atomic checkpoint manager with integrity verification and rotation.

    Attributes:
        directory: Path to the checkpoint directory.
        max_checkpoints: Maximum number of retained checkpoint files.
    """

    def __init__(self, directory: Path, max_checkpoints: int = 5) -> None:
        """Initialize the checkpoint manager.

        Args:
            directory: Path to the checkpoint directory (created if needed).
            max_checkpoints: Maximum number of retained checkpoints.
        """
        self.directory = directory
        self.max_checkpoints = max_checkpoints
        self.directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _compute_checksum(data: bytes) -> str:
        """Compute SHA-256 hex digest for the given bytes.

        Args:
            data: Raw bytes to hash.

        Returns:
            Hex-encoded SHA-256 digest.
        """
        return hashlib.sha256(data).hexdigest()

    def _checkpoint_path(self, checkpoint_id: str) -> Path:
        """Return the file path for a checkpoint.

        Args:
            checkpoint_id: Unique checkpoint identifier.

        Returns:
            Path to the checkpoint JSON file.
        """
        return self.directory / f"{checkpoint_id}.json"

    def _metadata_path(self, checkpoint_id: str) -> Path:
        """Return the file path for checkpoint metadata.

        Args:
            checkpoint_id: Unique checkpoint identifier.

        Returns:
            Path to the metadata JSON file.
        """
        return self.directory / f"{checkpoint_id}.meta.json"

    def save(
        self,
        checkpoint_id: str,
        state: dict[str, Any],
        step_index: int = 0,
        step_name: str = "",
    ) -> CheckpointMetadata:
        """Atomically save a checkpoint with integrity metadata.

        Uses the temp-file -> fsync -> os.replace pattern to ensure
        atomic writes that survive crashes.

        Args:
            checkpoint_id: Unique identifier for this checkpoint.
            state: Serializable state dictionary to persist.
            step_index: Current graph step index.
            step_name: Current graph node name.

        Returns:
            Metadata for the saved checkpoint.

        Raises:
            CheckpointError: If the save operation fails.
        """
        payload = json.dumps(state, default=str, sort_keys=True).encode("utf-8")
        checksum = self._compute_checksum(payload)

        metadata = CheckpointMetadata(
            checkpoint_id=checkpoint_id,
            step_index=step_index,
            step_name=step_name,
            sha256=checksum,
            state_size_bytes=len(payload),
        )

        try:
            self._atomic_write(self._checkpoint_path(checkpoint_id), payload)
            self._atomic_write(
                self._metadata_path(checkpoint_id),
                metadata.model_dump_json(indent=2).encode("utf-8"),
            )
        except OSError as exc:
            raise CheckpointError(f"Failed to save checkpoint {checkpoint_id}") from exc

        self._rotate()
        logger.info(
            "checkpoint_saved",
            checkpoint_id=checkpoint_id,
            step_index=step_index,
            size_bytes=len(payload),
        )
        return metadata

    def load(self, checkpoint_id: str) -> dict[str, Any]:
        """Load and verify a checkpoint.

        Args:
            checkpoint_id: Identifier of the checkpoint to load.

        Returns:
            The deserialized state dictionary.

        Raises:
            CheckpointCorruptionError: If the checksum does not match.
            CheckpointError: If the checkpoint cannot be read.
        """
        cp_path = self._checkpoint_path(checkpoint_id)
        meta_path = self._metadata_path(checkpoint_id)

        if not cp_path.exists():
            raise CheckpointError(f"Checkpoint not found: {checkpoint_id}")

        payload = cp_path.read_bytes()

        if meta_path.exists():
            meta = CheckpointMetadata.model_validate_json(meta_path.read_bytes())
            actual = self._compute_checksum(payload)
            if actual != meta.sha256:
                raise CheckpointCorruptionError(
                    f"Checkpoint {checkpoint_id} is corrupt: "
                    f"expected {meta.sha256}, got {actual}"
                )

        state: dict[str, Any] = json.loads(payload)
        logger.info("checkpoint_loaded", checkpoint_id=checkpoint_id)
        return state

    def latest(self) -> str | None:
        """Return the ID of the most recent checkpoint, or ``None``.

        Returns:
            The checkpoint ID string, or ``None`` if no checkpoints exist.
        """
        meta_files = sorted(
            self.directory.glob("*.meta.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not meta_files:
            return None
        return meta_files[0].stem.removesuffix(".meta")

    def list_checkpoints(self) -> list[CheckpointMetadata]:
        """List all available checkpoints, newest first.

        Returns:
            List of checkpoint metadata, sorted by creation time descending.
        """
        metas: list[CheckpointMetadata] = []
        for meta_path in sorted(
            self.directory.glob("*.meta.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        ):
            try:
                metas.append(
                    CheckpointMetadata.model_validate_json(meta_path.read_bytes())
                )
            except Exception:
                logger.warning("corrupt_metadata", path=str(meta_path))
        return metas

    def _rotate(self) -> None:
        """Remove oldest checkpoints exceeding ``max_checkpoints``.

        Always retains at least 2 checkpoints regardless of configuration
        to prevent total state loss if the most recent checkpoint is corrupt.
        """
        effective_max = max(self.max_checkpoints, 2)
        checkpoints = self.list_checkpoints()
        if len(checkpoints) <= effective_max:
            return

        for old in checkpoints[effective_max:]:
            cp_path = self._checkpoint_path(old.checkpoint_id)
            meta_path = self._metadata_path(old.checkpoint_id)
            for path in (cp_path, meta_path):
                if path.exists():
                    path.unlink()
            logger.debug("checkpoint_rotated", checkpoint_id=old.checkpoint_id)

    @staticmethod
    def _atomic_write(path: Path, data: bytes) -> None:
        """Write data atomically using temp file -> fsync -> os.replace.

        Args:
            path: Target file path.
            data: Bytes to write.
        """
        fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        fd_closed = False
        try:
            os.write(fd, data)
            os.fsync(fd)
            os.close(fd)
            fd_closed = True
            os.replace(tmp_path, str(path))
        except BaseException:
            if not fd_closed:
                os.close(fd)
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise
