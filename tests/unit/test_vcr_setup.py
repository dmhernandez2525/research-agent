"""Unit tests for VCR.py cassette recording infrastructure."""

from __future__ import annotations

import os
import socket
from typing import TYPE_CHECKING, Any

import pytest
import vcr

if TYPE_CHECKING:
    from pathlib import Path

from tests.conftest import CASSETTE_DIR, FILTERED_HEADERS


def _dns_available(host: str) -> bool:
    try:
        socket.getaddrinfo(host, 443)
        return True
    except OSError:
        return False


_HTTPBIN_AVAILABLE = _dns_available("httpbin.org")

# ---------------------------------------------------------------------------
# Cassette directory
# ---------------------------------------------------------------------------


class TestCassetteDirectory:
    """The cassette directory exists and is correctly configured."""

    def test_directory_exists(self) -> None:
        assert CASSETTE_DIR.exists()
        assert CASSETTE_DIR.is_dir()

    def test_gitkeep_present(self) -> None:
        gitkeep = CASSETTE_DIR / ".gitkeep"
        assert gitkeep.exists()


# ---------------------------------------------------------------------------
# Filtered headers
# ---------------------------------------------------------------------------


class TestFilteredHeaders:
    """Sensitive headers are filtered from cassette recordings."""

    def test_authorization_filtered(self) -> None:
        assert "authorization" in FILTERED_HEADERS

    def test_api_key_filtered(self) -> None:
        assert "x-api-key" in FILTERED_HEADERS

    def test_anthropic_api_key_filtered(self) -> None:
        assert "anthropic-api-key" in FILTERED_HEADERS

    def test_openai_api_key_filtered(self) -> None:
        assert "openai-api-key" in FILTERED_HEADERS

    def test_generic_api_key_filtered(self) -> None:
        assert "api-key" in FILTERED_HEADERS


# ---------------------------------------------------------------------------
# VCR config fixture
# ---------------------------------------------------------------------------


class TestVcrConfig:
    """vcr_config fixture returns valid configuration."""

    def test_cassette_library_dir(self, vcr_config: dict[str, Any]) -> None:
        assert vcr_config["cassette_library_dir"] == str(CASSETTE_DIR)

    def test_filter_headers_present(self, vcr_config: dict[str, Any]) -> None:
        assert "authorization" in vcr_config["filter_headers"]
        assert "anthropic-api-key" in vcr_config["filter_headers"]

    def test_decode_compressed(self, vcr_config: dict[str, Any]) -> None:
        assert vcr_config["decode_compressed_response"] is True

    def test_default_record_mode_none(self, vcr_config: dict[str, Any]) -> None:
        # When VCR_RECORD_MODE is not set, default to "none"
        if "VCR_RECORD_MODE" not in os.environ:
            assert vcr_config["record_mode"] == "none"


# ---------------------------------------------------------------------------
# VCR cassette recording/replay
# ---------------------------------------------------------------------------


class TestCassetteRecordReplay:
    """VCR cassettes can record and replay HTTP interactions."""

    @pytest.mark.skipif(
        not _HTTPBIN_AVAILABLE,
        reason="Network unavailable for live VCR cassette recording tests.",
    )
    def test_record_and_replay(self, tmp_path: Path) -> None:
        """Verify a cassette records and replays correctly."""
        cassette_path = tmp_path / "test_cassette.yaml"

        my_vcr = vcr.VCR(
            record_mode="all",
            decode_compressed_response=True,
        )

        # Record phase: make a simple HTTP request
        import httpx

        with my_vcr.use_cassette(str(cassette_path)):
            resp = httpx.get("https://httpbin.org/get")
            assert resp.status_code == 200
            recorded_data = resp.json()

        # Cassette file should exist
        assert cassette_path.exists()

        # Replay phase: same request should return cached response
        with my_vcr.use_cassette(str(cassette_path), record_mode="none"):
            resp2 = httpx.get("https://httpbin.org/get")
            assert resp2.status_code == 200
            replayed_data = resp2.json()

        assert recorded_data["url"] == replayed_data["url"]

    @pytest.mark.skipif(
        not _HTTPBIN_AVAILABLE,
        reason="Network unavailable for live VCR cassette recording tests.",
    )
    def test_header_filtering(self, tmp_path: Path) -> None:
        """Verify sensitive headers are stripped from recorded request headers."""
        import yaml

        cassette_path = tmp_path / "filtered_cassette.yaml"

        my_vcr = vcr.VCR(
            record_mode="all",
            filter_headers=FILTERED_HEADERS,
            decode_compressed_response=True,
        )

        import httpx

        with my_vcr.use_cassette(str(cassette_path)):
            httpx.get(
                "https://httpbin.org/get",
                headers={"Authorization": "Bearer secret-token-123"},
            )

        # Parse cassette and check request headers only (response body
        # may echo headers back, which is expected from httpbin)
        cassette_data = yaml.safe_load(cassette_path.read_text())
        req_headers = cassette_data["interactions"][0]["request"]["headers"]
        header_keys = [k.lower() for k in req_headers]
        assert "authorization" not in header_keys

    def test_cassette_naming_convention(self) -> None:
        """Cassette names should be derivable from test names."""
        test_name = "test_search_pipeline"
        cassette_name = f"{test_name}.yaml"
        expected_path = CASSETTE_DIR / cassette_name
        # Just verify the convention produces a valid path
        assert expected_path.name == "test_search_pipeline.yaml"
        assert expected_path.parent == CASSETTE_DIR
