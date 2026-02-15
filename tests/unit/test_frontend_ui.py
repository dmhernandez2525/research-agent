"""Unit tests for Phase 18/19 frontend scaffolding and route wiring."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FRONTEND = ROOT / "frontend"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_frontend_scaffold_files_exist() -> None:
    required = [
        FRONTEND / "package.json",
        FRONTEND / "vite.config.ts",
        FRONTEND / "tailwind.config.ts",
        FRONTEND / "src/main.tsx",
        FRONTEND / "src/components/layout/AppLayout.tsx",
        FRONTEND / "src/pages/SessionsPage.tsx",
        FRONTEND / "src/pages/SessionDetailPage.tsx",
        FRONTEND / "src/pages/AnalyticsPage.tsx",
        FRONTEND / "src/pages/HistoryPage.tsx",
    ]
    missing = [path for path in required if not path.exists()]
    assert missing == []


def test_package_json_has_expected_frontend_dependencies() -> None:
    payload = json.loads(_read(FRONTEND / "package.json"))
    deps = payload["dependencies"]
    assert "react" in deps
    assert "react-router-dom" in deps
    assert "react-markdown" in deps

    scripts = payload["scripts"]
    assert scripts["dev"] == "vite"
    assert "build" in scripts
    assert "generate:api" in scripts


def test_main_router_wires_all_pages() -> None:
    source = _read(FRONTEND / "src/main.tsx")
    assert "AppLayout" in source
    assert "SessionsPage" in source
    assert "SessionDetailPage" in source
    assert "AnalyticsPage" in source
    assert "HistoryPage" in source


def test_realtime_and_report_components_present() -> None:
    progress = _read(FRONTEND / "src/components/progress/ProgressDashboard.tsx")
    report = _read(FRONTEND / "src/components/reports/ReportViewer.tsx")

    assert "websocket" in progress.lower()
    assert "transport" in progress
    assert "react-markdown" in report.lower()
    assert "Table of Contents" in report


def test_advanced_ui_components_cover_phase_19_features() -> None:
    plan_editor = _read(FRONTEND / "src/components/plan/PlanEditor.tsx")
    source_browser = _read(FRONTEND / "src/components/sources/SourceBrowser.tsx")
    analytics = _read(FRONTEND / "src/components/analytics/CostAnalytics.tsx")
    history = _read(FRONTEND / "src/components/history/HistoryReplay.tsx")

    assert "localStorage" in plan_editor
    assert "draggable" in plan_editor
    assert "freshness" in source_browser
    assert "Near-duplicate" in source_browser
    assert "Export CSV" in analytics
    assert "Budget utilization" in analytics
    assert "Session Comparison" in history
    assert "Export JSON" in history
