"""Tests for the public JSON API routers."""
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

import router.api.run_history as run_history_module
import router.api.run_metrics as run_metrics_module
from router.api import router as api_router


class TestApiRoutes(unittest.TestCase):
    """Smoke tests for /api endpoints."""

    def setUp(self) -> None:
        self.app = FastAPI()
        self.app.include_router(api_router, prefix="/api")
        self.client = TestClient(self.app)

    def test_list_run_history(self) -> None:
        """Should return run history rows as JSON."""
        run_history_module.run_history_service = MagicMock()
        run_history_module.run_history_service.run_history_table_rows.return_value = [
            SimpleNamespace(
                id=1,
                run_id=101,
                service_name="wikipedia",
                status="Run Started",
                metadata={"message": "started"},
                timestamp=datetime(2026, 8, 10, 12, 0, tzinfo=timezone.utc),
            )
        ]

        response = self.client.get("/api/run-history")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()[0]["id"], 1)
        self.assertEqual(response.json()[0]["service_name"], "wikipedia")

    def test_list_run_metrics(self) -> None:
        """Should return run metrics rows as JSON."""
        run_metrics_module.run_metrics_service = MagicMock()
        run_metrics_module.run_metrics_service.run_metrics_table_rows.return_value = [
            SimpleNamespace(
                id=1,
                run_history_id=1,
                metadata={"progress": 50},
                timestamp=datetime(2026, 8, 10, 12, 5, tzinfo=timezone.utc),
            ),
            SimpleNamespace(
                id=2,
                run_history_id=1,
                metadata={"progress": 100},
                timestamp=datetime(2026, 8, 10, 12, 10, tzinfo=timezone.utc),
            )
        ]

        response = self.client.get("/api/run-metrics")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()), 2)
        self.assertEqual(response.json()[0]["id"], 1)
        self.assertEqual(response.json()[0]["metadata"]["progress"], 50)
        self.assertEqual(response.json()[1]["id"], 2)
        self.assertEqual(response.json()[1]["metadata"]["progress"], 100)
