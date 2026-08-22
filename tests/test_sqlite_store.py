"""Tests for SQLite experiment history storage."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.agent.service import run_agent
from src.storage.sqlite_store import SQLiteExperimentStore
from src.tools.contracts import ToolResult


class SQLiteExperimentStoreTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        database_path = (
            Path(self.temporary_directory.name) / "history.sqlite3"
        )
        self.store = SQLiteExperimentStore(database_path)
        self.addCleanup(self.store.close)

    def _save_experiment_and_execution(self):
        self.store.save_experiment(
            "experiment-1",
            {"model": "demo-model"},
            "2026-08-22T10:00:00+00:00",
        )
        self.store.save_execution(
            "execution-1",
            "experiment-1",
            "completed",
            "2026-08-22T10:00:00+00:00",
            "2026-08-22T10:01:00+00:00",
            '{"raw": "output"}',
        )

    def test_initializes_sqlite_database(self):
        rows = self.store._connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
            """
        ).fetchall()

        self.assertEqual(
            {row["name"] for row in rows},
            {
                "experiments",
                "executions",
                "tool_results",
                "diagnosis_results",
            },
        )

    def test_saves_experiment(self):
        self.store.save_experiment(
            "experiment-1",
            {"dataset": "demo-dataset"},
            "2026-08-22T10:00:00+00:00",
        )

        row = self.store._connection.execute(
            "SELECT * FROM experiments WHERE experiment_id = ?",
            ("experiment-1",),
        ).fetchone()
        self.assertIsNotNone(row)
        self.assertIn("demo-dataset", row["input_context"])

    def test_saves_execution(self):
        self._save_experiment_and_execution()

        execution = self.store.get_execution("execution-1")

        self.assertEqual(execution["status"], "completed")
        self.assertEqual(execution["experiment_id"], "experiment-1")
        self.assertEqual(
            execution["raw_model_output"],
            '{"raw": "output"}',
        )

    def test_saves_tool_result(self):
        self._save_experiment_and_execution()
        tool_result = ToolResult.success(
            "metric_analysis",
            features={"accuracy_macro_f1_gap": 0.1},
            provenance={"module": "existing.calculator"},
        )

        self.store.save_tool_result("execution-1", tool_result)
        execution = self.store.get_execution("execution-1")

        stored = execution["tool_results"]["metric_analysis"]
        self.assertEqual(stored, tool_result.to_dict())

    def test_saves_diagnosis_and_report(self):
        self._save_experiment_and_execution()
        diagnosis = {
            "task_type": "experiment_diagnosis",
            "primary_issue": "class_imbalance",
        }

        self.store.save_diagnosis_result(
            "execution-1",
            diagnosis,
            "Diagnosis report",
        )
        execution = self.store.get_execution("execution-1")

        self.assertEqual(execution["diagnosis"], diagnosis)
        self.assertEqual(execution["report"], "Diagnosis report")

    def test_queries_experiment_history(self):
        self._save_experiment_and_execution()
        self.store.save_execution(
            "execution-2",
            "experiment-1",
            "failed",
            "2026-08-22T11:00:00+00:00",
            "2026-08-22T11:01:00+00:00",
        )

        history = self.store.get_experiment_history("experiment-1")

        self.assertEqual(
            [item["execution_id"] for item in history],
            ["execution-1", "execution-2"],
        )

    def test_agent_persists_complete_result(self):
        result = run_agent(
            "agent-storage-test",
            {"model_name": "demo-model"},
            store=self.store,
        )

        execution = self.store.get_execution(result["execution_id"])

        self.assertEqual(result["workflow_status"], "completed")
        self.assertIsNone(result["persistence_error"])
        self.assertEqual(execution["status"], "completed")
        self.assertEqual(execution["diagnosis"], result["diagnosis"])
        self.assertEqual(execution["report"], result["report"])
        self.assertTrue(execution["tool_results"])

    def test_storage_failure_does_not_change_diagnosis(self):
        self.store.close()

        result = run_agent(
            "storage-failure-test",
            {"model_name": "demo-model"},
            store=self.store,
        )

        self.assertEqual(result["workflow_status"], "completed")
        self.assertIsNotNone(result["diagnosis"])
        self.assertIsNotNone(result["report"])
        self.assertIsNone(result["error"])
        self.assertIn("persistence failed", result["persistence_error"])


if __name__ == "__main__":
    unittest.main()
