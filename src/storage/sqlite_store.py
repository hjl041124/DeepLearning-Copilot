"""SQLite-backed experiment history storage."""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3
from typing import Any

from src.storage.schema import SCHEMA_STATEMENTS
from src.tools.contracts import ToolResult


class SQLiteExperimentStore:
    """Record Agent runs without participating in diagnosis decisions."""

    def __init__(self, database_path: str | Path):
        self.database_path = str(database_path)

        if self.database_path != ":memory:":
            Path(self.database_path).parent.mkdir(
                parents=True,
                exist_ok=True,
            )

        self._connection = sqlite3.connect(self.database_path)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self.initialize_database()

    def initialize_database(self) -> None:
        """Create the history tables when they do not yet exist."""

        with self._connection:
            for statement in SCHEMA_STATEMENTS:
                self._connection.execute(statement)

    def close(self) -> None:
        """Close the underlying SQLite connection."""

        self._connection.close()

    def __enter__(self) -> "SQLiteExperimentStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @staticmethod
    def _to_json(value: Any) -> str:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
        )

    def save_experiment(
        self,
        experiment_id: str,
        input_context: dict[str, Any],
        created_at: str,
    ) -> None:
        """Save an experiment, retaining its original creation time."""

        with self._connection:
            self._connection.execute(
                """
                INSERT INTO experiments (
                    experiment_id,
                    input_context,
                    created_at
                ) VALUES (?, ?, ?)
                ON CONFLICT(experiment_id) DO UPDATE SET
                    input_context = excluded.input_context
                """,
                (
                    experiment_id,
                    self._to_json(input_context),
                    created_at,
                ),
            )

    def save_execution(
        self,
        execution_id: str,
        experiment_id: str,
        status: str,
        started_at: str,
        completed_at: str | None = None,
        raw_model_output: str | None = None,
    ) -> None:
        """Save or update one execution of an experiment."""

        with self._connection:
            self._connection.execute(
                """
                INSERT INTO executions (
                    execution_id,
                    experiment_id,
                    status,
                    started_at,
                    completed_at,
                    raw_model_output
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(execution_id) DO UPDATE SET
                    status = excluded.status,
                    completed_at = excluded.completed_at,
                    raw_model_output = excluded.raw_model_output
                """,
                (
                    execution_id,
                    experiment_id,
                    status,
                    started_at,
                    completed_at,
                    raw_model_output,
                ),
            )

    def save_tool_result(
        self,
        execution_id: str,
        tool_result: ToolResult,
    ) -> None:
        """Save one ToolResult as JSON for an execution."""

        result = tool_result.to_dict()
        with self._connection:
            self._connection.execute(
                """
                INSERT INTO tool_results (
                    execution_id,
                    tool_name,
                    tool_result
                ) VALUES (?, ?, ?)
                ON CONFLICT(execution_id, tool_name) DO UPDATE SET
                    tool_result = excluded.tool_result
                """,
                (
                    execution_id,
                    tool_result.tool_name,
                    self._to_json(result),
                ),
            )

    def save_diagnosis_result(
        self,
        execution_id: str,
        diagnosis: dict[str, Any],
        report: str,
    ) -> None:
        """Save the validated diagnosis and formatted report."""

        with self._connection:
            self._connection.execute(
                """
                INSERT INTO diagnosis_results (
                    execution_id,
                    diagnosis,
                    report
                ) VALUES (?, ?, ?)
                ON CONFLICT(execution_id) DO UPDATE SET
                    diagnosis = excluded.diagnosis,
                    report = excluded.report
                """,
                (
                    execution_id,
                    self._to_json(diagnosis),
                    report,
                ),
            )

    def get_execution(
        self,
        execution_id: str,
    ) -> dict[str, Any] | None:
        """Return a complete execution record, or None when absent."""

        execution = self._connection.execute(
            """
            SELECT
                execution_id,
                experiment_id,
                status,
                started_at,
                completed_at,
                raw_model_output
            FROM executions
            WHERE execution_id = ?
            """,
            (execution_id,),
        ).fetchone()

        if execution is None:
            return None

        tool_rows = self._connection.execute(
            """
            SELECT tool_name, tool_result
            FROM tool_results
            WHERE execution_id = ?
            ORDER BY tool_name
            """,
            (execution_id,),
        ).fetchall()
        diagnosis_row = self._connection.execute(
            """
            SELECT diagnosis, report
            FROM diagnosis_results
            WHERE execution_id = ?
            """,
            (execution_id,),
        ).fetchone()

        result = dict(execution)
        result["tool_results"] = {
            row["tool_name"]: json.loads(row["tool_result"])
            for row in tool_rows
        }
        result["diagnosis"] = (
            json.loads(diagnosis_row["diagnosis"])
            if diagnosis_row is not None
            else None
        )
        result["report"] = (
            diagnosis_row["report"]
            if diagnosis_row is not None
            else None
        )
        return result

    def get_experiment_history(
        self,
        experiment_id: str,
    ) -> list[dict[str, Any]]:
        """Return all executions for an experiment, oldest first."""

        rows = self._connection.execute(
            """
            SELECT execution_id
            FROM executions
            WHERE experiment_id = ?
            ORDER BY started_at, execution_id
            """,
            (experiment_id,),
        ).fetchall()

        history = []
        for row in rows:
            execution = self.get_execution(row["execution_id"])
            if execution is not None:
                history.append(execution)
        return history
