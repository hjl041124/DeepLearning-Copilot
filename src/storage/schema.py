"""SQLite schema for experiment execution history."""


SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS experiments (
        experiment_id TEXT PRIMARY KEY,
        input_context TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS executions (
        execution_id TEXT PRIMARY KEY,
        experiment_id TEXT NOT NULL,
        status TEXT NOT NULL,
        started_at TEXT NOT NULL,
        completed_at TEXT,
        raw_model_output TEXT,
        FOREIGN KEY (experiment_id)
            REFERENCES experiments (experiment_id)
            ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS tool_results (
        execution_id TEXT NOT NULL,
        tool_name TEXT NOT NULL,
        tool_result TEXT NOT NULL,
        PRIMARY KEY (execution_id, tool_name),
        FOREIGN KEY (execution_id)
            REFERENCES executions (execution_id)
            ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS diagnosis_results (
        execution_id TEXT PRIMARY KEY,
        diagnosis TEXT NOT NULL,
        report TEXT NOT NULL,
        FOREIGN KEY (execution_id)
            REFERENCES executions (execution_id)
            ON DELETE CASCADE
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_executions_experiment_started
    ON executions (experiment_id, started_at)
    """,
)
