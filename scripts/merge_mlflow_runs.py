"""Merge or synchronize metrics from two MLflow runs into a single continuous run.

Usage:
    python scripts/merge_mlflow_runs.py --source-run-id <RUN_2> --target-run-id <RUN_1>
Or auto-merge the two most recent runs of the same experiment:
    python scripts/merge_mlflow_runs.py
"""

import argparse
import logging
from pathlib import Path
import sqlite3
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "mlflow.db"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("merge_mlflow_runs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge two MLflow runs in SQLite database")
    parser.add_argument("--db-path", type=str, default=str(DB_PATH), help="Path to mlflow.db SQLite database")
    parser.add_argument("--source-run-id", type=str, default=None, help="Source Run ID (e.g. Run 2 resumed)")
    parser.add_argument("--target-run-id", type=str, default=None, help="Target Run ID (e.g. Run 1 original)")
    return parser.parse_args()


def auto_detect_runs(conn: sqlite3.Connection) -> tuple[str, str]:
    """Auto-detect the two most recent runs for vehicle-damage-detection."""
    c = conn.cursor()
    runs = c.execute(
        """
        SELECT run_uuid, name, start_time 
        FROM runs 
        WHERE name LIKE '%faster_rcnn%'
        ORDER BY start_time DESC 
        LIMIT 2
        """
    ).fetchall()

    if len(runs) < 2:
        raise ValueError("Could not find at least two Faster R-CNN runs to merge.")

    # runs[0] is the more recent run (Run 2 / resumed)
    # runs[1] is the older run (Run 1 / original)
    source_run_id = runs[0][0]
    target_run_id = runs[1][0]
    logger.info("Auto-detected source run (resumed): %s (%s)", source_run_id, runs[0][1])
    logger.info("Auto-detected target run (original): %s (%s)", target_run_id, runs[1][1])
    return source_run_id, target_run_id


def merge_runs(db_path: Path, source_run_id: str, target_run_id: str) -> int:
    """Copy all metric records from source_run_id into target_run_id and update latest_metrics."""
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    conn = sqlite3.connect(str(db_path))
    c = conn.cursor()

    try:
        # 1. Fetch metrics from source run
        source_metrics = c.execute(
            """
            SELECT key, value, timestamp, step, is_nan 
            FROM metrics 
            WHERE run_uuid = ?
            """,
            (source_run_id,),
        ).fetchall()

        if not source_metrics:
            logger.warning("No metrics found in source run: %s", source_run_id)
            return 0

        logger.info("Found %d metric entries in source run %s", len(source_metrics), source_run_id)

        # 2. Insert into target run (ignore if duplicate step already exists)
        inserted = 0
        for key, value, ts, step, is_nan in source_metrics:
            existing = c.execute(
                """
                SELECT COUNT(*) FROM metrics 
                WHERE run_uuid = ? AND key = ? AND step = ?
                """,
                (target_run_id, key, step),
            ).fetchone()[0]

            if existing == 0:
                c.execute(
                    """
                    INSERT INTO metrics (key, value, timestamp, run_uuid, step, is_nan)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (key, value, ts, target_run_id, step, is_nan),
                )
                inserted += 1

        logger.info("Inserted %d new metric entries into target run %s", inserted, target_run_id)

        # 3. Update latest_metrics for target run
        # Find highest step for each key in target run
        keys = [r[0] for r in c.execute("SELECT DISTINCT key FROM metrics WHERE run_uuid = ?", (target_run_id,)).fetchall()]
        for k in keys:
            latest_row = c.execute(
                """
                SELECT key, value, timestamp, step, is_nan, run_uuid 
                FROM metrics 
                WHERE run_uuid = ? AND key = ? 
                ORDER BY step DESC 
                LIMIT 1
                """,
                (target_run_id, k),
            ).fetchone()

            if latest_row:
                c.execute(
                    """
                    INSERT OR REPLACE INTO latest_metrics (key, value, timestamp, step, is_nan, run_uuid)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    latest_row,
                )

        conn.commit()
        logger.info("Successfully merged metrics! Target run %s now has a continuous chart.", target_run_id)
        return inserted
    finally:
        conn.close()


def main() -> None:
    args = parse_args()
    db_path = Path(args.db_path)

    conn = sqlite3.connect(str(db_path))
    try:
        source_id = args.source_run_id
        target_id = args.target_run_id
        if not source_id or not target_id:
            detected_source, detected_target = auto_detect_runs(conn)
            source_id = source_id or detected_source
            target_id = target_id or detected_target
    finally:
        conn.close()

    merge_runs(db_path, source_id, target_id)


if __name__ == "__main__":
    main()
