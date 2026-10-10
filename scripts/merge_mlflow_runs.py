"""Merge or synchronize metrics from multiple fragmented MLflow runs into a single continuous run.

Supports:
- Merging 2 runs: python scripts/merge_mlflow_runs.py --source-run-id <RUN_2> --target-run-id <RUN_1>
- Merging ALL runs of a model (e.g. 1-15, 15-20, 20-25 -> 1 single run):
    python scripts/merge_mlflow_runs.py --auto
"""

import argparse
import logging
from pathlib import Path
import sqlite3
import sys
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "mlflow.db"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("merge_mlflow_runs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge multiple MLflow runs in SQLite database")
    parser.add_argument("--db-path", type=str, default=str(DB_PATH), help="Path to mlflow.db SQLite database")
    parser.add_argument("--source-run-id", type=str, default=None, help="Specific source Run ID")
    parser.add_argument("--target-run-id", type=str, default=None, help="Specific target Run ID")
    parser.add_argument("--run-name-pattern", type=str, default="faster_rcnn_resnet50_fpn_amp_accum_b2_ep40", help="Run name pattern to group and merge")
    parser.add_argument("--update-checkpoints", action="store_true", default=True, help="Update checkpoint files to reference root target run ID")
    return parser.parse_args()


def get_grouped_runs(conn: sqlite3.Connection, pattern: str) -> list[tuple[str, str, int]]:
    """Get all runs matching pattern ordered chronologically by start_time."""
    c = conn.cursor()
    runs = c.execute(
        """
        SELECT run_uuid, name, start_time 
        FROM runs 
        WHERE name LIKE ?
        ORDER BY start_time ASC
        """,
        (f"%{pattern}%",),
    ).fetchall()
    return runs


def merge_single_pair(c: sqlite3.Cursor, source_run_id: str, target_run_id: str) -> int:
    """Copy non-duplicate metric rows from source to target."""
    source_metrics = c.execute(
        """
        SELECT key, value, timestamp, step, is_nan 
        FROM metrics 
        WHERE run_uuid = ?
        ORDER BY step ASC
        """,
        (source_run_id,),
    ).fetchall()

    if not source_metrics:
        return 0

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

    return inserted


def update_latest_metrics(c: sqlite3.Cursor, target_run_id: str) -> None:
    """Refresh the latest_metrics table for target_run_id."""
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


def update_checkpoints_run_id(checkpoint_dir: Path, target_run_id: str) -> None:
    """Update mlflow_run_id field in existing .pth checkpoints to point to the root target run."""
    if not checkpoint_dir.exists():
        return

    for pth_file in checkpoint_dir.glob("*.pth"):
        try:
            chkpt = torch.load(pth_file, map_location="cpu", weights_only=False)
            current_id = chkpt.get("mlflow_run_id")
            if current_id != target_run_id:
                chkpt["mlflow_run_id"] = target_run_id
                torch.save(chkpt, pth_file)
                logger.info("Updated %s -> mlflow_run_id = %s", pth_file.name, target_run_id)
        except Exception as e:
            logger.warning("Could not patch checkpoint %s: %s", pth_file.name, e)


def main() -> None:
    args = parse_args()
    db_path = Path(args.db_path)

    if not db_path.exists():
        logger.error("Database not found: %s", db_path)
        sys.exit(1)

    conn = sqlite3.connect(str(db_path))
    c = conn.cursor()

    try:
        if args.source_run_id and args.target_run_id:
            logger.info("Merging explicit pair: %s -> %s", args.source_run_id, args.target_run_id)
            inserted = merge_single_pair(c, args.source_run_id, args.target_run_id)
            update_latest_metrics(c, args.target_run_id)
            conn.commit()
            logger.info("Inserted %d metric rows into target %s", inserted, args.target_run_id)
            target_id = args.target_run_id
        else:
            runs = get_grouped_runs(conn, args.run_name_pattern)
            if len(runs) < 2:
                logger.info("Found %d run(s) for pattern '%s'. No merge needed.", len(runs), args.run_name_pattern)
                return

            # Target is the earliest root run (Epochs starting from 1)
            target_run_id = runs[0][0]
            target_name = runs[0][1]
            logger.info("Root target run: %s (%s)", target_run_id, target_name)

            total_inserted = 0
            for source_run_id, source_name, start_time in runs[1:]:
                logger.info("Merging source run: %s (%s) into root %s", source_run_id, source_name, target_run_id)
                n = merge_single_pair(c, source_run_id, target_run_id)
                total_inserted += n
                logger.info("  -> Added %d new metric rows from %s", n, source_run_id)

            update_latest_metrics(c, target_run_id)
            conn.commit()
            logger.info("=" * 60)
            logger.info("MERGE COMPLETE! Total %d metrics merged into root run %s", total_inserted, target_run_id)
            logger.info("=" * 60)
            target_id = target_run_id

        # Update checkpoints so future resumes attach directly to target_id
        if args.update_checkpoints:
            chkpt_dir = PROJECT_ROOT / "artifacts" / "checkpoints" / args.run_name_pattern
            update_checkpoints_run_id(chkpt_dir, target_id)

    finally:
        conn.close()


if __name__ == "__main__":
    main()
