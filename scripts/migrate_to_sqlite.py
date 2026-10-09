"""Migrate all MLflow FileStore runs (./mlruns) losslessly to SQLite (sqlite:///mlflow.db)."""

import logging
import math
import os
from pathlib import Path
import sqlite3
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Cho phép đọc FileStore cũ để di chuyển dữ liệu
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

import mlflow

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("migrate_to_sqlite")


def migrate_filestore_to_sqlite(
    source_dir: Path | str = PROJECT_ROOT / "mlruns",
    db_path: Path | str = PROJECT_ROOT / "mlflow.db",
) -> None:
    source_path = Path(source_dir)
    db_file = Path(db_path)

    logger.info("=" * 65)
    logger.info("   MIGRATING MLFLOW FILESTORE TO SQLITE (mlflow.db)")
    logger.info("=" * 65)
    logger.info("Source FileStore: %s", source_path)
    logger.info("Target SQLite   : %s", db_file)

    # 1. Khởi tạo database SQLite thông qua MlflowClient để đảm bảo tất cả bảng và schema được tạo
    sqlite_uri = f"sqlite:///{db_file.as_posix()}"
    target_client = mlflow.tracking.MlflowClient(sqlite_uri)

    source_uri = "./mlruns"
    source_client = mlflow.tracking.MlflowClient(source_uri)

    # Kết nối SQLite trực tiếp để ghi dữ liệu nguyên bản (bảo toàn chính xác run_id và timestamp)
    conn = sqlite3.connect(str(db_file))
    cur = conn.cursor()

    experiments = source_client.search_experiments(view_type=mlflow.entities.ViewType.ALL)
    logger.info("Found %d experiment(s) in source FileStore.", len(experiments))

    total_runs_migrated = 0
    total_metrics_migrated = 0
    total_params_migrated = 0

    for exp in experiments:
        logger.info("Processing Experiment: %s (ID: %s)", exp.name, exp.experiment_id)

        # 1. Thêm hoặc cập nhật experiment trong SQLite
        cur.execute("SELECT experiment_id FROM experiments WHERE name = ?", (exp.name,))
        row = cur.fetchone()
        if row:
            exp_id = row[0]
            logger.info("  Experiment '%s' already exists in SQLite (ID: %s)", exp.name, exp_id)
        else:
            now_ms = int(time.time() * 1000)
            # Dùng exp.experiment_id nếu là số nguyên, hoặc tự tăng
            exp_id = int(exp.experiment_id) if exp.experiment_id.isdigit() else (
                (cur.execute("SELECT MAX(CAST(experiment_id AS INTEGER)) FROM experiments").fetchone()[0] or 0) + 1
            )
            cur.execute(
                """
                INSERT INTO experiments (experiment_id, name, artifact_location, lifecycle_stage, creation_time, last_update_time)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    exp_id,
                    exp.name,
                    exp.artifact_location,
                    exp.lifecycle_stage or "active",
                    exp.creation_time or now_ms,
                    exp.last_update_time or now_ms,
                ),
            )
            conn.commit()
            logger.info("  Created Experiment '%s' in SQLite (ID: %s)", exp.name, exp_id)

        # 2. Thêm tags của experiment
        if exp.tags:
            for tag_k, tag_v in exp.tags.items():
                cur.execute(
                    "INSERT OR REPLACE INTO experiment_tags (experiment_id, key, value) VALUES (?, ?, ?)",
                    (exp_id, tag_k, str(tag_v)),
                )
            conn.commit()

        # 3. Quét tất cả runs của experiment
        runs = source_client.search_runs([exp.experiment_id], run_view_type=mlflow.entities.ViewType.ALL)
        logger.info("  Found %d run(s) for experiment '%s'", len(runs), exp.name)

        for run in runs:
            run_id = run.info.run_id
            logger.info("    Migrating run: %s (ID: %s)", run.info.run_name or "unnamed", run_id)

            # Insert or replace run info
            cur.execute(
                """
                INSERT OR REPLACE INTO runs (
                    run_uuid, name, source_type, source_name, entry_point_name,
                    user_id, status, start_time, end_time, source_version,
                    lifecycle_stage, artifact_uri, experiment_id, deleted_time
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    run.info.run_name or "",
                    "LOCAL",
                    "",
                    "",
                    run.info.user_id or "",
                    run.info.status,
                    run.info.start_time,
                    run.info.end_time,
                    "",
                    run.info.lifecycle_stage or "active",
                    run.info.artifact_uri,
                    exp_id,
                    None,
                ),
            )

            # Insert Params
            for param_k, param_v in run.data.params.items():
                cur.execute(
                    "INSERT OR REPLACE INTO params (key, value, run_uuid) VALUES (?, ?, ?)",
                    (param_k, str(param_v), run_id),
                )
                total_params_migrated += 1

            # Insert Tags
            for tag_k, tag_v in run.data.tags.items():
                cur.execute(
                    "INSERT OR REPLACE INTO tags (key, value, run_uuid) VALUES (?, ?, ?)",
                    (tag_k, str(tag_v), run_id),
                )

            # Insert Metric histories (bảo toàn toàn bộ điểm epoch steps)
            for metric_k in run.data.metrics.keys():
                history = source_client.get_metric_history(run_id, metric_k)
                for m in history:
                    is_nan = 1 if math.isnan(m.value) else 0
                    cur.execute(
                        "INSERT INTO metrics (key, value, timestamp, run_uuid, step, is_nan) VALUES (?, ?, ?, ?, ?, ?)",
                        (metric_k, m.value, m.timestamp, run_id, m.step, is_nan),
                    )
                    total_metrics_migrated += 1

                # Latest metrics
                latest_val = run.data.metrics[metric_k]
                latest_is_nan = 1 if math.isnan(latest_val) else 0
                cur.execute(
                    "INSERT OR REPLACE INTO latest_metrics (key, value, timestamp, step, is_nan, run_uuid) VALUES (?, ?, ?, ?, ?, ?)",
                    (metric_k, latest_val, history[-1].timestamp if history else int(time.time()*1000), history[-1].step if history else 0, latest_is_nan, run_id),
                )

            conn.commit()
            total_runs_migrated += 1

    conn.close()

    logger.info("=" * 65)
    logger.info("MIGRATION COMPLETED SUCCESSFULLY!")
    logger.info("Total Runs Migrated   : %d", total_runs_migrated)
    logger.info("Total Params Migrated : %d", total_params_migrated)
    logger.info("Total Metrics Migrated: %d", total_metrics_migrated)
    logger.info("SQLite Database File  : %s", db_file.resolve())
    logger.info("=" * 65)


def verify_sqlite_database(db_path: Path | str = PROJECT_ROOT / "mlflow.db") -> None:
    """Verify that MLflow can read all migrated data directly via SQLite."""
    db_file = Path(db_path)
    sqlite_uri = f"sqlite:///{db_file.as_posix()}"
    client = mlflow.tracking.MlflowClient(sqlite_uri)

    exps = client.search_experiments()
    logger.info("Verifying SQLite backend: Found %d experiment(s):", len(exps))
    for e in exps:
        runs = client.search_runs([e.experiment_id])
        logger.info(" - Experiment '%s' (ID: %s): %d run(s)", e.name, e.experiment_id, len(runs))
        for r in runs:
            logger.info("    Run '%s' (ID: %s, Status: %s, Metrics: %d, Params: %d)",
                        r.info.run_name, r.info.run_id, r.info.status, len(r.data.metrics), len(r.data.params))


if __name__ == "__main__":
    migrate_filestore_to_sqlite()
    verify_sqlite_database()
