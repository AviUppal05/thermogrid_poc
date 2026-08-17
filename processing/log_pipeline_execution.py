#!/usr/bin/env python3
"""
ThermoGrid — Master Pipeline Execution Logging
==================================================
Stage 10 of the project roadmap: part of Master Pipeline & Monitoring
Setup. Appends ONE ROW per full pipeline run to
pipeline_logs/execution_history, recording the outcome of each stage.

This is an ACCUMULATING AUDIT TRAIL, same as dq_logs/ - it must never
be wiped, or the whole point of a persisted execution history is
defeated. No self-heal-via-wipe here (see append_log_row's docstring
for the same reasoning applied to dq_validation.py).

Usage (called from master-pipeline.yml with each job's outcome):
    python log_pipeline_execution.py \\
        --run-id 12345 \\
        --triggered-by workflow_dispatch \\
        --bronze success --staging success --dq success \\
        --fault_detection success --silver success --gold success \\
        --duration-seconds 245

Env vars required:
    HF_ACCESS_KEY_ID
    HF_SECRET_ACCESS_KEY
    HF_NAMESPACE
    HF_BUCKET_NAME
"""

import argparse
import os
import sys
import time
from datetime import datetime, timezone

import boto3
import pandas as pd
from botocore.client import Config
from deltalake import DeltaTable, write_deltalake
from deltalake.exceptions import DeltaError, TableNotFoundError

STAGES = ["bronze", "staging", "dq", "fault_detection", "silver", "gold"]


def deltalake_storage_options():
    return {
        "endpoint_url": f"https://s3.hf.co/{os.environ['HF_NAMESPACE']}",
        "AWS_ACCESS_KEY_ID": os.environ["HF_ACCESS_KEY_ID"],
        "AWS_SECRET_ACCESS_KEY": os.environ["HF_SECRET_ACCESS_KEY"],
        "AWS_REGION": "us-east-1",
        "AWS_S3_ALLOW_UNSAFE_RENAME": "true",
    }


def append_execution_log(bucket: str, log_row: dict, storage_options: dict,
                          max_attempts: int = 3) -> None:
    table_uri = f"s3://{bucket}/pipeline_logs/execution_history"
    log_df = pd.DataFrame([log_row])

    try:
        DeltaTable(table_uri, storage_options=storage_options)
        table_exists = True
    except TableNotFoundError:
        table_exists = False

    mode = "append" if table_exists else "overwrite"
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            write_deltalake(table_uri, log_df, mode=mode, storage_options=storage_options)
            print(f"Logged pipeline run {log_row['run_id']} to pipeline_logs/execution_history")
            return
        except DeltaError as e:
            last_error = e
            if attempt < max_attempts:
                wait = 2 ** attempt
                print(f"  append attempt {attempt}/{max_attempts} failed ({e}), retrying in {wait}s...")
                time.sleep(wait)

    # Deliberately NOT self-healing (would wipe history) - just warn.
    print(f"warning: could not append to pipeline_logs/execution_history after "
          f"{max_attempts} attempts ({last_error}). This run's outcome won't be "
          f"in the log, but the actual pipeline stages already ran regardless.")


def main():
    parser = argparse.ArgumentParser(description="Log a full pipeline run's outcome")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--triggered-by", required=True)
    parser.add_argument("--duration-seconds", type=int, required=True)
    for stage in STAGES:
        parser.add_argument(f"--{stage}", required=True,
                             help="success / failure / skipped / cancelled")
    args = parser.parse_args()

    bucket = os.environ["HF_BUCKET_NAME"]
    storage_options = deltalake_storage_options()

    stage_statuses = {stage: getattr(args, stage) for stage in STAGES}
    overall_status = "success" if all(s == "success" for s in stage_statuses.values()) else "failure"

    log_row = {
        "run_id": args.run_id,
        "triggered_by": args.triggered_by,
        "logged_at": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": args.duration_seconds,
        "overall_status": overall_status,
        **{f"{stage}_status": status for stage, status in stage_statuses.items()},
    }

    print("Pipeline run summary:")
    for k, v in log_row.items():
        print(f"  {k}: {v}")

    append_execution_log(bucket, log_row, storage_options)


if __name__ == "__main__":
    try:
        main()
    except KeyError as e:
        print(f"::error::Missing required environment variable: {e}", file=sys.stderr)
        sys.exit(1)