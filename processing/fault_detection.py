#!/usr/bin/env python3
"""
ThermoGrid — DQ Validated to Fault Detection
=================================================
Stage 7 of the project roadmap: Exception & Fault Detection Development.

Reads dq_validated/hvac_telemetry and computes REAL, derived anomaly
detection logic - not just passing through the generator's fault_code
field. Three distinct signals get added:

    overheating_flag   -> supply/heating coil temps operationally too
                            hot, even though individually still inside
                            the "physically plausible" range DQ checks
                            for. This is the key distinction: DQ's
                            critical checks catch sensor CORRUPTION
                            (900F - impossible). Fault detection here
                            catches values that are real, physically
                            possible readings, but still indicate a
                            genuine equipment problem (85F supply air
                            is achievable, and also means something is
                            wrong).
    coolant_leak_flag  -> the cooling coil isn't doing its job: fan is
                            actively running (implying a cooling call)
                            but the coil temperature is too high for
                            that to be working properly - a proxy for
                            lost refrigerant/coolant.
    operational_fault_flag -> re-surfaces fault_code/status_flag here
                            explicitly, so this table is a complete
                            one-stop view of equipment health (both the
                            generator-reported fault AND the derived
                            anomalies), without redefining what that
                            field already means.

Honest limitation: this is single-reading, threshold-based detection.
A real fault-detection system would also look at DURATION (has this
been true for the last N minutes, not just this one reading) and
trends across time. That's a natural enhancement, not built here -
noted rather than glossed over.

Utility meter data has no physical equipment to overheat or leak
coolant, so this stage only applies to hvac_telemetry.

No Spark, no Databricks - runs entirely on a GitHub Actions runner.

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

import boto3
import pandas as pd
from botocore.client import Config
from deltalake import DeltaTable, write_deltalake
from deltalake.exceptions import DeltaError, TableNotFoundError


# ------------------------------------------------------------------
# Storage helpers
# ------------------------------------------------------------------
def get_s3_client():
    namespace = os.environ["HF_NAMESPACE"]
    return boto3.client(
        "s3",
        endpoint_url=f"https://s3.hf.co/{namespace}",
        aws_access_key_id=os.environ["HF_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["HF_SECRET_ACCESS_KEY"],
        region_name="us-east-1",
        config=Config(s3={"addressing_style": "path"}, signature_version="s3v4"),
    )


def deltalake_storage_options():
    return {
        "endpoint_url": f"https://s3.hf.co/{os.environ['HF_NAMESPACE']}",
        "AWS_ACCESS_KEY_ID": os.environ["HF_ACCESS_KEY_ID"],
        "AWS_SECRET_ACCESS_KEY": os.environ["HF_SECRET_ACCESS_KEY"],
        "AWS_REGION": "us-east-1",
        "AWS_S3_ALLOW_UNSAFE_RENAME": "true",
    }


def delete_prefix_objects(s3, bucket: str, prefix: str) -> int:
    keys = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    deleted = 0
    for i in range(0, len(keys), 1000):
        batch = keys[i:i + 1000]
        resp = s3.delete_objects(
            Bucket=bucket,
            Delete={"Objects": [{"Key": k} for k in batch], "Quiet": True},
        )
        deleted += len(batch) - len(resp.get("Errors", []))
    return deleted


def write_fault_table(s3, bucket: str, df: pd.DataFrame, storage_options: dict,
                       max_attempts: int = 4) -> None:
    table_uri = f"s3://{bucket}/fault_detection/hvac_telemetry"
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            write_deltalake(
                table_uri, df, mode="overwrite",
                partition_by=["building_id", "dt"], storage_options=storage_options,
            )
            return
        except DeltaError as e:
            last_error = e
            if attempt < max_attempts:
                wait = 2 ** attempt
                print(f"  write attempt {attempt}/{max_attempts} failed "
                      f"({e}), retrying in {wait}s...")
                time.sleep(wait)

    print(f"  all {max_attempts} write attempts failed ({last_error}). "
          f"Self-healing: wiping fault_detection/hvac_telemetry and recreating fresh...")
    deleted = delete_prefix_objects(s3, bucket, "fault_detection/hvac_telemetry/")
    print(f"  deleted {deleted} object(s), retrying write once more...")
    write_deltalake(
        table_uri, df, mode="overwrite",
        partition_by=["building_id", "dt"], storage_options=storage_options,
    )
    print(f"  self-heal succeeded - fault_detection/hvac_telemetry recreated cleanly.")


# ------------------------------------------------------------------
# Anomaly detection logic
# ------------------------------------------------------------------
# Thresholds are deliberately NARROWER than DQ's critical-check bounds.
# DQ asks "is this reading even physically possible" (very wide bounds,
# e.g. supply_air_temp between 30-100). Fault detection asks "is this
# reading, while entirely plausible, indicative of an equipment
# problem" - a much narrower, operationally-meaningful band.
OVERHEATING_SUPPLY_AIR_TEMP_THRESHOLD = 78.0    # normal range tops out ~66
OVERHEATING_HEATING_COIL_TEMP_THRESHOLD = 130.0  # normal range tops out ~118

COOLANT_LEAK_FAN_ACTIVE_THRESHOLD = 30.0         # fan running -> cooling call likely active
COOLANT_LEAK_COIL_TEMP_THRESHOLD = 55.0          # a healthy active cooling coil should run cooler


def detect_overheating(df: pd.DataFrame) -> pd.Series:
    return (
        (df["supply_air_temp"] > OVERHEATING_SUPPLY_AIR_TEMP_THRESHOLD) |
        (df["heating_coil_temp"] > OVERHEATING_HEATING_COIL_TEMP_THRESHOLD)
    )


def detect_coolant_leak(df: pd.DataFrame) -> pd.Series:
    fan_running = df["fan_speed_pct"] > COOLANT_LEAK_FAN_ACTIVE_THRESHOLD
    coil_too_warm = df["cooling_coil_temp"] > COOLANT_LEAK_COIL_TEMP_THRESHOLD
    return fan_running & coil_too_warm


def run_fault_detection(s3, bucket: str, storage_options: dict) -> dict:
    print("\n=== hvac_telemetry (fault detection) ===")
    dq_uri = f"s3://{bucket}/dq_validated/hvac_telemetry"

    try:
        dq_table = DeltaTable(dq_uri, storage_options=storage_options)
    except TableNotFoundError:
        print(f"  no dq_validated table found at {dq_uri}, skipping")
        return {"rows": 0, "overheating": 0, "coolant_leak": 0, "operational_fault": 0}

    df = dq_table.to_pandas()
    total_rows = len(df)
    print(f"  read {total_rows} rows from dq_validated (version {dq_table.version()})")

    df["overheating_flag"] = detect_overheating(df)
    df["coolant_leak_flag"] = detect_coolant_leak(df)
    # re-surfacing the generator's own signal explicitly in this table,
    # not redefining it - status_flag == 'Fault' already means "the
    # equipment reported a fault"
    df["operational_fault_flag"] = df["status_flag"].str.lower() == "fault"

    overheating_count = int(df["overheating_flag"].sum())
    coolant_leak_count = int(df["coolant_leak_flag"].sum())
    operational_fault_count = int(df["operational_fault_flag"].sum())

    print(f"  overheating: {overheating_count}, coolant_leak: {coolant_leak_count}, "
          f"operational_fault (from generator): {operational_fault_count}")

    print(f"  writing {total_rows} rows to fault_detection/hvac_telemetry...")
    write_fault_table(s3, bucket, df, storage_options)

    return {"rows": total_rows, "overheating": overheating_count,
            "coolant_leak": coolant_leak_count, "operational_fault": operational_fault_count}


def main():
    bucket = os.environ["HF_BUCKET_NAME"]
    storage_options = deltalake_storage_options()
    s3 = get_s3_client()

    result = run_fault_detection(s3, bucket, storage_options)

    print("\n=== Summary ===")
    print(f"hvac_telemetry: {result['rows']} rows, {result['overheating']} overheating, "
          f"{result['coolant_leak']} coolant leak signals, {result['operational_fault']} operational faults")

    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            f.write(f"total_rows={result['rows']}\n")
            f.write(f"overheating_count={result['overheating']}\n")
            f.write(f"coolant_leak_count={result['coolant_leak']}\n")
            f.write(f"operational_fault_count={result['operational_fault']}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyError as e:
        print(f"::error::Missing required environment variable: {e}", file=sys.stderr)
        sys.exit(1)