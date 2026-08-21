#!/usr/bin/env python3
"""
ThermoGrid — Equipment-Level Fact Table
============================================
Stage 11 support. gold/hvac_daily_summary aggregates ALL equipment in
a building into one row per building/day - useful for a high-level
view, but useless for maintenance reporting, since you can't tell
WHICH AHU needs attention, only which building. This builds
gold/fact_equipment_daily: one row per equipment_id per day, so a
maintenance table in Power BI can actually point at a specific unit.

Reads fault_detection/hvac_telemetry (already has the DQ flag, fault
detection flags, and everything needed - no need to touch Silver's
enriched fields for this).

Same aggregation rule as the rest of Gold: numeric averages exclude
critical (corrupted-reading) rows.

Env vars required:
    HF_ACCESS_KEY_ID
    HF_SECRET_ACCESS_KEY
    HF_NAMESPACE
    HF_BUCKET_NAME
"""

import os
import sys
import time

import boto3
import pandas as pd
from botocore.client import Config
from deltalake import DeltaTable, write_deltalake
from deltalake.exceptions import DeltaError, TableNotFoundError


def deltalake_storage_options():
    return {
        "endpoint_url": f"https://s3.hf.co/{os.environ['HF_NAMESPACE']}",
        "AWS_ACCESS_KEY_ID": os.environ["HF_ACCESS_KEY_ID"],
        "AWS_SECRET_ACCESS_KEY": os.environ["HF_SECRET_ACCESS_KEY"],
        "AWS_REGION": "us-east-1",
        "AWS_S3_ALLOW_UNSAFE_RENAME": "true",
    }


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


def delete_prefix_objects(s3, bucket: str, prefix: str) -> int:
    keys = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    deleted = 0
    for i in range(0, len(keys), 1000):
        batch = keys[i:i + 1000]
        resp = s3.delete_objects(Bucket=bucket, Delete={"Objects": [{"Key": k} for k in batch], "Quiet": True})
        deleted += len(batch) - len(resp.get("Errors", []))
    return deleted


def write_fact_table(s3, bucket: str, df: pd.DataFrame, storage_options: dict, max_attempts: int = 4) -> None:
    table_uri = f"s3://{bucket}/gold/fact_equipment_daily"
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            write_deltalake(table_uri, df, mode="overwrite",
                             partition_by=["building_id", "dt"], storage_options=storage_options)
            return
        except DeltaError as e:
            last_error = e
            if attempt < max_attempts:
                wait = 2 ** attempt
                print(f"  write attempt {attempt}/{max_attempts} failed ({e}), retrying in {wait}s...")
                time.sleep(wait)

    print(f"  all {max_attempts} write attempts failed ({last_error}). "
          f"Self-healing: wiping gold/fact_equipment_daily and recreating fresh...")
    deleted = delete_prefix_objects(s3, bucket, "gold/fact_equipment_daily/")
    print(f"  deleted {deleted} object(s), retrying write once more...")
    write_deltalake(table_uri, df, mode="overwrite",
                     partition_by=["building_id", "dt"], storage_options=storage_options)
    print(f"  self-heal succeeded - gold/fact_equipment_daily recreated cleanly.")


def build_fact_equipment_daily(df: pd.DataFrame) -> pd.DataFrame:
    reliable = df[df["data_quality_flag"] != "critical"]

    counts = df.groupby(["building_id", "equipment_id", "dt"]).agg(
        total_readings=("equipment_id", "count"),
        clean_readings=("data_quality_flag", lambda s: (s == "clean").sum()),
        warning_readings=("data_quality_flag", lambda s: (s == "warning").sum()),
        critical_readings=("data_quality_flag", lambda s: (s == "critical").sum()),
        overheating_count=("overheating_flag", lambda s: (s == True).sum()),
        coolant_leak_count=("coolant_leak_flag", lambda s: (s == True).sum()),
        operational_fault_count=("operational_fault_flag", lambda s: (s == True).sum()),
    ).reset_index()

    reliable_agg = reliable.groupby(["building_id", "equipment_id", "dt"]).agg(
        avg_filter_pressure_drop=("filter_pressure_drop", "mean"),
        operational_readings=("status_flag", lambda s: (s.str.lower() == "operational").sum()),
    ).reset_index()

    summary = counts.merge(reliable_agg, on=["building_id", "equipment_id", "dt"], how="left")
    summary["uptime_pct"] = (summary["operational_readings"] / summary["total_readings"] * 100).round(1)
    summary["avg_filter_pressure_drop"] = summary["avg_filter_pressure_drop"].round(2)
    summary = summary.drop(columns=["operational_readings"])

    return summary


def main():
    bucket = os.environ["HF_BUCKET_NAME"]
    storage_options = deltalake_storage_options()
    s3 = get_s3_client()

    source_uri = f"s3://{bucket}/fault_detection/hvac_telemetry"
    try:
        table = DeltaTable(source_uri, storage_options=storage_options)
    except TableNotFoundError:
        print(f"no fault_detection table found at {source_uri}, skipping")
        return

    df = table.to_pandas()
    print(f"read {len(df)} rows from fault_detection (version {table.version()})")

    summary = build_fact_equipment_daily(df)
    print(f"writing {len(summary)} equipment/day row(s) to gold/fact_equipment_daily...")
    write_fact_table(s3, bucket, summary, storage_options)

    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            f.write(f"fact_equipment_daily_rows={len(summary)}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyError as e:
        print(f"::error::Missing required environment variable: {e}", file=sys.stderr)
        sys.exit(1)