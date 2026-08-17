#!/usr/bin/env python3
"""
ThermoGrid — Silver to Gold
================================
Stage 9 of the project roadmap: Gold Layer KPI & Reporting Tables.

Reads silver/hvac_telemetry and silver/utility_meter (row-level, one
row per reading) and produces THREE dashboard-ready outputs:

    gold/hvac_daily_summary     -> one row per building_id/dt: equipment
                                    count, DQ health mix, uptime %,
                                    overheating/coolant-leak/fault event
                                    counts, filter-replacement counts
    gold/utility_daily_summary  -> one row per building_id/dt/utility_type:
                                    total consumption, total estimated
                                    cost, avg variance, over-consuming count
    gold/alarm_log               -> DETAIL level, not aggregated: one
                                    row per individual alarm event
                                    (overheating / coolant_leak /
                                    operational_fault), for drill-down
                                    in a dashboard rather than just a
                                    summary count

Aggregation rule: numeric sums/averages EXCLUDE critical-flagged rows
(their underlying values may be corrupted/garbage - e.g. a negative
consumption reading would poison a building's total if included). DQ
health counts (clean/warning/critical) themselves obviously include
all three tiers, since the point is to report on data quality, not
hide it.

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


def write_gold_table(s3, bucket: str, table_name: str, df: pd.DataFrame,
                      partition_cols: list[str], storage_options: dict,
                      max_attempts: int = 4) -> None:
    table_uri = f"s3://{bucket}/gold/{table_name}"
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            write_deltalake(
                table_uri, df, mode="overwrite",
                partition_by=partition_cols, storage_options=storage_options,
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
          f"Self-healing: wiping gold/{table_name} and recreating fresh...")
    deleted = delete_prefix_objects(s3, bucket, f"gold/{table_name}/")
    print(f"  deleted {deleted} object(s), retrying write once more...")
    write_deltalake(
        table_uri, df, mode="overwrite",
        partition_by=partition_cols, storage_options=storage_options,
    )
    print(f"  self-heal succeeded - gold/{table_name} recreated cleanly.")


# ------------------------------------------------------------------
# HVAC daily summary
# ------------------------------------------------------------------
def build_hvac_daily_summary(df: pd.DataFrame) -> pd.DataFrame:
    reliable = df[df["data_quality_flag"] != "critical"]

    grouped = df.groupby(["building_id", "dt"])
    summary = grouped.agg(
        equipment_count=("equipment_id", "nunique"),
        total_readings=("equipment_id", "count"),
        clean_readings=("data_quality_flag", lambda s: (s == "clean").sum()),
        warning_readings=("data_quality_flag", lambda s: (s == "warning").sum()),
        critical_readings=("data_quality_flag", lambda s: (s == "critical").sum()),
        overheating_events=("overheating_flag", "sum"),
        coolant_leak_events=("coolant_leak_flag", "sum"),
        operational_fault_events=("operational_fault_flag", "sum"),
    ).reset_index()

    # metrics that must exclude critical (garbage-value) rows - computed
    # separately on the `reliable` subset, then merged back in
    reliable_grouped = reliable.groupby(["building_id", "dt"]).agg(
        avg_temp_differential=("temp_differential", "mean"),
        operational_readings=("status_flag", lambda s: (s.str.lower() == "operational").sum()),
        filter_replace_now_count=("filter_health_status", lambda s: (s == "Replace Now").sum()),
    ).reset_index()

    summary = summary.merge(reliable_grouped, on=["building_id", "dt"], how="left")
    summary["uptime_pct"] = (summary["operational_readings"] / summary["total_readings"] * 100).round(1)
    summary["avg_temp_differential"] = summary["avg_temp_differential"].round(1)
    summary = summary.drop(columns=["operational_readings"])

    return summary


# ------------------------------------------------------------------
# Utility daily summary
# ------------------------------------------------------------------
def build_utility_daily_summary(df: pd.DataFrame) -> pd.DataFrame:
    reliable = df[df["data_quality_flag"] != "critical"].copy()
    # is_over_consuming arrives as object dtype (True/False/<NA> mix from
    # Silver's .where()). Aggregating an object-dtype boolean column with
    # .sum() is unreliable across group sizes: a multi-row group correctly
    # sums to a real int, but a SINGLE-row group just returns that one
    # object unchanged - a raw Python bool, not a computed sum. Mixing
    # int and bool in the same output column then breaks the PyArrow
    # write ("Expected integer, got bool"). Casting to a clean boolean
    # dtype first makes .sum() behave consistently regardless of group
    # size. Safe here since `reliable` has already excluded critical
    # rows, which is the only source of nulls in this column.
    reliable["is_over_consuming"] = reliable["is_over_consuming"].astype(bool)

    counts = df.groupby(["building_id", "dt", "utility_type"]).agg(
        total_readings=("meter_id", "count"),
        clean_readings=("data_quality_flag", lambda s: (s == "clean").sum()),
        warning_readings=("data_quality_flag", lambda s: (s == "warning").sum()),
        critical_readings=("data_quality_flag", lambda s: (s == "critical").sum()),
    ).reset_index()

    reliable_agg = reliable.groupby(["building_id", "dt", "utility_type"]).agg(
        total_actual_consumption_kwh=("actual_consumption_kwh", "sum"),
        total_estimated_cost=("estimated_cost", "sum"),
        avg_variance_pct=("consumption_variance_pct", "mean"),
        over_consuming_count=("is_over_consuming", "sum"),
    ).reset_index()

    summary = counts.merge(reliable_agg, on=["building_id", "dt", "utility_type"], how="left")
    summary["total_actual_consumption_kwh"] = summary["total_actual_consumption_kwh"].round(1)
    summary["total_estimated_cost"] = summary["total_estimated_cost"].round(2)
    summary["avg_variance_pct"] = summary["avg_variance_pct"].round(1)

    return summary


# ------------------------------------------------------------------
# Alarm log (detail, not aggregated)
# ------------------------------------------------------------------
def build_alarm_log(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (reading, alarm_type) where that alarm fired. A
    single reading with two simultaneous alarms produces two rows -
    intentional, since a dashboard drilling into "coolant leak events"
    shouldn't have to know about unrelated overheating alarms on the
    same reading, and vice versa."""
    alarm_types = {
        "overheating": "overheating_flag",
        "coolant_leak": "coolant_leak_flag",
        "operational_fault": "operational_fault_flag",
    }

    base_cols = ["timestamp", "building_id", "equipment_id", "dt", "status_flag", "fault_code"]
    rows = []
    for alarm_type, flag_col in alarm_types.items():
        fired = df[df[flag_col] == True]
        if fired.empty:
            continue
        subset = fired[base_cols].copy()
        subset["alarm_type"] = alarm_type
        rows.append(subset)

    if not rows:
        return pd.DataFrame(columns=base_cols + ["alarm_type"])

    return pd.concat(rows, ignore_index=True)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def run_gold(s3, bucket: str, storage_options: dict) -> dict:
    results = {}

    print("\n=== hvac_telemetry -> gold/hvac_daily_summary + gold/alarm_log ===")
    hvac_uri = f"s3://{bucket}/silver/hvac_telemetry"
    try:
        hvac_table = DeltaTable(hvac_uri, storage_options=storage_options)
        hvac_df = hvac_table.to_pandas()
        print(f"  read {len(hvac_df)} rows from silver/hvac_telemetry (version {hvac_table.version()})")

        hvac_summary = build_hvac_daily_summary(hvac_df)
        print(f"  writing {len(hvac_summary)} building/day summary row(s)...")
        write_gold_table(s3, bucket, "hvac_daily_summary", hvac_summary,
                          ["building_id", "dt"], storage_options)

        alarm_log = build_alarm_log(hvac_df)
        print(f"  writing {len(alarm_log)} alarm event row(s)...")
        write_gold_table(s3, bucket, "alarm_log", alarm_log,
                          ["building_id", "dt"], storage_options)

        results["hvac_summary_rows"] = len(hvac_summary)
        results["alarm_log_rows"] = len(alarm_log)
    except TableNotFoundError:
        print(f"  no silver/hvac_telemetry table found, skipping")
        results["hvac_summary_rows"] = 0
        results["alarm_log_rows"] = 0

    print("\n=== utility_meter -> gold/utility_daily_summary ===")
    utility_uri = f"s3://{bucket}/silver/utility_meter"
    try:
        utility_table = DeltaTable(utility_uri, storage_options=storage_options)
        utility_df = utility_table.to_pandas()
        print(f"  read {len(utility_df)} rows from silver/utility_meter (version {utility_table.version()})")

        utility_summary = build_utility_daily_summary(utility_df)
        print(f"  writing {len(utility_summary)} building/day/utility_type summary row(s)...")
        write_gold_table(s3, bucket, "utility_daily_summary", utility_summary,
                          ["building_id", "dt"], storage_options)

        results["utility_summary_rows"] = len(utility_summary)
    except TableNotFoundError:
        print(f"  no silver/utility_meter table found, skipping")
        results["utility_summary_rows"] = 0

    return results


def main():
    bucket = os.environ["HF_BUCKET_NAME"]
    storage_options = deltalake_storage_options()
    s3 = get_s3_client()

    results = run_gold(s3, bucket, storage_options)

    print("\n=== Summary ===")
    for k, v in results.items():
        print(f"{k}: {v}")

    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            for k, v in results.items():
                f.write(f"{k}={v}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyError as e:
        print(f"::error::Missing required environment variable: {e}", file=sys.stderr)
        sys.exit(1)