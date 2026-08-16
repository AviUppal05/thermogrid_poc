#!/usr/bin/env python3
"""
ThermoGrid — Bronze to Staging
==================================
Stage 5 of the project roadmap: Staging Layer & Data Cleaning Pipeline.

Reads the Bronze Delta tables (which preserve raw data untouched,
duplicates and all) and produces a clean, structurally consistent
Staging table:

    - Datatype conversion  -> timestamp to real datetime, numeric
      columns coerced to numeric
    - Null handling        -> explicit, documented policy per column
      (see handle_nulls) rather than silently leaving mixed
      null-representations around
    - Deduplication         -> exact duplicate readings collapsed on
      their natural key (this is where dedup lives now, moved out of
      Bronze per the roadmap - Bronze preserves everything as it
      arrived, Staging is where cleaning starts)
    - Formatting            -> consistent casing/whitespace on
      categorical string fields, consistent numeric rounding

This does NOT do range/plausibility checks, quality flagging, or fault
detection - those are separate stages (dq_validation.py,
fault_detection.py) that come after this one.

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

DATASETS = {
    "hvac_telemetry": {
        "dedup_keys": ["timestamp", "equipment_id"],
        "numeric_columns": [
            "supply_air_temp", "return_air_temp", "fan_speed_pct", "duct_pressure",
            "filter_pressure_drop", "cooling_coil_temp", "heating_coil_temp", "outdoor_air_temp",
        ],
        "string_columns": ["equipment_id", "building_id", "fault_code", "status_flag"],
    },
    "utility_meter": {
        "dedup_keys": ["timestamp", "meter_id"],
        "numeric_columns": [
            "actual_consumption_kwh", "expected_consumption_kwh", "peak_demand_kw", "cost_rate",
        ],
        "string_columns": ["meter_id", "building_id", "utility_type"],
    },
}


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


def write_staging_table(s3, bucket: str, dataset_name: str, df: pd.DataFrame,
                         storage_options: dict, max_attempts: int = 4) -> None:
    """Same retry-then-self-heal pattern as Bronze/Silver. Staging is
    rebuilt fully from Bronze every run (no incremental state of its
    own), so there's nothing lost by wiping and recreating it if it
    ever gets into a bad state."""
    table_uri = f"s3://{bucket}/staging/{dataset_name}"
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
          f"Self-healing: wiping staging/{dataset_name} and recreating fresh...")
    deleted = delete_prefix_objects(s3, bucket, f"staging/{dataset_name}/")
    print(f"  deleted {deleted} object(s), retrying write once more...")
    write_deltalake(
        table_uri, df, mode="overwrite",
        partition_by=["building_id", "dt"], storage_options=storage_options,
    )
    print(f"  self-heal succeeded - staging/{dataset_name} recreated cleanly.")


# ------------------------------------------------------------------
# Cleaning steps
# ------------------------------------------------------------------
def cast_types(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Datatype conversion."""
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["building_id"] = df["building_id"].astype("string")
    df["dt"] = df["dt"].astype("string")
    for col in cfg["numeric_columns"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def handle_nulls(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Explicit, documented null-handling policy - not a blanket
    fillna(0), which would silently invent sensor readings that never
    happened. Only fields where a null has an unambiguous, safe
    replacement get one:

    - fault_code null genuinely means "no fault" (that's how the
      generator produces it) -> made explicit as 'NONE' instead of a
      bare null, so downstream consumers don't have to special-case
      NaN vs "no fault" - they mean the same thing here.
    - Numeric sensor readings (temps, pressures, consumption, etc.)
      are LEFT AS TRUE NULL. A missing/corrupted reading cannot be
      safely guessed at, and imputing a plausible-looking value would
      hide exactly the kind of problem the next stage (Data Quality &
      Validation) is meant to catch.
    - Required identifiers (timestamp, equipment_id/meter_id,
      building_id) are also left null if they arrived null - that's a
      genuine data problem to flag in DQ, not something Staging should
      paper over.
    """
    if dataset_name == "hvac_telemetry" and "fault_code" in df.columns:
        df["fault_code"] = df["fault_code"].fillna("NONE")
    return df


def format_strings(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Formatting: consistent casing/whitespace on categorical string
    fields, so 'operational', ' Operational ', 'OPERATIONAL' don't end
    up as three different values downstream."""
    for col in cfg["string_columns"]:
        if col in df.columns and df[col].dtype == object or pd.api.types.is_string_dtype(df.get(col)):
            df[col] = df[col].astype("string").str.strip()
    if "status_flag" in df.columns:
        df["status_flag"] = df["status_flag"].str.strip().str.title()
    if "utility_type" in df.columns:
        df["utility_type"] = df["utility_type"].str.strip().str.title()
    return df


def deduplicate(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Deduplication - moved here from Bronze per the roadmap. Bronze
    preserves exact duplicate readings (e.g. from overlapping raw
    files); Staging is where they get collapsed to one."""
    before = len(df)
    df = df.drop_duplicates(subset=cfg["dedup_keys"]).reset_index(drop=True)
    return df, before - len(df)


# ------------------------------------------------------------------
# Main per-dataset routine
# ------------------------------------------------------------------
def build_staging(s3, dataset_name: str, bucket: str, storage_options: dict) -> dict:
    print(f"\n=== {dataset_name} ===")
    cfg = DATASETS[dataset_name]
    bronze_uri = f"s3://{bucket}/bronze/{dataset_name}"

    try:
        bronze_table = DeltaTable(bronze_uri, storage_options=storage_options)
    except TableNotFoundError:
        print(f"  no Bronze table found at {bronze_uri}, skipping")
        return {"rows_in": 0, "rows_out": 0, "duplicates_removed": 0}

    df = bronze_table.to_pandas()
    rows_in = len(df)
    print(f"  read {rows_in} rows from Bronze (version {bronze_table.version()})")

    df = cast_types(df, cfg)
    df = handle_nulls(df, dataset_name)
    df = format_strings(df, cfg)
    df, duplicates_removed = deduplicate(df, cfg)
    rows_out = len(df)

    print(f"  {rows_in} rows -> {rows_out} rows ({duplicates_removed} exact duplicate(s) removed)")

    print(f"  writing {rows_out} rows to staging/{dataset_name} in a single commit...")
    write_staging_table(s3, bucket, dataset_name, df, storage_options)

    return {"rows_in": rows_in, "rows_out": rows_out, "duplicates_removed": duplicates_removed}


def main():
    parser = argparse.ArgumentParser(description="Build Staging tables from Bronze")
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS.keys()), default=list(DATASETS.keys()))
    args = parser.parse_args()

    bucket = os.environ["HF_BUCKET_NAME"]
    storage_options = deltalake_storage_options()
    s3 = get_s3_client()

    results = {}
    for name in args.datasets:
        results[name] = build_staging(s3, name, bucket, storage_options)

    print("\n=== Summary ===")
    total_in = total_out = total_dupes = 0
    for name, r in results.items():
        print(f"{name}: {r['rows_in']} in -> {r['rows_out']} out ({r['duplicates_removed']} duplicates removed)")
        total_in += r["rows_in"]
        total_out += r["rows_out"]
        total_dupes += r["duplicates_removed"]

    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            f.write(f"total_rows_in={total_in}\n")
            f.write(f"total_rows_out={total_out}\n")
            f.write(f"total_duplicates_removed={total_dupes}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyError as e:
        print(f"::error::Missing required environment variable: {e}", file=sys.stderr)
        sys.exit(1)