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
import json
import os
import sys
import time
from datetime import datetime, timezone

import boto3
import pandas as pd
from botocore.client import Config
from botocore.exceptions import ClientError
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


# ------------------------------------------------------------------
# Watermark manifest - tracks which Bronze _pipeline_run_id values
# have already been incorporated into Staging, so each run only
# transforms and writes rows from NEW Bronze runs.
#
# Important limitation, stated plainly: deltalake's Python reader has
# no cheap way to read only "new" rows from a Delta table (that needs
# Change Data Feed, which isn't set up here) - so this still reads the
# ENTIRE Bronze table into memory every run. The win is real but
# partial: only new rows get run through cast/null-handling/dedup, and
# only new rows get WRITTEN (appended, not a full table rewrite) -
# which is where the bulk of the time was actually going for a large
# accumulated history.
# ------------------------------------------------------------------
def manifest_key(dataset_name: str) -> str:
    return f"staging/_manifests/{dataset_name}_processed_run_ids.json"


def get_manifest(s3, bucket: str, dataset_name: str) -> set[str]:
    try:
        obj = s3.get_object(Bucket=bucket, Key=manifest_key(dataset_name))
        data = json.loads(obj["Body"].read())
        return set(data.get("processed_run_ids", []))
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in ("NoSuchKey", "404"):
            return set()
        raise


def put_manifest(s3, bucket: str, dataset_name: str, processed_run_ids: set[str]) -> None:
    body = json.dumps({
        "processed_run_ids": sorted(processed_run_ids),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }).encode("utf-8")
    s3.put_object(Bucket=bucket, Key=manifest_key(dataset_name), Body=body,
                  ContentType="application/json")


def staging_table_exists(table_uri: str, storage_options: dict) -> bool:
    try:
        DeltaTable(table_uri, storage_options=storage_options)
        return True
    except TableNotFoundError:
        return False


def write_staging_table(s3, bucket: str, dataset_name: str, df: pd.DataFrame,
                         storage_options: dict, mode: str, max_attempts: int = 4) -> None:
    """mode is 'append' (normal incremental case) or 'overwrite' (first
    write, or a self-heal rebuild)."""
    table_uri = f"s3://{bucket}/staging/{dataset_name}"
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            write_deltalake(
                table_uri, df, mode=mode,
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
    staging_uri = f"s3://{bucket}/staging/{dataset_name}"

    try:
        bronze_table = DeltaTable(bronze_uri, storage_options=storage_options)
    except TableNotFoundError:
        print(f"  no Bronze table found at {bronze_uri}, skipping")
        return {"rows_in": 0, "rows_out": 0, "duplicates_removed": 0, "new_runs": 0}

    full_bronze_df = bronze_table.to_pandas()
    print(f"  read {len(full_bronze_df)} total rows from Bronze (version {bronze_table.version()})")

    processed_run_ids = get_manifest(s3, bucket, dataset_name)
    all_run_ids = set(full_bronze_df["_pipeline_run_id"].dropna().unique())
    new_run_ids = all_run_ids - processed_run_ids

    if not new_run_ids:
        print(f"  watermark: {len(processed_run_ids)} Bronze run(s) already processed, "
              f"0 new run(s) - nothing to do")
        return {"rows_in": 0, "rows_out": 0, "duplicates_removed": 0, "new_runs": 0}

    print(f"  watermark: {len(processed_run_ids)} Bronze run(s) already processed, "
          f"{len(new_run_ids)} new run(s) to incorporate")

    df = full_bronze_df[full_bronze_df["_pipeline_run_id"].isin(new_run_ids)].copy()
    rows_in = len(df)

    df = cast_types(df, cfg)
    df = handle_nulls(df, dataset_name)
    df = format_strings(df, cfg)
    # Dedup only within THIS new batch, not against the full Staging
    # history - acceptable here because Bronze is itself watermarked
    # per raw file, so the same reading shouldn't arrive as "new"
    # twice across separate Bronze runs in the first place.
    df, duplicates_removed = deduplicate(df, cfg)
    rows_out = len(df)

    print(f"  {rows_in} new row(s) -> {rows_out} row(s) "
          f"({duplicates_removed} exact duplicate(s) removed within this batch)")

    table_exists = staging_table_exists(staging_uri, storage_options)
    mode = "append" if table_exists else "overwrite"
    print(f"  writing {rows_out} row(s) to staging/{dataset_name} ({mode})...")

    try:
        write_staging_table(s3, bucket, dataset_name, df, storage_options, mode)
        processed_run_ids |= new_run_ids
    except DeltaError as e:
        print(f"  write failed even after retries+self-heal attempt ({e}). "
              f"Falling back to a full rebuild from all of Bronze...")
        delete_prefix_objects(s3, bucket, f"staging/{dataset_name}/")
        full_df = full_bronze_df.copy()
        full_df = cast_types(full_df, cfg)
        full_df = handle_nulls(full_df, dataset_name)
        full_df = format_strings(full_df, cfg)
        full_df, _ = deduplicate(full_df, cfg)
        write_staging_table(s3, bucket, dataset_name, full_df, storage_options, "overwrite")
        processed_run_ids = all_run_ids
        rows_out = len(full_df)
        print(f"  full rebuild succeeded - staging/{dataset_name} recreated fresh "
              f"from all of Bronze ({rows_out} rows).")

    put_manifest(s3, bucket, dataset_name, processed_run_ids)

    return {"rows_in": rows_in, "rows_out": rows_out, "duplicates_removed": duplicates_removed,
            "new_runs": len(new_run_ids)}


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
    total_in = total_out = total_dupes = total_new_runs = 0
    for name, r in results.items():
        print(f"{name}: {r['new_runs']} new Bronze run(s), {r['rows_in']} in -> "
              f"{r['rows_out']} out ({r['duplicates_removed']} duplicates removed)")
        total_in += r["rows_in"]
        total_out += r["rows_out"]
        total_dupes += r["duplicates_removed"]
        total_new_runs += r["new_runs"]

    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            f.write(f"total_rows_in={total_in}\n")
            f.write(f"total_rows_out={total_out}\n")
            f.write(f"total_duplicates_removed={total_dupes}\n")
            f.write(f"total_new_runs={total_new_runs}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyError as e:
        print(f"::error::Missing required environment variable: {e}", file=sys.stderr)
        sys.exit(1)