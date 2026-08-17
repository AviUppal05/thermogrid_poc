#!/usr/bin/env python3
"""
ThermoGrid — Raw Data Ingestion Pipeline (Bronze)
=====================================================
Stages 3-4 of the project roadmap: Bronze layer creation + raw data
ingestion pipeline with incremental load logic and watermark tracking.

This is a pure LANDING layer. It does:
    - schema drift detection (missing/extra columns vs. expected)
    - audit/lineage metadata (_source_file, _ingested_at, _pipeline_run_id)
    - incremental append of only NEW raw files since the last run,
      tracked via a watermark manifest stored alongside the table
    - file compaction (OPTIMIZE) as part of the ingestion pipeline

It deliberately does NOT do:
    - deduplication (moved to Staging - Bronze preserves exactly what
      arrived, duplicates and all)
    - any business-rule validation, range checks, or type coercion

Watermark tracking: a small JSON manifest at
bronze/_manifests/<dataset>_ingested_files.json records every raw file
key already ingested. Each run only reads and appends files NOT in
that manifest, instead of reprocessing the entire raw/ history every
time.

No Spark, no Databricks - runs entirely on a GitHub Actions runner.

Env vars required:
    HF_ACCESS_KEY_ID
    HF_SECRET_ACCESS_KEY
    HF_NAMESPACE
    HF_BUCKET_NAME
"""

import argparse
import io
import json
import os
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import boto3
import pandas as pd
from botocore.client import Config
from botocore.exceptions import ClientError
from deltalake import DeltaTable, write_deltalake
from deltalake.exceptions import DeltaError, TableNotFoundError

MAX_PARALLEL_FETCHES = 16

# ------------------------------------------------------------------
# Dataset config: source prefix and the expected schema used for
# basic validation / schema-drift handling on the way into Bronze.
# (dedup_keys live in staging_build.py now, not here.)
# ------------------------------------------------------------------
DATASETS = {
    "hvac_telemetry": {
        "raw_prefix": "raw/hvac_telemetry/",
        "expected_columns": [
            "timestamp", "equipment_id", "building_id", "supply_air_temp",
            "return_air_temp", "fan_speed_pct", "duct_pressure", "filter_pressure_drop",
            "cooling_coil_temp", "heating_coil_temp", "outdoor_air_temp",
            "fault_code", "status_flag",
        ],
    },
    "utility_meter": {
        "raw_prefix": "raw/utility_meter/",
        "expected_columns": [
            "timestamp", "meter_id", "building_id", "utility_type",
            "actual_consumption_kwh", "expected_consumption_kwh",
            "peak_demand_kw", "cost_rate",
        ],
    },
}

PARTITION_RE = re.compile(r"building_id=([^/]+)/dt=([^/]+)/")


# ------------------------------------------------------------------
# S3 (HF bucket) helpers
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
    """AWS_S3_ALLOW_UNSAFE_RENAME is required because delta-rs normally
    uses a DynamoDB locking provider to guarantee safe concurrent writes
    on S3. The HF bucket doesn't support that, but our GitHub Actions
    jobs run one writer at a time, so unsafe rename is safe here."""
    return {
        "endpoint_url": f"https://s3.hf.co/{os.environ['HF_NAMESPACE']}",
        "AWS_ACCESS_KEY_ID": os.environ["HF_ACCESS_KEY_ID"],
        "AWS_SECRET_ACCESS_KEY": os.environ["HF_SECRET_ACCESS_KEY"],
        "AWS_REGION": "us-east-1",
        "AWS_S3_ALLOW_UNSAFE_RENAME": "true",
    }


def list_raw_files(s3, bucket: str, prefix: str) -> list[str]:
    keys = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".parquet"):
                keys.append(obj["Key"])
    return keys


def group_by_partition(keys: list[str]) -> dict[tuple[str, str], list[str]]:
    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for key in keys:
        m = PARTITION_RE.search(key)
        if not m:
            print(f"  skipping (no building_id/dt in path): {key}")
            continue
        building_id, dt = m.group(1), m.group(2)
        groups[(building_id, dt)].append(key)
    return groups


def read_parquet_from_s3(s3, bucket: str, key: str) -> pd.DataFrame:
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_parquet(io.BytesIO(obj["Body"].read()))


# ------------------------------------------------------------------
# Watermark manifest - tracks which raw files have already been
# ingested, so each run only processes what's new.
# ------------------------------------------------------------------
def manifest_key(dataset_name: str) -> str:
    return f"bronze/_manifests/{dataset_name}_ingested_files.json"


def get_manifest(s3, bucket: str, dataset_name: str) -> set[str]:
    try:
        obj = s3.get_object(Bucket=bucket, Key=manifest_key(dataset_name))
        data = json.loads(obj["Body"].read())
        return set(data.get("ingested_files", []))
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in ("NoSuchKey", "404"):
            return set()  # no manifest yet - first run
        raise


def put_manifest(s3, bucket: str, dataset_name: str, ingested_files: set[str]) -> None:
    body = json.dumps({
        "ingested_files": sorted(ingested_files),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }).encode("utf-8")
    s3.put_object(Bucket=bucket, Key=manifest_key(dataset_name), Body=body,
                  ContentType="application/json")


# ------------------------------------------------------------------
# Schema validation + audit metadata
# ------------------------------------------------------------------
def validate_schema(df: pd.DataFrame, expected_columns: list[str], source_key: str) -> pd.DataFrame:
    """Basic schema validation / schema-drift handling: check the file
    has the columns it's supposed to, without judging the VALUES.

    - Expected column missing -> filled with null, logged (visible
      drift, not silent data loss).
    - Unexpected extra column present -> kept, logged (schema
      evolution - Bronze preserves what arrived, doesn't decide it
      doesn't belong).
    """
    missing = [c for c in expected_columns if c not in df.columns]
    extra = [c for c in df.columns if c not in expected_columns]

    if missing:
        print(f"    schema drift in {source_key}: missing columns {missing} - filling with null")
        for col in missing:
            df[col] = pd.NA
    if extra:
        print(f"    schema drift in {source_key}: unexpected columns {extra} - keeping them")

    return df


def tag_audit_metadata(df: pd.DataFrame, source_key: str, ingested_at: str, run_id: str) -> pd.DataFrame:
    """Audit/lineage metadata - doesn't touch the actual data, just
    records where and when it came from."""
    df["_source_file"] = source_key
    df["_ingested_at"] = ingested_at
    df["_pipeline_run_id"] = run_id
    return df


def fetch_files_parallel(s3, bucket: str, file_keys: list[str],
                          max_workers: int = MAX_PARALLEL_FETCHES) -> dict[str, pd.DataFrame]:
    """Reads multiple raw files concurrently instead of one at a time.

    This was the single biggest bottleneck in Bronze: each raw file
    was fetched with its own sequential network round-trip to HF's S3
    endpoint, so a backlog of hundreds/thousands of files meant
    hundreds/thousands of round-trips waited on one after another.
    boto3 clients are safe to share across threads for read calls like
    get_object, so this uses a thread pool - this is I/O-bound work
    (waiting on network), which is exactly what threads (not
    processes) are good for in Python, since the GIL is released
    while waiting on I/O.
    """
    results: dict[str, pd.DataFrame] = {}
    errors: dict[str, Exception] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_key = {
            executor.submit(read_parquet_from_s3, s3, bucket, k): k
            for k in file_keys
        }
        for future in as_completed(future_to_key):
            key = future_to_key[future]
            try:
                results[key] = future.result()
            except Exception as e:
                errors[key] = e

    if errors:
        # Surface every failed file, not just the first one - makes
        # debugging a partial-failure batch much easier than a single
        # opaque stack trace.
        for key, e in errors.items():
            print(f"  ERROR fetching {key}: {e}")
        raise RuntimeError(f"Failed to fetch {len(errors)}/{len(file_keys)} raw file(s)")

    return results


def process_files_into_df(s3, bucket: str, file_keys: list[str], cfg: dict,
                           ingested_at: str, run_id: str):
    """Reads a set of raw files, applies schema validation + audit
    tagging, and stamps building_id/dt from the path. Deliberately no
    dedup here - that's Staging's job now."""
    print(f"  fetching {len(file_keys)} raw file(s) ({MAX_PARALLEL_FETCHES} at a time)...")
    fetched = fetch_files_parallel(s3, bucket, file_keys)

    groups = group_by_partition(file_keys)
    total_raw_rows = 0
    partition_dfs = []

    for (building_id, dt), keys in sorted(groups.items()):
        dfs = []
        for k in keys:
            file_df = fetched[k]
            file_df = validate_schema(file_df, cfg["expected_columns"], k)
            file_df = tag_audit_metadata(file_df, k, ingested_at, run_id)
            dfs.append(file_df)
        df = pd.concat(dfs, ignore_index=True)
        raw_rows = len(df)

        # 'dt' (and building_id, for safety) only exist in the S3 path,
        # not inside the Parquet files - Delta needs them as real
        # columns to partition by.
        df["building_id"] = building_id
        df["dt"] = dt

        print(f"  building_id={building_id} dt={dt}: {len(keys)} file(s), {raw_rows} row(s)")

        total_raw_rows += raw_rows
        partition_dfs.append(df)

    full_df = pd.concat(partition_dfs, ignore_index=True) if partition_dfs else pd.DataFrame()
    return full_df, total_raw_rows, groups


# ------------------------------------------------------------------
# Delta write
# ------------------------------------------------------------------
def delete_prefix_objects(s3, bucket: str, prefix: str) -> int:
    """Deletes every object under a prefix. Used for self-healing a
    corrupted Bronze table."""
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


def bronze_table_exists(table_uri: str, storage_options: dict) -> bool:
    try:
        DeltaTable(table_uri, storage_options=storage_options)
        return True
    except TableNotFoundError:
        return False


def write_bronze_delta(table_uri: str, df: pd.DataFrame, storage_options: dict,
                        mode: str, max_attempts: int = 4) -> None:
    """Write with retry-and-backoff. mode is 'overwrite' (table doesn't
    exist yet, or a self-heal rebuild) or 'append' (normal incremental
    case). See compact_dataset for the self-heal fallback if all
    retries here are exhausted - it needs to fully reprocess raw
    files, which this function doesn't have access to, so that lives
    one level up."""
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            write_deltalake(
                table_uri,
                df,
                mode=mode,
                partition_by=["building_id", "dt"],
                storage_options=storage_options,
            )
            return
        except DeltaError as e:
            last_error = e
            if attempt < max_attempts:
                wait = 2 ** attempt  # 2s, 4s, 8s
                print(f"  write attempt {attempt}/{max_attempts} failed "
                      f"({e}), retrying in {wait}s...")
                time.sleep(wait)
    raise last_error


# ------------------------------------------------------------------
# Main ingestion routine
# ------------------------------------------------------------------
def compact_dataset(s3, bucket: str, dataset_name: str, cfg: dict,
                     storage_options: dict, run_optimize: bool,
                     ingested_at: str, run_id: str) -> dict:
    print(f"\n=== {dataset_name} ===")
    all_keys = list_raw_files(s3, bucket, cfg["raw_prefix"])
    if not all_keys:
        print("  no raw files found, skipping")
        return {"new_files": 0, "raw_rows": 0, "bronze_rows": 0, "source_files": 0}

    table_uri = f"s3://{bucket}/bronze/{dataset_name}"
    manifest = get_manifest(s3, bucket, dataset_name)
    new_keys = sorted(k for k in all_keys if k not in manifest)

    if not new_keys:
        print(f"  watermark: {len(manifest)} file(s) already ingested, "
              f"0 new file(s) since last run - nothing to do")
        return {"new_files": 0, "raw_rows": 0, "bronze_rows": 0, "source_files": len(all_keys)}

    print(f"  watermark: {len(manifest)} file(s) already ingested, "
          f"{len(new_keys)} new file(s) to process")

    table_exists = bronze_table_exists(table_uri, storage_options)
    full_df, total_raw_rows, groups = process_files_into_df(
        s3, bucket, new_keys, cfg, ingested_at, run_id)
    total_bronze_rows = len(full_df)

    mode = "append" if table_exists else "overwrite"
    print(f"  writing {total_bronze_rows} new row(s) across {len(groups)} "
          f"partition(s) ({mode})...")

    ingested_this_run = new_keys
    try:
        write_bronze_delta(table_uri, full_df, storage_options, mode)
    except DeltaError as e:
        # Retries exhausted. Rather than fail the whole run, wipe this
        # dataset's Bronze table and rebuild it from ALL raw files
        # (not just the new ones) - raw/ is never deleted, so nothing
        # is actually lost, just re-read. See historical note: this
        # error has recurred consistently blaming a specific stale
        # version regardless of how many commits the table is on,
        # pointing to a corrupted checkpoint rather than a one-off race.
        print(f"  all retries failed ({e}). Self-healing: wiping "
              f"bronze/{dataset_name} and reprocessing ALL {len(all_keys)} "
              f"raw file(s) from scratch...")
        delete_prefix_objects(s3, bucket, f"bronze/{dataset_name}/")
        full_df, total_raw_rows, groups = process_files_into_df(
            s3, bucket, all_keys, cfg, ingested_at, run_id)
        total_bronze_rows = len(full_df)
        write_bronze_delta(table_uri, full_df, storage_options, "overwrite")
        ingested_this_run = all_keys
        print(f"  self-heal succeeded - bronze/{dataset_name} recreated "
              f"fresh from all raw data.")

    manifest |= set(ingested_this_run)
    put_manifest(s3, bucket, dataset_name, manifest)

    if run_optimize:
        try:
            print(f"  running OPTIMIZE on bronze/{dataset_name} ...")
            dt_table = DeltaTable(table_uri, storage_options=storage_options)
            dt_table.optimize.compact()
        except Exception as e:
            # OPTIMIZE only merges/adds files - it never deletes, so a
            # failure here is annoying but not risky, and shouldn't take
            # down a run that already wrote the actual data successfully.
            print(f"  warning: OPTIMIZE failed, skipping (data write "
                  f"already succeeded above): {e}")

    # VACUUM deletes files, so it is deliberately NOT run automatically -
    # see vacuum_bronze.py for the manual, safe-retention alternative.

    return {
        "new_files": len(new_keys),
        "raw_rows": total_raw_rows,
        "bronze_rows": total_bronze_rows,
        "source_files": len(all_keys),
    }


def main():
    parser = argparse.ArgumentParser(description="Incrementally ingest raw ThermoGrid data into Bronze")
    parser.add_argument("--optimize", action="store_true",
                         help="Run Delta OPTIMIZE (file compaction) after writing")
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS.keys()), default=list(DATASETS.keys()),
                         help="Which datasets to ingest (default: all)")
    args = parser.parse_args()

    bucket = os.environ["HF_BUCKET_NAME"]
    s3 = get_s3_client()
    storage_options = deltalake_storage_options()

    ingested_at = datetime.now(timezone.utc).isoformat()
    run_id = os.environ.get("GITHUB_RUN_ID", "local")

    results = {}
    for name in args.datasets:
        results[name] = compact_dataset(s3, bucket, name, DATASETS[name], storage_options,
                                         args.optimize, ingested_at, run_id)

    print("\n=== Summary ===")
    total_new_files = total_raw = total_bronze = total_files = 0
    for name, r in results.items():
        print(f"{name}: {r['new_files']} new file(s) ingested, "
              f"{r['raw_rows']} raw row(s) -> {r['bronze_rows']} bronze row(s) "
              f"({r['source_files']} total raw files seen)")
        total_new_files += r["new_files"]
        total_raw += r["raw_rows"]
        total_bronze += r["bronze_rows"]
        total_files += r["source_files"]

    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            f.write(f"total_new_files={total_new_files}\n")
            f.write(f"total_source_files={total_files}\n")
            f.write(f"total_raw_rows={total_raw}\n")
            f.write(f"total_bronze_rows={total_bronze}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyError as e:
        print(f"::error::Missing required environment variable: {e}", file=sys.stderr)
        sys.exit(1)