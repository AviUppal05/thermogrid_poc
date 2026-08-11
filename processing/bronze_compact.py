#!/usr/bin/env python3
"""
ThermoGrid — Raw to Bronze Compaction
=======================================
Reads every small raw Parquet file under raw/<dataset>/building_id=*/dt=*/
in the Hugging Face Storage Bucket, consolidates each partition into a
single deduplicated batch, and writes it as a proper Delta table under
bronze/<dataset> using the deltalake (delta-rs) package.

No Spark, no Databricks — this runs entirely on a GitHub Actions runner.

Env vars required (same ones used by the generator):
    HF_ACCESS_KEY_ID
    HF_SECRET_ACCESS_KEY
    HF_NAMESPACE
    HF_BUCKET_NAME
"""

import argparse
import io
import os
import re
import sys
import time
from collections import defaultdict

import boto3
import pandas as pd
from botocore.client import Config
from deltalake import DeltaTable, write_deltalake
from deltalake.exceptions import DeltaError

# ------------------------------------------------------------------
# Dataset config: source prefix + natural key used to drop duplicate
# rows (raw files are never deleted, so re-running must be idempotent)
# ------------------------------------------------------------------
DATASETS = {
    "hvac_telemetry": {
        "raw_prefix": "raw/hvac_telemetry/",
        "dedup_keys": ["timestamp", "equipment_id"],
    },
    "utility_meter": {
        "raw_prefix": "raw/utility_meter/",
        "dedup_keys": ["timestamp", "meter_id"],
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
    """storage_options for the deltalake (delta-rs) writer.

    AWS_S3_ALLOW_UNSAFE_RENAME is required because delta-rs normally uses
    a DynamoDB locking provider to guarantee safe concurrent writes on S3.
    The HF bucket doesn't support that, but our GitHub Actions jobs run
    one writer at a time, so unsafe rename is safe in this setup.
    """
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
# Delta write
# ------------------------------------------------------------------
def write_bronze_table(table_uri: str, df: pd.DataFrame, storage_options: dict,
                        max_attempts: int = 4) -> None:
    """Write the full dataset in a single Delta commit, with retries.

    Earlier versions of this script wrote one commit per partition,
    reusing a live DeltaTable handle across the loop. That still failed
    against HF's S3-compatible storage with "Invalid table version"
    errors - each commit requires correctly reading the table's current
    version from the storage backend, and that backend doesn't guarantee
    the same strict read-after-write consistency AWS S3 does. Switching
    to a single full-table overwrite per run removed most of the
    problem, but the same underlying lag can still occasionally cause
    a single write to fail if it lands right after another operation
    (e.g. a prior run's OPTIMIZE) touched the log. A short retry with
    backoff, forcing a fresh read of the table state each attempt,
    resolves these transient cases without needing manual intervention.
    """
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            write_deltalake(
                table_uri,
                df,
                mode="overwrite",
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
# Main compaction routine
# ------------------------------------------------------------------
def compact_dataset(s3, bucket: str, dataset_name: str, cfg: dict,
                     storage_options: dict, run_optimize: bool) -> dict:
    print(f"\n=== {dataset_name} ===")
    keys = list_raw_files(s3, bucket, cfg["raw_prefix"])
    if not keys:
        print("  no raw files found, skipping")
        return {"partitions": 0, "raw_rows": 0, "bronze_rows": 0, "source_files": 0}

    groups = group_by_partition(keys)
    table_uri = f"s3://{bucket}/bronze/{dataset_name}"

    total_raw_rows = 0
    partition_dfs = []

    for (building_id, dt), file_keys in sorted(groups.items()):
        dfs = [read_parquet_from_s3(s3, bucket, k) for k in file_keys]
        df = pd.concat(dfs, ignore_index=True)
        raw_rows = len(df)

        df = df.drop_duplicates(subset=cfg["dedup_keys"]).reset_index(drop=True)

        # 'dt' (and building_id, for safety) only exist in the S3 path, not
        # inside the Parquet files themselves - Delta needs them as real
        # columns to partition by, so stamp them on from what we parsed
        # out of the path.
        df["building_id"] = building_id
        df["dt"] = dt

        print(f"  building_id={building_id} dt={dt}: "
              f"{len(file_keys)} files, {raw_rows} raw rows -> {len(df)} deduped rows")

        total_raw_rows += raw_rows
        partition_dfs.append(df)

    full_df = pd.concat(partition_dfs, ignore_index=True)
    total_bronze_rows = len(full_df)

    print(f"  writing {total_bronze_rows} rows across {len(groups)} partitions in a single commit...")
    write_bronze_table(table_uri, full_df, storage_options)

    if run_optimize:
        try:
            print(f"  running OPTIMIZE on bronze/{dataset_name} ...")
            dt_table = DeltaTable(table_uri, storage_options=storage_options)
            dt_table.optimize.compact()
        except Exception as e:
            # OPTIMIZE only merges/adds files - it never deletes, so a
            # failure here is annoying but not risky, and shouldn't take
            # down a run that already wrote the actual data successfully.
            # This has been observed against HF's S3-compatible storage
            # when the follow-up reload lands on a listing that hasn't
            # caught up with the write that just happened.
            print(f"  warning: OPTIMIZE failed, skipping (data write already "
                  f"succeeded above): {e}")

    # VACUUM deletes files, so it is deliberately NOT run automatically
    # here. Running it with a stale view of the table (same storage-
    # consistency lag as above) risks deleting a file that's still part
    # of the live snapshot - a failed compaction is annoying, a bad
    # vacuum is data loss. Run vacuum_bronze.py by hand occasionally
    # instead, once things have settled.

    return {
        "partitions": len(groups),
        "raw_rows": total_raw_rows,
        "bronze_rows": total_bronze_rows,
        "source_files": len(keys),
    }


def main():
    parser = argparse.ArgumentParser(description="Compact raw ThermoGrid data into Bronze Delta tables")
    parser.add_argument("--optimize", action="store_true",
                         help="Run Delta OPTIMIZE (file compaction) + VACUUM after writing")
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS.keys()), default=list(DATASETS.keys()),
                         help="Which datasets to compact (default: all)")
    args = parser.parse_args()

    bucket = os.environ["HF_BUCKET_NAME"]
    s3 = get_s3_client()
    storage_options = deltalake_storage_options()

    results = {}
    for name in args.datasets:
        results[name] = compact_dataset(s3, bucket, name, DATASETS[name], storage_options, args.optimize)

    print("\n=== Summary ===")
    total_partitions = total_raw = total_bronze = total_files = 0
    for name, r in results.items():
        print(f"{name}: {r['partitions']} partitions, {r['source_files']} source files, "
              f"{r['raw_rows']} raw rows -> {r['bronze_rows']} bronze rows")
        total_partitions += r["partitions"]
        total_raw += r["raw_rows"]
        total_bronze += r["bronze_rows"]
        total_files += r["source_files"]

    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            f.write(f"total_partitions={total_partitions}\n")
            f.write(f"total_source_files={total_files}\n")
            f.write(f"total_raw_rows={total_raw}\n")
            f.write(f"total_bronze_rows={total_bronze}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyError as e:
        print(f"::error::Missing required environment variable: {e}", file=sys.stderr)
        sys.exit(1)