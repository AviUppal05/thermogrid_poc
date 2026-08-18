#!/usr/bin/env python3
"""
ThermoGrid — Staging to DQ Validation
=========================================
Stage 6 of the project roadmap: Data Quality & Validation Implementation.

Reads the Staging tables (already type-cast, null-handled, deduped,
formatted) and produces THREE outputs:

    dq_validated/<dataset>  -> every row, with data_quality_flag
                                ('clean'/'warning'/'critical') and
                                data_quality_issues columns. Nothing is
                                dropped from this table.
    quarantine/<dataset>    -> a FILTERED COPY of just the critical
                                rows - "rejected records handling", but
                                as a copy for easy isolated inspection,
                                not a deletion. The row still exists in
                                dq_validated/ too.
    dq_logs/<dataset>       -> one APPENDED row per pipeline run with
                                summary counts (clean/warning/critical,
                                run id, timestamp) - an actual persisted
                                audit trail across runs, not just
                                console output that disappears when the
                                job log expires.

Also does "duplicate validation": a sanity re-check that Staging's
dedup actually left no duplicate natural keys behind. Should always
find zero - this is a validation of an assumption, not a second
dedup pass.

Important distinction preserved from before: fault_code/status_flag
are operational signal, not a data quality problem, and are never
factored into data_quality_flag.

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
    "hvac_telemetry": {"dedup_keys": ["timestamp", "equipment_id"]},
    "utility_meter": {"dedup_keys": ["timestamp", "meter_id"]},
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


def write_snapshot_table(s3, bucket: str, prefix: str, dataset_name: str, df: pd.DataFrame,
                          storage_options: dict, mode: str, max_attempts: int = 4) -> None:
    """mode is 'append' (normal incremental case) or 'overwrite' (first
    write). Retries only - does NOT self-heal internally. df here is
    only the NEW batch, not the full table, so wiping-and-rewriting
    just this df would destroy all previously accumulated history.
    The caller (which has access to the full upstream data) is
    responsible for the full-reprocess fallback on failure - see
    run_dq_validation."""
    table_uri = f"s3://{bucket}/{prefix}/{dataset_name}"
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
    raise last_error



def table_exists(table_uri: str, storage_options: dict) -> bool:
    try:
        DeltaTable(table_uri, storage_options=storage_options)
        return True
    except TableNotFoundError:
        return False


# ------------------------------------------------------------------
# Watermark manifest - tracks which Staging _pipeline_run_id values
# have already been processed by DQ Validation.
# ------------------------------------------------------------------
def manifest_key(dataset_name: str) -> str:
    return f"dq_validated/_manifests/{dataset_name}_processed_run_ids.json"


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


def append_log_row(s3, bucket: str, dataset_name: str, log_row: dict,
                    storage_options: dict, max_attempts: int = 3) -> None:
    """dq_logs is an ACCUMULATING AUDIT TRAIL, not a current-state
    snapshot - it must never be wiped, or the whole point of having a
    persisted history is defeated. So this deliberately has NO
    self-heal fallback: if the append keeps failing, log a warning and
    move on rather than touching the table destructively. The actual
    DQ results (dq_validated/quarantine) already succeeded by this
    point regardless."""
    table_uri = f"s3://{bucket}/dq_logs/{dataset_name}"
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
            print(f"  logged run summary to dq_logs/{dataset_name}")
            return
        except DeltaError as e:
            last_error = e
            if attempt < max_attempts:
                wait = 2 ** attempt
                print(f"  dq_log append attempt {attempt}/{max_attempts} failed "
                      f"({e}), retrying in {wait}s...")
                time.sleep(wait)
    print(f"  warning: could not append to dq_logs/{dataset_name} after "
          f"{max_attempts} attempts ({last_error}). Skipping the log entry for "
          f"this run - the actual DQ results above are unaffected.")


# ------------------------------------------------------------------
# Duplicate validation (sanity re-check, not a second dedup pass)
# ------------------------------------------------------------------
def validate_no_duplicates(df: pd.DataFrame, dedup_keys: list[str]) -> int:
    dup_count = df.duplicated(subset=dedup_keys, keep=False).sum()
    if dup_count > 0:
        print(f"  WARNING: {dup_count} duplicate row(s) found on {dedup_keys} - "
              f"Staging's dedup should have caught these. Investigate staging_build.py.")
    else:
        print(f"  duplicate validation: 0 duplicates found on {dedup_keys} (as expected)")
    return int(dup_count)


# ------------------------------------------------------------------
# Data quality checks (range checks)
# ------------------------------------------------------------------
def apply_quality_checks(df: pd.DataFrame, checks: list[tuple]) -> pd.DataFrame:
    """checks: list of (name, severity, boolean_mask_series) where the
    mask is True for ROWS THAT FAIL the check."""
    issues = pd.Series([[] for _ in range(len(df))], index=df.index)
    severities = pd.Series(["clean"] * len(df), index=df.index)

    for name, severity, mask in checks:
        mask = mask.fillna(True)  # NaN inputs count as failing
        for idx in df.index[mask]:
            issues.at[idx].append(name)
        if severity == "critical":
            severities = severities.mask(mask, "critical")
        else:  # warning - only upgrade if not already critical
            severities = severities.mask(mask & (severities != "critical"), "warning")

    df["data_quality_flag"] = severities
    df["data_quality_issues"] = issues.apply(lambda x: ";".join(x))
    return df


def hvac_quality_checks(df: pd.DataFrame) -> list[tuple]:
    return [
        ("missing_identifier", "critical",
         df["timestamp"].isna() | df["equipment_id"].isna() | df["building_id"].isna()),
        ("supply_air_temp_out_of_range", "critical",
         ~df["supply_air_temp"].between(30, 100)),
        ("return_air_temp_out_of_range", "critical",
         ~df["return_air_temp"].between(50, 100)),
        ("fan_speed_pct_out_of_range", "critical",
         ~df["fan_speed_pct"].between(0, 100)),
        ("duct_pressure_out_of_range", "critical",
         ~df["duct_pressure"].between(0, 8)),
        ("filter_pressure_drop_out_of_range", "critical",
         ~df["filter_pressure_drop"].between(0, 3)),
        ("cooling_coil_temp_out_of_range", "critical",
         ~df["cooling_coil_temp"].between(30, 75)),
        ("heating_coil_temp_out_of_range", "critical",
         ~df["heating_coil_temp"].between(70, 150)),
        ("outdoor_air_temp_out_of_range", "critical",
         ~df["outdoor_air_temp"].between(-30, 135)),
        ("supply_warmer_than_return", "warning",
         df["supply_air_temp"] > df["return_air_temp"]),
    ]


def utility_quality_checks(df: pd.DataFrame) -> list[tuple]:
    return [
        ("missing_identifier", "critical",
         df["timestamp"].isna() | df["meter_id"].isna() | df["building_id"].isna()),
        ("negative_actual_consumption", "critical",
         df["actual_consumption_kwh"] < 0),
        ("negative_peak_demand", "critical",
         df["peak_demand_kw"] < 0),
        ("non_positive_cost_rate", "critical",
         df["cost_rate"] <= 0),
        ("non_positive_expected_consumption", "warning",
         df["expected_consumption_kwh"] <= 0),
        ("peak_below_actual", "warning",
         df["peak_demand_kw"] < df["actual_consumption_kwh"]),
    ]


# ------------------------------------------------------------------
# Main per-dataset routine
# ------------------------------------------------------------------
def run_dq_validation(s3, dataset_name: str, bucket: str, storage_options: dict, run_id: str) -> dict:
    print(f"\n=== {dataset_name} ===")
    staging_uri = f"s3://{bucket}/staging/{dataset_name}"
    dq_validated_uri = f"s3://{bucket}/dq_validated/{dataset_name}"

    try:
        staging_table = DeltaTable(staging_uri, storage_options=storage_options)
    except TableNotFoundError:
        print(f"  no Staging table found at {staging_uri}, skipping")
        return {"rows": 0, "clean": 0, "warning": 0, "critical": 0, "duplicates_found": 0}

    full_staging_df = staging_table.to_pandas()
    print(f"  read {len(full_staging_df)} total rows from Staging (version {staging_table.version()})")

    processed_run_ids = get_manifest(s3, bucket, dataset_name)
    all_run_ids = set(full_staging_df["_pipeline_run_id"].dropna().unique())
    new_run_ids = all_run_ids - processed_run_ids

    if not new_run_ids:
        print(f"  watermark: {len(processed_run_ids)} run(s) already processed, "
              f"0 new run(s) - nothing to do")
        return {"rows": 0, "clean": 0, "warning": 0, "critical": 0, "duplicates_found": 0}

    print(f"  watermark: {len(processed_run_ids)} run(s) already processed, "
          f"{len(new_run_ids)} new run(s) to validate")

    df = full_staging_df[full_staging_df["_pipeline_run_id"].isin(new_run_ids)].copy()
    total_rows = len(df)

    duplicates_found = validate_no_duplicates(df, DATASETS[dataset_name]["dedup_keys"])

    checks = hvac_quality_checks(df) if dataset_name == "hvac_telemetry" else utility_quality_checks(df)
    df = apply_quality_checks(df, checks)

    counts = df["data_quality_flag"].value_counts().to_dict()
    clean = counts.get("clean", 0)
    warning = counts.get("warning", 0)
    critical = counts.get("critical", 0)
    print(f"  quality summary (this batch): {clean} clean, {warning} warning, {critical} critical "
          f"({total_rows} new rows, all kept in dq_validated)")

    mode = "append" if table_exists(dq_validated_uri, storage_options) else "overwrite"
    print(f"  writing {total_rows} row(s) to dq_validated/{dataset_name} ({mode})...")

    try:
        write_snapshot_table(s3, bucket, "dq_validated", dataset_name, df, storage_options, mode)
        quarantine_df = df[df["data_quality_flag"] == "critical"].reset_index(drop=True)
        quarantine_uri = f"s3://{bucket}/quarantine/{dataset_name}"
        if len(quarantine_df) > 0:
            q_mode = "append" if table_exists(quarantine_uri, storage_options) else "overwrite"
            print(f"  writing {len(quarantine_df)} row(s) to quarantine/{dataset_name} ({q_mode})...")
            write_snapshot_table(s3, bucket, "quarantine", dataset_name, quarantine_df, storage_options, q_mode)
        else:
            print(f"  no critical rows in this batch, nothing to add to quarantine/{dataset_name}")
        final_processed_run_ids = processed_run_ids | new_run_ids
    except DeltaError as e:
        print(f"  write failed even after retries ({e}). Falling back to a full "
              f"rebuild of dq_validated/{dataset_name} and quarantine/{dataset_name} "
              f"from all of Staging...")
        delete_prefix_objects(s3, bucket, f"dq_validated/{dataset_name}/")
        delete_prefix_objects(s3, bucket, f"quarantine/{dataset_name}/")

        full_df = full_staging_df.copy()
        validate_no_duplicates(full_df, DATASETS[dataset_name]["dedup_keys"])
        checks = hvac_quality_checks(full_df) if dataset_name == "hvac_telemetry" else utility_quality_checks(full_df)
        full_df = apply_quality_checks(full_df, checks)

        write_snapshot_table(s3, bucket, "dq_validated", dataset_name, full_df, storage_options, "overwrite")
        full_quarantine_df = full_df[full_df["data_quality_flag"] == "critical"].reset_index(drop=True)
        if len(full_quarantine_df) > 0:
            write_snapshot_table(s3, bucket, "quarantine", dataset_name, full_quarantine_df,
                                  storage_options, "overwrite")

        final_processed_run_ids = all_run_ids
        counts = full_df["data_quality_flag"].value_counts().to_dict()
        clean, warning, critical = counts.get("clean", 0), counts.get("warning", 0), counts.get("critical", 0)
        total_rows = len(full_df)
        print(f"  full rebuild succeeded - dq_validated/{dataset_name} recreated fresh "
              f"from all of Staging ({total_rows} rows).")

    put_manifest(s3, bucket, dataset_name, final_processed_run_ids)

    log_row = {
        "run_id": run_id,
        "logged_at": datetime.now(timezone.utc).isoformat(),
        "total_rows": total_rows,
        "clean": int(clean),
        "warning": int(warning),
        "critical": int(critical),
        "duplicates_found": duplicates_found,
    }
    append_log_row(s3, bucket, dataset_name, log_row, storage_options)

    return {"rows": total_rows, "clean": clean, "warning": warning, "critical": critical,
            "duplicates_found": duplicates_found}


def main():
    parser = argparse.ArgumentParser(description="Run DQ validation against Staging tables")
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS.keys()), default=list(DATASETS.keys()))
    args = parser.parse_args()

    bucket = os.environ["HF_BUCKET_NAME"]
    storage_options = deltalake_storage_options()
    s3 = get_s3_client()
    run_id = os.environ.get("GITHUB_RUN_ID", "local")

    results = {}
    for name in args.datasets:
        results[name] = run_dq_validation(s3, name, bucket, storage_options, run_id)

    print("\n=== Summary ===")
    total_rows = total_clean = total_warning = total_critical = total_dupes = 0
    for name, r in results.items():
        print(f"{name}: {r['rows']} rows ({r['clean']} clean, {r['warning']} warning, "
              f"{r['critical']} critical, {r['duplicates_found']} duplicates found)")
        total_rows += r["rows"]
        total_clean += r["clean"]
        total_warning += r["warning"]
        total_critical += r["critical"]
        total_dupes += r["duplicates_found"]

    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            f.write(f"total_rows={total_rows}\n")
            f.write(f"total_clean={total_clean}\n")
            f.write(f"total_warning={total_warning}\n")
            f.write(f"total_critical={total_critical}\n")
            f.write(f"total_duplicates_found={total_dupes}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyError as e:
        print(f"::error::Missing required environment variable: {e}", file=sys.stderr)
        sys.exit(1)