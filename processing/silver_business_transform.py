#!/usr/bin/env python3
"""
ThermoGrid — Silver Layer: Business Transformation & KPI Enrichment
========================================================================
Stage 8 of the project roadmap.

Reads fault_detection/hvac_telemetry and dq_validated/utility_meter and
adds business-meaningful derived columns on top of the already-cleaned,
already-validated, already-fault-checked data:

HVAC:
    temp_differential   -> return_air_temp - supply_air_temp
    operating_mode      -> 'Cooling' or 'Heating', from that relationship
    filter_health_status -> Good / Monitor / Replace Soon / Replace Now,
                            translating a raw pressure-drop number into
                            an operational category

Utility:
    consumption_variance_pct -> how far actual is from the expected baseline
    is_over_consuming        -> business flag off that variance
    estimated_cost            -> actual_consumption_kwh * cost_rate
    load_factor                -> actual consumption relative to peak demand
    time_of_use_period         -> Peak / Shoulder / Off-Peak, from the hour

Design rule: KPI columns are only computed for rows where
data_quality_flag != 'critical'. The ORIGINAL row and all its original
columns are still kept either way (nothing dropped, per project
policy) - but a derived KPI computed from a known-corrupted reading
(e.g. estimated_cost from a negative consumption value) would just be
manufacturing a second piece of bad data on top of the first. Garbage
in, null KPI out - not garbage out.

No Spark, no Databricks - runs entirely on a GitHub Actions runner.

Env vars required:
    HF_ACCESS_KEY_ID
    HF_SECRET_ACCESS_KEY
    HF_NAMESPACE
    HF_BUCKET_NAME
"""

import argparse
import json
import math
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


def table_exists(table_uri: str, storage_options: dict) -> bool:
    try:
        DeltaTable(table_uri, storage_options=storage_options)
        return True
    except TableNotFoundError:
        return False


def manifest_key(dataset_name: str) -> str:
    return f"silver/_manifests/{dataset_name}_processed_run_ids.json"


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


def write_silver_table(s3, bucket: str, dataset_name: str, df: pd.DataFrame,
                        storage_options: dict, mode: str, max_attempts: int = 4) -> None:
    """Retries only - no internal self-heal. df here is only the new
    batch; see build_silver_hvac/build_silver_utility for the
    full-reprocess fallback."""
    table_uri = f"s3://{bucket}/silver/{dataset_name}"
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



# ------------------------------------------------------------------
# HVAC enrichment
# ------------------------------------------------------------------
def enrich_hvac(df: pd.DataFrame) -> pd.DataFrame:
    computable = df["data_quality_flag"] != "critical"

    temp_differential = df["return_air_temp"] - df["supply_air_temp"]
    df["temp_differential"] = temp_differential.where(computable)

    operating_mode = pd.Series(pd.NA, index=df.index, dtype="object")
    cooling = computable & (df["supply_air_temp"] < df["return_air_temp"])
    heating = computable & (df["supply_air_temp"] >= df["return_air_temp"])
    operating_mode = operating_mode.mask(cooling, "Cooling").mask(heating, "Heating")
    df["operating_mode"] = operating_mode

    def bucket_filter_health(x):
        if pd.isna(x):
            return pd.NA
        if x < 0.3:
            return "Good"
        if x < 0.5:
            return "Monitor"
        if x < 0.8:
            return "Replace Soon"
        return "Replace Now"

    filter_health = df["filter_pressure_drop"].where(computable).apply(bucket_filter_health)
    df["filter_health_status"] = filter_health

    return df


# ------------------------------------------------------------------
# Utility enrichment
# ------------------------------------------------------------------
def tou_period(hour: int) -> str:
    if 14 <= hour <= 19:
        return "Peak"
    if hour in (10, 11, 12, 13, 20, 21):
        return "Shoulder"
    return "Off-Peak"


def enrich_utility(df: pd.DataFrame) -> pd.DataFrame:
    computable = df["data_quality_flag"] != "critical"

    variance_pct = ((df["actual_consumption_kwh"] - df["expected_consumption_kwh"])
                     / df["expected_consumption_kwh"] * 100)
    variance_pct = variance_pct.replace([float("inf"), float("-inf")], math.nan)
    df["consumption_variance_pct"] = variance_pct.where(computable, math.nan).round(1)

    is_over = (df["actual_consumption_kwh"] > df["expected_consumption_kwh"])
    df["is_over_consuming"] = is_over.where(computable)

    estimated_cost = df["actual_consumption_kwh"] * df["cost_rate"]
    df["estimated_cost"] = estimated_cost.where(computable, math.nan).round(2)

    load_factor = df["actual_consumption_kwh"] / df["peak_demand_kw"]
    load_factor = load_factor.replace([float("inf"), float("-inf")], math.nan)
    df["load_factor"] = load_factor.where(computable, math.nan).round(3)

    hours = pd.to_datetime(df["timestamp"]).dt.hour
    df["time_of_use_period"] = hours.apply(tou_period)
    df["time_of_use_period"] = df["time_of_use_period"].where(computable)

    return df


# ------------------------------------------------------------------
# Main per-dataset routine
# ------------------------------------------------------------------
def build_silver_hvac(s3, bucket: str, storage_options: dict) -> dict:
    print("\n=== hvac_telemetry (Silver enrichment) ===")
    source_uri = f"s3://{bucket}/fault_detection/hvac_telemetry"
    silver_uri = f"s3://{bucket}/silver/hvac_telemetry"
    try:
        table = DeltaTable(source_uri, storage_options=storage_options)
    except TableNotFoundError:
        print(f"  no fault_detection table found at {source_uri}, skipping")
        return {"rows": 0, "enriched": 0}

    full_df = table.to_pandas()
    print(f"  read {len(full_df)} total rows from fault_detection (version {table.version()})")

    processed_run_ids = get_manifest(s3, bucket, "hvac_telemetry")
    all_run_ids = set(full_df["_pipeline_run_id"].dropna().unique())
    new_run_ids = all_run_ids - processed_run_ids

    if not new_run_ids:
        print(f"  watermark: {len(processed_run_ids)} run(s) already processed, "
              f"0 new run(s) - nothing to do")
        return {"rows": 0, "enriched": 0}

    print(f"  watermark: {len(processed_run_ids)} run(s) already processed, "
          f"{len(new_run_ids)} new run(s) to enrich")

    df = full_df[full_df["_pipeline_run_id"].isin(new_run_ids)].copy()
    total_rows = len(df)

    df = enrich_hvac(df)
    enriched = int((df["data_quality_flag"] != "critical").sum())
    print(f"  computed KPIs for {enriched}/{total_rows} rows "
          f"({total_rows - enriched} critical row(s) kept with null KPIs)")

    mode = "append" if table_exists(silver_uri, storage_options) else "overwrite"
    print(f"  writing {total_rows} row(s) to silver/hvac_telemetry ({mode})...")

    try:
        write_silver_table(s3, bucket, "hvac_telemetry", df, storage_options, mode)
        final_processed_run_ids = processed_run_ids | new_run_ids
    except DeltaError as e:
        print(f"  write failed even after retries ({e}). Falling back to a full "
              f"rebuild of silver/hvac_telemetry from all of fault_detection...")
        delete_prefix_objects(s3, bucket, "silver/hvac_telemetry/")
        full_reprocess_df = enrich_hvac(full_df.copy())
        write_silver_table(s3, bucket, "hvac_telemetry", full_reprocess_df, storage_options, "overwrite")
        final_processed_run_ids = all_run_ids
        total_rows = len(full_reprocess_df)
        enriched = int((full_reprocess_df["data_quality_flag"] != "critical").sum())
        print(f"  full rebuild succeeded - silver/hvac_telemetry recreated fresh "
              f"from all of fault_detection ({total_rows} rows).")

    put_manifest(s3, bucket, "hvac_telemetry", final_processed_run_ids)

    return {"rows": total_rows, "enriched": enriched}


def build_silver_utility(s3, bucket: str, storage_options: dict) -> dict:
    print("\n=== utility_meter (Silver enrichment) ===")
    source_uri = f"s3://{bucket}/dq_validated/utility_meter"
    silver_uri = f"s3://{bucket}/silver/utility_meter"
    try:
        table = DeltaTable(source_uri, storage_options=storage_options)
    except TableNotFoundError:
        print(f"  no dq_validated table found at {source_uri}, skipping")
        return {"rows": 0, "enriched": 0}

    full_df = table.to_pandas()
    print(f"  read {len(full_df)} total rows from dq_validated (version {table.version()})")

    processed_run_ids = get_manifest(s3, bucket, "utility_meter")
    all_run_ids = set(full_df["_pipeline_run_id"].dropna().unique())
    new_run_ids = all_run_ids - processed_run_ids

    if not new_run_ids:
        print(f"  watermark: {len(processed_run_ids)} run(s) already processed, "
              f"0 new run(s) - nothing to do")
        return {"rows": 0, "enriched": 0}

    print(f"  watermark: {len(processed_run_ids)} run(s) already processed, "
          f"{len(new_run_ids)} new run(s) to enrich")

    df = full_df[full_df["_pipeline_run_id"].isin(new_run_ids)].copy()
    total_rows = len(df)

    df = enrich_utility(df)
    enriched = int((df["data_quality_flag"] != "critical").sum())
    print(f"  computed KPIs for {enriched}/{total_rows} rows "
          f"({total_rows - enriched} critical row(s) kept with null KPIs)")

    mode = "append" if table_exists(silver_uri, storage_options) else "overwrite"
    print(f"  writing {total_rows} row(s) to silver/utility_meter ({mode})...")

    try:
        write_silver_table(s3, bucket, "utility_meter", df, storage_options, mode)
        final_processed_run_ids = processed_run_ids | new_run_ids
    except DeltaError as e:
        print(f"  write failed even after retries ({e}). Falling back to a full "
              f"rebuild of silver/utility_meter from all of dq_validated...")
        delete_prefix_objects(s3, bucket, "silver/utility_meter/")
        full_reprocess_df = enrich_utility(full_df.copy())
        write_silver_table(s3, bucket, "utility_meter", full_reprocess_df, storage_options, "overwrite")
        final_processed_run_ids = all_run_ids
        total_rows = len(full_reprocess_df)
        enriched = int((full_reprocess_df["data_quality_flag"] != "critical").sum())
        print(f"  full rebuild succeeded - silver/utility_meter recreated fresh "
              f"from all of dq_validated ({total_rows} rows).")

    put_manifest(s3, bucket, "utility_meter", final_processed_run_ids)

    return {"rows": total_rows, "enriched": enriched}


def main():
    parser = argparse.ArgumentParser(description="Build Silver (business/KPI) tables")
    parser.add_argument("--datasets", nargs="+", choices=["hvac_telemetry", "utility_meter"],
                         default=["hvac_telemetry", "utility_meter"])
    args = parser.parse_args()

    bucket = os.environ["HF_BUCKET_NAME"]
    storage_options = deltalake_storage_options()
    s3 = get_s3_client()

    results = {}
    if "hvac_telemetry" in args.datasets:
        results["hvac_telemetry"] = build_silver_hvac(s3, bucket, storage_options)
    if "utility_meter" in args.datasets:
        results["utility_meter"] = build_silver_utility(s3, bucket, storage_options)

    print("\n=== Summary ===")
    total_rows = total_enriched = 0
    for name, r in results.items():
        print(f"{name}: {r['rows']} rows, {r['enriched']} with computed KPIs")
        total_rows += r["rows"]
        total_enriched += r["enriched"]

    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            f.write(f"total_rows={total_rows}\n")
            f.write(f"total_enriched={total_enriched}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyError as e:
        print(f"::error::Missing required environment variable: {e}", file=sys.stderr)
        sys.exit(1)