#!/usr/bin/env python3
"""
ThermoGrid — Dimension Tables for Power BI Star Schema
============================================================
Stage 11 support. Every table built so far (Bronze through Gold) is
fact-style: building_id/equipment_id/meter_id sit as plain repeated
text in each row, with no separate table describing what those
entities actually ARE. That was a deliberate simplification made
early in this project ("just use what's in the raw data as-is").

This script builds proper dimension tables so Power BI can relate
multiple Gold fact tables through shared, clean lookup tables instead
of joining directly on repeated text columns:

    dim_building   -> one row per building. equipment_count and
                      meter_count are pulled from the actual generator
                      config (EQUIPMENT_BY_BUILDING / METERS_BY_BUILDING
                      in simulate_sensors.py), not invented. A couple of
                      descriptive fields (building_name, building_type)
                      are added purely for portfolio/demo readability -
                      the source data has no real facility metadata to
                      draw from, so these are clearly illustrative.
    dim_equipment  -> one row per AHU, linked to its building.
    dim_meter      -> one row per utility meter, linked to its building.
    dim_date       -> a full calendar year, the standard Power BI
                      best-practice table for proper time intelligence
                      (week/month/quarter comparisons) that a raw date
                      string column can't give you on its own.
    dim_alarm_type -> the 3 alarm types fault_detection.py can produce,
                      with a plain-language description.

These are small, cheap, effectively-static tables - full overwrite
every run is fine, no incremental/watermark complexity needed here.

Env vars required:
    HF_ACCESS_KEY_ID
    HF_SECRET_ACCESS_KEY
    HF_NAMESPACE
    HF_BUCKET_NAME
"""

import os
import sys
import time
from datetime import date, timedelta

import boto3
import pandas as pd
from botocore.client import Config
from deltalake import write_deltalake
from deltalake.exceptions import DeltaError

# ------------------------------------------------------------------
# Ground truth from the generator (generators/simulate_sensors.py) -
# not invented. Kept here as a literal copy since dimension-building
# doesn't need to import the generator module itself.
# ------------------------------------------------------------------
EQUIPMENT_BY_BUILDING = {
    "B1": ["AHU-B1-01", "AHU-B1-02", "AHU-B1-03"],
    "B2": ["AHU-B2-01", "AHU-B2-02", "AHU-B2-03", "AHU-B2-04"],
    "B3": ["AHU-B3-01", "AHU-B3-02"],
    "B4": ["AHU-B4-01", "AHU-B4-02", "AHU-B4-03", "AHU-B4-04", "AHU-B4-05"],
    "B5": ["AHU-B5-01", "AHU-B5-02", "AHU-B5-03"],
}

METERS_BY_BUILDING = {
    b: [(f"MTR-{b}-ELEC", "Electricity"), (f"MTR-{b}-CHW", "Chilled Water"), (f"MTR-{b}-GAS", "Gas")]
    for b in EQUIPMENT_BY_BUILDING
}

# Illustrative only - the source data has no real facility metadata.
# Clearly labeled as such in the docstring above and worth saying out
# loud if anyone asks about this table.
BUILDING_TYPES = {"B1": "Office", "B2": "Retail", "B3": "Warehouse", "B4": "Office", "B5": "Mixed-Use"}


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


def write_dimension_table(s3, bucket: str, table_name: str, df: pd.DataFrame,
                           storage_options: dict, max_attempts: int = 4) -> None:
    """No partitioning - these are small enough (single digits to low
    thousands of rows) that partitioning would create more overhead
    than it saves. Full overwrite every run; retry then self-heal,
    same pattern as everywhere else in this pipeline."""
    table_uri = f"s3://{bucket}/dimensions/{table_name}"
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            write_deltalake(table_uri, df, mode="overwrite", storage_options=storage_options)
            return
        except DeltaError as e:
            last_error = e
            if attempt < max_attempts:
                wait = 2 ** attempt
                print(f"  write attempt {attempt}/{max_attempts} failed ({e}), retrying in {wait}s...")
                time.sleep(wait)

    print(f"  all {max_attempts} write attempts failed ({last_error}). "
          f"Self-healing: wiping dimensions/{table_name} and recreating fresh...")
    deleted = delete_prefix_objects(s3, bucket, f"dimensions/{table_name}/")
    print(f"  deleted {deleted} object(s), retrying write once more...")
    write_deltalake(table_uri, df, mode="overwrite", storage_options=storage_options)
    print(f"  self-heal succeeded - dimensions/{table_name} recreated cleanly.")


# ------------------------------------------------------------------
# Dimension builders
# ------------------------------------------------------------------
def build_dim_building() -> pd.DataFrame:
    rows = []
    for building_id, equipment_list in EQUIPMENT_BY_BUILDING.items():
        rows.append({
            "building_id": building_id,
            "building_name": f"Building {building_id[1:]}",
            "building_type": BUILDING_TYPES[building_id],
            "equipment_count": len(equipment_list),
            "meter_count": len(METERS_BY_BUILDING[building_id]),
        })
    return pd.DataFrame(rows)


def build_dim_equipment() -> pd.DataFrame:
    rows = []
    for building_id, equipment_list in EQUIPMENT_BY_BUILDING.items():
        for equipment_id in equipment_list:
            equipment_number = int(equipment_id.split("-")[-1])
            rows.append({
                "equipment_id": equipment_id,
                "building_id": building_id,
                "equipment_type": "AHU",
                "equipment_number": equipment_number,
            })
    return pd.DataFrame(rows)


def build_dim_meter() -> pd.DataFrame:
    rows = []
    for building_id, meters in METERS_BY_BUILDING.items():
        for meter_id, utility_type in meters:
            rows.append({
                "meter_id": meter_id,
                "building_id": building_id,
                "utility_type": utility_type,
            })
    return pd.DataFrame(rows)


def build_dim_date(year: int) -> pd.DataFrame:
    start = date(year, 1, 1)
    end = date(year, 12, 31)
    rows = []
    d = start
    while d <= end:
        rows.append({
            "date_key": d.isoformat(),
            "year": d.year,
            "month": d.month,
            "month_name": d.strftime("%B"),
            "day": d.day,
            "day_of_week": d.weekday(),  # 0=Monday
            "day_name": d.strftime("%A"),
            "is_weekend": d.weekday() >= 5,
            "week_of_year": d.isocalendar()[1],
            "quarter": (d.month - 1) // 3 + 1,
        })
        d += timedelta(days=1)
    return pd.DataFrame(rows)


def build_dim_alarm_type() -> pd.DataFrame:
    return pd.DataFrame([
        {"alarm_type": "overheating", "description": "Supply air or heating coil temperature "
         "operationally too hot, even though physically plausible", "category": "Thermal"},
        {"alarm_type": "coolant_leak", "description": "Fan actively running but cooling coil "
         "temperature too high for that to be working properly", "category": "Refrigerant"},
        {"alarm_type": "operational_fault", "description": "Equipment reported its own fault "
         "code / non-operational status", "category": "Equipment-Reported"},
    ])


DIMENSIONS = {
    "dim_building": build_dim_building,
    "dim_equipment": build_dim_equipment,
    "dim_meter": build_dim_meter,
    "dim_alarm_type": build_dim_alarm_type,
}


def main():
    bucket = os.environ["HF_BUCKET_NAME"]
    s3 = get_s3_client()
    storage_options = deltalake_storage_options()

    results = {}
    for name, builder in DIMENSIONS.items():
        df = builder()
        print(f"{name}: {len(df)} row(s)")
        write_dimension_table(s3, bucket, name, df, storage_options)
        results[name] = len(df)

    dim_date_df = build_dim_date(2026)
    print(f"dim_date: {len(dim_date_df)} row(s)")
    write_dimension_table(s3, bucket, "dim_date", dim_date_df, storage_options)
    results["dim_date"] = len(dim_date_df)

    print("\n=== Summary ===")
    for name, count in results.items():
        print(f"{name}: {count} rows")

    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            for name, count in results.items():
                f.write(f"{name}_rows={count}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyError as e:
        print(f"::error::Missing required environment variable: {e}", file=sys.stderr)
        sys.exit(1)