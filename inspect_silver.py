#!/usr/bin/env python3
"""
ThermoGrid — Inspect Silver KPI Columns
===========================================
Reads the full Silver Delta table (all partitions, not a single
Parquet file) and shows sample rows grouped by data_quality_flag, so
you can confirm KPI columns are populated for clean/warning rows and
null for critical rows.

Usage:
    python inspect_silver.py hvac_telemetry
    python inspect_silver.py utility_meter
    python inspect_silver.py hvac_telemetry --limit 10

Env vars required:
    HF_ACCESS_KEY_ID
    HF_SECRET_ACCESS_KEY
    HF_NAMESPACE
    HF_BUCKET_NAME
"""

import argparse
import os

import pandas as pd
from deltalake import DeltaTable

KPI_COLUMNS = {
    "hvac_telemetry": ["temp_differential", "operating_mode", "filter_health_status"],
    "utility_meter": ["consumption_variance_pct", "is_over_consuming", "estimated_cost",
                       "load_factor", "time_of_use_period"],
}

ID_COLUMN = {"hvac_telemetry": "equipment_id", "utility_meter": "meter_id"}


def deltalake_storage_options():
    return {
        "endpoint_url": f"https://s3.hf.co/{os.environ['HF_NAMESPACE']}",
        "AWS_ACCESS_KEY_ID": os.environ["HF_ACCESS_KEY_ID"],
        "AWS_SECRET_ACCESS_KEY": os.environ["HF_SECRET_ACCESS_KEY"],
        "AWS_REGION": "us-east-1",
        "AWS_S3_ALLOW_UNSAFE_RENAME": "true",
    }


def main():
    parser = argparse.ArgumentParser(description="Inspect Silver KPI columns by data quality flag")
    parser.add_argument("dataset", choices=["hvac_telemetry", "utility_meter"])
    parser.add_argument("--limit", type=int, default=5, help="Rows to show per flag category (default: 5)")
    args = parser.parse_args()

    bucket = os.environ["HF_BUCKET_NAME"]
    table_uri = f"s3://{bucket}/silver/{args.dataset}"
    storage_options = deltalake_storage_options()

    table = DeltaTable(table_uri, storage_options=storage_options)
    df = table.to_pandas()
    print(f"Loaded {len(df)} total rows from silver/{args.dataset} (table version {table.version()})\n")

    kpi_cols = KPI_COLUMNS[args.dataset]
    id_col = ID_COLUMN[args.dataset]
    display_cols = [id_col, "data_quality_flag"] + kpi_cols

    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.width", None)

    for flag in ["clean", "warning", "critical"]:
        subset = df[df["data_quality_flag"] == flag]
        print(f"--- {flag} ({len(subset)} row(s)) ---")
        if subset.empty:
            print("  (none)\n")
            continue

        # Quick check: are the KPI columns null (critical) or populated (clean/warning)?
        null_counts = subset[kpi_cols].isna().sum()
        print(f"  null counts per KPI column: {null_counts.to_dict()}")
        if flag == "critical":
            all_null = (null_counts == len(subset)).all()
            print(f"  all KPI columns fully null for critical rows: {all_null}")
        else:
            any_populated = (null_counts < len(subset)).any()
            print(f"  at least some KPI values populated: {any_populated}")

        print(subset[display_cols].head(args.limit).to_string(index=False))
        print()


if __name__ == "__main__":
    main()