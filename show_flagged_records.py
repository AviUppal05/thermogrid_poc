#!/usr/bin/env python3
"""
ThermoGrid — Show Flagged Silver Records
===========================================
Reads the full Silver Delta table (across all partitions, not just one
Parquet file) and shows the rows that got flagged warning/critical,
plus a breakdown by issue type.

Usage:
    python show_flagged_records.py hvac_telemetry
    python show_flagged_records.py utility_meter --limit 50
    python show_flagged_records.py hvac_telemetry --severity critical

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


def deltalake_storage_options():
    return {
        "endpoint_url": f"https://s3.hf.co/{os.environ['HF_NAMESPACE']}",
        "AWS_ACCESS_KEY_ID": os.environ["HF_ACCESS_KEY_ID"],
        "AWS_SECRET_ACCESS_KEY": os.environ["HF_SECRET_ACCESS_KEY"],
        "AWS_REGION": "us-east-1",
        "AWS_S3_ALLOW_UNSAFE_RENAME": "true",
    }


def main():
    parser = argparse.ArgumentParser(description="Show flagged records from a Silver table")
    parser.add_argument("dataset", choices=["hvac_telemetry", "utility_meter"])
    parser.add_argument("--severity", choices=["warning", "critical", "any"], default="any",
                         help="Which severity to show (default: any non-clean)")
    parser.add_argument("--limit", type=int, default=25, help="Max rows to print (default: 25)")
    args = parser.parse_args()

    bucket = os.environ["HF_BUCKET_NAME"]
    table_uri = f"s3://{bucket}/silver/{args.dataset}"
    storage_options = deltalake_storage_options()

    table = DeltaTable(table_uri, storage_options=storage_options)
    df = table.to_pandas()
    print(f"Loaded {len(df)} total rows from silver/{args.dataset} (table version {table.version()})\n")

    if args.severity == "any":
        flagged = df[df["data_quality_flag"] != "clean"]
    else:
        flagged = df[df["data_quality_flag"] == args.severity]

    print(f"{len(flagged)} row(s) match severity={args.severity}\n")

    if flagged.empty:
        return

    print("Breakdown by issue type:")
    issue_counts = flagged["data_quality_issues"].str.split(";").explode().value_counts()
    for issue, count in issue_counts.items():
        print(f"  {issue}: {count}")

    print(f"\nBreakdown by partition (building_id/dt):")
    print(flagged.groupby(["building_id", "dt"]).size().sort_values(ascending=False).head(10))

    print(f"\nFirst {min(args.limit, len(flagged))} flagged row(s):")
    display_cols = ["timestamp", "building_id", "data_quality_flag", "data_quality_issues"]
    id_col = "equipment_id" if args.dataset == "hvac_telemetry" else "meter_id"
    display_cols.insert(2, id_col)
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.width", None)
    print(flagged[display_cols].head(args.limit).to_string(index=False))


if __name__ == "__main__":
    main()