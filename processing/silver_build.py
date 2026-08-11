#!/usr/bin/env python3
"""
ThermoGrid — Bronze to Silver
================================
Reads the Bronze Delta tables, type-casts columns, and adds two
non-destructive data-quality columns:

    data_quality_flag   -> 'clean' | 'warning' | 'critical'
    data_quality_issues -> semicolon-joined list of failed checks, or ''

No rows are ever dropped. This is a deliberate design choice: a bad
sensor reading is still evidence something happened, and dropping it
would make troubleshooting equipment (or the pipeline itself) harder,
not easier. Downstream consumers (Gold, dashboards) decide for
themselves whether/how to filter on these flags.

Important distinction this script preserves rather than conflating:
    - fault_code / status_flag = 'Fault' or 'Warning' is OPERATIONAL
      SIGNAL - the equipment telling you something real. It is left
      completely untouched and is NOT treated as a data quality issue.
    - data_quality_flag is only about whether the READING ITSELF is
      trustworthy (physically plausible, structurally complete) -
      independent of whether the equipment is healthy.

No Spark, no Databricks - runs entirely on a GitHub Actions runner,
same as bronze_compact.py.

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

import pandas as pd
from deltalake import DeltaTable, write_deltalake
from deltalake.exceptions import DeltaError, TableNotFoundError


# ------------------------------------------------------------------
# Storage helpers
# ------------------------------------------------------------------
def deltalake_storage_options():
    return {
        "endpoint_url": f"https://s3.hf.co/{os.environ['HF_NAMESPACE']}",
        "AWS_ACCESS_KEY_ID": os.environ["HF_ACCESS_KEY_ID"],
        "AWS_SECRET_ACCESS_KEY": os.environ["HF_SECRET_ACCESS_KEY"],
        "AWS_REGION": "us-east-1",
        "AWS_S3_ALLOW_UNSAFE_RENAME": "true",
    }


def write_silver_table(table_uri: str, df: pd.DataFrame, storage_options: dict,
                        max_attempts: int = 4) -> None:
    """Same retry-with-backoff pattern used for Bronze - see bronze_compact.py
    for the full reasoning. HF's S3-compatible storage doesn't guarantee
    strict read-after-write consistency, so an occasional transient
    version mismatch is expected and worth retrying rather than failing
    the whole run over."""
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
                wait = 2 ** attempt
                print(f"  write attempt {attempt}/{max_attempts} failed "
                      f"({e}), retrying in {wait}s...")
                time.sleep(wait)
    raise last_error


# ------------------------------------------------------------------
# Type casting
# ------------------------------------------------------------------
def cast_common(df: pd.DataFrame) -> pd.DataFrame:
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["building_id"] = df["building_id"].astype("string")
    df["dt"] = df["dt"].astype("string")
    return df


NUMERIC_COLUMNS = {
    "hvac_telemetry": [
        "supply_air_temp", "return_air_temp", "fan_speed_pct", "duct_pressure",
        "filter_pressure_drop", "cooling_coil_temp", "heating_coil_temp", "outdoor_air_temp",
    ],
    "utility_meter": [
        "actual_consumption_kwh", "expected_consumption_kwh", "peak_demand_kw", "cost_rate",
    ],
}


def cast_numeric(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    for col in NUMERIC_COLUMNS[dataset_name]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ------------------------------------------------------------------
# Data quality checks
# ------------------------------------------------------------------
def apply_quality_checks(df: pd.DataFrame, checks: list[tuple]) -> pd.DataFrame:
    """checks: list of (name, severity, boolean_mask_series) where the mask
    is True for ROWS THAT FAIL the check."""
    issues = pd.Series([[] for _ in range(len(df))], index=df.index)
    severities = pd.Series(["clean"] * len(df), index=df.index)

    for name, severity, mask in checks:
        mask = mask.fillna(True)  # NaN inputs (e.g. from failed numeric cast) count as failing
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
        # logical consistency: supply air should normally be cooler than
        # return air when the unit is actively cooling - not a hard physical
        # law (heating mode flips it), so this is a warning, not critical
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
        # peak demand within the hour should generally be >= the average
        # power implied by that hour's total consumption - not impossible
        # to violate with sub-hourly metering quirks, so warning not critical
        ("peak_below_actual", "warning",
         df["peak_demand_kw"] < df["actual_consumption_kwh"]),
    ]


# ------------------------------------------------------------------
# Main per-dataset routine
# ------------------------------------------------------------------
def build_silver(dataset_name: str, bucket: str, storage_options: dict) -> dict:
    print(f"\n=== {dataset_name} ===")
    bronze_uri = f"s3://{bucket}/bronze/{dataset_name}"
    silver_uri = f"s3://{bucket}/silver/{dataset_name}"

    try:
        bronze_table = DeltaTable(bronze_uri, storage_options=storage_options)
    except TableNotFoundError:
        print(f"  no Bronze table found at {bronze_uri}, skipping")
        return {"rows": 0, "clean": 0, "warning": 0, "critical": 0}

    df = bronze_table.to_pandas()
    total_rows = len(df)
    print(f"  read {total_rows} rows from Bronze (version {bronze_table.version()})")

    df = cast_common(df)
    df = cast_numeric(df, dataset_name)

    checks = hvac_quality_checks(df) if dataset_name == "hvac_telemetry" else utility_quality_checks(df)
    df = apply_quality_checks(df, checks)

    counts = df["data_quality_flag"].value_counts().to_dict()
    clean = counts.get("clean", 0)
    warning = counts.get("warning", 0)
    critical = counts.get("critical", 0)
    print(f"  quality summary: {clean} clean, {warning} warning, {critical} critical "
          f"(all {total_rows} rows kept)")

    print(f"  writing {total_rows} rows to Silver in a single commit...")
    write_silver_table(silver_uri, df, storage_options)

    return {"rows": total_rows, "clean": clean, "warning": warning, "critical": critical}


def main():
    parser = argparse.ArgumentParser(description="Build Silver tables from Bronze")
    parser.add_argument("--datasets", nargs="+", choices=["hvac_telemetry", "utility_meter"],
                         default=["hvac_telemetry", "utility_meter"])
    args = parser.parse_args()

    bucket = os.environ["HF_BUCKET_NAME"]
    storage_options = deltalake_storage_options()

    results = {}
    for name in args.datasets:
        results[name] = build_silver(name, bucket, storage_options)

    print("\n=== Summary ===")
    total_rows = total_clean = total_warning = total_critical = 0
    for name, r in results.items():
        print(f"{name}: {r['rows']} rows ({r['clean']} clean, {r['warning']} warning, {r['critical']} critical)")
        total_rows += r["rows"]
        total_clean += r["clean"]
        total_warning += r["warning"]
        total_critical += r["critical"]

    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            f.write(f"total_rows={total_rows}\n")
            f.write(f"total_clean={total_clean}\n")
            f.write(f"total_warning={total_warning}\n")
            f.write(f"total_critical={total_critical}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyError as e:
        print(f"::error::Missing required environment variable: {e}", file=sys.stderr)
        sys.exit(1)