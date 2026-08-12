#!/usr/bin/env python3
"""
ThermoGrid IoT Sensor Data Generator
=====================================
Generates two synthetic datasets that mimic a real Azure/IoT ingestion feed:

  1. HVAC telemetry   -> minute-level, one row per AHU per simulated minute
  2. Utility meter     -> hourly-level, one row per meter per completed hour

Both are written as Parquet batches and uploaded to a Hugging Face
Storage Bucket using the S3-compatible API (https://s3.hf.co/<namespace>).

Env vars required:
    HF_ACCESS_KEY_ID
    HF_SECRET_ACCESS_KEY
    HF_NAMESPACE
    HF_BUCKET_NAME
"""

import argparse
import io
import os
import random
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone

import boto3
import pandas as pd
from botocore.client import Config
from botocore.exceptions import ClientError

# ------------------------------------------------------------------
# Portfolio / equipment dimension
# ------------------------------------------------------------------
BUILDINGS = ["B1", "B2", "B3", "B4", "B5"]

EQUIPMENT_BY_BUILDING = {
    "B1": ["AHU-B1-01", "AHU-B1-02", "AHU-B1-03"],
    "B2": ["AHU-B2-01", "AHU-B2-02", "AHU-B2-03", "AHU-B2-04"],
    "B3": ["AHU-B3-01", "AHU-B3-02"],
    "B4": ["AHU-B4-01", "AHU-B4-02", "AHU-B4-03", "AHU-B4-04", "AHU-B4-05"],
    "B5": ["AHU-B5-01", "AHU-B5-02", "AHU-B5-03"],
}

METERS_BY_BUILDING = {
    b: [(f"MTR-{b}-ELEC", "Electricity"),
        (f"MTR-{b}-CHW", "Chilled Water"),
        (f"MTR-{b}-GAS", "Gas")]
    for b in BUILDINGS
}

FAULT_CODES = ["HC-412", "CC-118", "FAN-207", "DP-305", "SENS-004"]
FAULT_PROBABILITY = 0.015      # per equipment per tick
WARNING_PROBABILITY = 0.03     # per equipment per tick

# Sensor GLITCH is intentionally separate from equipment FAULTS above.
# A fault_code means the equipment is genuinely malfunctioning - that's
# real operational signal and should read as physically plausible data
# (a stalled fan really does report near-zero speed). A glitch means the
# SENSOR OR TRANSMISSION corrupted the reading itself - a stuck ADC, a
# bit-flip, a dropped/garbled packet - producing a value that shouldn't
# be trusted regardless of what the equipment is actually doing. This is
# what Silver's data_quality_flag is designed to catch, independent of
# fault_code/status_flag.
SENSOR_GLITCH_PROBABILITY = 0.04   # per equipment per tick - hard/impossible values -> critical

# SOFT_ANOMALY is different again from both faults and hard glitches: it
# nudges values into an implausible RELATIONSHIP while every individual
# field stays inside its normal physical range (e.g. supply air a couple
# degrees above return air - odd, but each number on its own looks fine).
# This is what actually exercises the warning tier - hard glitches always
# also break a critical check, so a row with one never surfaces as
# "warning" (critical takes priority once any critical check fails).
SOFT_ANOMALY_PROBABILITY = 0.04   # per equipment per tick - plausible-but-odd -> warning

# in-memory equipment state so readings drift smoothly instead of jumping
_equipment_state: dict[str, dict] = {}


def apply_hvac_soft_anomaly(row: dict) -> dict:
    """Nudges supply_air_temp a few degrees above return_air_temp while
    keeping both within their normal, physically valid ranges - odd
    enough to warrant a warning, not implausible enough to be critical."""
    if random.random() >= SOFT_ANOMALY_PROBABILITY:
        return row
    return_temp = row["return_air_temp"]
    row["supply_air_temp"] = round(min(return_temp + random.uniform(1, 6), 99.0), 1)
    return row


def apply_utility_soft_anomaly(row: dict) -> dict:
    """Two independent plausible-but-odd relationships: peak demand
    dipping just under the hour's average consumption, or a missing
    baseline (expected_consumption_kwh = 0). Both fields stay positive
    and in-range individually."""
    if random.random() >= SOFT_ANOMALY_PROBABILITY:
        return row
    if random.random() < 0.5:
        row["peak_demand_kw"] = round(row["actual_consumption_kwh"] * random.uniform(0.7, 0.95), 1)
    else:
        row["expected_consumption_kwh"] = 0.0
    return row


def apply_hvac_glitch(row: dict) -> dict:
    """With low probability, corrupt one field into a physically
    implausible value - simulating a bad sensor/transmission rather than
    a genuine equipment fault. fault_code/status_flag are left untouched."""
    if random.random() >= SENSOR_GLITCH_PROBABILITY:
        return apply_hvac_soft_anomaly(row)

    glitch_type = random.choice([
        "extreme_temp", "negative_pressure", "fan_out_of_range", "missing_equipment_id",
    ])
    if glitch_type == "extreme_temp":
        field = random.choice(["supply_air_temp", "return_air_temp", "outdoor_air_temp"])
        row[field] = round(random.choice([random.uniform(-200, -50), random.uniform(300, 500)]), 1)
    elif glitch_type == "negative_pressure":
        row["duct_pressure"] = round(-abs(random.uniform(1, 20)), 2)
    elif glitch_type == "fan_out_of_range":
        row["fan_speed_pct"] = round(random.choice([random.uniform(-50, -1), random.uniform(150, 300)]), 1)
    elif glitch_type == "missing_equipment_id":
        row["equipment_id"] = None
    return row


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def outdoor_air_temp(ts: datetime) -> float:
    """Synthetic diurnal outdoor temp curve (replaces a NOAA feed)."""
    hour_frac = ts.hour + ts.minute / 60.0
    seasonal_base = 68 + 12 * math_sin((ts.timetuple().tm_yday / 365.0) * 2 * 3.14159 - 1.4)
    daily_swing = 8 * math_sin((hour_frac - 9) / 24.0 * 2 * 3.14159)
    return round(seasonal_base + daily_swing + random.uniform(-1.0, 1.0), 1)


def math_sin(x: float) -> float:
    import math
    return math.sin(x)


def get_equipment_state(equipment_id: str) -> dict:
    if equipment_id not in _equipment_state:
        _equipment_state[equipment_id] = {
            "supply_air_temp": random.uniform(58.0, 62.0),
            "return_air_temp": random.uniform(72.0, 75.0),
            "fan_speed_pct": random.uniform(40.0, 70.0),
            "filter_pressure_drop": random.uniform(0.15, 0.35),
            "active_fault": None,
            "fault_ticks_remaining": 0,
        }
    return _equipment_state[equipment_id]


def drift(value: float, lo: float, hi: float, step: float) -> float:
    value += random.uniform(-step, step)
    return max(lo, min(hi, value))


# ------------------------------------------------------------------
# Generators
# ------------------------------------------------------------------
def generate_hvac_reading(ts: datetime, building_id: str, equipment_id: str) -> dict:
    state = get_equipment_state(equipment_id)
    oat = outdoor_air_temp(ts)

    # resolve/advance any active fault
    if state["fault_ticks_remaining"] > 0:
        state["fault_ticks_remaining"] -= 1
        if state["fault_ticks_remaining"] == 0:
            state["active_fault"] = None
    elif random.random() < FAULT_PROBABILITY:
        state["active_fault"] = random.choice(FAULT_CODES)
        state["fault_ticks_remaining"] = random.randint(5, 20)

    fault_code = state["active_fault"]

    if fault_code:
        status_flag = "Fault"
    elif random.random() < WARNING_PROBABILITY:
        status_flag = "Warning"
    else:
        status_flag = "Operational"

    # normal drift
    state["fan_speed_pct"] = drift(state["fan_speed_pct"], 20.0, 100.0, 3.0)
    state["supply_air_temp"] = drift(state["supply_air_temp"], 54.0, 66.0, 0.6)
    state["return_air_temp"] = drift(state["return_air_temp"], 70.0, 78.0, 0.4)
    state["filter_pressure_drop"] = drift(state["filter_pressure_drop"], 0.10, 0.60, 0.01)

    supply_air_temp = state["supply_air_temp"]
    return_air_temp = state["return_air_temp"]
    fan_speed_pct = state["fan_speed_pct"]
    filter_pressure_drop = state["filter_pressure_drop"]

    # fault-specific distortion so downstream anomaly detection has signal
    if fault_code == "HC-412":       # heating coil fault
        heating_coil_temp = random.uniform(85.0, 95.0)
        cooling_coil_temp = random.uniform(44.0, 50.0)
    elif fault_code == "CC-118":     # cooling coil fault
        cooling_coil_temp = random.uniform(58.0, 68.0)
        heating_coil_temp = random.uniform(100.0, 110.0)
    else:
        cooling_coil_temp = round(random.uniform(42.0, 50.0), 1)
        heating_coil_temp = round(random.uniform(100.0, 118.0), 1)

    if fault_code == "DP-305":
        duct_pressure = round(random.uniform(3.5, 5.0), 2)
    else:
        duct_pressure = round(random.uniform(0.8, 2.2), 2)

    if fault_code == "FAN-207":
        fan_speed_pct = round(random.uniform(0.0, 10.0), 1)
        status_flag = "Fault"

    if status_flag == "Fault" and fault_code == "FAN-207":
        # a stalled fan often reads as offline
        if random.random() < 0.3:
            status_flag = "Offline"

    return apply_hvac_glitch({
        "timestamp": ts.strftime("%Y-%m-%d %H:%M:00"),
        "equipment_id": equipment_id,
        "building_id": building_id,
        "supply_air_temp": round(supply_air_temp, 1),
        "return_air_temp": round(return_air_temp, 1),
        "fan_speed_pct": round(fan_speed_pct, 1),
        "duct_pressure": duct_pressure,
        "filter_pressure_drop": round(filter_pressure_drop, 2),
        "cooling_coil_temp": round(cooling_coil_temp, 1),
        "heating_coil_temp": round(heating_coil_temp, 1),
        "outdoor_air_temp": oat,
        "fault_code": fault_code,
        "status_flag": status_flag,
    })


def tou_rate(hour: int, utility_type: str) -> float:
    """Simple time-of-use pricing curve, $/kWh (or $/therm for gas)."""
    if utility_type == "Gas":
        return round(random.uniform(0.9, 1.1), 3)
    is_peak = 14 <= hour <= 19
    is_shoulder = hour in (10, 11, 12, 13, 20, 21)
    if is_peak:
        base = 0.34
    elif is_shoulder:
        base = 0.22
    else:
        base = 0.11
    return round(base + random.uniform(-0.01, 0.01), 3)


def apply_utility_glitch(row: dict) -> dict:
    """Same idea as apply_hvac_glitch - simulates a bad meter reading or
    transmission error, independent of the actual consumption pattern."""
    if random.random() >= SENSOR_GLITCH_PROBABILITY:
        return apply_utility_soft_anomaly(row)

    glitch_type = random.choice(["negative_consumption", "zero_cost_rate", "negative_peak_demand"])
    if glitch_type == "negative_consumption":
        row["actual_consumption_kwh"] = round(-abs(random.uniform(10, 200)), 1)
    elif glitch_type == "zero_cost_rate":
        row["cost_rate"] = 0.0
    elif glitch_type == "negative_peak_demand":
        row["peak_demand_kw"] = round(-abs(random.uniform(10, 200)), 1)
    return row


def generate_utility_reading(ts: datetime, building_id: str, meter_id: str, utility_type: str) -> dict:
    hour = ts.hour
    weekday = ts.weekday() < 5

    # occupancy-driven daily load shape (peaks mid-afternoon on weekdays)
    occupancy_factor = 0.35 + 0.65 * math_sin(max(0.0, (hour - 6) / 16.0 * 3.14159)) if 6 <= hour <= 22 else 0.25
    weekday_factor = 1.0 if weekday else 0.6

    base_load = {"Electricity": 180.0, "Chilled Water": 90.0, "Gas": 40.0}[utility_type]
    expected_consumption_kwh = round(base_load * occupancy_factor * weekday_factor, 1)

    # inject variance so Dashboard 2 has something to flag
    variance_multiplier = random.choices(
        [random.uniform(0.85, 1.05), random.uniform(1.15, 1.45)],
        weights=[0.85, 0.15],
    )[0]
    actual_consumption_kwh = round(expected_consumption_kwh * variance_multiplier, 1)
    peak_demand_kw = round(actual_consumption_kwh * random.uniform(1.05, 1.35), 1)

    return apply_utility_glitch({
        "timestamp": ts.strftime("%Y-%m-%d %H:00:00"),
        "meter_id": meter_id,
        "building_id": building_id,
        "utility_type": utility_type,
        "actual_consumption_kwh": actual_consumption_kwh,
        "expected_consumption_kwh": expected_consumption_kwh,
        "peak_demand_kw": peak_demand_kw,
        "cost_rate": tou_rate(hour, utility_type),
    })


# ------------------------------------------------------------------
# HF Storage Bucket (S3-compatible) upload
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


def ensure_bucket_exists(namespace: str, bucket: str) -> None:
    """Create the Storage Bucket if it doesn't exist yet.

    Buckets are their own repo type on the Hub — referencing a name in
    HF_BUCKET_NAME does not create it. This uses the regular Hub API
    (HF_TOKEN), which is separate from the S3 access key/secret used
    for the actual uploads. If HF_TOKEN isn't set, this is skipped and
    you must create the bucket manually (web UI or `hf bucket create`)
    before running the script.
    """
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("  (HF_TOKEN not set - skipping auto-create; "
              "make sure the bucket already exists)")
        return
    try:
        from huggingface_hub import create_bucket
        create_bucket(f"{namespace}/{bucket}", token=token, exist_ok=True)
        print(f"  confirmed bucket exists: {namespace}/{bucket}")
    except Exception as e:
        print(f"  warning: could not verify/create bucket via Hub API: {e}")


def upload_batch(s3, bucket: str, dataset: str, building_id: str, dt: str, rows: list[dict]) -> str:
    df = pd.DataFrame(rows)
    buf = io.BytesIO()
    df.to_parquet(buf, index=False, engine="pyarrow")
    buf.seek(0)

    key = f"raw/{dataset}/building_id={building_id}/dt={dt}/{dataset}_{uuid.uuid4().hex}.parquet"
    s3.put_object(Bucket=bucket, Key=key, Body=buf.getvalue())
    return key


def flush_grouped(s3, bucket: str, dataset: str, rows: list[dict], batch_size: int) -> tuple[int, int]:
    """Group rows by building_id + date, then upload in batch_size chunks."""
    if not rows:
        return 0, 0

    df = pd.DataFrame(rows)
    df["_dt"] = df["timestamp"].str[:10]
    total_batches = 0

    for (building_id, dt), group in df.groupby(["building_id", "_dt"]):
        group = group.drop(columns=["_dt"])
        records = group.to_dict("records")
        for i in range(0, len(records), batch_size):
            chunk = records[i:i + batch_size]
            key = upload_batch(s3, bucket, dataset, building_id, dt, chunk)
            print(f"  uploaded {len(chunk):>4} rows -> s3://{bucket}/{key}")
            total_batches += 1

    return len(rows), total_batches


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="ThermoGrid sensor data generator")
    parser.add_argument("--duration-minutes", type=int, default=5,
                        help="Simulated minutes of HVAC telemetry to generate")
    parser.add_argument("--batch-size", type=int, default=500,
                        help="Max rows per uploaded Parquet file")
    parser.add_argument("--tick-seconds", type=float, default=30,
                        help="Real-world seconds to sleep between ticks")
    parser.add_argument("--hours-back", type=int, default=6,
                        help="How many completed hours of utility meter data to backfill")
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    namespace = os.environ["HF_NAMESPACE"]
    bucket = os.environ["HF_BUCKET_NAME"]
    ensure_bucket_exists(namespace, bucket)
    s3 = get_s3_client()

    # ---------------- HVAC telemetry (minute-level) ----------------
    print(f"Generating HVAC telemetry for {args.duration_minutes} simulated minute(s)...")
    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    hvac_rows = []
    for m in range(args.duration_minutes):
        ts = now - timedelta(minutes=(args.duration_minutes - 1 - m))
        for building_id, equipment_list in EQUIPMENT_BY_BUILDING.items():
            for equipment_id in equipment_list:
                hvac_rows.append(generate_hvac_reading(ts, building_id, equipment_id))
        if args.tick_seconds > 0 and m < args.duration_minutes - 1:
            time.sleep(args.tick_seconds)

    hvac_count, hvac_batches = flush_grouped(s3, bucket, "hvac_telemetry", hvac_rows, args.batch_size)

    # ---------------- Utility meter (hourly-level) ------------------
    print(f"Generating utility meter data for the last {args.hours_back} completed hour(s)...")
    current_hour = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    utility_rows = []
    for h in range(args.hours_back, 0, -1):
        ts = current_hour - timedelta(hours=h)
        for building_id, meters in METERS_BY_BUILDING.items():
            for meter_id, utility_type in meters:
                utility_rows.append(generate_utility_reading(ts, building_id, meter_id, utility_type))

    utility_count, utility_batches = flush_grouped(s3, bucket, "utility_meter", utility_rows, args.batch_size)

    total_readings = hvac_count + utility_count
    total_batches = hvac_batches + utility_batches

    print(f"\nDone. HVAC rows: {hvac_count} | Utility rows: {utility_count} | "
          f"Total batches: {total_batches}")

    # Expose outputs to the GitHub Actions step summary
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            f.write(f"total_readings={total_readings}\n")
            f.write(f"total_batches={total_batches}\n")
            f.write(f"hvac_readings={hvac_count}\n")
            f.write(f"utility_readings={utility_count}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyError as e:
        print(f"::error::Missing required environment variable: {e}", file=sys.stderr)
        sys.exit(1)
    except ClientError as e:
        print(f"::error::S3 upload failed: {e}", file=sys.stderr)
        sys.exit(1)