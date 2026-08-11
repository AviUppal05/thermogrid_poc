#!/usr/bin/env python3
"""
ThermoGrid — Manual Bronze Vacuum
====================================
Deletes old, no-longer-referenced data files from a Bronze Delta table
(the ones left behind after OPTIMIZE compacts data into new files).

This is deliberately a separate, manually-run script rather than part
of the automatic compact-bronze workflow. VACUUM deletes files, and
running it immediately after a write against S3-compatible storage
that doesn't guarantee strict read-after-write consistency risks
deleting a file that's still genuinely part of the current snapshot.
Run this by hand occasionally once the table has been stable for a
while (default retention is 7 days, i.e. only removes files that
became orphaned more than a week ago - comfortably outside any
consistency-lag window).

Usage:
    python vacuum_bronze.py hvac_telemetry              # dry run
    python vacuum_bronze.py hvac_telemetry --yes         # actually delete
    python vacuum_bronze.py utility_meter --yes --retention-hours 24

Env vars required:
    HF_ACCESS_KEY_ID
    HF_SECRET_ACCESS_KEY
    HF_NAMESPACE
    HF_BUCKET_NAME
"""

import argparse
import os
import sys

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
    parser = argparse.ArgumentParser(description="Manually vacuum a Bronze Delta table")
    parser.add_argument("dataset", choices=["hvac_telemetry", "utility_meter"])
    parser.add_argument("--yes", action="store_true",
                         help="Actually delete. Without this flag, only lists what would be removed.")
    parser.add_argument("--retention-hours", type=int, default=168,
                         help="Only remove files orphaned longer than this (default: 168 = 7 days)")
    args = parser.parse_args()

    bucket = os.environ["HF_BUCKET_NAME"]
    table_uri = f"s3://{bucket}/bronze/{args.dataset}"
    storage_options = deltalake_storage_options()

    table = DeltaTable(table_uri, storage_options=storage_options)
    print(f"Table: {table_uri}")
    print(f"Current version: {table.version()}")

    candidates = table.vacuum(
        retention_hours=args.retention_hours,
        dry_run=True,
        enforce_retention_duration=True,
    )
    if not candidates:
        print("No orphaned files old enough to remove. Nothing to do.")
        return

    print(f"\n{len(candidates)} file(s) eligible for removal (older than {args.retention_hours}h):")
    for f in candidates[:20]:
        print(f"  {f}")
    if len(candidates) > 20:
        print(f"  ... and {len(candidates) - 20} more")

    if not args.yes:
        print(f"\nDry run only. Re-run with --yes to actually delete these {len(candidates)} file(s).")
        return

    print(f"\nDeleting {len(candidates)} file(s)...")
    table.vacuum(
        retention_hours=args.retention_hours,
        dry_run=False,
        enforce_retention_duration=True,
    )
    print("Done.")


if __name__ == "__main__":
    try:
        main()
    except KeyError as e:
        print(f"::error::Missing required environment variable: {e}", file=sys.stderr)
        sys.exit(1)