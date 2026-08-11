#!/usr/bin/env python3
"""
ThermoGrid — Delete a Prefix from the HF Bucket
=================================================
Deletes every object under a given prefix (e.g. "bronze/") in the
Hugging Face Storage Bucket. Defaults to a dry run that only lists
what *would* be deleted - pass --yes to actually delete.

Usage:
    python delete_prefix.py bronze/                # dry run, lists only
    python delete_prefix.py bronze/ --yes           # actually deletes

Env vars required:
    HF_ACCESS_KEY_ID
    HF_SECRET_ACCESS_KEY
    HF_NAMESPACE
    HF_BUCKET_NAME
"""

import argparse
import os
import sys

import boto3
from botocore.client import Config


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


def list_all_keys(s3, bucket: str, prefix: str) -> list[str]:
    keys = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return keys


def delete_keys(s3, bucket: str, keys: list[str]) -> int:
    """Deletes in batches of 1000 (the S3 DeleteObjects limit)."""
    deleted = 0
    for i in range(0, len(keys), 1000):
        batch = keys[i:i + 1000]
        resp = s3.delete_objects(
            Bucket=bucket,
            Delete={"Objects": [{"Key": k} for k in batch], "Quiet": True},
        )
        errors = resp.get("Errors", [])
        if errors:
            for e in errors:
                print(f"  FAILED to delete {e['Key']}: {e.get('Message')}")
        deleted += len(batch) - len(errors)
    return deleted


def main():
    parser = argparse.ArgumentParser(description="Delete all objects under a prefix in the HF bucket")
    parser.add_argument("prefix", help="Prefix to delete, e.g. 'bronze/' (trailing slash recommended)")
    parser.add_argument("--yes", action="store_true",
                         help="Actually delete. Without this flag, only lists what would be deleted.")
    args = parser.parse_args()

    prefix = args.prefix if args.prefix.endswith("/") else args.prefix + "/"

    bucket = os.environ["HF_BUCKET_NAME"]
    s3 = get_s3_client()

    print(f"Listing objects under s3://{bucket}/{prefix} ...")
    keys = list_all_keys(s3, bucket, prefix)

    if not keys:
        print("No objects found under that prefix. Nothing to do.")
        return

    print(f"Found {len(keys)} objects:")
    for k in keys[:20]:
        print(f"  {k}")
    if len(keys) > 20:
        print(f"  ... and {len(keys) - 20} more")

    if not args.yes:
        print(f"\nDry run only - {len(keys)} objects would be deleted.")
        print("Re-run with --yes to actually delete them.")
        return

    print(f"\nDeleting {len(keys)} objects...")
    deleted = delete_keys(s3, bucket, keys)
    print(f"Done. Deleted {deleted}/{len(keys)} objects under {prefix}")


if __name__ == "__main__":
    try:
        main()
    except KeyError as e:
        print(f"::error::Missing required environment variable: {e}", file=sys.stderr)
        sys.exit(1)