#!/usr/bin/env python3
"""
ThermoGrid — Delete a Single Object
=======================================
Deletes one exact object by key (not a prefix/folder) from the HF
bucket. Use this for things like a single manifest file, where
delete_prefix.py's folder-style matching doesn't apply.

Usage:
    python delete_object.py fault_detection/_manifests/hvac_telemetry_processed_run_ids.json

Env vars required:
    HF_ACCESS_KEY_ID
    HF_SECRET_ACCESS_KEY
    HF_NAMESPACE
    HF_BUCKET_NAME
"""

import argparse
import os

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError


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


def main():
    parser = argparse.ArgumentParser(description="Delete a single exact object key")
    parser.add_argument("key", help="Exact object key, e.g. fault_detection/_manifests/hvac_telemetry_processed_run_ids.json")
    args = parser.parse_args()

    bucket = os.environ["HF_BUCKET_NAME"]
    s3 = get_s3_client()

    try:
        s3.head_object(Bucket=bucket, Key=args.key)
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("404", "NoSuchKey"):
            print(f"Object not found: {args.key}")
            print("Nothing to delete.")
            return
        raise

    s3.delete_object(Bucket=bucket, Key=args.key)
    print(f"Deleted: {args.key}")


if __name__ == "__main__":
    main()