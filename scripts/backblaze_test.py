"""
example_b2_roundtrip_with_list.py

Uploads a CSV to Backblaze B2 (S3-compatible),
lists objects under a prefix,
downloads the file back,
and verifies integrity.

Requires:
  helpers.py (in same directory)
  pip install boto3 botocore

Env vars:
  B2_S3_ENDPOINT
  B2_S3_ACCESS_KEY_ID
  B2_S3_SECRET_ACCESS_KEY
  STRAND_BUCKET
"""

from __future__ import annotations

import csv
import pathlib
import tempfile
import hashlib

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.helpers import (
    B2Config,
    make_s3_client,
    join_key,
    upload_file,
    download_file,
    list_objects,
)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main() -> None:
    # ---- Connect to Backblaze ----
    cfg = B2Config.from_env(bucket_env="STRAND_BUCKET")
    s3 = make_s3_client(cfg)
    bucket = cfg.bucket or "strand"

    # Store under a scratch prefix
    prefix = "scratch/"
    key = join_key(prefix, "example_roundtrip.csv")

    with tempfile.TemporaryDirectory() as td:
        td_path = pathlib.Path(td)

        local_csv = td_path / "example_roundtrip.csv"
        downloaded_csv = td_path / "downloaded_example_roundtrip.csv"

        # ---- Create local CSV ----
        rows = [
            {"id": "1", "name": "alpha", "value": "10"},
            {"id": "2", "name": "beta", "value": "20"},
            {"id": "3", "name": "gamma", "value": "30"},
        ]

        with local_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "name", "value"])
            writer.writeheader()
            writer.writerows(rows)

        print(f"Created local file: {local_csv}")

        # ---- Upload ----
        upload_file(
            s3=s3,
            bucket=bucket,
            local_path=local_csv,
            key=key,
            content_type="text/csv; charset=utf-8",
        )

        print(f"Uploaded to s3://{bucket}/{key}")

        # ---- List objects under prefix ----
        print("\nObjects under prefix 'scratch/':")
        for obj in list_objects(s3, bucket, prefix=prefix, recursive=True):
            print(f"  - {obj['Key']} ({obj['Size']} bytes)")

        # ---- Download ----
        download_file(
            s3=s3,
            bucket=bucket,
            key=key,
            local_path=downloaded_csv,
            overwrite=True,
        )

        print(f"\nDownloaded to: {downloaded_csv}")

        # ---- Verify ----
        original_hash = sha256_bytes(local_csv.read_bytes())
        downloaded_hash = sha256_bytes(downloaded_csv.read_bytes())

        print("Original SHA256  :", original_hash)
        print("Downloaded SHA256:", downloaded_hash)

        if original_hash != downloaded_hash:
            raise RuntimeError("❌ File mismatch after roundtrip.")

        print("\n✅ Success: Upload, list, download, and verification complete.")


if __name__ == "__main__":
    main()