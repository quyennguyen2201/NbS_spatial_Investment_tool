"""
helpers.py — Backblaze B2 (S3-compatible) helper utilities for STRAND-style data lakes.

Design goals:
- Simple, safe wrappers around common S3 ops (list, upload, download, copy, delete).
- Works with Backblaze B2 S3 endpoints (endpoint_url required).
- Sensible defaults for research: immutable/versioned paths, manifests, and local caching.

Dependencies:
- boto3
- botocore

Recommended env vars:
- B2_S3_ENDPOINT           https://s3.us-west-002.backblazeb2.com
- B2_S3_ACCESS_KEY_ID (contact Greg Bodeker)
- B2_S3_SECRET_ACCESS_KEY (contact Greg Bodeker)
- STRAND_BUCKET            e.g. STRAND2
"""

from __future__ import annotations

import os
from dotenv import load_dotenv
import io
import json
import time
import hashlib
import logging
import pathlib
import datetime as _dt
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Union

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError, EndpointConnectionError, ConnectionClosedError
from boto3.s3.transfer import TransferConfig


# --------------------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------------------

logger = logging.getLogger("b2_helpers")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)


# --------------------------------------------------------------------------------------
# Config + Client
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class B2Config:
    endpoint_url: str
    access_key_id: str
    secret_access_key: str
    region: str = "us-west-002"
    bucket: Optional[str] = None

    @staticmethod
    def from_env(bucket_env: str = "STRAND_BUCKET") -> "B2Config":
        

        load_dotenv()
        endpoint = os.getenv("B2_S3_ENDPOINT", "").strip()
        ak = os.getenv("B2_S3_ACCESS_KEY_ID", "").strip()
        sk = os.getenv("B2_S3_SECRET_ACCESS_KEY", "").strip()
        region = os.getenv("B2_S3_REGION", "us-west-002").strip()
        bucket = os.getenv(bucket_env, "").strip() or None

        missing = [k for k, v in {
            "B2_S3_ENDPOINT": endpoint,
            "B2_S3_ACCESS_KEY_ID": ak,
            "B2_S3_SECRET_ACCESS_KEY": sk,
        }.items() if not v]
        if missing:
            raise ValueError(f"Missing env vars: {', '.join(missing)}")
        return B2Config(endpoint_url=endpoint, access_key_id=ak, secret_access_key=sk, region=region, bucket=bucket)


def make_s3_client(cfg: B2Config):
    """
    Creates a boto3 S3 client configured for Backblaze B2 S3.
    """
    # B2 is S3-compatible; path-style often avoids edge cases with custom endpoints.
    # (Virtual-hosted style can work too, but path-style is robust.)
    config = Config(
        region_name=cfg.region,
        s3={"addressing_style": "path"},
        retries={"max_attempts": 10, "mode": "standard"},
        signature_version="s3v4",
    )

    return boto3.client(
        "s3",
        endpoint_url=cfg.endpoint_url,
        aws_access_key_id=cfg.access_key_id,
        aws_secret_access_key=cfg.secret_access_key,
        config=config,
    )


# Multipart thresholds for big files (Parquet, GeoTIFF, etc.)
_DEFAULT_TRANSFER = TransferConfig(
    multipart_threshold=64 * 1024 * 1024,     # 64MB
    multipart_chunksize=64 * 1024 * 1024,     # 64MB
    max_concurrency=8,
    use_threads=True,
)


# --------------------------------------------------------------------------------------
# Small utilities
# --------------------------------------------------------------------------------------

def now_utc_compact() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_file(path: Union[str, pathlib.Path], chunk_size: int = 8 * 1024 * 1024) -> str:
    p = pathlib.Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def join_key(*parts: str) -> str:
    """
    Join S3 key parts with "/" and remove double slashes.
    """
    cleaned = []
    for part in parts:
        if part is None:
            continue
        s = str(part).strip().replace("\\", "/")
        s = s.strip("/")
        if s:
            cleaned.append(s)
    return "/".join(cleaned)


def is_probably_dir_key(key: str) -> bool:
    return key.endswith("/")


# --------------------------------------------------------------------------------------
# Retry wrapper (network-ish errors)
# --------------------------------------------------------------------------------------

_RETRYABLE = (EndpointConnectionError, ConnectionClosedError)


def _with_retries(fn, *args, max_tries: int = 5, base_sleep: float = 0.5, **kwargs):
    tries = 0
    while True:
        tries += 1
        try:
            return fn(*args, **kwargs)
        except _RETRYABLE as e:
            if tries >= max_tries:
                raise
            sleep = base_sleep * (2 ** (tries - 1))
            logger.warning("Retryable error (%s). Retry %d/%d after %.1fs", type(e).__name__, tries, max_tries, sleep)
            time.sleep(sleep)


# --------------------------------------------------------------------------------------
# Core S3 operations
# --------------------------------------------------------------------------------------

def list_objects(
    s3,
    bucket: str,
    prefix: str = "",
    recursive: bool = True,
    max_keys: int = 1000,
) -> Iterator[Dict[str, Any]]:
    """
    Yields objects under prefix. For huge prefixes, this paginates safely.
    """
    prefix = prefix.strip("/")
    if prefix and not prefix.endswith("/"):
        prefix += "/"

    continuation = None
    while True:
        kwargs = {"Bucket": bucket, "Prefix": prefix, "MaxKeys": max_keys}
        if continuation:
            kwargs["ContinuationToken"] = continuation

        resp = _with_retries(s3.list_objects_v2, **kwargs)
        for obj in resp.get("Contents", []) or []:
            key = obj["Key"]
            if not recursive:
                # Non-recursive: only immediate children under prefix
                rest = key[len(prefix):]
                if "/" in rest:
                    continue
            yield obj

        if resp.get("IsTruncated"):
            continuation = resp.get("NextContinuationToken")
        else:
            break


def object_exists(s3, bucket: str, key: str) -> bool:
    try:
        _with_retries(s3.head_object, Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in ("404", "NoSuchKey", "NotFound"):
            return False
        raise


def get_text(s3, bucket: str, key: str, encoding: str = "utf-8") -> str:
    resp = _with_retries(s3.get_object, Bucket=bucket, Key=key)
    return resp["Body"].read().decode(encoding)


def put_text(
    s3,
    bucket: str,
    key: str,
    text: str,
    content_type: str = "text/plain; charset=utf-8",
    cache_control: Optional[str] = None,
    metadata: Optional[Dict[str, str]] = None,
) -> None:
    extra = {"ContentType": content_type}
    if cache_control:
        extra["CacheControl"] = cache_control
    if metadata:
        extra["Metadata"] = {str(k): str(v) for k, v in metadata.items()}
    _with_retries(s3.put_object, Bucket=bucket, Key=key, Body=text.encode("utf-8"), **extra)


def get_json(s3, bucket: str, key: str) -> Any:
    return json.loads(get_text(s3, bucket, key))


def put_json(
    s3,
    bucket: str,
    key: str,
    data: Any,
    indent: int = 2,
    sort_keys: bool = True,
    metadata: Optional[Dict[str, str]] = None,
) -> None:
    text = json.dumps(data, indent=indent, sort_keys=sort_keys, ensure_ascii=False)
    put_text(
        s3,
        bucket=bucket,
        key=key,
        text=text,
        content_type="application/json; charset=utf-8",
        metadata=metadata,
    )


def upload_file(
    s3,
    bucket: str,
    local_path: Union[str, pathlib.Path],
    key: str,
    content_type: Optional[str] = None,
    metadata: Optional[Dict[str, str]] = None,
    transfer_config: TransferConfig = _DEFAULT_TRANSFER,
) -> None:
    """
    Upload local file to S3 key.
    """
    local_path = pathlib.Path(local_path)
    extra = {}
    if content_type:
        extra["ContentType"] = content_type
    if metadata:
        extra["Metadata"] = {str(k): str(v) for k, v in metadata.items()}

    logger.info("Uploading %s -> s3://%s/%s", local_path, bucket, key)
    _with_retries(
        s3.upload_file,
        Filename=str(local_path),
        Bucket=bucket,
        Key=key,
        ExtraArgs=extra if extra else None,
        Config=transfer_config,
    )


def download_file(
    s3,
    bucket: str,
    key: str,
    local_path: Union[str, pathlib.Path],
    overwrite: bool = False,
    transfer_config: TransferConfig = _DEFAULT_TRANSFER,
) -> pathlib.Path:
    """
    Download S3 key to local path.
    """
    local_path = pathlib.Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    if local_path.exists() and not overwrite:
        logger.info("Download skipped (exists): %s", local_path)
        return local_path

    logger.info("Downloading s3://%s/%s -> %s", bucket, key, local_path)
    _with_retries(
        s3.download_file,
        Bucket=bucket,
        Key=key,
        Filename=str(local_path),
        Config=transfer_config,
    )
    return local_path


def copy_object(s3, bucket: str, src_key: str, dst_key: str, metadata_directive: str = "COPY") -> None:
    """
    Copy within the same bucket.
    """
    logger.info("Copying s3://%s/%s -> s3://%s/%s", bucket, src_key, bucket, dst_key)
    _with_retries(
        s3.copy_object,
        Bucket=bucket,
        CopySource={"Bucket": bucket, "Key": src_key},
        Key=dst_key,
        MetadataDirective=metadata_directive,
    )


def delete_object(s3, bucket: str, key: str) -> None:
    logger.info("Deleting s3://%s/%s", bucket, key)
    _with_retries(s3.delete_object, Bucket=bucket, Key=key)


def delete_prefix(s3, bucket: str, prefix: str, batch_size: int = 1000) -> int:
    """
    Delete all objects under prefix. Returns number deleted.
    """
    to_delete = []
    deleted = 0
    for obj in list_objects(s3, bucket, prefix=prefix, recursive=True):
        to_delete.append({"Key": obj["Key"]})
        if len(to_delete) >= batch_size:
            resp = _with_retries(s3.delete_objects, Bucket=bucket, Delete={"Objects": to_delete})
            deleted += len(resp.get("Deleted", []) or [])
            to_delete = []
    if to_delete:
        resp = _with_retries(s3.delete_objects, Bucket=bucket, Delete={"Objects": to_delete})
        deleted += len(resp.get("Deleted", []) or [])
    return deleted


# --------------------------------------------------------------------------------------
# Data-lake conventions: versioned paths + manifests
# DON"T WORRY ABOUT THESE UNLESS YOU'RE WORKING WITH MANY OUTPUTS/FILE VERSIONS
# --------------------------------------------------------------------------------------

def make_run_id(prefix: str = "") -> str:
    """
    Generates a run_id suitable for foldering outputs: 20260220T010203Z_ab12cd
    """
    ts = now_utc_compact()
    rand = hashlib.sha1(os.urandom(16)).hexdigest()[:6]
    rid = f"{ts}_{rand}"
    return f"{prefix}{rid}" if prefix else rid


def write_run_manifest(
    s3,
    bucket: str,
    run_prefix: str,
    manifest: Dict[str, Any],
    *,
    manifest_name: str = "manifest.json",
) -> str:
    """
    Writes manifest JSON to run_prefix/manifest.json and returns the S3 key.
    """
    run_prefix = run_prefix.strip("/")
    key = join_key(run_prefix, manifest_name)
    put_json(s3, bucket, key, manifest, metadata={"strand_manifest": "true"})
    return key


def set_latest_pointer(
    s3,
    bucket: str,
    latest_key: str,
    target_run_prefix: str,
) -> None:
    """
    Writes a small text pointer file that indicates the current run location.
    Example:
      latest_key = "strand/uk_stress_testing/model_outputs/LATEST.txt"
      target_run_prefix = "strand/uk_stress_testing/model_outputs/run_id=2026.../"
    """
    target_run_prefix = target_run_prefix.strip("/") + "/"
    put_text(s3, bucket, latest_key, target_run_prefix, content_type="text/plain; charset=utf-8")


# --------------------------------------------------------------------------------------
# Local cache helpers (treat S3 as source-of-truth, local as cache)
# --------------------------------------------------------------------------------------

def cache_download(
    s3,
    bucket: str,
    key: str,
    cache_dir: Union[str, pathlib.Path],
    overwrite: bool = False,
) -> pathlib.Path:
    """
    Downloads an object into a deterministic local cache path:
        <cache_dir>/<bucket>/<key>
    """
    cache_dir = pathlib.Path(cache_dir)
    local_path = cache_dir / bucket / key
    return download_file(s3, bucket, key, local_path, overwrite=overwrite)


def upload_dir(
    s3,
    bucket: str,
    local_dir: Union[str, pathlib.Path],
    dst_prefix: str,
    *,
    include_globs: Tuple[str, ...] = ("**/*",),
    exclude_globs: Tuple[str, ...] = (".DS_Store", "__pycache__/**", "*.tmp"),
    transfer_config: TransferConfig = _DEFAULT_TRANSFER,
) -> int:
    """
    Upload directory tree to dst_prefix, preserving relative paths.
    Returns number of files uploaded.
    """
    local_dir = pathlib.Path(local_dir)
    if not local_dir.exists():
        raise FileNotFoundError(local_dir)

    # Build exclude matcher
    def _excluded(rel: pathlib.Path) -> bool:
        s = rel.as_posix()
        for pat in exclude_globs:
            if pathlib.PurePosixPath(s).match(pat):
                return True
        return False

    uploaded = 0
    for inc in include_globs:
        for p in local_dir.glob(inc):
            if p.is_dir():
                continue
            rel = p.relative_to(local_dir)
            if _excluded(rel):
                continue
            key = join_key(dst_prefix, rel.as_posix())
            upload_file(s3, bucket, p, key, transfer_config=transfer_config)
            uploaded += 1
    return uploaded


def download_prefix(
    s3,
    bucket: str,
    prefix: str,
    local_dir: Union[str, pathlib.Path],
    *,
    overwrite: bool = False,
) -> int:
    """
    Download all objects under prefix into local_dir, preserving key structure under that prefix.
    Returns number of objects downloaded (or skipped if exists and overwrite=False).
    """
    local_dir = pathlib.Path(local_dir)
    prefix = prefix.strip("/")
    if prefix and not prefix.endswith("/"):
        prefix += "/"

    n = 0
    for obj in list_objects(s3, bucket, prefix=prefix, recursive=True):
        key = obj["Key"]
        rel = key[len(prefix):] if key.startswith(prefix) else key
        local_path = local_dir / rel
        download_file(s3, bucket, key, local_path, overwrite=overwrite)
        n += 1
    return n


# --------------------------------------------------------------------------------------
# Optional: convenience "STRAND" setup
# --------------------------------------------------------------------------------------

@dataclass
class StrandStore:
    """
    Convenience wrapper for a single bucket (e.g., "strand") with a fixed root prefix.
    """
    s3: Any
    bucket: str = "strand"
    root_prefix: str = "strand"  # if you *also* want a root folder inside bucket, set this; otherwise "".

    def key(self, *parts: str) -> str:
        return join_key(self.root_prefix, *parts) if self.root_prefix else join_key(*parts)

    def upload(self, local_path: Union[str, pathlib.Path], *key_parts: str, **kwargs) -> None:
        return upload_file(self.s3, self.bucket, local_path, self.key(*key_parts), **kwargs)

    def download(self, *key_parts: str, local_path: Union[str, pathlib.Path], **kwargs) -> pathlib.Path:
        return download_file(self.s3, self.bucket, self.key(*key_parts), local_path, **kwargs)

    def exists(self, *key_parts: str) -> bool:
        return object_exists(self.s3, self.bucket, self.key(*key_parts))

    def ls(self, *prefix_parts: str, recursive: bool = True) -> List[str]:
        prefix = self.key(*prefix_parts)
        return [o["Key"] for o in list_objects(self.s3, self.bucket, prefix=prefix, recursive=recursive)]

    def put_manifest(self, run_prefix_parts: Tuple[str, ...], manifest: Dict[str, Any]) -> str:
        rp = self.key(*run_prefix_parts)
        return write_run_manifest(self.s3, self.bucket, rp, manifest)

    def set_latest(self, latest_key_parts: Tuple[str, ...], target_run_prefix_parts: Tuple[str, ...]) -> None:
        latest_key = self.key(*latest_key_parts)
        target = self.key(*target_run_prefix_parts)
        return set_latest_pointer(self.s3, self.bucket, latest_key, target)
