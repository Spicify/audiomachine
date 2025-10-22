import json
import datetime
from typing import Any, Dict, List, Optional

import boto3
from botocore.exceptions import ClientError
from settings import (
    AWS_S3_BUCKET,
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    AWS_DEFAULT_REGION,
)


_cached_client = None


def get_s3_client():
    global _cached_client
    if _cached_client is not None:
        return _cached_client
    # Prefer explicit creds/region from settings; fall back to default provider chain
    client_kwargs = {}
    if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
        client_kwargs["aws_access_key_id"] = AWS_ACCESS_KEY_ID
        client_kwargs["aws_secret_access_key"] = AWS_SECRET_ACCESS_KEY
    if AWS_DEFAULT_REGION:
        client_kwargs["region_name"] = AWS_DEFAULT_REGION
    _cached_client = boto3.client("s3", **client_kwargs)
    return _cached_client


def get_bucket_defaults():
    if not AWS_S3_BUCKET:
        raise RuntimeError(
            "AWS_S3_BUCKET is not configured in settings/secrets")
    return AWS_S3_BUCKET


def s3_upload_bytes(key: str, data: bytes, content_type: str = "application/octet-stream", bucket: Optional[str] = None):
    s3 = get_s3_client()
    bucket = bucket or get_bucket_defaults()
    s3.put_object(Bucket=bucket, Key=key, Body=data, ContentType=content_type)


def s3_read_json(key: str, bucket: Optional[str] = None) -> Optional[Dict[str, Any]]:
    s3 = get_s3_client()
    bucket = bucket or get_bucket_defaults()
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        body = obj["Body"].read().decode("utf-8")
        return json.loads(body)
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
            return None
        raise


def s3_write_json(key: str, payload: Dict[str, Any], bucket: Optional[str] = None):
    s3 = get_s3_client()
    bucket = bucket or get_bucket_defaults()
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    s3.put_object(Bucket=bucket, Key=key, Body=body,
                  ContentType="application/json")


def s3_generate_presigned_url(key: str, expires_seconds: int = 3600, bucket: Optional[str] = None) -> Optional[str]:
    s3 = get_s3_client()
    bucket = bucket or get_bucket_defaults()
    try:
        url = s3.generate_presigned_url(
            ClientMethod="get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=expires_seconds,
        )
        return url
    except Exception:
        return None


def s3_list_json(prefix: str = "projects/", bucket: Optional[str] = None) -> List[str]:
    s3 = get_s3_client()
    bucket = bucket or get_bucket_defaults()
    keys: List[str] = []
    continuation_token = None
    while True:
        kwargs = {"Bucket": bucket, "Prefix": prefix}
        if continuation_token:
            kwargs["ContinuationToken"] = continuation_token
        resp = s3.list_objects_v2(**kwargs)
        for item in resp.get("Contents", []):
            key = item["Key"]
            if key.endswith(".json"):
                keys.append(key)
        if resp.get("IsTruncated"):
            continuation_token = resp.get("NextContinuationToken")
        else:
            break
    return keys


def s3_get_bytes(key: str, bucket: Optional[str] = None) -> Optional[bytes]:
    s3 = get_s3_client()
    bucket = bucket or get_bucket_defaults()
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        return obj["Body"].read()
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
            return None
        raise


def s3_list_objects(prefix: str = "projects/", bucket: Optional[str] = None) -> List[str]:
    """List object keys for a given prefix (no extension filtering)."""
    s3 = get_s3_client()
    bucket = bucket or get_bucket_defaults()
    keys: List[str] = []
    continuation_token = None
    while True:
        kwargs = {"Bucket": bucket, "Prefix": prefix}
        if continuation_token:
            kwargs["ContinuationToken"] = continuation_token
        resp = s3.list_objects_v2(**kwargs)
        for item in resp.get("Contents", []):
            keys.append(item["Key"])
        if resp.get("IsTruncated"):
            continuation_token = resp.get("NextContinuationToken")
        else:
            break
    return keys


def s3_list_objects_with_meta(prefix: str = "projects/", bucket: Optional[str] = None) -> List[Dict[str, Any]]:
    """List objects with minimal metadata (Key, LastModified).

    Returns a list of dicts: {"Key": str, "LastModified": datetime}.
    """
    s3 = get_s3_client()
    bucket = bucket or get_bucket_defaults()
    out: List[Dict[str, Any]] = []
    continuation_token = None
    while True:
        kwargs: Dict[str, Any] = {"Bucket": bucket, "Prefix": prefix}
        if continuation_token:
            kwargs["ContinuationToken"] = continuation_token
        resp = s3.list_objects_v2(**kwargs)
        for item in resp.get("Contents", []):
            out.append(
                {"Key": item["Key"], "LastModified": item["LastModified"]})
        if resp.get("IsTruncated"):
            continuation_token = resp.get("NextContinuationToken")
        else:
            break
    return out


def s3_delete_object(key: str, bucket: Optional[str] = None) -> None:
    s3 = get_s3_client()
    bucket = bucket or get_bucket_defaults()
    s3.delete_object(Bucket=bucket, Key=key)


def s3_list_objects_page(prefix: str, max_keys: int = 10, continuation_token: Optional[str] = None, bucket: Optional[str] = None) -> Dict[str, Any]:
    """Return a single page of S3 objects under prefix, sorted by LastModified descending.

    Returns dict with keys: {"keys": List[str], "next_token": Optional[str]}.
    Only fetches max_keys objects, no full listing.
    """
    s3 = get_s3_client()
    bucket = bucket or get_bucket_defaults()
    kwargs: Dict[str, Any] = {
        "Bucket": bucket, "Prefix": prefix, "MaxKeys": max(1, min(1000, max_keys))}
    if continuation_token:
        kwargs["ContinuationToken"] = continuation_token
    resp = s3.list_objects_v2(**kwargs)

    # Sort by LastModified descending (newest first)
    contents = resp.get("Contents", [])
    sorted_contents = sorted(
        contents, key=lambda x: x["LastModified"], reverse=True)
    out_keys: List[str] = [item["Key"] for item in sorted_contents]

    next_token = resp.get("NextContinuationToken") if resp.get(
        "IsTruncated") else None
    return {"keys": out_keys, "next_token": next_token}


def s3_list_recent_json(prefix: str = "projects/", max_items: int = 20, bucket: Optional[str] = None) -> List[str]:
    """Fast limited listing of JSON files, sorted by LastModified descending."""
    s3 = get_s3_client()
    bucket = bucket or get_bucket_defaults()
    kwargs: Dict[str, Any] = {
        "Bucket": bucket, "Prefix": prefix, "MaxKeys": max_items * 2}  # Get extra to account for non-JSON files
    resp = s3.list_objects_v2(**kwargs)

    # Filter JSON files and sort by LastModified descending
    contents = resp.get("Contents", [])
    json_objects = [item for item in contents if item["Key"].endswith(".json")]
    sorted_objects = sorted(
        json_objects, key=lambda x: x["LastModified"], reverse=True)

    # Return only the requested number of items
    return [item["Key"] for item in sorted_objects[:max_items]]


def s3_object_exists(key: str, bucket: Optional[str] = None) -> bool:
    """Fast HEAD check for object existence."""
    s3 = get_s3_client()
    bucket = bucket or get_bucket_defaults()
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
            return False
        raise


def s3_list_projects_page(prefix: str = "projects/", max_keys: int = 10,
                          continuation_token: Optional[str] = None, bucket: Optional[str] = None) -> Dict[str, Any]:
    """
    Return project JSON keys globally sorted by LastModified (newest first).
    Implements stable pagination: page N always shows next 10 newest items.
    """
    s3 = get_s3_client()
    bucket = bucket or get_bucket_defaults()

    # Fetch *all* JSON objects once
    all_objects: List[Dict[str, Any]] = []
    token = None
    while True:
        kwargs = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kwargs)
        all_objects.extend(resp.get("Contents", []))
        if not resp.get("IsTruncated"):
            break
        token = resp.get("NextContinuationToken")

    # Keep only .json files and sort globally
    json_objs = [obj for obj in all_objects if obj["Key"].endswith(".json")]
    json_objs.sort(key=lambda o: o["LastModified"], reverse=True)

    # Convert continuation_token into a numeric page index
    page_index = int(continuation_token or 1)
    start = (page_index - 1) * max_keys
    end = start + max_keys
    page_slice = json_objs[start:end]

    keys = [obj["Key"] for obj in page_slice]
    next_token = str(page_index + 1) if end < len(json_objs) else None

    return {"keys": keys, "next_token": next_token}
