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
    List exactly `max_keys` project JSON files in descending LastModified order (most recent first),
    supporting continuation for next-page retrieval.
    """
    s3 = get_s3_client()
    bucket = bucket or get_bucket_defaults()
    kwargs = {"Bucket": bucket, "Prefix": prefix,
              "MaxKeys": max(1, min(1000, max_keys))}
    if continuation_token:
        kwargs["ContinuationToken"] = continuation_token

    resp = s3.list_objects_v2(**kwargs)
    contents = resp.get("Contents", [])
    # Sort descending by LastModified for consistent reverse ordering
    contents = sorted(contents, key=lambda x: x["LastModified"], reverse=True)
    keys = [item["Key"] for item in contents if item["Key"].endswith(".json")]

    # Ensure we always attempt to return exactly max_keys if possible
    while len(keys) < max_keys and resp.get("IsTruncated"):
        resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix,
                                  ContinuationToken=resp["NextContinuationToken"],
                                  MaxKeys=max(1, min(1000, max_keys - len(keys))))
        next_contents = sorted(resp.get("Contents", []),
                               key=lambda x: x["LastModified"], reverse=True)
        next_keys = [item["Key"]
                     for item in next_contents if item["Key"].endswith(".json")]
        keys.extend(next_keys)
        if not resp.get("IsTruncated") or len(keys) >= max_keys:
            break

    # Final sort to guarantee strict reverse chronological order
    # This ensures consistent ordering even when combining multiple S3 responses
    keys = keys[:max_keys]

    # Re-sort the final keys by their LastModified timestamps to guarantee order
    # We need to fetch metadata for each key to sort properly
    if len(keys) > 1:
        key_metadata = []
        for key in keys:
            try:
                head_resp = s3.head_object(Bucket=bucket, Key=key)
                key_metadata.append((key, head_resp["LastModified"]))
            except Exception:
                # If head_object fails, use a default timestamp
                key_metadata.append((key, datetime.datetime.min))

        # Sort by LastModified descending (newest first)
        key_metadata.sort(key=lambda x: x[1], reverse=True)
        keys = [item[0] for item in key_metadata]

    next_token = resp.get("NextContinuationToken") if resp.get(
        "IsTruncated") else None
    return {"keys": keys, "next_token": next_token}
