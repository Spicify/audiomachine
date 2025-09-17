import json
from typing import Any, Dict, List, Optional

import boto3
from botocore.exceptions import ClientError
from settings import (
    AWS_S3_BUCKET,
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    AWS_DEFAULT_REGION,
)


def get_s3_client():
    # Prefer explicit creds/region from settings; fall back to default provider chain
    client_kwargs = {}
    if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
        client_kwargs["aws_access_key_id"] = AWS_ACCESS_KEY_ID
        client_kwargs["aws_secret_access_key"] = AWS_SECRET_ACCESS_KEY
    if AWS_DEFAULT_REGION:
        client_kwargs["region_name"] = AWS_DEFAULT_REGION
    return boto3.client("s3", **client_kwargs)


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
