from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


def upload_outputs(output_dir: str | Path, s3_output: str | None) -> list[str]:
    if not s3_output:
        return []
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError("boto3 is required for S3 upload") from exc
    bucket, prefix = parse_s3_uri(s3_output)
    client = boto3.client("s3")
    uploaded: list[str] = []
    for path in Path(output_dir).rglob("*"):
        if not path.is_file():
            continue
        key = f"{prefix}{path.relative_to(output_dir).as_posix()}"
        client.upload_file(str(path), bucket, key)
        uploaded.append(f"s3://{bucket}/{key}")
    ready_key = f"{prefix}checkpoint/READY.json"
    client.put_object(
        Bucket=bucket,
        Key=ready_key,
        Body=json.dumps({"created_unix_ms": int(time.time() * 1000), "output_dir": str(output_dir)}).encode("utf-8"),
        ContentType="application/json",
    )
    uploaded.append(f"s3://{bucket}/{ready_key}")
    return uploaded


def upload_training_checkpoint(output_dir: str | Path, s3_output: str | None, *, step: int) -> list[str]:
    if not s3_output:
        return []
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError("boto3 is required for S3 checkpoint upload") from exc
    output_path = Path(output_dir)
    bucket, prefix = parse_s3_uri(s3_output)
    client = boto3.client("s3")
    files = [
        output_path / "checkpoint-latest.pt",
        output_path / f"checkpoint-step-{step:08d}.pt",
        output_path / "training_report.json",
    ]
    uploaded: list[str] = []
    uploaded_keys: list[str] = []
    for path in files:
        if not path.exists():
            continue
        key = f"{prefix}{path.name}"
        client.upload_file(str(path), bucket, key)
        uploaded.append(f"s3://{bucket}/{key}")
        uploaded_keys.append(key)
    ready_key = f"{prefix}checkpoint/READY.json"
    client.put_object(
        Bucket=bucket,
        Key=ready_key,
        Body=json.dumps(
            {
                "created_unix_ms": int(time.time() * 1000),
                "latest_step": step,
                "latest_checkpoint": "checkpoint-latest.pt",
                "numbered_checkpoint": f"checkpoint-step-{step:08d}.pt",
                "uploaded_keys": uploaded_keys,
            }
        ).encode("utf-8"),
        ContentType="application/json",
    )
    uploaded.append(f"s3://{bucket}/{ready_key}")
    return uploaded


def restore_training_checkpoint_from_s3(s3_output: str | None, output_dir: str | Path) -> list[str]:
    if not s3_output:
        return []
    output_path = Path(output_dir)
    latest_path = output_path / "checkpoint-latest.pt"
    if latest_path.exists():
        return []
    try:
        import boto3
        from botocore.exceptions import ClientError
    except ImportError as exc:
        raise RuntimeError("boto3 is required for S3 checkpoint restore") from exc

    bucket, prefix = parse_s3_uri(s3_output)
    client = boto3.client("s3")
    ready = get_ready_json(client, bucket, f"{prefix}checkpoint/READY.json", ClientError)
    if not ready:
        return []
    checkpoint_names = [
        ready.get("latest_checkpoint", "checkpoint-latest.pt"),
        ready.get("numbered_checkpoint"),
        "training_report.json",
    ]
    downloaded: list[str] = []
    output_path.mkdir(parents=True, exist_ok=True)
    for name in checkpoint_names:
        if not name:
            continue
        key = f"{prefix}{name}"
        destination = output_path / Path(name).name
        try:
            client.download_file(bucket, key, str(destination))
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code")
            if code in {"NoSuchKey", "404", "NotFound"} and name == "training_report.json":
                continue
            raise
        downloaded.append(str(destination))
    return downloaded


def sync_dataset_from_s3(s3_dataset: str | None, dataset_dir: str | Path) -> list[str]:
    if not s3_dataset:
        return []
    output_path = Path(dataset_dir)
    if dataset_ready(output_path):
        return []
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError("boto3 is required for S3 dataset sync") from exc

    bucket, prefix = parse_s3_uri(s3_dataset)
    client = boto3.client("s3")
    output_path.mkdir(parents=True, exist_ok=True)
    downloaded: list[str] = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            relative = key[len(prefix) :]
            if not relative or relative.endswith("/"):
                continue
            if should_skip_dataset_key(relative):
                continue
            destination = output_path / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() and destination.stat().st_size == obj.get("Size"):
                continue
            client.download_file(bucket, key, str(destination))
            downloaded.append(str(destination))
    if not dataset_ready(output_path):
        raise RuntimeError(f"S3 dataset sync finished but dataset is incomplete: {output_path}")
    return downloaded


def dataset_ready(dataset_dir: str | Path) -> bool:
    path = Path(dataset_dir)
    metadata_path = path / "metadata.json"
    if not metadata_path.exists():
        return False
    try:
        metadata = json.loads(metadata_path.read_text())
    except json.JSONDecodeError:
        return False
    for relative in expected_dataset_files(metadata):
        file_path = path / relative
        if not file_path.exists() or not file_path.is_file():
            return False
    return True


def expected_dataset_files(metadata: dict[str, Any]) -> list[str]:
    files: list[str] = ["metadata.json"]
    if (metadata.get("output_mode") == "pretrain") or metadata.get("dtype"):
        files.append("manifest.json")
    for shard in metadata.get("shards", []):
        path = shard.get("path")
        if path:
            files.append(path)
    return files


def get_ready_json(client, bucket: str, key: str, client_error_type) -> dict[str, Any] | None:
    try:
        response = client.get_object(Bucket=bucket, Key=key)
    except client_error_type as exc:
        code = exc.response.get("Error", {}).get("Code")
        if code in {"NoSuchKey", "404", "NotFound"}:
            return None
        raise
    body = response["Body"].read().decode("utf-8")
    return json.loads(body)


def should_skip_dataset_key(relative: str) -> bool:
    return relative.startswith("checkpoint/") or relative.endswith(".partial.json")


def parse_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"not an S3 URI: {uri}")
    rest = uri[len("s3://") :]
    bucket, _, prefix = rest.partition("/")
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    return bucket, prefix
