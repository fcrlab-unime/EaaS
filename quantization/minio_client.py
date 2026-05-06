import os
import logging
from pathlib import Path
from minio import Minio
from minio.error import S3Error

logger = logging.getLogger(__name__)


def make_client() -> Minio:
    return Minio(
        os.environ["MINIO_ENDPOINT"],
        access_key=os.environ["MINIO_ROOT_USER"],
        secret_key=os.environ["MINIO_ROOT_PASSWORD"],
        secure=os.environ.get("MINIO_SECURE", "false").lower() == "true",
    )


def download(client: Minio, bucket: str, key: str, dest: str) -> Path:
    path = Path(dest)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        client.fget_object(bucket, key, str(path))
        logger.info("Downloaded s3://%s/%s → %s", bucket, key, path)
    except S3Error as e:
        logger.error("MinIO download failed: %s", e)
        raise
    return path


def upload(client: Minio, bucket: str, key: str, src: str) -> None:
    try:
        client.fput_object(bucket, key, src)
        logger.info("Uploaded %s → s3://%s/%s", src, bucket, key)
    except S3Error as e:
        logger.error("MinIO upload failed: %s", e)
        raise


def list_objects(client: Minio, bucket: str, prefix: str = "") -> list[str]:
    try:
        objects = client.list_objects(bucket, prefix=prefix, recursive=True)
        return [obj.object_name for obj in objects]
    except S3Error as e:
        logger.error("MinIO list failed: %s", e)
        raise