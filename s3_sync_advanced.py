import os
import argparse
import hashlib
import boto3
import logging
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ---------------- Config ----------------
MAX_WORKERS = 8
LOCAL_FOLDER = "./sync-folder"
# ----------------------------------------

# ---------- Logging Setup ---------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("sync.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
# ----------------------------------------

def md5_checksum(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def list_local_files(folder):
    files = {}
    for root, _, filenames in os.walk(folder):
        for filename in filenames:
            full_path = os.path.join(root, filename)
            rel_path = os.path.relpath(full_path, folder).replace("\", "/")
            mtime = os.path.getmtime(full_path)
            files[rel_path] = {
                "path": full_path,
                "md5": md5_checksum(full_path),
                "mtime": datetime.utcfromtimestamp(mtime).replace(tzinfo=timezone.utc)
            }
    return files

def list_s3_files(s3, bucket):
    paginator = s3.get_paginator("list_objects_v2")
    page_iterator = paginator.paginate(Bucket=bucket)
    files = {}
    for page in page_iterator:
        for obj in page.get("Contents", []):
            files[obj["Key"]] = {
                "etag": obj["ETag"].strip('"'),
                "last_modified": obj["LastModified"]
            }
    return files

def upload_file(s3, bucket, key, path):
    s3.upload_file(path, bucket, key, ExtraArgs={"ServerSideEncryption": "aws:kms"})
    logger.info(f"Uploaded: {key}")

def download_file(s3, bucket, key, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    s3.download_file(bucket, key, path)
    logger.info(f"Downloaded: {key}")

def delete_local_file(path):
    os.remove(path)
    logger.info(f"Deleted local file: {path}")

def delete_s3_file(s3, bucket, key):
    s3.delete_object(Bucket=bucket, Key=key)
    logger.info(f"Deleted S3 file: {key}")

def resolve_conflict(local_meta, remote_meta, strategy):
    if strategy == "prefer-local":
        return "upload"
    elif strategy == "prefer-remote":
        return "download"
    elif strategy == "backup":
        return "backup"
    else:  # default: mtime
        return "upload" if local_meta["mtime"] > remote_meta["last_modified"] else "download"

def sync(s3, bucket, folder, delete_local=False, delete_remote=False, conflict_mode="mtime"):
    local_files = list_local_files(folder)
    s3_files = list_s3_files(s3, bucket)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []

        # Upload or resolve conflict
        for key, local_meta in local_files.items():
            if key not in s3_files:
                futures.append(executor.submit(upload_file, s3, bucket, key, local_meta["path"]))
            else:
                remote_meta = s3_files[key]
                if local_meta["md5"] != remote_meta["etag"]:
                    action = resolve_conflict(local_meta, remote_meta, conflict_mode)
                    if action == "upload":
                        futures.append(executor.submit(upload_file, s3, bucket, key, local_meta["path"]))
                    elif action == "download":
                        dest = os.path.join(folder, key)
                        futures.append(executor.submit(download_file, s3, bucket, key, dest))
                    elif action == "backup":
                        backup_path = local_meta["path"] + ".bak"
                        os.rename(local_meta["path"], backup_path)
                        logger.info(f"Backed up local file to {backup_path}")
                        futures.append(executor.submit(download_file, s3, bucket, key, local_meta["path"]))

        # Download missing files
        for key in s3_files:
            if key not in local_files:
                dest = os.path.join(folder, key)
                futures.append(executor.submit(download_file, s3, bucket, key, dest))

        # Delete local files not in S3
        if delete_local:
            for key in local_files:
                if key not in s3_files:
                    futures.append(executor.submit(delete_local_file, local_files[key]["path"]))

        # Delete S3 files not in local
        if delete_remote:
            for key in s3_files:
                if key not in local_files:
                    futures.append(executor.submit(delete_s3_file, s3, bucket, key))

        for _ in as_completed(futures): pass

    logger.info("✅ Sync complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Advanced S3 Sync Tool")
    parser.add_argument("--bucket", required=True, help="S3 bucket name")
    parser.add_argument("--region", default="ap-east-1")
    parser.add_argument("--delete-local", action="store_true")
    parser.add_argument("--delete-remote", action="store_true")
    parser.add_argument("--conflict-mode", choices=["mtime", "prefer-local", "prefer-remote", "backup"], default="mtime")
    parser.add_argument("--folder", default=LOCAL_FOLDER)
    args = parser.parse_args()

    s3 = boto3.client("s3", region_name=args.region)
    sync(s3, args.bucket, args.folder, delete_local=args.delete_local, delete_remote=args.delete_remote, conflict_mode=args.conflict_mode)