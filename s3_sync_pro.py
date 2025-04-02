
import os
import boto3
import hashlib
import argparse
from datetime import datetime, timezone

AWS_REGION = "ap-northeast-1"
BUCKET_NAME = "your-bucket-name"
LOCAL_FOLDER = "./sync-folder"

s3 = boto3.client(
    's3',
    aws_access_key_id='YOUR_ACCESS_KEY',
    aws_secret_access_key='YOUR_SECRET_KEY',
    region_name=AWS_REGION
)

def md5_checksum(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def list_local_files(include, exclude):
    files = {}
    for root, _, filenames in os.walk(LOCAL_FOLDER):
        for filename in filenames:
            rel_path = os.path.relpath(os.path.join(root, filename), LOCAL_FOLDER).replace("\", "/")
            if any(pattern in rel_path for pattern in exclude):
                continue
            if include and not any(pattern in rel_path for pattern in include):
                continue
            full_path = os.path.join(root, filename)
            mtime = os.path.getmtime(full_path)
            files[rel_path] = {
                "md5": md5_checksum(full_path),
                "mtime": datetime.utcfromtimestamp(mtime).replace(tzinfo=timezone.utc),
                "path": full_path
            }
    return files

def list_s3_objects():
    paginator = s3.get_paginator('list_objects_v2')
    page_iterator = paginator.paginate(Bucket=BUCKET_NAME)

    files = {}
    for page in page_iterator:
        for obj in page.get("Contents", []):
            key = obj["Key"]
            files[key] = {
                "etag": obj["ETag"].strip('"'),
                "last_modified": obj["LastModified"]
            }
    return files

def sync_upload(local_files, s3_files, dryrun):
    for key, meta in local_files.items():
        s3_meta = s3_files.get(key)
        should_upload = False
        if not s3_meta:
            should_upload = True
        elif meta["md5"] != s3_meta["etag"]:
            should_upload = True
        elif meta["mtime"] > s3_meta["last_modified"]:
            should_upload = True

        if should_upload:
            print(f"{'[DRYRUN] ' if dryrun else ''}Uploading {key}")
            if not dryrun:
                s3.upload_file(meta["path"], BUCKET_NAME, key)

def sync_download(local_files, s3_files, include, exclude, dryrun):
    for key, meta in s3_files.items():
        if any(pattern in key for pattern in exclude):
            continue
        if include and not any(pattern in key for pattern in include):
            continue

        local_meta = local_files.get(key)
        local_path = os.path.join(LOCAL_FOLDER, key)
        if not local_meta:
            print(f"{'[DRYRUN] ' if dryrun else ''}Downloading {key}")
            if not dryrun:
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                s3.download_file(BUCKET_NAME, key, local_path)
        elif meta["etag"] != local_meta["md5"] or meta["last_modified"] > local_meta["mtime"]:
            print(f"{'[DRYRUN] ' if dryrun else ''}Overwriting local {key}")
            if not dryrun:
                s3.download_file(BUCKET_NAME, key, local_path)

def delete_local_extras(local_files, s3_files, dryrun):
    for key in local_files:
        if key not in s3_files:
            print(f"{'[DRYRUN] ' if dryrun else ''}Deleting local {key}")
            if not dryrun:
                os.remove(local_files[key]["path"])

def delete_s3_extras(local_files, s3_files, dryrun):
    for key in s3_files:
        if key not in local_files:
            print(f"{'[DRYRUN] ' if dryrun else ''}Deleting S3 {key}")
            if not dryrun:
                s3.delete_object(Bucket=BUCKET_NAME, Key=key)

def sync(include, exclude, dryrun, delete_local, delete_remote):
    print("Listing local files...")
    local_files = list_local_files(include, exclude)
    print("Listing S3 files...")
    s3_files = list_s3_objects()
    print("Uploading changes...")
    sync_upload(local_files, s3_files, dryrun)
    print("Downloading new/updated files...")
    sync_download(local_files, s3_files, include, exclude, dryrun)
    if delete_local:
        print("Deleting local extras...")
        delete_local_extras(local_files, s3_files, dryrun)
    if delete_remote:
        print("Deleting S3 extras...")
        delete_s3_extras(local_files, s3_files, dryrun)
    print("Sync complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Python-based AWS S3 sync tool")
    parser.add_argument("--include", nargs="*", default=[], help="Include pattern(s)")
    parser.add_argument("--exclude", nargs="*", default=[], help="Exclude pattern(s)")
    parser.add_argument("--dryrun", action="store_true", help="Preview changes without execution")
    parser.add_argument("--delete-local", action="store_true", help="Delete local files not in S3")
    parser.add_argument("--delete-remote", action="store_true", help="Delete S3 files not in local")
    args = parser.parse_args()

    sync(args.include, args.exclude, args.dryrun, args.delete_local, args.delete_remote)
