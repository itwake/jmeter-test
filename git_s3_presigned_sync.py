import os
import sys
import time
import json
import logging
import hashlib
import argparse
import subprocess
import requests
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

MAX_WORKERS = 8

def get_git_email():
    result = subprocess.run(["git", "config", "user.email"], capture_output=True, text=True)
    email = result.stdout.strip()
    logging.info(f"Detected Git user email: {email}")
    return email

def md5_checksum(path):
    hash_md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def list_local_files(folder):
    files = {}
    for root, _, filenames in os.walk(folder):
        for filename in filenames:
            full = os.path.join(root, filename)
            rel = os.path.relpath(full, folder).replace("\", "/")
            files[rel] = {
                "path": full,
                "md5": md5_checksum(full),
                "mtime": datetime.utcfromtimestamp(os.path.getmtime(full)).replace(tzinfo=timezone.utc)
            }
    logging.info(f"Found {len(files)} local files")
    return files

def list_s3_files(api_url, email):
    resp = requests.post(f"{api_url}/list", json={"email": email})
    resp.raise_for_status()
    files = resp.json()
    result = {}
    for f in files:
        result[f["Key"]] = {
            "etag": f["ETag"].strip('"'),
            "last_modified": datetime.fromisoformat(f["LastModified"])
        }
    logging.info(f"Fetched {len(result)} S3 files")
    return result

def get_presigned_url(api_url, email, key, method):
    resp = requests.post(f"{api_url}/presign", json={
        "email": email,
        "key": key,
        "method": method
    })
    resp.raise_for_status()
    return resp.json()["url"]

def upload_file(file_meta, url):
    with open(file_meta["path"], "rb") as f:
        headers = {"Content-Type": "application/octet-stream"}
        r = requests.put(url, data=f, headers=headers)
    if r.status_code == 200:
        logging.info(f"Uploaded {file_meta['path']} -> {url}")
    else:
        logging.error(f"Upload failed for {file_meta['path']} -> {r.status_code}")

def download_file(url, path):
    r = requests.get(url)
    if r.status_code == 200:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(r.content)
        logging.info(f"Downloaded {path}")
    else:
        logging.error(f"Download failed for {path} -> {r.status_code}")

def sync(api_url, local_folder):
    email = get_git_email()
    local_files = list_local_files(local_folder)
    remote_files = list_s3_files(api_url, email)

    with ThreadPoolExecutor(max_workers=8) as executor:
        # 上传
        for key, meta in local_files.items():
            if key not in remote_files or meta["md5"] != remote_files[key]["etag"]:
                url = get_presigned_url(api_url, email, key, "PUT")
                executor.submit(upload_file, meta, url)

        # 下载
        for key, meta in remote_files.items():
            if key not in local_files:
                local_path = os.path.join(local_folder, key)
                url = get_presigned_url(api_url, email, key, "GET")
                executor.submit(download_file, url, local_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Git-aware S3 sync via Presigned URL")
    parser.add_argument("--api-url", required=True, help="API Gateway base URL")
    parser.add_argument("--local-folder", default="./sync-folder", help="Local sync folder")
    args = parser.parse_args()

    try:
        sync(args.api_url.rstrip("/"), args.local_folder)
    except Exception as e:
        logging.error(f"Sync failed: {e}")
        sys.exit(1)