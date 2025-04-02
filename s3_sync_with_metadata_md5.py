import os, boto3, hashlib, argparse, logging
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
MAX_WORKERS = 8

def md5sum(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""): h.update(chunk)
    return h.hexdigest()

def list_local_files(folder):
    result = {}
    for root, _, files in os.walk(folder):
        for f in files:
            path = os.path.join(root, f)
            rel = os.path.relpath(path, folder).replace("\", "/")
            result[rel] = {
                "path": path,
                "md5": md5sum(path),
                "mtime": datetime.utcfromtimestamp(os.path.getmtime(path)).replace(tzinfo=timezone.utc)
            }
    return result

def get_s3_meta(s3, bucket, key):
    try:
        obj = s3.head_object(Bucket=bucket, Key=key)
        return {"md5": obj["Metadata"].get("md5"), "last_modified": obj["LastModified"]}
    except Exception:
        return None

def upload(s3, bucket, key, path, md5):
    s3.upload_file(path, bucket, key, ExtraArgs={
        "ServerSideEncryption": "aws:kms",
        "Metadata": {"md5": md5}
    })
    logger.info(f"Uploaded: {key}")

def download(s3, bucket, key, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    s3.download_file(bucket, key, path)
    logger.info(f"Downloaded: {key}")

def sync(s3, bucket, folder, mode="mtime", delete_remote=False, delete_local=False):
    local = list_local_files(folder)
    remote_keys = {o["Key"] for o in s3.list_objects_v2(Bucket=bucket).get("Contents", [])}
    logger.info("Checking files...")

    with ThreadPoolExecutor(MAX_WORKERS) as pool:
        futures = []

        for key, meta in local.items():
            s3meta = get_s3_meta(s3, bucket, key)
            if not s3meta:
                futures.append(pool.submit(upload, s3, bucket, key, meta["path"], meta["md5"]))
            elif s3meta["md5"] != meta["md5"]:
                if mode == "prefer-local" or (mode == "mtime" and meta["mtime"] > s3meta["last_modified"]):
                    futures.append(pool.submit(upload, s3, bucket, key, meta["path"], meta["md5"]))
                elif mode == "prefer-remote":
                    futures.append(pool.submit(download, s3, bucket, key, meta["path"]))

        for key in remote_keys - set(local.keys()):
            path = os.path.join(folder, key)
            futures.append(pool.submit(download, s3, bucket, key, path)) if not delete_local else None
            if delete_remote: futures.append(pool.submit(lambda: s3.delete_object(Bucket=bucket, Key=key)))

        for key in set(local.keys()) - remote_keys:
            if delete_local: os.remove(local[key]["path"]); logger.info(f"Deleted local: {key}")

        for f in futures: f.result()

    logger.info("✅ Sync completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--region", default="ap-east-1")
    parser.add_argument("--folder", default="./sync-folder")
    parser.add_argument("--delete-remote", action="store_true")
    parser.add_argument("--delete-local", action="store_true")
    parser.add_argument("--conflict-mode", choices=["mtime", "prefer-local", "prefer-remote"], default="mtime")
    args = parser.parse_args()
    s3 = boto3.client("s3", region_name=args.region)
    sync(s3, args.bucket, args.folder, args.conflict_mode, args.delete_remote, args.delete_local)