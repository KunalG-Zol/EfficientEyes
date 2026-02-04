import os
import boto3
from botocore import UNSIGNED
from botocore.config import Config

# --- CONFIGURATION ---
BUCKET_NAME = "rareplanes-public"
PREFIX = "real/train/PS-RGB_tiled/"
LOCAL_DIR = "./mini_dataset/images/"
COUNT = 100

# 1. Setup Connection (Anonymous / No Login Needed)
# This config tells Python: "Don't look for a password, this is a public bucket"
s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

# 2. Create Directory
os.makedirs(LOCAL_DIR, exist_ok=True)

print(f"Fetching list of first {COUNT} images...")

# 3. List Files
response = s3.list_objects_v2(
    Bucket=BUCKET_NAME,
    Prefix=PREFIX,
    MaxKeys=1000
)

# 4. Download Loop

downloaded = 0
if 'Contents' in response:
    print(f"Found files. Starting download to {LOCAL_DIR}...")

    for item in response['Contents']:
        key = item['Key']
        filename = os.path.basename(key)

        # Only download PNGs (skip folder names)
        if filename.endswith(".png"):
            local_path = os.path.join(LOCAL_DIR, filename)

            # The Magic Line: Downloads directly
            s3.download_file(BUCKET_NAME, key, local_path)
            downloaded += 1
            if (downloaded + 1) % 10 == 0:
                print(f"Downloaded {downloaded + 1} files...")

            if downloaded >= COUNT:
                break
print("Batch Download Complete!")