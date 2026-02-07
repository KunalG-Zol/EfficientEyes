import os
import boto3
from botocore import UNSIGNED
from botocore.config import Config

# --- CONFIGURATION ---
BUCKET_NAME = "rareplanes-public"
DOWNLOAD_TARGETS = [
    ("real/train/PS-RGB_tiled/", "./dataset/images/train/"),
    ("real/test/PS-RGB_tiled/", "./dataset/images/val/")
]


def download_dataset():
    # 1. Setup Connection
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    paginator = s3.get_paginator('list_objects_v2')

    for prefix, local_dir in DOWNLOAD_TARGETS:
        print(f"\n--- Scanning folder: {prefix} ---")
        os.makedirs(local_dir, exist_ok=True)

        count = 0
        skipped = 0


        for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix=prefix):
            if 'Contents' not in page: continue

            for item in page['Contents']:
                key = item['Key']
                filename = os.path.basename(key)

                # Filter for PNGs
                if not filename.endswith(".png"):
                    continue

                local_path = os.path.join(local_dir, filename)

                # RESUME LOGIC: Don't re-download if we have it
                if os.path.exists(local_path):
                    skipped += 1
                    if skipped % 2000 == 0:
                        print(f"Skipped {skipped} existing files...")
                    continue

                # Download
                s3.download_file(BUCKET_NAME, key, local_path)
                count += 1

                if count % 100 == 0:
                    print(f"Downloaded {count} new files...")

    print("\n✅ Batch Download Complete!")


if __name__ == "__main__":
    download_dataset()