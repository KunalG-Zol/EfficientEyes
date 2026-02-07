import boto3
from botocore import UNSIGNED
from botocore.config import Config

s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

print("Downloading Train Annotations...")
s3.download_file(
    "rareplanes-public",
    "real/metadata_annotations/RarePlanes_Train_Coco_Annotations_tiled.json",
    "RarePlanes_Train_Coco_Annotations_tiled.json"
)

print("Downloading Test Annotations...")
s3.download_file(
    "rareplanes-public",
    "real/metadata_annotations/RarePlanes_Test_Coco_Annotations_tiled.json",
    "RarePlanes_Test_Coco_Annotations_tiled.json"
)
print("Done!")