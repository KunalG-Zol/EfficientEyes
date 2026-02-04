import json
import os
import pandas as pd
from tqdm import tqdm

# --- CONFIGURATION ---
JSON_FILE = 'RarePlanes_Train_Coco_Annotations_tiled.json'
IMAGE_DIR = './mini_dataset/images/'
LABEL_DIR = './mini_dataset/labels/'


def create_labels():
    # 1. Get list of images we actually have locally
    if not os.path.exists(IMAGE_DIR):
        print(f"Error: Directory {IMAGE_DIR} not found.")
        return

    local_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith('.png')]
    print(f"Found {len(local_files)} images in {IMAGE_DIR}")

    # 2. Load the Master JSON
    print(f"Loading {JSON_FILE} (this takes a few seconds)...")
    with open(JSON_FILE, 'r') as f:
        data = json.load(f)

    # 3. Create DataFrames
    df_imgs = pd.DataFrame(data['images'])
    df_anns = pd.DataFrame(data['annotations'])

    # 4. Filter: Keep only the rows for the images we have
    # This matches the local filename to the 'file_name' column in the JSON
    df_imgs = df_imgs[df_imgs['file_name'].isin(local_files)]

    # Create a fast lookup for Image ID -> Width/Height
    # We need this to normalize coordinates (0.0 to 1.0)
    img_details = df_imgs.set_index('id')[['width', 'height', 'file_name']].to_dict('index')

    # Filter annotations to only include our 100 images
    valid_ids = df_imgs['id'].tolist()
    df_anns = df_anns[df_anns['image_id'].isin(valid_ids)]

    # Group annotations by image for fast processing
    grouped_anns = df_anns.groupby('image_id')

    # 5. Create Labels Directory
    os.makedirs(LABEL_DIR, exist_ok=True)

    print("Generating .txt labels...")
    created_count = 0

    for img_id, info in tqdm(img_details.items()):
        filename = info['file_name']
        img_w = info['width']
        img_h = info['height']

        # Output file path (change .png to .txt)
        txt_name = os.path.splitext(filename)[0] + ".txt"
        out_path = os.path.join(LABEL_DIR, txt_name)

        yolo_lines = []

        # If this image has planes, write them
        if img_id in grouped_anns.groups:
            planes = grouped_anns.get_group(img_id)

            for _, plane in planes.iterrows():
                # COCO bbox: [x, y, w, h] (Pixels)
                x, y, w, h = plane['bbox']

                # Convert to YOLO: [x_center, y_center, w, h] (Normalized)
                x_center = (x + w / 2) / img_w
                y_center = (y + h / 2) / img_h
                w_norm = w / img_w
                h_norm = h / img_h

                # Class 0 = Plane
                yolo_lines.append(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

        # Write file (Even if empty! Empty file = "No planes here")
        with open(out_path, 'w') as f:
            f.write('\n'.join(yolo_lines))

        created_count += 1

    print(f"Success! Created {created_count} label files.")

    # 6. Generate the 'mini_dataset.yaml' file for YOLO
    yaml_content = f"""
path: {os.path.abspath('./mini_dataset')} # Absolute path to dataset
train: images  # Train images (relative to 'path')
val: images    # Val images (using same 100 images for the Overfit Test)

names:
  0: plane
"""
    with open('mini_dataset.yaml', 'w') as f:
        f.write(yaml_content)
    print("Created 'mini_dataset.yaml'. Ready to train!")


if __name__ == "__main__":
    create_labels()