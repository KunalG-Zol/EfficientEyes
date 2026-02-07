import json
import os

# --- CONFIGURATION ---
DATASET_ROOT = "./dataset"
SETS = {
    "train": {
        "json": "RarePlanes_Train_Coco_Annotations_tiled.json",
        "images": os.path.join(DATASET_ROOT, "images/train"),
        "labels": os.path.join(DATASET_ROOT, "labels/train")
    },
    "val": {
        "json": "RarePlanes_Test_Coco_Annotations_tiled.json",
        "images": os.path.join(DATASET_ROOT, "images/val"),
        "labels": os.path.join(DATASET_ROOT, "labels/val")
    }
}

def convert_coco_to_yolo(box, img_w, img_h):
    x, y, w, h = box
    if img_w == 0 or img_h == 0: return None
    
    # Normalize to 0-1
    center_x = (x + w / 2) / img_w
    center_y = (y + h / 2) / img_h
    norm_w = w / img_w
    norm_h = h / img_h
    
    # Clamp just in case
    return f"0 {max(0, min(1, center_x)):.6f} {max(0, min(1, center_y)):.6f} {max(0, min(1, norm_w)):.6f} {max(0, min(1, norm_h)):.6f}"

def process_set(set_name, config):
    print(f"\n--- Processing {set_name.upper()} Set ---")
    
    if not os.path.exists(config["json"]):
        print(f"❌ Error: {config['json']} not found.")
        return

    # 1. Load Data
    print(f"Loading JSON...")
    with open(config["json"], 'r') as f:
        data = json.load(f)
    
    images_info = {img['id']: img for img in data['images']}
    
    # 2. Check Downloaded Images
    if not os.path.exists(config["images"]):
        print(f"❌ Error: Image folder {config['images']} not found.")
        return
    downloaded_files = set(os.listdir(config["images"]))
    print(f"Found {len(downloaded_files)} images on disk.")

    # 3. Create Labels Folder
    os.makedirs(config["labels"], exist_ok=True)
    
    # 4. Generate Labels
    labels_to_write = {}
    count = 0
    
    print("Generating labels...")
    for ann in data['annotations']:
        image_id = ann['image_id']
        img_info = images_info.get(image_id)
        
        if not img_info: continue
        
        filename = img_info['file_name']
        
        # Only process if image exists on disk
        if filename not in downloaded_files:
            continue
            
        yolo_line = convert_coco_to_yolo(ann['bbox'], img_info['width'], img_info['height'])
        
        if yolo_line:
            if filename not in labels_to_write:
                labels_to_write[filename] = []
            labels_to_write[filename].append(yolo_line)
            
        count += 1
        if count % 10000 == 0:
            print(f"Processed {count} annotations...")

    # 5. Write to Disk
    print(f"Writing {len(labels_to_write)} label files to disk...")
    for filename, lines in labels_to_write.items():
        txt_name = filename.replace('.png', '.txt')
        with open(os.path.join(config["labels"], txt_name), 'w') as f:
            f.write('\n'.join(lines))
            
    print(f"✅ Done with {set_name}.")

# --- RUN ---
process_set("train", SETS['train'])
process_set("val", SETS['val'])