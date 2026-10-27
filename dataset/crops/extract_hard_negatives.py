import os
import json
import glob
import cv2
import numpy as np

IMAGES_DIR = r"C:\All_Projects\esp32_cam_ai_system\images"
OUTPUT_CROPS_DIR = r"C:\All_Projects\esp32_cam_ai_system\dataset\crops"
ANNOTATIONS_FILE = r"C:\All_Projects\esp32_cam_ai_system\dataset\annotations\annotations.json"
SPLIT_FILE = r"C:\All_Projects\esp32_cam_ai_system\dataset\splits\split_definition.json"

with open(ANNOTATIONS_FILE, "r") as f:
    annotations = json.load(f)
with open(SPLIT_FILE, "r") as f:
    splits = json.load(f)

# Clear existing background crops
for split in ["train", "val", "test"]:
    bg_dir = os.path.join(OUTPUT_CROPS_DIR, split, "background")
    for f in glob.glob(os.path.join(bg_dir, "*.png")):
        try: os.remove(f)
        except: pass

counts = {"train": 0, "val": 0, "test": 0}

for fname, split in splits.items():
    fpath = os.path.join(IMAGES_DIR, fname)
    img = cv2.imread(fpath)
    h_img, w_img = img.shape[:2]
    gt_boxes = annotations.get(fname, [])
    
    # Generate grid of candidate background boxes (size 80x80 to 120x120)
    step_x = 140
    step_y = 120
    crop_idx = 0
    
    for y in range(40, h_img - 100, step_y):
        for x in range(40, w_img - 100, step_x):
            box_w, box_h = 96, 96
            # Check overlap with ANY ground truth object
            overlap = False
            for gt in gt_boxes:
                gx, gy, gw, gh = gt["bbox"]
                # Expanded check
                if (x < gx + gw + 20 and x + box_w > gx - 20 and
                    y < gy + gh + 20 and y + box_h > gy - 20):
                    overlap = True
                    break
            if not overlap:
                crop = img[y:y+box_h, x:x+box_w]
                if crop.size > 0:
                    out_name = f"{split}_bg_{fname[:10]}_{crop_idx}.png"
                    out_path = os.path.join(OUTPUT_CROPS_DIR, split, "background", out_name)
                    cv2.imwrite(out_path, crop)
                    counts[split] += 1
                    crop_idx += 1

print("Extracted extensive negative background crops:")
for s, c in counts.items():
    print(f" Split [{s.upper()}]: {c} background patches")
