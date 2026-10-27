import os
import glob
import json
import cv2
import numpy as np

IMAGES_DIR = r"C:\All_Projects\esp32_cam_ai_system\images"
OUTPUT_CROPS_DIR = r"C:\All_Projects\esp32_cam_ai_system\dataset\crops"
ANNOTATIONS_DIR = r"C:\All_Projects\esp32_cam_ai_system\dataset\annotations"
SCREENSHOTS_DIR = r"C:\All_Projects\esp32_cam_ai_system\screenshots"
SPLITS_DIR = r"C:\All_Projects\esp32_cam_ai_system\dataset\splits"

for d in [OUTPUT_CROPS_DIR, ANNOTATIONS_DIR, SCREENSHOTS_DIR, SPLITS_DIR]:
    os.makedirs(d, exist_ok=True)

# 21 Source images sorted alphabetically
files = sorted(glob.glob(os.path.join(IMAGES_DIR, "*.jpeg")))
filenames = [os.path.basename(f) for f in files]

# Train/Val/Test split strictly grouped by source image:
# Train: 15 images
# Val: 3 images
# Test: 3 images
SPLIT_MAPPING = {
    filenames[0]: "train",   # 1
    filenames[1]: "train",   # 2
    filenames[2]: "val",     # 3 (Validation)
    filenames[3]: "train",   # 4
    filenames[4]: "train",   # 5
    filenames[5]: "train",   # 6
    filenames[6]: "train",   # 7
    filenames[7]: "train",   # 8
    filenames[8]: "val",     # 9 (Validation)
    filenames[9]: "train",   # 10
    filenames[10]: "train",  # 11
    filenames[11]: "test",   # 12 (Held-out Test)
    filenames[12]: "train",  # 13
    filenames[13]: "train",  # 14
    filenames[14]: "test",   # 15 (Held-out Test)
    filenames[15]: "train",  # 16
    filenames[16]: "train",  # 17
    filenames[17]: "val",    # 18 (Validation)
    filenames[18]: "train",  # 19
    filenames[19]: "train",  # 20
    filenames[20]: "test"    # 21 (Held-out Test)
}

with open(os.path.join(SPLITS_DIR, "split_definition.json"), "w") as f:
    json.dump(SPLIT_MAPPING, f, indent=2)

# Detailed Ground-Truth Bounding Box Database [x, y, w, h] for 1280x960
# Hand-checked against image pixel coordinates:
ANNOTATIONS = {
    # 1. WhatsApp Image 2026-10-27 at 8.01.57 PM.jpeg
    filenames[0]: [
        {"class": "cream_container", "bbox": [1010, 500, 100, 95]},
        {"class": "mouse", "bbox": [945, 680, 115, 175]},
        {"class": "microcontroller_board", "bbox": [860, 615, 75, 90]}
    ],
    # 2. WhatsApp Image 2026-10-27 at 8.01.58 PM (1).jpeg
    filenames[1]: [
        {"class": "cream_container", "bbox": [745, 335, 70, 60]},
        {"class": "mouse", "bbox": [755, 410, 100, 130]},
        {"class": "glue_stick", "bbox": [550, 360, 25, 75]}
    ],
    # 3. WhatsApp Image 2026-10-27 at 8.01.58 PM (2).jpeg (VAL)
    filenames[2]: [
        {"class": "cream_container", "bbox": [1055, 670, 85, 75]},
        {"class": "glue_stick", "bbox": [1015, 420, 30, 80]},
        {"class": "microcontroller_board", "bbox": [1030, 630, 45, 55]}
    ],
    # 4. WhatsApp Image 2026-10-27 at 8.01.58 PM.jpeg
    filenames[3]: [
        {"class": "cream_container", "bbox": [338, 875, 120, 100]}, # near front right
        {"class": "mouse", "bbox": [190, 715, 160, 205]},
        {"class": "glue_stick", "bbox": [115, 390, 35, 90]},
        {"class": "microcontroller_board", "bbox": [230, 615, 65, 75]}
    ],
    # 5. WhatsApp Image 2026-10-27 at 8.01.59 PM.jpeg
    filenames[4]: [
        {"class": "cream_container", "bbox": [745, 370, 50, 45]}
    ],
    # 6. WhatsApp Image 2026-10-27 at 8.02.00 PM.jpeg
    filenames[5]: [
        {"class": "cream_container", "bbox": [1040, 700, 75, 65]},
        {"class": "mouse", "bbox": [890, 780, 125, 140]},
        {"class": "microcontroller_board", "bbox": [1010, 640, 40, 55]}
    ],
    # 7. WhatsApp Image 2026-10-27 at 8.02.08 PM.jpeg
    filenames[6]: [
        {"class": "cream_container", "bbox": [178, 785, 95, 90]},
        {"class": "earbuds_case", "bbox": [900, 630, 110, 115]},
        {"class": "mouse", "bbox": [655, 770, 130, 110]}
    ],
    # 8. WhatsApp Image 2026-10-27 at 8.02.09 PM.jpeg
    filenames[7]: [
        {"class": "cream_container", "bbox": [430, 755, 65, 55]},
        {"class": "earbuds_case", "bbox": [735, 695, 45, 50]},
        {"class": "mouse", "bbox": [610, 755, 95, 65]}
    ],
    # 9. WhatsApp Image 2026-10-27 at 8.02.36 PM.jpeg (VAL)
    filenames[8]: [
        {"class": "earbuds_case", "bbox": [1125, 730, 60, 50]},
        {"class": "mouse", "bbox": [995, 730, 85, 55]},
        {"class": "glue_stick", "bbox": [1190, 705, 30, 55]}
    ],
    # 10. WhatsApp Image 2026-10-27 at 8.02.37 PM (1).jpeg
    filenames[9]: [
        {"class": "earbuds_case", "bbox": [90, 635, 80, 105]},
        {"class": "glue_stick", "bbox": [255, 615, 35, 70]},
        {"class": "microcontroller_board", "bbox": [320, 660, 55, 50]}
    ],
    # 11. WhatsApp Image 2026-10-27 at 8.02.37 PM (2).jpeg
    filenames[10]: [
        {"class": "earbuds_case", "bbox": [765, 570, 45, 40]},
        {"class": "mouse", "bbox": [815, 630, 40, 70]}
    ],
    # 12. WhatsApp Image 2026-10-27 at 8.02.37 PM.jpeg (TEST)
    filenames[11]: [
        {"class": "earbuds_case", "bbox": [1035, 660, 55, 45]},
        {"class": "glue_stick", "bbox": [1135, 655, 25, 55]}
    ],
    # 13. WhatsApp Image 2026-10-27 at 8.02.38 PM (1).jpeg
    filenames[12]: [
        {"class": "cream_container", "bbox": [355, 790, 60, 50]}
    ],
    # 14. WhatsApp Image 2026-10-27 at 8.02.38 PM (2).jpeg
    filenames[13]: [
        {"class": "glue_stick", "bbox": [675, 845, 40, 70]}
    ],
    # 15. WhatsApp Image 2026-10-27 at 8.02.38 PM.jpeg (TEST)
    filenames[14]: [
        {"class": "cream_container", "bbox": [960, 790, 50, 40]}
    ],
    # 16. WhatsApp Image 2026-10-27 at 8.02.39 PM (1).jpeg
    filenames[15]: [
        {"class": "cream_container", "bbox": [10, 825, 60, 55]},
        {"class": "earbuds_case", "bbox": [350, 785, 65, 50]}
    ],
    # 17. WhatsApp Image 2026-10-27 at 8.02.39 PM.jpeg
    filenames[16]: [
        {"class": "cream_container", "bbox": [470, 775, 45, 25]}
    ],
    # 18. WhatsApp Image 2026-10-27 at 8.02.40 PM (1).jpeg (VAL)
    filenames[17]: [
        {"class": "cream_container", "bbox": [900, 770, 35, 30]},
        {"class": "earbuds_case", "bbox": [1035, 830, 40, 35]},
        {"class": "mouse", "bbox": [900, 795, 65, 50]}
    ],
    # 19. WhatsApp Image 2026-10-27 at 8.02.40 PM.jpeg
    filenames[18]: [
        {"class": "cream_container", "bbox": [90, 970, 45, 35]},
        {"class": "earbuds_case", "bbox": [305, 945, 40, 40]}
    ],
    # 20. WhatsApp Image 2026-10-27 at 8.02.41 PM (1).jpeg
    filenames[19]: [
        {"class": "cream_container", "bbox": [455, 940, 50, 45]},
        {"class": "earbuds_case", "bbox": [735, 935, 55, 50]},
        {"class": "mouse", "bbox": [635, 970, 45, 35]}
    ],
    # 21. WhatsApp Image 2026-10-27 at 8.02.41 PM.jpeg (TEST)
    filenames[20]: [
        {"class": "cream_container", "bbox": [940, 965, 45, 40]},
        {"class": "earbuds_case", "bbox": [1160, 905, 45, 45]},
        {"class": "mouse", "bbox": [1085, 945, 65, 50]}
    ]
}

# Negative background sampling coordinates per image (regions that contain no target objects):
# Keyboard keys, laptop palm rest, world map fabric, desk wood, cloth
BACKGROUND_REGIONS = [
    [150, 500, 80, 80],   # Keyboard left
    [250, 520, 80, 80],   # Keyboard center
    [50, 700, 90, 90],    # Desk mat lower left
    [400, 300, 100, 80],  # Laptop body
    [100, 200, 90, 90],   # Cloth / wall
    [500, 750, 80, 80],   # Desk mat ocean area
    [700, 550, 80, 80]    # Desk mat text area
]

# Color mapping for visualization
COLORS = {
    "cream_container": (0, 165, 255),    # Orange
    "mouse": (255, 0, 255),              # Magenta
    "earbuds_case": (255, 255, 0),       # Cyan
    "glue_stick": (0, 255, 255),         # Yellow
    "microcontroller_board": (0, 0, 255),# Red
    "background": (128, 128, 128)        # Gray
}

# Save full annotations JSON
with open(os.path.join(ANNOTATIONS_DIR, "annotations.json"), "w") as f:
    json.dump(ANNOTATIONS, f, indent=2)

print("Saved annotations.json. Extracting crops...")

# Extract crops and save into train/val/test directories
target_classes = ["cream_container", "mouse", "earbuds_case", "glue_stick", "microcontroller_board", "background"]
crop_stats = {"train": {c: 0 for c in target_classes},
              "val": {c: 0 for c in target_classes},
              "test": {c: 0 for c in target_classes}}

for split in ["train", "val", "test"]:
    for cls in target_classes:
        os.makedirs(os.path.join(OUTPUT_CROPS_DIR, split, cls), exist_ok=True)

annotated_thumbs = []

for idx, fpath in enumerate(files):
    fname = os.path.basename(fpath)
    split = SPLIT_MAPPING[fname]
    img = cv2.imread(fpath)
    h_img, w_img = img.shape[:2]
    
    # Copy for annotation visualization
    vis_img = img.copy()
    
    # Extract annotated object crops
    boxes = ANNOTATIONS.get(fname, [])
    for b_idx, item in enumerate(boxes):
        cls = item["class"]
        x, y, w, h = item["bbox"]
        
        # Clamp to image boundaries
        x = max(0, min(x, w_img - 1))
        y = max(0, min(y, h_img - 1))
        w = max(10, min(w, w_img - x))
        h = max(10, min(h, h_img - y))
        
        crop = img[y:y+h, x:x+w]
        if crop.size > 0:
            crop_fname = f"{split}_{cls}_{idx+1}_{b_idx}.png"
            cv2.imwrite(os.path.join(OUTPUT_CROPS_DIR, split, cls, crop_fname), crop)
            crop_stats[split][cls] += 1
            
        # Draw on vis_img
        col = COLORS.get(cls, (0, 255, 0))
        cv2.rectangle(vis_img, (x, y), (x+w, y+h), col, 3)
        cv2.putText(vis_img, cls, (x, max(20, y-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        
    # Extract background crops
    for bg_idx, bg_box in enumerate(BACKGROUND_REGIONS):
        bx, by, bw, bh = bg_box
        bx = max(0, min(bx, w_img - 1))
        by = max(0, min(by, h_img - 1))
        bw = max(10, min(bw, w_img - bx))
        bh = max(10, min(bh, h_img - by))
        bg_crop = img[by:by+bh, bx:bx+bw]
        if bg_crop.size > 0:
            bg_fname = f"{split}_background_{idx+1}_{bg_idx}.png"
            cv2.imwrite(os.path.join(OUTPUT_CROPS_DIR, split, "background", bg_fname), bg_crop)
            crop_stats[split]["background"] += 1

    # Thumbnail for verification contact sheet
    thumb = cv2.resize(vis_img, (320, 240), interpolation=cv2.INTER_AREA)
    cv2.rectangle(thumb, (0, 0), (320, 24), (0, 0, 0), -1)
    split_col = (0, 255, 0) if split == "train" else ((0, 255, 255) if split == "val" else (0, 0, 255))
    cv2.putText(thumb, f"#{idx+1} [{split.upper()}] {len(boxes)} objs", (5, 17), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, split_col, 1, cv2.LINE_AA)
    annotated_thumbs.append(thumb)

# Generate contact sheet grid
cols = 3
rows = int(np.ceil(len(annotated_thumbs) / cols))
grid_h = rows * 240
grid_w = cols * 320
annotated_sheet = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

for i, thumb in enumerate(annotated_thumbs):
    r = i // cols
    c = i % cols
    annotated_sheet[r*240:(r+1)*240, c*320:(c+1)*320] = thumb

out_sheet_path = os.path.join(SCREENSHOTS_DIR, "annotated_dataset_verification.png")
cv2.imwrite(out_sheet_path, annotated_sheet)

print(f"Annotated verification sheet saved to {out_sheet_path}")
print("Crop extraction statistics:")
for s, counts in crop_stats.items():
    print(f" Split [{s.upper()}]:", counts)
