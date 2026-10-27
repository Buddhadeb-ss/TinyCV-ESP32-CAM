import os
import json
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

IMAGES_DIR = r"C:\All_Projects\esp32_cam_ai_system\images"
INVENTORY_DIR = r"C:\All_Projects\esp32_cam_ai_system\dataset\inventory"
SCREENSHOTS_DIR = r"C:\All_Projects\esp32_cam_ai_system\screenshots"

os.makedirs(INVENTORY_DIR, exist_ok=True)
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)

files = sorted([f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
inventory = []
thumbs = []

print(f"Analyzing {len(files)} images...")

for idx, fname in enumerate(files):
    fpath = os.path.join(IMAGES_DIR, fname)
    size_bytes = os.path.getsize(fpath)
    
    # Load using OpenCV to calculate image quality metrics
    img = cv2.imread(fpath)
    if img is None:
        print(f"Failed to read {fname}")
        continue
    h, w, c = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_val = float(np.mean(gray))
    std_val = float(np.std(gray))
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    
    # Aspect ratio
    aspect = round(w / h, 3)
    
    item = {
        "id": idx + 1,
        "filename": fname,
        "width": w,
        "height": h,
        "channels": c,
        "size_bytes": size_bytes,
        "aspect_ratio": aspect,
        "mean_intensity": round(mean_val, 2),
        "std_intensity": round(std_val, 2),
        "blur_laplacian_var": round(laplacian_var, 2),
        "sharpness_category": "sharp" if laplacian_var > 100 else ("acceptable" if laplacian_var > 40 else "blurry")
    }
    inventory.append(item)
    
    # Thumbnail for contact sheet
    thumb_w, thumb_h = 320, 240
    thumb = cv2.resize(img, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
    # Overlay label
    cv2.rectangle(thumb, (0, 0), (thumb_w, 24), (0, 0, 0), -1)
    cv2.putText(thumb, f"#{idx+1} {fname[:18]}... ({w}x{h})", (5, 17), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
    thumbs.append(thumb)

# Save inventory JSON
inventory_path = os.path.join(INVENTORY_DIR, "image_inventory.json")
with open(inventory_path, "w") as f:
    json.dump(inventory, f, indent=2)
print(f"Inventory saved to {inventory_path}")

# Create contact sheet grid (3 columns x 7 rows = 21 images)
cols = 3
rows = int(np.ceil(len(thumbs) / cols))
grid_h = rows * 240
grid_w = cols * 320
contact_sheet = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

for i, thumb in enumerate(thumbs):
    r = i // cols
    c = i % cols
    y = r * 240
    x = c * 320
    contact_sheet[y:y+240, x:x+320] = thumb

sheet_path1 = os.path.join(INVENTORY_DIR, "contact_sheet.png")
sheet_path2 = os.path.join(SCREENSHOTS_DIR, "dataset_contact_sheet.png")
cv2.imwrite(sheet_path1, contact_sheet)
cv2.imwrite(sheet_path2, contact_sheet)
print(f"Contact sheet saved to {sheet_path1} and {sheet_path2}")
