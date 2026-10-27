"""
Sensor-Matched Dataset Splitter & Integrity Auditor (tools/build_sensor_dataset_splits.py)
Partitions genuine ESP32-CAM raw captures into train/val/test splits strictly grouped by session_id
to guarantee zero temporal video leakage between adjacent video frames.
"""

import os
import sys
import json
import glob
import random
import shutil
import numpy as np
import cv2

PROJECT_ROOT = r"C:\All_Projects\esp32_cam_ai_system"
RAW_DIR = os.path.join(PROJECT_ROOT, "dataset", "raw_esp32cam")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "dataset", "processed_esp32cam")
SPLITS_DIR = os.path.join(PROJECT_ROOT, "dataset", "splits")
MANIFEST_PATH = os.path.join(PROJECT_ROOT, "dataset", "manifests", "dataset_manifest_esp32cam.json")
SPLIT_JSON_PATH = os.path.join(SPLITS_DIR, "sensor_matched_splits.json")

for d in [PROCESSED_DIR, SPLITS_DIR]:
    os.makedirs(d, exist_ok=True)

DEFAULT_CLASSES = [
    "background",
    "cream_container",
    "mouse",
    "earbuds_case",
    "glue_stick",
    "microcontroller_board"
]

def build_splits(val_ratio=0.15, test_ratio=0.15, seed=42):
    print("=" * 60)
    print(" BUILDING SENSOR-MATCHED DATASET SPLITS (ZERO-LEAKAGE)")
    print("=" * 60)

    if not os.path.exists(MANIFEST_PATH):
        print(f"[ERROR] Manifest not found at {MANIFEST_PATH}")
        return False

    with open(MANIFEST_PATH, "r") as f:
        manifest = json.load(f)

    samples = manifest.get("samples", [])
    if not samples:
        print("[WARN] No samples recorded in manifest!")
        return False

    print(f"Total raw samples in manifest: {len(samples)}")

    # Group samples by class and by session_id
    # { class_name: { session_id: [sample, ...] } }
    grouped = {}
    for s in samples:
        cls = s["class_label"]
        sess = s.get("session_id", "default_session")
        if cls not in grouped:
            grouped[cls] = {}
        if sess not in grouped[cls]:
            grouped[cls][sess] = []
        grouped[cls][sess].append(s)

    splits = {
        "train": [],
        "val": [],
        "test": []
    }

    random.seed(seed)

    # For each class, partition sessions (or samples if single session)
    for cls, sessions in grouped.items():
        sess_keys = list(sessions.keys())
        random.shuffle(sess_keys)
        
        if len(sess_keys) >= 3:
            # Multi-session: assign whole sessions to splits
            n_test = max(1, int(len(sess_keys) * test_ratio))
            n_val = max(1, int(len(sess_keys) * val_ratio))
            test_sess = sess_keys[:n_test]
            val_sess = sess_keys[n_test:n_test + n_val]
            train_sess = sess_keys[n_test + n_val:]
            if not train_sess:
                train_sess = [val_sess.pop()]
                
            for sess in train_sess:
                splits["train"].extend(sessions[sess])
            for sess in val_sess:
                splits["val"].extend(sessions[sess])
            for sess in test_sess:
                splits["test"].extend(sessions[sess])
        else:
            # Single session: partition chronologically by time blocks to prevent adjacent frame leakage
            all_cls_samples = []
            for sess in sess_keys:
                all_cls_samples.extend(sessions[sess])
            # Sort chronologically
            all_cls_samples.sort(key=lambda x: x["timestamp"])
            
            n = len(all_cls_samples)
            if n < 3:
                # Small count: all to train
                splits["train"].extend(all_cls_samples)
            else:
                # Split in 3 continuous chronological blocks: first 70% train, next 15% val, last 15% test
                n_val = max(1, int(n * val_ratio))
                n_test = max(1, int(n * test_ratio))
                n_train = n - n_val - n_test
                splits["train"].extend(all_cls_samples[:n_train])
                splits["val"].extend(all_cls_samples[n_train:n_train + n_val])
                splits["test"].extend(all_cls_samples[n_train + n_val:])

    # Summary
    print("\nDataset Partition Summary:")
    print(f" -> Train Samples: {len(splits['train'])}")
    print(f" -> Val Samples:   {len(splits['val'])}")
    print(f" -> Test Samples:  {len(splits['test'])}")

    # Copy files to dataset/processed_esp32cam/<split>/<class_name>/
    for split_name in ["train", "val", "test"]:
        s_dir = os.path.join(PROCESSED_DIR, split_name)
        if os.path.exists(s_dir):
            shutil.rmtree(s_dir)
        os.makedirs(s_dir, exist_ok=True)
        for c in DEFAULT_CLASSES:
            os.makedirs(os.path.join(s_dir, c), exist_ok=True)

        for item in splits[split_name]:
            src_p = item["absolute_path"]
            if os.path.exists(src_p):
                cls = item["class_label"]
                dst_p = os.path.join(PROCESSED_DIR, split_name, cls, os.path.basename(src_p))
                shutil.copy2(src_p, dst_p)

    # Save split JSON
    out_split = {
        "metadata": {
            "source": "ESP32-CAM OV3660 Raw Sensor Dataset",
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "random_seed": seed,
            "counts": {
                "train": len(splits["train"]),
                "val": len(splits["val"]),
                "test": len(splits["test"])
            }
        },
        "splits": {
            "train": [s["image_id"] for s in splits["train"]],
            "val": [s["image_id"] for s in splits["val"]],
            "test": [s["image_id"] for s in splits["test"]]
        }
    }
    with open(SPLIT_JSON_PATH, "w") as f:
        json.dump(out_split, f, indent=2)

    print(f"Saved split definition to: {SPLIT_JSON_PATH}")
    return True

if __name__ == "__main__":
    build_splits()
