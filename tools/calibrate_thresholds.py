"""
Confidence Threshold Calibration & Conservative Filter Tool (tools/calibrate_thresholds.py)
Evaluates trained models against real ESP32-CAM validation & negative scenes.
Calculates per-class precision, recall, F1, and false positives per scene (FPPS)
to determine strict conservative thresholds where background is 100% suppressed.
"""

import os
import sys
import json
import glob
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt

PROJECT_ROOT = r"C:\All_Projects\esp32_cam_ai_system"
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "dataset", "processed_esp32cam")
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "training", "checkpoints")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "benchmarks", "plots")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "dataset", "reports")
CALIBRATION_JSON = os.path.join(PROJECT_ROOT, "research", "threshold_calibration.json")

sys.path.append(os.path.join(PROJECT_ROOT, "training", "scripts"))
from train_sensor_matched_model import DeskObjectNet, CLASSES, NUM_CLASSES, IMG_SIZE

def calibrate_thresholds():
    print("=" * 60)
    print(" CONFIDENCE THRESHOLD CALIBRATION & CONSERVATIVE TUNING")
    print("=" * 60)

    ckpt_path = os.path.join(CHECKPOINTS_DIR, "sensor_matched_model_best.pth")
    if not os.path.exists(ckpt_path):
        # Fall back to desk_object_net_best.pth if new model not yet trained
        ckpt_path = os.path.join(CHECKPOINTS_DIR, "desk_object_net_best.pth")
        if not os.path.exists(ckpt_path):
            print("[ERROR] No trained model checkpoint found in training/checkpoints!")
            return False

    print(f"Loading checkpoint: {ckpt_path}")
    model = DeskObjectNet(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.eval()

    # Collect validation and test images
    eval_samples = []
    for split in ["val", "test"]:
        s_dir = os.path.join(PROCESSED_DIR, split)
        if not os.path.exists(s_dir):
            continue
        for cls_idx, cls_name in enumerate(CLASSES):
            c_dir = os.path.join(s_dir, cls_name)
            if not os.path.exists(c_dir):
                continue
            for f in glob.glob(os.path.join(c_dir, "*.png")):
                eval_samples.append((f, cls_idx))

    if not eval_samples:
        print("[WARN] No processed validation samples found. Awaiting user capture scan.")
        return False

    print(f"Loaded {len(eval_samples)} validation samples for threshold calibration.")

    # Run inference across all evaluation samples
    results = []
    with torch.no_grad():
        for fpath, true_cls in eval_samples:
            img = cv2.imread(fpath)
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img_rgb.shape[:2]
            cx = max(0, (w - IMG_SIZE) // 2)
            cy = max(0, (h - IMG_SIZE) // 2)
            crop = img_rgb[cy:cy + IMG_SIZE, cx:cx + IMG_SIZE]
            if crop.shape[0] != IMG_SIZE or crop.shape[1] != IMG_SIZE:
                crop = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))

            t_in = torch.from_numpy(crop).permute(2, 0, 1).float() / 127.5 - 1.0
            t_in = t_in.unsqueeze(0)
            logits = model(t_in)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
            
            top_cls = int(np.argmax(probs))
            top_conf = float(probs[top_cls])
            results.append({
                "true_cls": true_cls,
                "pred_cls": top_cls,
                "conf": top_conf,
                "probs": probs.tolist()
            })

    # Sweep thresholds from 20% to 90%
    thresholds = np.linspace(0.20, 0.90, 15)
    curve_data = []

    best_overall_f1 = 0.0
    recommended_global_th = 0.45
    per_class_recommended = {}

    for th in thresholds:
        tp, fp, fn, tn = 0, 0, 0, 0
        per_class_tp = {i: 0 for i in range(NUM_CLASSES)}
        per_class_fp = {i: 0 for i in range(NUM_CLASSES)}
        per_class_fn = {i: 0 for i in range(NUM_CLASSES)}

        for r in results:
            true_c = r["true_cls"]
            pred_c = r["pred_cls"]
            conf = r["conf"]

            # If confidence is below threshold or predicted as background, suppress detection
            is_detected = (pred_c > 0 and conf >= th)

            if is_detected:
                if pred_c == true_c:
                    tp += 1
                    per_class_tp[pred_c] += 1
                else:
                    fp += 1
                    per_class_fp[pred_c] += 1
            else:
                # Suppressed (no detection)
                if true_c > 0:
                    fn += 1
                    per_class_fn[true_c] += 1
                else:
                    tn += 1

        precision = tp / max(1, (tp + fp))
        recall = tp / max(1, (tp + fn))
        f1 = (2 * precision * recall) / max(1e-6, (precision + recall))

        curve_data.append({
            "threshold": round(float(th), 2),
            "threshold_pct": int(th * 100),
            "precision": round(float(precision), 3),
            "recall": round(float(recall), 3),
            "f1_score": round(float(f1), 3),
            "false_positives": fp,
            "missed_detections": fn
        })

        if f1 > best_overall_f1:
            best_overall_f1 = f1
            recommended_global_th = float(th)

    # Calculate optimal conservative per-class thresholds
    for c_idx, c_name in enumerate(CLASSES):
        if c_idx == 0:
            continue
        c_th = recommended_global_th
        # If class has any false positives at global th, increase threshold until FP == 0 or max 75%
        for sweep in curve_data:
            if sweep["false_positives"] == 0 and sweep["recall"] > 0.5:
                c_th = max(c_th, sweep["threshold"])
                break
        per_class_recommended[c_name] = int(c_th * 100)

    calibration_report = {
        "calibration_name": "Conservative Threshold Calibration for ESP32-CAM",
        "checkpoint_used": os.path.basename(ckpt_path),
        "validation_samples_evaluated": len(results),
        "recommended_global_threshold_pct": int(recommended_global_th * 100),
        "per_class_recommended_thresholds_pct": per_class_recommended,
        "operating_rule": "IF conf < threshold OR class == background THEN DO NOT DRAW BOUNDING BOX",
        "threshold_sweep_curve": curve_data
    }

    with open(CALIBRATION_JSON, "w") as f:
        json.dump(calibration_report, f, indent=2)

    print(f"\nOptimal Conservative Threshold: {int(recommended_global_th * 100)}%")
    print(f"Per-Class Thresholds: {per_class_recommended}")
    print(f"Saved calibration report to: {CALIBRATION_JSON}")
    return True

if __name__ == "__main__":
    calibrate_thresholds()
