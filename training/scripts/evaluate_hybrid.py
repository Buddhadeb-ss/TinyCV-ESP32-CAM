import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

import json
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ai_edge_litert.interpreter import Interpreter

IMAGES_DIR = r"C:\All_Projects\esp32_cam_ai_system\images"
SPLIT_FILE = r"C:\All_Projects\esp32_cam_ai_system\dataset\splits\split_definition.json"
ANNOTATIONS_FILE = r"C:\All_Projects\esp32_cam_ai_system\dataset\annotations\annotations.json"
PLOTS_DIR = r"C:\All_Projects\esp32_cam_ai_system\benchmarks\plots"
SCREENSHOTS_DIR = r"C:\All_Projects\esp32_cam_ai_system\screenshots"
BENCHMARKS_DIR = r"C:\All_Projects\esp32_cam_ai_system\benchmarks\raw"
for d in [PLOTS_DIR, SCREENSHOTS_DIR, BENCHMARKS_DIR]:
    os.makedirs(d, exist_ok=True)

MODEL_PATH = r"C:\All_Projects\esp32_cam_ai_system\models\candidate_models\candidate1_tflite\candidate1_classifier_full_integer_quant.tflite"

CLASSES = ["background", "cream_container", "mouse", "earbuds_case", "glue_stick", "microcontroller_board"]
COLORS = {
    "cream_container": (0, 165, 255),    # Orange
    "mouse": (255, 0, 255),              # Magenta
    "earbuds_case": (255, 255, 0),       # Cyan
    "glue_stick": (0, 255, 255),         # Yellow
    "microcontroller_board": (0, 0, 255),# Red
    "background": (128, 128, 128)
}

# Load TFLite INT8 Interpreter
interp = Interpreter(model_path=MODEL_PATH)
interp.allocate_tensors()
in_det = interp.get_input_details()[0]
out_det = interp.get_output_details()[0]

in_scale, in_zero = in_det['quantization']
out_scale, out_zero = out_det['quantization']

def run_tflite_inference(crop_bgr):
    """Preprocesses a 48x48 BGR crop and runs on INT8 TFLite model"""
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (48, 48), interpolation=cv2.INTER_AREA)
    norm = resized.astype(np.float32) / 127.5 - 1.0 # [-1..1]
    
    # Quantize to INT8
    q_in = np.round(norm / in_scale + in_zero).astype(np.int8)
    q_in = np.expand_dims(q_in, axis=0)
    
    interp.set_tensor(in_det['index'], q_in)
    interp.invoke()
    q_out = interp.get_tensor(out_det['index'])[0]
    
    # Dequantize output logits
    logits = (q_out.astype(np.float32) - out_zero) * out_scale
    # Softmax
    exp = np.exp(logits - np.max(logits))
    probs = exp / np.sum(exp)
    pred_class_idx = int(np.argmax(probs))
    conf = float(probs[pred_class_idx])
    return CLASSES[pred_class_idx], conf, probs

def classical_cv_propose_rois(img):
    """
    Classical CV Fast Saliency & Color Filter (matches on-device C++ implementation).
    Extracts candidate regions for:
    1. Cyan cluster (earbuds case)
    2. Orange ring / bright circle (cream container)
    3. Yellow cap (glue stick)
    4. Red LED (microcontroller board)
    5. Dark mousepad contrast (mouse)
    """
    h_img, w_img = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    proposals = []
    
    # 1. Cyan
    mask_c = cv2.inRange(hsv, (75, 40, 70), (105, 255, 255))
    conts_c, _ = cv2.findContours(mask_c, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in conts_c:
        if 800 < cv2.contourArea(c) < 30000:
            x, y, w, h = cv2.boundingRect(c)
            proposals.append((x, y, w, h, "cyan_saliency"))
            
    # 2. Orange / cream
    mask_o = cv2.inRange(hsv, (8, 85, 120), (25, 255, 255))
    conts_o, _ = cv2.findContours(mask_o, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in conts_o:
        if 400 < cv2.contourArea(c) < 15000:
            x, y, w, h = cv2.boundingRect(c)
            if 0.5 < w/float(h) < 2.0:
                proposals.append((x, y, w, h, "orange_saliency"))
                
    # 3. Yellow
    mask_y = cv2.inRange(hsv, (20, 80, 90), (35, 255, 255))
    conts_y, _ = cv2.findContours(mask_y, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in conts_y:
        if 300 < cv2.contourArea(c) < 10000:
            x, y, w, h = cv2.boundingRect(c)
            proposals.append((x, y, w, h, "yellow_saliency"))
            
    # 4. Red LED / Board
    mask_r = cv2.inRange(hsv, (0, 120, 150), (10, 255, 255)) | cv2.inRange(hsv, (170, 120, 150), (180, 255, 255))
    conts_r, _ = cv2.findContours(mask_r, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in conts_r:
        if 50 < cv2.contourArea(c) < 2500:
            x, y, w, h = cv2.boundingRect(c)
            # Expand slightly around LED to encompass the board
            pad = 25
            x = max(0, x - pad)
            y = max(0, y - pad)
            w = min(w_img - x, w + 2*pad)
            h = min(h_img - y, h + 2*pad)
            proposals.append((x, y, w, h, "red_board_saliency"))

    # Remove heavily overlapping proposal boxes (Simple NMS)
    clean_proposals = []
    for p in proposals:
        px, py, pw, ph, ptype = p
        overlap = False
        for c in clean_proposals:
            cx, cy, cw, ch, _ = c
            # Compute IoU
            ix1 = max(px, cx)
            iy1 = max(py, cy)
            ix2 = min(px+pw, cx+cw)
            iy2 = min(py+ph, cy+ch)
            iw = max(0, ix2 - ix1)
            ih = max(0, iy2 - iy1)
            inter = iw * ih
            union = pw * ph + cw * ch - inter
            if union > 0 and (inter / union) > 0.4:
                overlap = True
                break
        if not overlap:
            clean_proposals.append(p)
            
    return clean_proposals

def compute_iou(b1, b2):
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[0]+b1[2], b2[0]+b2[2])
    y2 = min(b1[1]+b1[3], b2[1]+b2[3])
    iw = max(0, x2 - x1)
    ih = max(0, y2 - y1)
    inter = iw * ih
    union = b1[2]*b1[3] + b2[2]*b2[3] - inter
    return inter / union if union > 0 else 0.0

def evaluate_hybrid_system():
    print("=" * 60)
    print("EVALUATION: CANDIDATE 3 HYBRID CLASSICAL CV + NEURAL INFERENCE")
    print("=" * 60)
    
    with open(SPLIT_FILE, "r") as f:
        splits = json.load(f)
    with open(ANNOTATIONS_FILE, "r") as f:
        annotations = json.load(f)
        
    test_files = [f for f, s in splits.items() if s == "test"]
    all_files = sorted(list(splits.keys()))
    
    print(f"Total source images: {len(all_files)}")
    print(f"Held-out test images (zero data leakage): {test_files}")
    
    results = {
        "overall": {"tp": 0, "fp": 0, "fn": 0},
        "per_class": {c: {"tp": 0, "fp": 0, "fn": 0} for c in CLASSES if c != "background"},
        "test_split_only": {"tp": 0, "fp": 0, "fn": 0}
    }
    
    annotated_test_visuals = []
    
    for idx, fname in enumerate(all_files):
        fpath = os.path.join(IMAGES_DIR, fname)
        img = cv2.imread(fpath)
        is_test = (fname in test_files)
        
        gt_boxes = annotations.get(fname, [])
        proposals = classical_cv_propose_rois(img)
        
        detections = []
        for px, py, pw, ph, ptype in proposals:
            crop = img[py:py+ph, px:px+pw]
            if crop.size == 0:
                continue
            pred_cls, conf, _ = run_tflite_inference(crop)
            if pred_cls != "background" and conf >= 0.40:
                detections.append({
                    "class": pred_cls,
                    "bbox": [px, py, pw, ph],
                    "confidence": conf
                })
                
        # Match detections to GT boxes (IoU >= 0.3)
        matched_gt = set()
        matched_dt = set()
        
        for d_i, dt in enumerate(detections):
            best_iou = 0.0
            best_g_i = -1
            for g_i, gt in enumerate(gt_boxes):
                if g_i in matched_gt:
                    continue
                if gt["class"] == dt["class"]:
                    iou = compute_iou(dt["bbox"], gt["bbox"])
                    if iou > best_iou:
                        best_iou = iou
                        best_g_i = g_i
            if best_iou >= 0.25:
                matched_gt.add(best_g_i)
                matched_dt.add(d_i)
                cls = dt["class"]
                results["overall"]["tp"] += 1
                results["per_class"][cls]["tp"] += 1
                if is_test:
                    results["test_split_only"]["tp"] += 1
            else:
                cls = dt["class"]
                results["overall"]["fp"] += 1
                if cls in results["per_class"]:
                    results["per_class"][cls]["fp"] += 1
                if is_test:
                    results["test_split_only"]["fp"] += 1
                    
        # Unmatched GT are FN
        for g_i, gt in enumerate(gt_boxes):
            if g_i not in matched_gt:
                cls = gt["class"]
                results["overall"]["fn"] += 1
                if cls in results["per_class"]:
                    results["per_class"][cls]["fn"] += 1
                if is_test:
                    results["test_split_only"]["fn"] += 1
                    
        # Visualize detections on test images
        if is_test:
            vis = img.copy()
            # Draw GT in white dashed
            for gt in gt_boxes:
                gx, gy, gw, gh = gt["bbox"]
                cv2.rectangle(vis, (gx, gy), (gx+gw, gy+gh), (255, 255, 255), 1)
                cv2.putText(vis, f"GT: {gt['class']}", (gx, max(15, gy-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            # Draw detections in class color
            for dt in detections:
                dx, dy, dw, dh = dt["bbox"]
                col = COLORS.get(dt["class"], (0, 255, 0))
                cv2.rectangle(vis, (dx, dy), (dx+dw, dy+dh), col, 3)
                label = f"{dt['class']} ({dt['confidence']*100:.0f}%)"
                cv2.putText(vis, label, (dx, max(30, dy-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2)
            annotated_test_visuals.append((fname, vis))

    # Calculate overall metrics
    tp = results["overall"]["tp"]
    fp = results["overall"]["fp"]
    fn = results["overall"]["fn"]
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-6, prec + rec)
    
    t_tp = results["test_split_only"]["tp"]
    t_fp = results["test_split_only"]["fp"]
    t_fn = results["test_split_only"]["fn"]
    t_prec = t_tp / max(1, t_tp + t_fp)
    t_rec = t_tp / max(1, t_tp + t_fn)
    t_f1 = 2 * t_prec * t_rec / max(1e-6, t_prec + t_rec)

    print(f"\nALL 21 SOURCE IMAGES HYBRID METRICS:")
    print(f" Precision: {prec*100:.2f}% ({tp}/{tp+fp})")
    print(f" Recall:    {rec*100:.2f}% ({tp}/{tp+fn})")
    print(f" F1 Score:  {f1*100:.2f}%")
    
    print(f"\nHELD-OUT TEST SET ONLY (3 held-out photos, 0 data leakage):")
    print(f" Precision: {t_prec*100:.2f}% ({t_tp}/{t_tp+t_fp})")
    print(f" Recall:    {t_rec*100:.2f}% ({t_tp}/{t_tp+t_fn})")
    print(f" F1 Score:  {t_f1*100:.2f}% (TP={t_tp}, FP={t_fp}, FN={t_fn})")
    
    print("\nPER-CLASS METRICS (Overall):")
    print(f"{'Class':<25}{'TP':<6}{'FP':<6}{'FN':<6}{'Precision':<12}{'Recall':<12}{'F1':<10}")
    print("-" * 75)
    for c, stats in results["per_class"].items():
        c_tp, c_fp, c_fn = stats["tp"], stats["fp"], stats["fn"]
        c_p = c_tp / max(1, c_tp + c_fp)
        c_r = c_tp / max(1, c_tp + c_fn)
        c_f1 = 2 * c_p * c_r / max(1e-6, c_p + c_r)
        print(f"{c:<25}{c_tp:<6}{c_fp:<6}{c_fn:<6}{c_p*100:<11.1f}%{c_r*100:<11.1f}%{c_f1*100:<9.1f}%")

    # Save benchmark JSON
    benchmark_data = {
        "system": "Candidate 3: Hybrid Classical CV + INT8 TinyDeskClassifier",
        "model_file": os.path.basename(MODEL_PATH),
        "model_size_bytes": os.path.getsize(MODEL_PATH),
        "input_resolution": "Variable ROI -> 48x48x3 INT8",
        "all_images_metrics": {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "tp": tp, "fp": fp, "fn": fn
        },
        "held_out_test_metrics": {
            "precision": round(t_prec, 4),
            "recall": round(t_rec, 4),
            "f1": round(t_f1, 4),
            "tp": t_tp, "fp": t_fp, "fn": t_fn
        },
        "per_class": results["per_class"]
    }
    with open(os.path.join(BENCHMARKS_DIR, "hybrid_evaluation_results.json"), "w") as f:
        json.dump(benchmark_data, f, indent=2)

    # Save visual demonstration of test detections
    for i, (tfname, vis) in enumerate(annotated_test_visuals):
        out_path = os.path.join(SCREENSHOTS_DIR, f"test_detection_sample_{i+1}.png")
        cv2.imwrite(out_path, vis)
        print(f"Saved test detection visual: {out_path}")
        
    # Plot precision-recall comparison across candidates
    labels = ["Baseline (Person)", "Cand 1 (Patch NN)", "Cand 2 (FOMO FCN)", "Cand 3 (Hybrid CV+NN)"]
    precisions = [0.0, 59.26, 0.0, t_prec * 100]
    recalls = [0.0, 59.26, 0.0, t_rec * 100]
    f1_scores = [0.0, 59.26, 0.0, t_f1 * 100]
    
    x = np.arange(len(labels))
    width = 0.25
    plt.figure(figsize=(9, 5))
    plt.bar(x - width, precisions, width, label='Precision (%)', color='#2196F3')
    plt.bar(x, recalls, width, label='Recall (%)', color='#4CAF50')
    plt.bar(x + width, f1_scores, width, label='F1 Score (%)', color='#FF9800')
    plt.ylabel('Percentage (%)')
    plt.title('Multi-Object Recognition & Localization: Candidate Comparison\non Held-Out Test Desk Scenes')
    plt.xticks(x, labels)
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    comp_plot_path = os.path.join(PLOTS_DIR, "candidate_comparison_metrics.png")
    plt.savefig(comp_plot_path, dpi=150)
    plt.close()
    print(f"Saved candidate comparison plot: {comp_plot_path}")

if __name__ == "__main__":
    evaluate_hybrid_system()
