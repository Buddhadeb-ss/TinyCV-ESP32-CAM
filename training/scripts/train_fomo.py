import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

import time
import json
import glob
import subprocess
import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

CLASSES = ["background", "cream_container", "mouse", "earbuds_case", "glue_stick", "microcontroller_board"]
NUM_CLASSES = len(CLASSES)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

IMAGES_DIR = r"C:\All_Projects\esp32_cam_ai_system\images"
ANNOTATIONS_FILE = r"C:\All_Projects\esp32_cam_ai_system\dataset\annotations\annotations.json"
SPLIT_FILE = r"C:\All_Projects\esp32_cam_ai_system\dataset\splits\split_definition.json"

CHECKPOINTS_DIR = r"C:\All_Projects\esp32_cam_ai_system\training\checkpoints"
PLOTS_DIR = r"C:\All_Projects\esp32_cam_ai_system\benchmarks\plots"
MODELS_DIR = r"C:\All_Projects\esp32_cam_ai_system\models\candidate_models"
for d in [CHECKPOINTS_DIR, PLOTS_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

GRID_SIZE = 12  # 96 / 8 = 12x12 grid
IMG_SIZE = 96

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(in_c, in_c, 3, stride=stride, padding=1, groups=in_c, bias=False)
        self.pw = nn.Conv2d(in_c, out_c, 1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))

class TinyFOMO(nn.Module):
    """
    FOMO-style Fully Convolutional Network for On-Device Centroid Object Localization.
    Input: [B, 3, 96, 96]
    Output: [B, NUM_CLASSES, 12, 12] spatial grid of class logits.
    """
    def __init__(self, num_classes=NUM_CLASSES, in_channels=3):
        super().__init__()
        # Initial stem downsamples 96 -> 48 (stride 2)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU6(inplace=True)
        )
        # Stage 1: 48 -> 24 (stride 2)
        self.stage1 = nn.Sequential(
            DepthwiseSeparableConv(16, 24, stride=1),
            DepthwiseSeparableConv(24, 24, stride=2)
        )
        # Stage 2: 24 -> 12 (stride 2)
        self.stage2 = nn.Sequential(
            DepthwiseSeparableConv(24, 32, stride=1),
            DepthwiseSeparableConv(32, 40, stride=2)
        )
        # Stage 3: feature refinement at 12x12
        self.refine = DepthwiseSeparableConv(40, 48, stride=1)
        
        # 1x1 classification head outputting class logits per 12x12 cell
        self.head = nn.Conv2d(48, num_classes, kernel_size=1, stride=1, bias=True)

    def forward(self, x):
        x = self.stem(x)    # 96 -> 48
        x = self.stage1(x)  # 48 -> 24
        x = self.stage2(x)  # 24 -> 12
        x = self.refine(x)  # 12 -> 12
        x = self.head(x)    # [B, 6, 12, 12]
        return x

class FOMODataset(Dataset):
    def __init__(self, split="train", augment=False, repeats=1):
        self.augment = augment
        with open(ANNOTATIONS_FILE, "r") as f:
            annotations = json.load(f)
        with open(SPLIT_FILE, "r") as f:
            splits = json.load(f)
            
        self.samples = []
        for fname, s in splits.items():
            if s == split:
                boxes = annotations.get(fname, [])
                fpath = os.path.join(IMAGES_DIR, fname)
                self.samples.append((fpath, boxes))
                
        # Repeat small training set to form meaningful epochs
        if repeats > 1:
            self.samples = self.samples * repeats
            
        print(f"Loaded {len(self.samples)} samples for FOMO [{split.upper()}] (augment={augment})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fpath, boxes = self.samples[idx]
        img = cv2.imread(fpath)
        h_orig, w_orig = img.shape[:2]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Build ground-truth 12x12 grid target
        target_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int64) # class 0 = background
        
        # Resize image to 96x96
        resized_img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        
        # Map centroids to 12x12 grid
        for item in boxes:
            cls_name = item["class"]
            if cls_name not in CLASS_TO_IDX:
                continue
            cls_id = CLASS_TO_IDX[cls_name]
            bx, by, bw, bh = item["bbox"]
            cx = bx + bw / 2.0
            cy = by + bh / 2.0
            
            # Map to grid coords [0..11]
            gx = int(np.clip(cx * GRID_SIZE / w_orig, 0, GRID_SIZE - 1))
            gy = int(np.clip(cy * GRID_SIZE / h_orig, 0, GRID_SIZE - 1))
            target_grid[gy, gx] = cls_id
            
        if self.augment:
            # Random horizontal flip
            if np.random.rand() > 0.5:
                resized_img = cv2.flip(resized_img, 1).copy()
                target_grid = np.fliplr(target_grid).copy()
            # Random brightness & contrast
            alpha = np.random.uniform(0.85, 1.15)
            beta = np.random.uniform(-15, 15)
            resized_img = np.clip(alpha * resized_img + beta, 0, 255).astype(np.uint8)
            
        tensor_img = torch.from_numpy(resized_img.copy()).permute(2, 0, 1).float() / 127.5 - 1.0
        tensor_target = torch.from_numpy(target_grid.copy()).long()
        return tensor_img, tensor_target

def train_fomo():
    print("=" * 60)
    print("TRAINING CANDIDATE 2: FOMO OBJECT CENTROID DETECTOR")
    print("=" * 60)
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 15 train images repeated 8x = 120 samples per epoch
    train_dataset = FOMODataset("train", augment=True, repeats=8)
    val_dataset = FOMODataset("val", augment=False, repeats=1)
    test_dataset = FOMODataset("test", augment=False, repeats=1)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=3, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=3, shuffle=False)
    
    model = TinyFOMO(num_classes=NUM_CLASSES)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"TinyFOMO Total Parameters: {total_params} (~{total_params/1024:.2f} KB FP32, ~{total_params/1024:.2f} KB INT8)")
    
    # Weighted Cross-Entropy Loss: give higher weight to foreground centroids vs background
    weights = torch.tensor([0.2, 2.5, 2.5, 2.5, 2.5, 2.5], dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.AdamW(model.parameters(), lr=0.004, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80)
    
    epochs = 80
    history = {"train_loss": [], "val_loss": [], "centroid_f1": []}
    best_f1 = 0.0
    best_weights = None
    
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        total = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            out = model(x) # [B, 6, 12, 12]
            loss = criterion(out, y) # y: [B, 12, 12]
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
            total += x.size(0)
            
        scheduler.step()
        train_loss = running_loss / total
        
        # Validation evaluation: centroid precision & recall
        model.eval()
        v_loss = 0.0
        v_total = 0
        tp = fp = fn = 0
        with torch.no_grad():
            for vx, vy in val_loader:
                vout = model(vx)
                loss = criterion(vout, vy)
                v_loss += loss.item() * vx.size(0)
                v_total += vx.size(0)
                
                preds = vout.argmax(dim=1).cpu().numpy() # [B, 12, 12]
                gts = vy.cpu().numpy()
                
                # Check foreground detections
                for b in range(vx.size(0)):
                    pred_fg = (preds[b] > 0)
                    gt_fg = (gts[b] > 0)
                    tp += np.sum(pred_fg & gt_fg & (preds[b] == gts[b]))
                    fp += np.sum(pred_fg & ~gt_fg)
                    fn += np.sum(~pred_fg & gt_fg)
                    
        val_loss = v_loss / max(1, v_total)
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-6, precision + recall)
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["centroid_f1"].append(f1)
        
        if f1 >= best_f1:
            best_f1 = f1
            best_weights = model.state_dict().copy()
            
        if epoch % 10 == 0 or epoch == epochs:
            print(f"Epoch {epoch:2d}/{epochs:2d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Centroid F1: {f1*100:.1f}% (P: {precision*100:.1f}%, R: {recall*100:.1f}%)")
            
    # Load best weights
    model.load_state_dict(best_weights)
    ckpt_path = os.path.join(CHECKPOINTS_DIR, "candidate2_fomo_best.pth")
    torch.save(model.state_dict(), ckpt_path)
    print(f"Best Validation Centroid F1: {best_f1*100:.1f}%. Saved to {ckpt_path}")
    
    # Evaluate on HELD-OUT TEST IMAGES
    model.eval()
    test_tp = test_fp = test_fn = 0
    test_detections = []
    with torch.no_grad():
        for tx, ty in test_loader:
            tout = model(tx)
            probs = torch.softmax(tout, dim=1).cpu().numpy()
            preds = tout.argmax(dim=1).cpu().numpy()
            gts = ty.cpu().numpy()
            for b in range(tx.size(0)):
                pred_fg = (preds[b] > 0)
                gt_fg = (gts[b] > 0)
                test_tp += np.sum(pred_fg & gt_fg & (preds[b] == gts[b]))
                test_fp += np.sum(pred_fg & ~gt_fg)
                test_fn += np.sum(~pred_fg & gt_fg)
                
    test_prec = test_tp / max(1, test_tp + test_fp)
    test_rec = test_tp / max(1, test_tp + test_fn)
    test_f1 = 2 * test_prec * test_rec / max(1e-6, test_prec + test_rec)
    print(f"\nHELD-OUT TEST SET EVALUATION (0 data leakage):")
    print(f" Centroid Precision: {test_prec*100:.2f}%")
    print(f" Centroid Recall:    {test_rec*100:.2f}%")
    print(f" Centroid F1 Score:  {test_f1*100:.2f}% (TP: {test_tp}, FP: {test_fp}, FN: {test_fn})")
    
    # Plot training loss & F1
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss", color="blue")
    plt.plot(history["val_loss"], label="Val Loss", color="red")
    plt.title("Candidate 2 (FOMO): Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history["centroid_f1"], label="Val F1", color="green")
    plt.title(f"Candidate 2 (FOMO): Centroid F1 (Test: {test_f1*100:.1f}%)")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, "candidate2_fomo_training_curves.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved FOMO training plot to {plot_path}")
    
    # Export to ONNX
    onnx_path = os.path.join(MODELS_DIR, "candidate2_fomo.onnx")
    dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=18
    )
    print(f"Exported ONNX model to {onnx_path} ({os.path.getsize(onnx_path)} bytes)")
    
    # Save calibration data
    calib_samples = []
    for x, _ in train_loader:
        calib_samples.append(x.numpy())
        if len(calib_samples) >= 8:
            break
    calib_array = np.concatenate(calib_samples, axis=0) # [N, 3, 96, 96]
    calib_path = os.path.join(MODELS_DIR, "candidate2_calib.npy")
    np.save(calib_path, calib_array)
    
    # Quantize to INT8 via onnx2tf
    tflite_out_dir = os.path.join(MODELS_DIR, "candidate2_tflite")
    cmd = [
        "python", "-m", "onnx2tf",
        "-i", onnx_path,
        "-o", tflite_out_dir,
        "-tb", "flatbuffer_direct",
        "-ett", "full_integer_quant",
        "-qt", "per-tensor",
        "-cind", "input", calib_path, "0.0", "1.0"
    ]
    print("Running onnx2tf INT8 quantization for FOMO...")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print("onnx2tf failed with returncode:", res.returncode)
        print("Stderr:", res.stderr[-500:])
    else:
        print("onnx2tf successfully generated INT8 quantized FOMO model!")
        int8_tflite = os.path.join(tflite_out_dir, "candidate2_fomo_full_integer_quant.tflite")
        if os.path.exists(int8_tflite):
            size_bytes = os.path.getsize(int8_tflite)
            print(f"Generated INT8 FOMO TFLite Model: {int8_tflite} ({size_bytes} bytes = {size_bytes/1024:.2f} KB)")
            
    # Save metrics JSON
    metrics = {
        "candidate": "Candidate 2 (TinyFOMO Centroid Detector)",
        "input_resolution": "96x96x3 (RGB)",
        "output_grid": "12x12x6 (Spatial Centroid Logits)",
        "total_parameters": total_params,
        "best_val_f1": float(best_f1),
        "test_precision": float(test_prec),
        "test_recall": float(test_rec),
        "test_f1": float(test_f1),
        "test_tp": int(test_tp),
        "test_fp": int(test_fp),
        "test_fn": int(test_fn)
    }
    with open(os.path.join(MODELS_DIR, "candidate2_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    train_fomo()
