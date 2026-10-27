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

CROPS_DIR = r"C:\All_Projects\esp32_cam_ai_system\dataset\crops"
CHECKPOINTS_DIR = r"C:\All_Projects\esp32_cam_ai_system\training\checkpoints"
PLOTS_DIR = r"C:\All_Projects\esp32_cam_ai_system\benchmarks\plots"
MODELS_DIR = r"C:\All_Projects\esp32_cam_ai_system\models\candidate_models"
for d in [CHECKPOINTS_DIR, PLOTS_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)

class TinyDeskClassifier(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, in_channels=3):
        super().__init__()
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU6(inplace=True)
        )
        self.stage1 = DepthwiseSeparableConv(16, 24, stride=1)
        self.stage2 = DepthwiseSeparableConv(24, 32, stride=2)
        self.stage3 = DepthwiseSeparableConv(32, 48, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.drop = nn.Dropout(0.25)
        self.fc = nn.Linear(48, num_classes)

    def forward(self, x):
        x = self.init_conv(x)  # 48 -> 24
        x = self.stage1(x)     # 24 -> 24
        x = self.stage2(x)     # 24 -> 12
        x = self.stage3(x)     # 12 -> 6
        x = self.pool(x)       # 6 -> 1
        x = torch.flatten(x, 1)
        x = self.drop(x)
        x = self.fc(x)
        return x

class CropDataset(Dataset):
    def __init__(self, split="train", img_size=48, augment=False):
        self.img_size = img_size
        self.augment = augment
        self.samples = []
        
        split_dir = os.path.join(CROPS_DIR, split)
        for cls_name in CLASSES:
            cls_dir = os.path.join(split_dir, cls_name)
            if not os.path.exists(cls_dir):
                continue
            files = glob.glob(os.path.join(cls_dir, "*.png"))
            for f in files:
                self.samples.append((f, CLASS_TO_IDX[cls_name]))
                
        # Oversample minority classes during training to balance class frequencies
        if augment and split == "train":
            balanced_samples = []
            class_counts = {i: 0 for i in range(NUM_CLASSES)}
            for s in self.samples:
                class_counts[s[1]] += 1
            max_count = max(class_counts.values())
            target_per_class = min(max_count, 60) # balance to ~60 samples per class
            
            for cls_idx in range(NUM_CLASSES):
                cls_samples = [s for s in self.samples if s[1] == cls_idx]
                if not cls_samples:
                    continue
                repeats = max(1, target_per_class // len(cls_samples))
                for _ in range(repeats):
                    balanced_samples.extend(cls_samples)
            self.samples = balanced_samples
            
        print(f"Loaded {len(self.samples)} samples for [{split.upper()}] (augment={augment})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fpath, label = self.samples[idx]
        img = cv2.imread(fpath)
        if img is None:
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        
        if self.augment:
            # Random horizontal flip
            if np.random.rand() > 0.5:
                img = cv2.flip(img, 1)
            # Random slight rotation
            angle = np.random.uniform(-15, 15)
            M = cv2.getRotationMatrix2D((self.img_size/2, self.img_size/2), angle, 1.0)
            img = cv2.warpAffine(img, M, (self.img_size, self.img_size), borderMode=cv2.BORDER_REFLECT_101)
            # Random brightness and contrast shift
            alpha = np.random.uniform(0.8, 1.2)
            beta = np.random.uniform(-20, 20)
            img = np.clip(alpha * img + beta, 0, 255).astype(np.uint8)
            
        # Normalize to [-1, 1] for TinyML deployment
        tensor_img = torch.from_numpy(img).permute(2, 0, 1).float() / 127.5 - 1.0
        return tensor_img, label

def train_candidate1():
    print("=" * 60)
    print("TRAINING CANDIDATE 1: TINY DESK PATCH CLASSIFIER")
    print("=" * 60)
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    train_dataset = CropDataset("train", img_size=48, augment=True)
    val_dataset = CropDataset("val", img_size=48, augment=False)
    test_dataset = CropDataset("test", img_size=48, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    model = TinyDeskClassifier(num_classes=NUM_CLASSES)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"TinyDeskClassifier Total Parameters: {total_params} (~{total_params/1024:.2f} KB FP32, ~{total_params/1024:.2f} KB INT8)")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=70)
    
    epochs = 70
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    best_weights = None
    
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * x.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)
            
        scheduler.step()
        train_loss = running_loss / total
        train_acc = correct / total
        
        # Validation
        model.eval()
        v_loss = 0.0
        v_correct = 0
        v_total = 0
        with torch.no_grad():
            for vx, vy in val_loader:
                vout = model(vx)
                loss = criterion(vout, vy)
                v_loss += loss.item() * vx.size(0)
                v_preds = vout.argmax(dim=1)
                v_correct += (v_preds == vy).sum().item()
                v_total += vx.size(0)
                
        val_loss = v_loss / max(1, v_total)
        val_acc = v_correct / max(1, v_total)
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_weights = model.state_dict().copy()
            
        if epoch % 10 == 0 or epoch == epochs:
            print(f"Epoch {epoch:2d}/{epochs:2d} | Train Loss: {train_loss:.4f} Acc: {train_acc*100:.1f}% | Val Loss: {val_loss:.4f} Acc: {val_acc*100:.1f}%")
            
    # Load best weights
    model.load_state_dict(best_weights)
    ckpt_path = os.path.join(CHECKPOINTS_DIR, "candidate1_best.pth")
    torch.save(model.state_dict(), ckpt_path)
    print(f"Best validation accuracy: {best_val_acc*100:.1f}%. Saved checkpoint to {ckpt_path}")
    
    # Evaluate on held-out TEST set
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for tx, ty in test_loader:
            tout = model(tx)
            preds = tout.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(ty.cpu().numpy())
            
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    test_acc = np.mean(all_preds == all_targets)
    print(f"\nHELD-OUT TEST SET ACCURACY (0 data leakage): {test_acc*100:.2f}% ({np.sum(all_preds==all_targets)}/{len(all_targets)})")
    
    # Compute Confusion Matrix
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for t, p in zip(all_targets, all_preds):
        cm[t, p] += 1
        
    print("\nTest Confusion Matrix:")
    header = f"{'True/Pred':<22}" + "".join([f"{c[:8]:<9}" for c in CLASSES])
    print(header)
    for i, row in enumerate(cm):
        row_str = f"{CLASSES[i]:<22}" + "".join([f"{val:<9}" for val in row])
        print(row_str)
        
    # Plot training curves
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss", color="blue")
    plt.plot(history["val_loss"], label="Val Loss", color="red")
    plt.title("Candidate 1: Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history["train_acc"], label="Train Acc", color="blue")
    plt.plot(history["val_acc"], label="Val Acc", color="red")
    plt.title(f"Candidate 1: Accuracy (Test: {test_acc*100:.1f}%)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, "candidate1_training_curves.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved training plot to {plot_path}")
    
    # Export to ONNX
    onnx_path = os.path.join(MODELS_DIR, "candidate1_classifier.onnx")
    dummy_input = torch.randn(1, 3, 48, 48)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=18
    )
    print(f"Exported ONNX model to {onnx_path} ({os.path.getsize(onnx_path)} bytes)")
    
    # Prepare representative calibration dataset for INT8 Post-Training Quantization
    calib_samples = []
    for x, _ in train_loader:
        calib_samples.append(x.numpy())
        if len(calib_samples) >= 10:
            break
    calib_array = np.concatenate(calib_samples, axis=0) # [N, 3, 48, 48]
    calib_path = os.path.join(MODELS_DIR, "candidate1_calib.npy")
    np.save(calib_path, calib_array)
    print(f"Saved calibration data to {calib_path} with shape {calib_array.shape}")
    
    # Run onnx2tf for INT8 Quantization
    tflite_out_dir = os.path.join(MODELS_DIR, "candidate1_tflite")
    cmd = [
        "python", "-m", "onnx2tf",
        "-i", onnx_path,
        "-o", tflite_out_dir,
        "-tb", "flatbuffer_direct",
        "-oiqt",
        "-cind", "input", calib_path, "0.0", "1.0"
    ]
    print("Running onnx2tf INT8 quantization...")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print("onnx2tf failed with returncode:", res.returncode)
        print("Stderr:", res.stderr[-500:])
    else:
        print("onnx2tf successfully generated INT8 quantized model!")
        int8_tflite = os.path.join(tflite_out_dir, "candidate1_classifier_full_integer_quant.tflite")
        if os.path.exists(int8_tflite):
            size_bytes = os.path.getsize(int8_tflite)
            print(f"Generated INT8 TFLite Model: {int8_tflite} ({size_bytes} bytes = {size_bytes/1024:.2f} KB)")
            
    # Save training record metrics
    metrics = {
        "candidate": "Candidate 1 (TinyDeskClassifier)",
        "input_resolution": "48x48x3 (RGB)",
        "total_parameters": total_params,
        "best_val_acc": float(best_val_acc),
        "test_acc": float(test_acc),
        "test_samples": len(all_targets),
        "classes": CLASSES,
        "confusion_matrix": cm.tolist()
    }
    with open(os.path.join(MODELS_DIR, "candidate1_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    train_candidate1()
