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
DEPLOY_DIR = r"C:\All_Projects\esp32_cam_ai_system\models\deployment"
for d in [CHECKPOINTS_DIR, PLOTS_DIR, MODELS_DIR, DEPLOY_DIR]:
    os.makedirs(d, exist_ok=True)

IMG_SIZE = 48

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(in_c, in_c, 3, stride=stride, padding=1, groups=in_c, bias=False)
        self.pw = nn.Conv2d(in_c, out_c, 1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))

class DeskObjectNet(nn.Module):
    """
    Optimized TinyML Architecture for ESP32-CAM:
    Input: [B, 3, 48, 48]
    Width: 24 -> 32 -> 48 -> 64 (with Depthwise Separable Convolutions)
    Target Size: ~35 KB INT8, < 50 ms latency on Xtensa LX6 @ 240 MHz.
    """
    def __init__(self, num_classes=NUM_CLASSES, in_channels=3):
        super().__init__()
        # Stem: 48 -> 24
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 24, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU6(inplace=True)
        )
        # Stage 1: 24 -> 24
        self.stage1 = DepthwiseSeparableConv(24, 32, stride=1)
        # Stage 2: 24 -> 12
        self.stage2 = DepthwiseSeparableConv(32, 48, stride=2)
        # Stage 3: 12 -> 6
        self.stage3 = DepthwiseSeparableConv(48, 64, stride=2)
        # Global pooling: 6 -> 1
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.3)
        self.head = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.head(x)
        return x

class BalancedCropDataset(Dataset):
    def __init__(self, split="train", img_size=IMG_SIZE, augment=False):
        self.img_size = img_size
        self.augment = augment
        
        split_dir = os.path.join(CROPS_DIR, split)
        class_samples = {i: [] for i in range(NUM_CLASSES)}
        for cls_name in CLASSES:
            c_dir = os.path.join(split_dir, cls_name)
            if not os.path.exists(c_dir):
                continue
            for f in glob.glob(os.path.join(c_dir, "*.png")):
                class_samples[CLASS_TO_IDX[cls_name]].append(f)
                
        # Balance dataset: For training, take 100 samples per class
        self.samples = []
        if split == "train":
            target_per_class = 120
            for c_idx in range(NUM_CLASSES):
                files = class_samples[c_idx]
                if not files:
                    continue
                if len(files) >= target_per_class:
                    selected = np.random.choice(files, target_per_class, replace=False).tolist()
                else:
                    repeats = int(np.ceil(target_per_class / len(files)))
                    selected = (files * repeats)[:target_per_class]
                for f in selected:
                    self.samples.append((f, c_idx))
        else:
            for c_idx in range(NUM_CLASSES):
                for f in class_samples[c_idx]:
                    self.samples.append((f, c_idx))
                    
        print(f"Loaded {len(self.samples)} samples for [{split.upper()}]")

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
            if np.random.rand() > 0.5:
                img = cv2.flip(img, 1)
            # Small random rotation (-15 to 15 deg)
            angle = np.random.uniform(-15, 15)
            M = cv2.getRotationMatrix2D((self.img_size/2, self.img_size/2), angle, 1.0)
            img = cv2.warpAffine(img, M, (self.img_size, self.img_size), borderMode=cv2.BORDER_REFLECT_101)
            # Color jitter
            alpha = np.random.uniform(0.8, 1.25)
            beta = np.random.uniform(-20, 20)
            img = np.clip(alpha * img + beta, 0, 255).astype(np.uint8)
            
        tensor_img = torch.from_numpy(img.copy()).permute(2, 0, 1).float() / 127.5 - 1.0
        return tensor_img, label

def train_desk_object_net():
    print("=" * 60)
    print("TRAINING OPTIMIZED DESKOBJECTNET (CANDIDATE 1B / PRODUCTION)")
    print("=" * 60)
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    train_ds = BalancedCropDataset("train", augment=True)
    val_ds = BalancedCropDataset("val", augment=False)
    test_ds = BalancedCropDataset("test", augment=False)
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)
    
    model = DeskObjectNet(num_classes=NUM_CLASSES)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"DeskObjectNet Total Parameters: {total_params} (~{total_params/1024:.2f} KB INT8)")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=75)
    
    epochs = 75
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
            
        if epoch % 15 == 0 or epoch == epochs:
            print(f"Epoch {epoch:2d}/{epochs:2d} | Train Loss: {train_loss:.4f} Acc: {train_acc*100:.1f}% | Val Loss: {val_loss:.4f} Acc: {val_acc*100:.1f}%")

    model.load_state_dict(best_weights)
    ckpt_path = os.path.join(CHECKPOINTS_DIR, "desk_object_net_best.pth")
    torch.save(model.state_dict(), ckpt_path)
    print(f"\nBest Validation Accuracy: {best_val_acc*100:.2f}%. Saved to {ckpt_path}")
    
    # Evaluate on HELD-OUT TEST SET (All 185 background + 6 foreground crops)
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
    
    # Per-class metrics
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for t, p in zip(all_targets, all_preds):
        cm[t, p] += 1
        
    print(f"\nHELD-OUT TEST SET OVERALL ACCURACY: {test_acc*100:.2f}% ({np.sum(all_preds==all_targets)}/{len(all_targets)})")
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
    plt.title("DeskObjectNet: Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history["train_acc"], label="Train Acc", color="blue")
    plt.plot(history["val_acc"], label="Val Acc", color="red")
    plt.title(f"DeskObjectNet: Accuracy (Test: {test_acc*100:.1f}%)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, "desk_object_net_curves.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    
    # Export to ONNX
    onnx_path = os.path.join(MODELS_DIR, "desk_object_net.onnx")
    dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=18
    )
    print(f"\nExported ONNX model: {onnx_path} ({os.path.getsize(onnx_path)} bytes)")
    
    # Calibration data for INT8 PTQ
    calib_samples = []
    for x, _ in train_loader:
        calib_samples.append(x.numpy())
        if len(calib_samples) >= 6:
            break
    calib_arr = np.concatenate(calib_samples, axis=0) # [N, 3, 48, 48]
    calib_path = os.path.join(MODELS_DIR, "desk_object_net_calib.npy")
    np.save(calib_path, calib_arr)
    
    # Quantize to INT8 via onnx2tf
    tflite_out_dir = os.path.join(MODELS_DIR, "desk_object_net_tflite")
    cmd = [
        "python", "-m", "onnx2tf",
        "-i", onnx_path,
        "-o", tflite_out_dir,
        "-tb", "flatbuffer_direct",
        "-ett", "full_integer_quant",
        "-qt", "per-tensor",
        "-cind", "input", calib_path, "0.0", "1.0"
    ]
    print("Running onnx2tf INT8 quantization for DeskObjectNet...")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print("onnx2tf failed with returncode:", res.returncode)
        print("Stderr:", res.stderr[-500:])
        return
        
    int8_tflite = os.path.join(tflite_out_dir, "desk_object_net_full_integer_quant.tflite")
    deploy_tflite = os.path.join(DEPLOY_DIR, "custom_desk_object_model_int8.tflite")
    import shutil
    shutil.copyfile(int8_tflite, deploy_tflite)
    
    size_bytes = os.path.getsize(deploy_tflite)
    print(f"\nSUCCESS! Deployment INT8 Model: {deploy_tflite} ({size_bytes} bytes = {size_bytes/1024:.2f} KB)")
    
    # Convert TFLite to C Array for ESP32-CAM firmware:
    with open(deploy_tflite, "rb") as f:
        tflite_bytes = f.read()
        
    c_source_path = os.path.join(DEPLOY_DIR, "custom_object_model_data.cc")
    c_header_path = os.path.join(DEPLOY_DIR, "custom_object_model_data.h")
    
    with open(c_header_path, "w") as f:
        f.write("#ifndef CUSTOM_OBJECT_MODEL_DATA_H_\n")
        f.write("#define CUSTOM_OBJECT_MODEL_DATA_H_\n\n")
        f.write("extern const unsigned char g_custom_object_model_data[];\n")
        f.write(f"extern const int g_custom_object_model_data_len;\n\n")
        f.write("#endif  // CUSTOM_OBJECT_MODEL_DATA_H_\n")
        
    with open(c_source_path, "w") as f:
        f.write('#include "custom_object_model_data.h"\n\n')
        f.write(f"// Size: {len(tflite_bytes)} bytes ({len(tflite_bytes)/1024:.2f} KB)\n")
        f.write("alignas(16) const unsigned char g_custom_object_model_data[] = {\n")
        for i, b in enumerate(tflite_bytes):
            if i % 12 == 0:
                f.write("  ")
            f.write(f"0x{b:02x}, ")
            if (i + 1) % 12 == 0 or i == len(tflite_bytes) - 1:
                f.write("\n")
        f.write("};\n")
        f.write(f"const int g_custom_object_model_data_len = {len(tflite_bytes)};\n")
        
    print(f"Generated C Array header: {c_header_path}")
    print(f"Generated C Array source: {c_source_path} ({os.path.getsize(c_source_path)} bytes)")

if __name__ == "__main__":
    train_desk_object_net()
