"""
Sensor-Matched Model Training Pipeline (training/scripts/train_sensor_matched_model.py)
Trains DeskObjectNet INT8 on images captured directly from the physical OV3660 sensor.
Performs PTQ quantization via onnx2tf and generates C deployment arrays for ESP32-CAM.
"""

import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

import time
import json
import glob
import subprocess
import shutil
import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = r"C:\All_Projects\esp32_cam_ai_system"
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "dataset", "processed_esp32cam")
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "training", "checkpoints")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "benchmarks", "plots")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "candidate_models")
DEPLOY_DIR = os.path.join(PROJECT_ROOT, "models", "deployment")
FIRMWARE_SRC = os.path.join(PROJECT_ROOT, "firmware", "src")
FIRMWARE_INC = os.path.join(PROJECT_ROOT, "firmware", "include")

for d in [CHECKPOINTS_DIR, PLOTS_DIR, MODELS_DIR, DEPLOY_DIR]:
    os.makedirs(d, exist_ok=True)

CLASSES = ["background", "cream_container", "mouse", "earbuds_case", "glue_stick", "microcontroller_board"]
NUM_CLASSES = len(CLASSES)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
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
    Upgraded Full-FOV TinyML Architecture for ESP32-CAM:
    Input: [B, 3, 48, 48] (Downsampled from Full 96x96 Camera Frame)
    Width: 24 -> 36 -> 54 -> 72 (Depthwise Separable Convolutions)
    Memory: ~20 KB INT8, ~150 ms latency on Xtensa LX6 @ 240 MHz.
    """
    def __init__(self, num_classes=NUM_CLASSES, in_channels=3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 24, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU6(inplace=True)
        )
        self.stage1 = DepthwiseSeparableConv(24, 36, stride=1)
        self.stage2 = DepthwiseSeparableConv(36, 54, stride=2)
        self.stage3 = DepthwiseSeparableConv(54, 72, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.25)
        self.head = nn.Linear(72, num_classes)

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


class SensorMatchedDataset(Dataset):
    def __init__(self, split="train", img_size=IMG_SIZE, augment=False, balance_target=90):
        self.img_size = img_size
        self.augment = augment
        self.split = split
        
        split_dir = os.path.join(PROCESSED_DIR, split)
        class_samples = {i: [] for i in range(NUM_CLASSES)}
        for cls_name in CLASSES:
            c_dir = os.path.join(split_dir, cls_name)
            if not os.path.exists(c_dir):
                continue
            for f in glob.glob(os.path.join(c_dir, "*.png")):
                class_samples[CLASS_TO_IDX[cls_name]].append(f)

        self.samples = []
        if split == "train" and balance_target > 0:
            for c_idx in range(NUM_CLASSES):
                files = class_samples[c_idx]
                if not files:
                    continue
                if len(files) >= balance_target:
                    selected = np.random.choice(files, balance_target, replace=False).tolist()
                else:
                    repeats = int(np.ceil(balance_target / len(files)))
                    selected = (files * repeats)[:balance_target]
                for f in selected:
                    self.samples.append((f, c_idx))
        else:
            for c_idx in range(NUM_CLASSES):
                for f in class_samples[c_idx]:
                    self.samples.append((f, c_idx))

        print(f"Loaded {len(self.samples)} sensor samples for [{split.upper()}] (Full-FOV)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fpath, label = self.samples[idx]
        img = cv2.imread(fpath)
        if img is None:
            img = np.zeros((96, 96, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h, w = img.shape[:2]

        if self.augment:
            # Random affine translation (+-8 pixels)
            tx = np.random.randint(-8, 9)
            ty = np.random.randint(-8, 9)
            M_trans = np.float32([[1, 0, tx], [0, 1, ty]])
            img = cv2.warpAffine(img, M_trans, (w, h), borderMode=cv2.BORDER_REFLECT_101)

            # Random scaling (0.85x to 1.15x)
            scale_factor = np.random.uniform(0.85, 1.15)
            M_scale = cv2.getRotationMatrix2D((w / 2, h / 2), 0, scale_factor)
            img = cv2.warpAffine(img, M_scale, (w, h), borderMode=cv2.BORDER_REFLECT_101)

            # Random horizontal flip
            if np.random.rand() > 0.5:
                img = cv2.flip(img, 1)

            # Small rotation (-12 to +12 deg)
            rot_deg = np.random.uniform(-12, 12)
            M_rot = cv2.getRotationMatrix2D((w / 2, h / 2), rot_deg, 1.0)
            img = cv2.warpAffine(img, M_rot, (w, h), borderMode=cv2.BORDER_REFLECT_101)

        # Full-FOV downsampling: entire 96x96 sensor frame downsampled to 48x48
        # ZERO blind spots: 100% of the object and context are visible!
        processed = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)

        if self.augment:
            # Exposure & lighting variation
            alpha = np.random.uniform(0.82, 1.18)
            beta = np.random.uniform(-18, 18)
            processed = np.clip(alpha * processed + beta, 0, 255).astype(np.uint8)
            # Subtle Gaussian sensor noise
            noise = np.random.normal(0, 2.0, processed.shape)
            processed = np.clip(processed + noise, 0, 255).astype(np.uint8)

        # Normalize to [-1.0, 1.0] matching INT8 quantization
        tensor_img = torch.from_numpy(processed.copy()).permute(2, 0, 1).float() / 127.5 - 1.0
        return tensor_img, label



def train_sensor_model(epochs=60, lr=0.003, batch_size=32):
    print("=" * 60)
    print(" TRAINING DESKOBJECTNET ON SENSOR-MATCHED DATASET")
    print("=" * 60)

    torch.manual_seed(42)
    np.random.seed(42)

    train_ds = SensorMatchedDataset("train", augment=True, balance_target=90)
    val_ds = SensorMatchedDataset("val", augment=False, balance_target=0)
    test_ds = SensorMatchedDataset("test", augment=False, balance_target=0)

    if len(train_ds) == 0:
        print("[ERROR] Training set is empty! Collect images using capture_tool.py first.")
        return False

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False) if len(val_ds) > 0 else None
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False) if len(test_ds) > 0 else None

    model = DeskObjectNet(num_classes=NUM_CLASSES)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_loss = 999.0
    best_weights = model.state_dict().copy()

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
        train_loss = running_loss / max(1, total)
        train_acc = correct / max(1, total)

        val_loss, val_acc = 0.0, 0.0
        if val_loader:
            model.eval()
            v_running = 0.0
            v_correct = 0
            v_total = 0
            with torch.no_grad():
                for vx, vy in val_loader:
                    vout = model(vx)
                    loss = criterion(vout, vy)
                    v_running += loss.item() * vx.size(0)
                    preds = vout.argmax(dim=1)
                    v_correct += (preds == vy).sum().item()
                    v_total += vx.size(0)
            val_loss = v_running / max(1, v_total)
            val_acc = v_correct / max(1, v_total)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if (val_loader and val_loss < best_loss) or (not val_loader and train_loss < best_loss):
            best_loss = val_loss if val_loader else train_loss
            best_weights = model.state_dict().copy()

        if epoch % 10 == 0 or epoch == epochs:
            print(f"Epoch {epoch:2d}/{epochs:2d} | Train Loss: {train_loss:.4f} Acc: {train_acc*100:.1f}% | Val Loss: {val_loss:.4f} Acc: {val_acc*100:.1f}%")

    model.load_state_dict(best_weights)
    ckpt_path = os.path.join(CHECKPOINTS_DIR, "sensor_matched_model_best.pth")
    torch.save(model.state_dict(), ckpt_path)
    print(f"\nSaved best model checkpoint to: {ckpt_path}")

    # Evaluate on held-out test set
    if test_loader and len(test_ds) > 0:
        model.eval()
        all_preds, all_targets, all_probs = [], [], []
        with torch.no_grad():
            for tx, ty in test_loader:
                tout = model(tx)
                probs = torch.softmax(tout, dim=1)
                all_preds.extend(tout.argmax(dim=1).cpu().numpy())
                all_targets.extend(ty.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        test_acc = np.mean(all_preds == all_targets)

        cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
        for t, p in zip(all_targets, all_preds):
            cm[t, p] += 1

        print(f"\nTEST SET OVERALL ACCURACY: {test_acc*100:.2f}% ({np.sum(all_preds==all_targets)}/{len(all_targets)})")
        print("Confusion Matrix:")
        header = f"{'True/Pred':<22}" + "".join([f"{c[:8]:<9}" for c in CLASSES])
        print(header)
        for i, row in enumerate(cm):
            print(f"{CLASSES[i]:<22}" + "".join([f"{val:<9}" for val in row]))

    # Export to ONNX
    onnx_path = os.path.join(MODELS_DIR, "sensor_matched_model.onnx")
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

    # Save calibration data
    calib_samples = []
    for x, _ in train_loader:
        calib_samples.append(x.numpy())
        if len(calib_samples) >= 5:
            break
    calib_arr = np.concatenate(calib_samples, axis=0)
    calib_path = os.path.join(MODELS_DIR, "sensor_calib.npy")
    np.save(calib_path, calib_arr)

    # Quantize to INT8 via onnx2tf
    tflite_out_dir = os.path.join(MODELS_DIR, "sensor_matched_tflite")
    cmd = [
        "python", "-m", "onnx2tf",
        "-i", onnx_path,
        "-o", tflite_out_dir,
        "-oiqt",
        "-qt", "per-tensor",
        "-cind", "input", calib_path, "0.0", "1.0"
    ]
    print("Running onnx2tf INT8 quantization for sensor-matched model...")
    res = subprocess.run(cmd, capture_output=True, text=True)

    # Find generated full integer quantized tflite file
    tflite_src = os.path.join(tflite_out_dir, "sensor_matched_model_full_integer_quant.tflite")
    if not os.path.exists(tflite_src):
        # search for any .tflite in output directory
        found = glob.glob(os.path.join(tflite_out_dir, "*integer_quant*.tflite"))
        if found:
            tflite_src = found[0]
        else:
            print("onnx2tf failed to generate INT8 model:", res.stderr[-500:])
            return False


    tflite_dest = os.path.join(DEPLOY_DIR, "custom_desk_object_model_int8.tflite")
    shutil.copy2(tflite_src, tflite_dest)
    print(f"Saved production INT8 model: {tflite_dest} ({os.path.getsize(tflite_dest)} bytes)")

    # Generate C deployment arrays
    print("Generating C deployment arrays for firmware...")
    with open(tflite_dest, "rb") as f:
        bytes_data = f.read()

    cc_content = f"""#include "custom_object_model_data.h"

// DeskObjectNet INT8 model trained directly on physical ESP32-CAM OV3660 sensor frames
alignas(16) const unsigned char g_custom_object_model_data[] = {{
"""
    for i in range(0, len(bytes_data), 12):
        chunk = bytes_data[i:i+12]
        hex_str = ", ".join([f"0x{b:02x}" for b in chunk])
        cc_content += f"  {hex_str},\n"
    cc_content += f"}};\nconst int g_custom_object_model_data_len = {len(bytes_data)};\n"

    h_content = f"""#ifndef CUSTOM_OBJECT_MODEL_DATA_H_
#define CUSTOM_OBJECT_MODEL_DATA_H_

extern const unsigned char g_custom_object_model_data[];
extern const int g_custom_object_model_data_len;

#endif  // CUSTOM_OBJECT_MODEL_DATA_H_
"""
    # Write to models/deployment and firmware/src + firmware/include
    with open(os.path.join(DEPLOY_DIR, "custom_object_model_data.cc"), "w") as f:
        f.write(cc_content)
    with open(os.path.join(DEPLOY_DIR, "custom_object_model_data.h"), "w") as f:
        f.write(h_content)
    with open(os.path.join(FIRMWARE_SRC, "custom_object_model_data.cc"), "w") as f:
        f.write(cc_content)
    with open(os.path.join(FIRMWARE_INC, "custom_object_model_data.h"), "w") as f:
        f.write(h_content)

    print("Updated firmware model data arrays successfully!")
    return True

if __name__ == "__main__":
    train_sensor_model()
