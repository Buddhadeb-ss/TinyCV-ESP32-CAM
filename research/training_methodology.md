# Training Methodology: Sensor-Matched TinyML on ESP32-CAM

## 1. Learning Problem Formulation

We evaluate the optimal formulation for localized multi-object recognition under resource constraints (**ESP32-D0WD-V3 @ 240 MHz, 520 KB SRAM, 4 MB PSRAM**):

1. **Full-Scene Global Classifier**:
   - Classifies the entire $96 \times 96$ image into one class.
   - *Limitation*: Cannot localize multiple objects or detect objects when off-center or surrounded by clutter.
2. **Dense Centroid Grid Detector (FOMO)**:
   - Evaluated as Candidate 2 in earlier research.
   - *Limitation*: Achieved 0.0% F1 score due to extreme sub-cell downsampling ($12 \times 12$ grid) and massive background cell class imbalance (~140:1).
3. **Candidate 3: Hybrid Saliency Proposals + Localized Patch Net (Selected Architecture)**:
   - **Stage 1 (Classical CV on Full Frame)**: Real-time spatial differencing and color saliency extracts bounding candidate regions $[bx, by, bw, bh]$ from the $96 \times 96$ sensor capture in **< 15 ms**.
   - **Stage 2 (DeskObjectNet INT8 Neural Verification)**: Evaluates the candidate region ($48 \times 48 \times 3$) using depthwise separable convolutions in **140 ms**.
   - **Stage 3 (Conservative Confidence Gating)**: Verifies that the predicted class probability exceeds the calibrated per-class threshold and is non-background. If not verified, the candidate is discarded with zero visual noise.

---

## 2. Model Architecture: DeskObjectNet INT8

`DeskObjectNet` is engineered specifically to maximize accuracy while fitting entirely within on-chip PSRAM and maintaining low latency with `esp-nn` assembly kernels:

```text
Input: [1, 3, 48, 48] (RGB Normalized [-1.0, 1.0])
  │
  ├── Stem: Conv2D(3 -> 24, k=3, s=2, p=1) + BatchNorm + ReLU6  --> [1, 24, 24, 24]
  ├── Stage 1: DepthwiseSeparableConv(24 -> 32, s=1)              --> [1, 32, 24, 24]
  ├── Stage 2: DepthwiseSeparableConv(32 -> 48, s=2)              --> [1, 48, 12, 12]
  ├── Stage 3: DepthwiseSeparableConv(48 -> 64, s=2)              --> [1, 64, 6, 6]
  ├── Global Average Pooling (6x6 -> 1x1)                         --> [1, 64, 1, 1]
  ├── Dropout (p=0.3)
  └── Linear Head (64 -> 6 Classes)                               --> [1, 6]
```

- **Parameters**: 13,854 weights.
- **Model Size (INT8 Flatbuffer)**: **14,808 bytes (14.46 KB)**.
- **Tensor Arena**: 96 KB PSRAM.
- **On-Device Execution**: 140.7 ms per inference on physical Xtensa LX6 @ 240 MHz.

---

## 3. Training & Quantization Pipeline

1. **PyTorch Training**:
   - Optimizer: `AdamW(lr=0.003, weight_decay=1e-4)` with `CosineAnnealingLR`.
   - Augmentations: Translation jitter ($\pm 6$ px), small rotation ($\pm 10^\circ$), horizontal flip (symmetric classes), dynamic gain simulation ($\pm 15\%$), and Gaussian sensor noise.
   - Batch size: 32. Epochs: 60.

2. **Full Integer Quantization (INT8 PTQ)**:
   - Exported to ONNX with opset 18.
   - Converted to TFLite via `onnx2tf` with `--ett full_integer_quant` and `--qt per-tensor`.
   - Calibrated using genuine sensor crop distributions.
   - Quantized parameters: `zero_point = 0`, `scale ≈ 1/128`, mapping directly to `model_input->data.int8[i] = (int8_t)(rgb_u8 - 128)`.

3. **C Array Export & PlatformIO Build**:
   - `models/deployment/generate_c_array.py` exports `alignas(16) const unsigned char g_custom_object_model_data[]`.
   - Compiled with PlatformIO against Espressif's optimized `esp-tflite-micro` and `esp-nn` assembly kernels.
   - Flash usage: 508 KB (16.5% of 3.14 MB partition).
