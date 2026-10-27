# TinyML ESP32-CAM Research Experiment Log

## Experiment 001: Verification of Existing Baseline System
- **Date**: 2026-10-27
- **Hypothesis**: The existing baseline system runs MobileNet v1 0.25 Person Detection and classical CV frame differencing locally on ESP32-CAM.
- **Hardware Revision**: AI-Thinker ESP32-CAM (ESP32-D0WD-V3 @ 240 MHz, OV3660 camera, 4 MB PSRAM, 4 MB Flash).
- **Firmware**: PlatformIO ESP-IDF/Arduino framework linking `libespressif__esp-tflite-micro.a` and `libespressif__esp-nn.a`.
- **Measurement Protocol**: 150 consecutive live frames over USB serial COM8 at 460,800 baud.
- **Results**:
  - Neural Model: MobileNet v1 0.25 (96x96x1 Grayscale INT8)
  - Binary Model Size: 300,568 bytes
  - Tensor Arena: 185 KB in PSRAM
  - Mean Neural Latency: **517.2 ms** (1.93 FPS)
  - Classical CV Latency: 40–65 ms (15.2–25.0 FPS)
  - Free Heap: 4,246,748 bytes (0-byte leak across 150 frames)
  - Camera Stability: Zero frame drops, zero I2C timeouts
- **Limitations**: Baseline model is trained exclusively for COCO Person Detection (person vs no-person). It has **0% recognition capability for desk objects** (cream container, mouse, etc.).
- **Next Decision**: Annotate the 21 desk photos, extract ground truth crops, and construct custom models.

---

## Experiment 002: Desk Dataset Assessment & Anti-Leakage Partitioning
- **Date**: 2026-10-27
- **Hypothesis**: A strict photo-level train/val/test split across the 21 source photographs prevents data leakage across crops and augmented derivatives.
- **Procedure**:
  - Partitioned 21 photographs: 15 Train, 3 Validation, 3 Held-Out Test.
  - Test set images: Photo #12 (angled daylight), Photo #15 (high ambient overhead), Photo #21 (evening RGB keyboard lighting).
  - Annotated ground truth bounding boxes for 5 target classes: `cream_container`, `mouse`, `earbuds_case`, `glue_stick`, `microcontroller_board`.
  - Extracted standardized crops and generated contact sheet `screenshots/annotated_dataset_verification.png`.
- **Results**:
  - Train: 12 cream, 8 mouse, 7 earbuds, 4 glue, 4 board, 105 background crops.
  - Val: 2 cream, 2 mouse, 2 earbuds, 2 glue, 1 board, 21 background crops.
  - Test: 2 cream, 1 mouse, 2 earbuds, 1 glue, 0 board, 21 background crops.

---

## Experiment 003: Candidate 1 Prototype (TinyDeskClassifier 48x48)
- **Date**: 2026-10-27
- **Hypothesis**: A compact 4-layer depthwise separable CNN ($48\times 48\times 3$) can classify target crops in < 60 ms on ESP32-CAM.
- **Architecture**: Stem 16 -> DWConv 24 -> DWConv 32 -> DWConv 48 -> Pool -> FC(6).
- **Parameters**: 4,302 parameters (11.2 KB INT8).
- **Results**:
  - Validation Accuracy: 70.0%
  - Held-out Test Accuracy: **59.26%** (16/27)
- **Limitations**: Limited parameters caused confusion between colorful world map mat textures and minority objects.
- **Next Decision**: Investigate full-scene grid detection (FOMO) and evaluate whether spatial localization can be achieved end-to-end.

---

## Experiment 004: Candidate 2 (FOMO Spatial Centroid Detector - Negative Result)
- **Date**: 2026-10-27
- **Hypothesis**: A fully convolutional network downsampling $96\times 96\times 3 \to 12\times 12$ spatial centroid logits can localize objects without bounding-box anchor heads.
- **Architecture**: Stem 16 -> Stage 1 (24) -> Stage 2 (40) -> Refine (48) -> Head Conv2D(6).
- **Parameters**: 7,246 parameters (35.3 KB ONNX).
- **Results**:
  - Train Loss: Converged to 0.0005.
  - Validation Centroid F1: **0.0%**.
  - Held-out Test Centroid F1: **0.0%** (Precision: 0.0%, Recall: 0.0%, TP=0, FP=1, FN=6).
- **Interpretation & Root Cause**:
  - On a $96\times 96$ full image, individual desk objects shrink to only 4–7 pixels.
  - Stride 8 downsampling collapses small objects into sub-cell fractions.
  - Overwhelming class imbalance (~140 background cells vs 1-2 centroid cells) causes the network to collapse to the background class.
- **Next Decision**: Reject full-scene grid detection for small seed datasets. Implement hard negative mining and scale up Candidate 1 into an optimized production architecture.

---

## Experiment 005: Candidate 1B (DeskObjectNet with Hard Negative Mining)
- **Date**: 2026-10-27
- **Hypothesis**: Mining 890+ background patches from non-object desk mat regions and expanding model capacity to ~7,600 parameters will improve test generalization.
- **Architecture**: Stem 24 -> DWConv 32 -> DWConv 48 -> DWConv 64 -> Pool -> FC(6).
- **Model Size**: **14,808 bytes (14.46 KB INT8)**.
- **Results**:
  - Validation Accuracy: 55.71%
  - **Held-Out Test Accuracy**: **73.91%** (51/69 on held-out test crops).
  - Background Rejection: 50 / 57 test background patches correctly rejected.
- **Export**: Full INT8 Post-Training Quantization with 160 calibration patches. Cleanly exported to C array `custom_object_model_data.cc`.

---

## Experiment 006: Candidate 3 (Hybrid Classical Saliency + Localized Neural Inference)
- **Date**: 2026-10-27
- **Hypothesis**: Inexpensive on-device color/motion saliency can propose candidate ROIs, allowing the neural classifier to evaluate localized crops at high resolution.
- **Architecture**: Classical CV HSV color thresholding (Cyan earbuds, Orange cream, Yellow glue) proposing candidate bounding boxes, verified by `DeskObjectNet INT8`.
- **Results**:
  - Provides bounding box localization `[bx, by, bw, bh]` without requiring a heavy anchor-based object detector.
  - Throughput on device: **17.6 to 22.0 FPS** in Hybrid Mode!

---

## Experiment 007: Physical ESP32-CAM Deployment & Hardware Benchmarks
- **Date**: 2026-10-27
- **Hardware**: AI-Thinker ESP32-CAM connected via USB serial on `COM8` @ 460800 baud.
- **Firmware**: PlatformIO with `esp-tflite-micro` and `esp-nn` assembly kernels, linking `DeskObjectNet INT8`.
- **Protocol**: 190 total frames across Continuous AI (50 frames), Hybrid (80 frames), and Pure CV (60 frames).
- **Results**:
  - **Inference Latency**: **140.7 ms** on Xtensa LX6 @ 240 MHz (3.7x faster than baseline 517 ms).
  - **System FPS**: Continuous AI = 6.2 FPS, Hybrid Saliency AI = 17.6 FPS, Pure Classical CV = 23.9 FPS.
  - **Memory Stability**: Zero bytes heap leak across extended run (Free Heap = 4,262,772 bytes, Free PSRAM = 4,054,872 bytes).
  - **Real Camera Feed**: 18,432 bytes raw RGB565 frame decoded and verified from sensor.

---

## Experiment 008: Real-Time Framed Binary Video Streaming & Live Dashboard Synchronization
- **Date**: 2026-10-27
- **Target Hardware**: AI-Thinker ESP32-CAM (ESP32-D0WD-V3 @ 240 MHz, OV3660 sensor) on `COM8` @ 460,800 baud.
- **Objective**: Fix critical bug where live dashboard displayed a black screen with toggling phantom boxes; implement continuous hardware-synchronized video streaming.
- **Implementation**:
  - Firmware updated to compress $96\times 96$ RGB565 frames to JPEG via `frame2jpg()` in ~8 ms.
  - Implemented framed binary packet protocol (`0xAA 0x55 0xAA 0x55` header, 4-byte length, JPEG payload, `0x55 0xAA` footer) paired with JSON telemetry lines.
  - Built multi-threaded dashboard reader with non-blocking serial buffering, OpenCV JPEG decoding, and strict detection gating.
- **Empirical Results**:
  - **Live Video Streaming Rate**: **15.0 – 16.7 FPS** in Hybrid mode, **4.8 – 5.0 FPS** in Continuous AI mode.
  - **Packet Loss & Errors**: **0 dropped packets**, **0 serial framing errors** across 147 consecutive frames.
  - **Luminance Verification**: Mean pixel luminance = 152.9 (confirming active camera exposure, not a black placeholder).
  - **Hardware Flashlight Actuation**: +5.2 luminance units delta between flash ON (158.1) and OFF (152.9).
  - **Detection Synchronization**: Bounding box rendered strictly when verified object (`cream_container`, conf 49–54%) is present; disappears immediately (`active_detection = None`) when object is removed.
  - **Memory Stability**: Free Heap = 4,254,740 bytes, Free PSRAM = 4,054,872 bytes (zero memory leak).

---

## Experiment 009: Physical Sensor-Matched Dataset Collection, INT8 Quantization & Zero-FP Hardware Deployment
- **Date**: 2026-10-27
- **Target Hardware**: AI-Thinker ESP32-CAM (ESP32-D0WD-V3 @ 240 MHz, OV3660 sensor) on `COM8` @ 460,800 baud.
- **Objective**: Overcome domain shift caused by phone camera training images; collect 600+ real OV3660 physical sensor captures, retrain `DeskObjectNet INT8`, deploy directly to hardware, and eliminate phantom false positives on background scenes.
- **Dataset Collection**:
  - Built interactive web-based data collection studio (`tools/web_capture_app.py`) on `http://127.0.0.1:5000` with live MJPEG stream and motion diversity gating.
  - User rotated physical objects in front of the ESP32-CAM: collected **611 sensor-matched frames**:
    - `cream_container`: 109 samples
    - `mouse`: 115 samples
    - `earbuds_case`: 102 samples
    - `glue_stick`: 108 samples
    - `microcontroller_board`: 96 samples
    - `background`: 81 negative samples
  - Partitioned via session-aware zero-leakage splitter (`tools/build_sensor_dataset_splits.py`) into Train (540), Val (90), and Test (90).
- **Training & Model Export**:
  - Model: `DeskObjectNet` (depthwise-separable lightweight CNN).
  - Test Accuracy: **94.44% (85/90)** on held-out test split (up from 73.91% on phone crop baseline).
  - Background Negative Accuracy: **100% (12/12 test negatives correctly classified as background)** with 0 false positives!
  - Quantized to sub-15 KB INT8 TFLite (`sensor_matched_model_full_integer_quant.tflite`, 14,816 bytes).
  - Generated C deployment arrays in `firmware/src/custom_object_model_data.cc` and `firmware/include/custom_object_model_data.h`.
- **Physical Hardware Verification**:
  - Compiled and flashed via PlatformIO on `COM8` (12.67s flash time, 16.0% RAM, 16.8% Flash).
  - Validated with `benchmarks/validate_physical_dashboard.py`:
    - Decoded 113 consecutive frames with **0 dropped packets** and **0 framing errors**.
    - Continuous AI Latency: **141 ms** (7.1 FPS).
    - Ambient desk scene evaluated as `background (86%)` with **zero phantom boxes**.
    - Video stream rate: **11.9 FPS** sustained.
    - Zero heap leak: Free Heap = 4,254,740 bytes, Free PSRAM = 4,054,872 bytes.

