# Sensor-Matched Dataset Specification: Physical OV3660 Capture Pipeline

## 1. Motivation: Eliminating the Camera Domain Gap

Prior TinyML models for this project were trained on downsampled crops from high-resolution smartphone photographs ($1280 \times 960$ RGB JPEG). While this permitted rapid prototyping, it introduced an insurmountable **sensor domain shift**:
- The deployment hardware uses an **OmniVision OV3660 / OV2640** CMOS sensor ($96 \times 96$ RGB565) paired with a miniature plastic lens.
- The physical sensor produces fixed-pattern noise (FPN), dynamic auto-gain swings (AGC/AEC), lens vignetting, and Bayer demosaicing artifacts.
- Training on smooth, high-fidelity smartphone photographs caused the neural network to confuse raw sensor noise on empty desk surfaces with target object features, leading to false positives.

**Mission 2 Solution**: Every single training, validation, and test image is captured directly from the **physical ESP32-CAM OV3660 sensor** that runs the final model.

---

## 2. Dataset Architecture & Directory Layout

The sensor-matched dataset is organized under `dataset/raw_esp32cam/` and partitioned without temporal leakage under `dataset/processed_esp32cam/`:

```text
dataset/
├── raw_esp32cam/                       # Direct raw sensor frame captures (PNG 96x96 RGB)
│   ├── cream_container/               # Physical Nivea Soft white container
│   ├── mouse/                         # Physical Portronics wireless optical mouse
│   ├── earbuds_case/                  # Physical mint-green charging case
│   ├── glue_stick/                    # Physical Fevistik yellow & blue glue stick
│   ├── microcontroller_board/         # Physical ESP32 / ESP32-CAM dev board
│   └── background/                    # Empty desk, laptop, keyboard, cables, hands, shadows
├── processed_esp32cam/                 # Grouped by capture session (Zero temporal leakage)
│   ├── train/                         # ~70% of independent recording sessions
│   ├── val/                           # ~15% of independent recording sessions
│   └── test/                          # ~15% of independent recording sessions
├── manifests/
│   ├── dataset_manifest_esp32cam.json # Complete metadata schema for every image
│   └── dataset_manifest_esp32cam.csv  # Flat tabular manifest
└── reports/
    ├── contact_sheet_*.png            # Visual verification contact sheets
    └── capture_tool_preview_verified.png
```

---

## 3. Real-Time Diversity & Quality Filtering Algorithm

To prevent collecting thousands of redundant, identical consecutive video frames or degraded motion-blurred images, [`capture_tool.py`](file:///C:/All_Projects/esp32_cam_ai_system/capture_tool.py) executes a multi-stage diversity filter in real time:

1. **Temporal Rate-Limiter ($\Delta t \ge 350$ ms)**:
   Enforces a minimum interval between consecutive accepted frames, allowing the user time to translate or rotate the camera.

2. **Laplacian Blur / Motion Denoise ($\text{Var}(\nabla^2 I) \ge 35.0$)**:
   $$\text{Sharpness} = \frac{1}{N} \sum (L(x, y) - \mu_L)^2$$
   Frames taken during rapid camera rotation or hand shake drop below 35.0 and are discarded immediately.

3. **Auto-Exposure / Luminance Window ($20.0 \le \mu_I \le 240.0$)**:
   Discards transient blown-out white frames caused by sudden lighting shifts or underexposed sensor frames.

4. **Mean Absolute Difference from Previous Accepted Sample ($\text{MAD} \ge 3.5$)**:
   $$\text{MAD} = \frac{1}{H \times W} \sum_{x, y} |I_{\text{current}}(x, y) - I_{\text{last\_accepted}}(x, y)|$$
   If the camera is held stationary, the difference drops below 3.5 and the duplicate frame is suppressed.

5. **Manual Capture Override (`[C]` key)**:
   Allows the user to force-capture any frame regardless of filter thresholds when a unique view is desired.

---

## 4. Metadata Schema Recorded Per Sample

Every sample saved by `capture_tool.py` is logged in `dataset/manifests/dataset_manifest_esp32cam.json`:

```json
{
  "image_id": "esp32_cream_container_20261027_230500_a1b2c3",
  "filename": "esp32cam_cream_container_20261027_230500_a1b2c3.png",
  "relative_path": "dataset/raw_esp32cam/cream_container/esp32cam_cream_container_20261027_230500_a1b2c3.png",
  "class_label": "cream_container",
  "session_id": "session_cream_20261027_230412",
  "timestamp": "2026-10-27 23:05:00",
  "resolution": [96, 96],
  "channels": 3,
  "sensor": "OmniVision OV3660",
  "sharpness": 142.8,
  "mean_luminance": 158.4,
  "diff_from_prev": 12.6,
  "capture_mode": "AUTO",
  "status": "VALID"
}
```

---

## 5. Physical Scanning Protocol for User

When using [`capture_tool.py`](file:///C:/All_Projects/esp32_cam_ai_system/capture_tool.py):

| Object Class | Physical Target Description | Recommended Scanning Motion |
| :--- | :--- | :--- |
| **`cream_container`** | Nivea Soft white round tub with orange rim | Rotate camera around tub: front brand text, angled lid, overhead, side profile (10–30 cm). |
| **`mouse`** | Black wireless Portronics optical mouse | Move camera across mouse buttons, top palm curvature, left/right grip, angled daylight. |
| **`earbuds_case`** | Mint-green / cyan charging case | Scan front charging LED, lid seam, angled perspective, close-up (8–20 cm). |
| **`glue_stick`** | Fevistik yellow cap and blue body | Capture standing vertically, laying horizontal on desk, cap facing lens, body tilted. |
| **`microcontroller_board`** | ESP32 / ESP32-CAM dev board | Scan PCB traces, ESP-WROOM module shielding, red power LED from multiple orientations. |
| **`background`** | Non-target desk environment | Pan across empty desk surface, mechanical keyboard, mousepad world map, laptop edge, shadows, cables, hands. |
