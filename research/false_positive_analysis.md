# Deep Dive Analysis: Sensor Domain Shift & False Positives on ESP32-CAM

## 1. Executive Summary

The initial version of `DeskObjectNet INT8` demonstrated on-device execution feasibility on the ESP32-CAM (**140.7 ms** latency on Xtensa LX6 @ 240 MHz, 508 KB flash usage, 16.5% partition). However, when evaluated against genuine live camera scenes from the physical **OmniVision OV3660** sensor, false positive detections occurred on un-target background regions (e.g. keyboard keys, desk grain, shadows, and reflections).

This investigation examines the root technical causes across the physical sensor, data distribution, optical pipeline, and quantization boundaries.

---

## 2. Quantitative Comparison: Smartphone Sensor vs. ESP32-CAM OV3660

The root cause of false positives is a massive **domain distribution shift** between the training dataset and deployment hardware:

| Optical & Sensor Parameter | Training Source (Smartphone Camera) | Deployment Sensor (OmniVision OV3660) | Impact on Learned Features |
| :--- | :--- | :--- | :--- |
| **Native Sensor Resolution** | $4032 \times 3024$ (12 MP BSI CMOS) | $2048 \times 1536$ scaled to **$96 \times 96$** | Extreme loss of high-frequency texture and text detail |
| **Pixel Pitch & Area** | ~1.4 $\mu$m with quad-binning | ~1.75 $\mu$m on miniature 1/5" optical format | Significantly lower signal-to-noise ratio (SNR) |
| **On-Chip ISP Processing** | Multi-frame HDR, edge sharpening, temporal denoise | Fixed hardware AGC/AEC, minimal smoothing | Raw CMOS noise patterns, sensor hot pixels, temporal grain |
| **Dynamic Range** | High Dynamic Range (> 12 EV) | Low Dynamic Range (< 8 EV) | Highlights blow out to pure white (255); shadows crush to near 0 |
| **Lens Optics & Aberration**| 6-element aspheric glass with IR cut filter | Miniature plastic lens module, fixed-focus | Chromatic aberration at peripheral pixels, radial vignetting |
| **Color Reproduction** | Calibrated sRGB, adaptive white balance | Auto White Balance (AWB) hunts continuously | Hue shifts depending on nearby ambient lighting and reflections |
| **Effective Patch Resolution**| Extracted from $1280 \times 960$ crops | Cropped from $96 \times 96$ sensor capture | Model trained on sharp downscaled crops, tested on noisy optical patches |

---

## 3. Visual Domain Discrepancy Breakdown

### 3.1 Frequency Distribution Mismatch
- **Smartphone Crops**: When downsampled to $48 \times 48$, phone photographs contain exceptionally clean, high-contrast anti-aliased edges and smooth gradients.
- **OV3660 Hardware Crops**: At $96 \times 96$, individual pixels from the OV3660 carry high-frequency Bayer demosaicing artifacts and fixed-pattern noise (FPN). The initial convolutional kernels ($3\times 3$ stride 2) activate aggressively on sensor noise patterns rather than semantic object boundaries.

### 3.2 Exposure & White Balance Instability
- The ESP32 camera driver runs autonomous gain control (`s->set_exposure_ctrl(s, 1)` and `s->set_awb_gain(s, 1)`).
- As the camera moves slightly across a desk, the average scene brightness changes, triggering auto-exposure shifts. A white desk surface can shift by $\pm 40$ RGB luminance units within 500 ms.
- In the initial phone dataset, all 21 photos were taken under static lighting, leaving the model unprepared for dynamic gain swings.

### 3.3 Training Data Saliency vs. Negative Diversity
- The phone dataset contained only 21 source scenes. Even with 891 hard-negative background crops extracted across those 21 photos, they all represented the **same physical desk mat and keyboard**.
- When the ESP32-CAM points at a wall, an arm, a cable, or an empty room, the model encountered textures entirely outside its convex training hull, leading to overconfident false positive predictions.

---

## 4. Why the Previous Live Dashboard Appeared Broken

During initial testing, the live GUI dashboard exhibited two severe symptoms:
1. **Black Screen**: `dashboard/live_gui_dashboard.py` initialized `last_frame_bgr` to all zeros and only read JSON telemetry lines. The firmware only sent images on manual `GET_FRAME` requests in slow ASCII hex. As a result, the video viewport remained permanently black.
2. **Flickering / Toggling Detection Overlays**: In Hybrid mode, classical motion/saliency detection continuously proposed candidate bounding boxes `[bx, by, bw, bh]`. Because `live_gui_dashboard.py` drew bounding boxes unconditionally without checking `det == 1` or filtering out `cls == "background"`, candidate search coordinates toggled over the black screen.

*Note*: Both transport and overlay gating issues were completely resolved in the previous fix using framed binary JPEG streaming (`0xAA 0x55 0xAA 0x55`) and strict `det == 1` gating.

---

## 5. Strategic Prescription: Mission 2 Sensor-Matched Architecture

To eradicate false positives and build a production-grade recognizer, Mission 2 enforces:
1. **100% Sensor-Matched Data**: All training, validation, and test samples must be captured directly through the physical OV3660 sensor on the ESP32-CAM via `capture_tool.py`.
2. **Session-Grouped Splitting**: Frames from the same scanning pass are never mixed across train and test sets, preventing temporal video leakage.
3. **Extensive Physical Negative Sampling**: Multiple distinct background sessions (empty desk, varied distances, clutter, hands, shadows, off-axis views).
4. **Calibrated Conservative Thresholds**: No detection box or label is ever drawn unless validated above its per-class threshold; background is strictly suppressed.
