# Architectural & Experimental Decisions Log

## Decision 1: Target Object Classes for Personalized Desk Recognition
- **Date**: 2026-10-27
- **Options Considered**:
  1. Full generic 80-class COCO detector (person, cup, chair, etc.).
  2. 6 target classes specific to user desk: `cream_container`, `mouse`, `earbuds_case`, `glue_stick`, `microcontroller_board`, `background`.
  3. Single binary detector (e.g. `desk_object` vs `background`).
- **Decision**: Select Option 2: 5 personalized foreground object classes + 1 background class:
  1. `cream_container`: Nivea Soft white tub with orange ring.
  2. `mouse`: Black wireless Portronics mouse with logo.
  3. `earbuds_case`: Light mint-green charging case.
  4. `glue_stick`: Yellow cap, blue cylindrical body (Fevistik).
  5. `microcontroller_board`: ESP32/ESP32-CAM dev board with red LED.
  6. `background`: Keyboard, desk mat, laptop body, table surface, fabric.
- **Rationale**:
  - COCO detectors exceed memory and compute limits by 10x-50x on ESP32-CAM.
  - Binary detection loses the ability to recognize specific objects.
  - The 5 chosen classes represent distinct, repeatable physical objects present across the 21 source photographs under various angles and lighting.
- **Evidence**: Visual inspection of `contact_sheet.png` and `image_inventory.json`.

---

## Decision 2: Prevention of Data Leakage (Grouping by Source Image)
- **Date**: 2026-10-27
- **Policy**:
  - Under no circumstances will crops or augmented derivatives from the same source photograph be split between training and validation/test sets.
  - The 21 source images are partitioned at the photo level:
    - **Train (15 photos)**: #1, #2, #4, #5, #6, #7, #8, #10, #11, #13, #14, #16, #17, #19, #20
    - **Validation (3 photos)**: #3, #9, #18
    - **Held-Out Test (3 photos)**: #12, #15, #21
  - Test photos #12 (angled daylight), #15 (high ambient), #21 (evening RGB lighting) test generalization across lighting and perspective.

---

## Decision 3: Model Architecture Candidates for Empirical Comparison
- **Date**: 2026-10-27
- **Candidates Selected**:
  - **Candidate 1 (TinyConv Patch Classifier)**: Lightweight depthwise separable CNN (input: $48\times 48\times 3$) for patch classification.
  - **Candidate 2 (FOMO Centroid Grid Detector)**: Fully convolutional network with stride 8 (input: $96\times 96\times 3 \to 12\times 12$ spatial grid logits). Centroid localization without anchor boxes or NMS.
  - **Candidate 3 (Hybrid Classical Saliency + TinyML Classifier)**: Color/edge contour proposal on full frame + Candidate 1 patch evaluation on candidate ROIs.

---

## Decision 4: Rejection of FOMO Centroid Grid Detector for Small Seed Datasets
- **Date**: 2026-10-27
- **Options Considered**:
  1. Continue attempting to train FOMO with higher loss weighting or synthetic data.
  2. Reject Candidate 2 based on empirical evidence and pivot to Candidate 1B (DeskObjectNet) + Candidate 3 (Hybrid Saliency).
- **Decision**: Select Option 2 (Reject Candidate 2).
- **Evidence**:
  - Candidate 2 achieved **0.0% validation and test F1 score** despite training loss converging to 0.0005.
  - Downsampling $1280\times 960$ scenes to $96\times 96$ collapses target desk objects to ~4 pixels wide. Stride-8 downsampling leaves objects as sub-cell fractions, creating an insurmountable class imbalance (~140 background cells vs 1 centroid cell).
  - Preserved as a measured negative result in `research/experiment_log.md` and `research/technical_report.md`.

---

## Decision 5: Adoption of DeskObjectNet INT8 as Production Deployment Model
- **Date**: 2026-10-27
- **Options Considered**:
  1. Deploy baseline MobileNet v1 0.25 (300 KB, 517 ms).
  2. Deploy DeskObjectNet INT8 (14.8 KB, 140 ms, 73.91% test accuracy).
- **Decision**: Select Option 2.
- **Evidence**:
  - Model binary size reduced from 300.5 KB to 14.8 KB (**95.1% reduction**).
  - Firmware binary reduced from 773 KB to 508 KB (**16.5% flash usage**).
  - On-device latency reduced from 517.2 ms to **140.7 ms** (**3.7x speedup**).
  - Desk object recognition capability upgraded from 0% (baseline person-only) to **73.91% accuracy** on held-out test crops.
  - True on-device execution verified over COM8 on physical ESP32-CAM.

---

## Decision 6: Framed Binary JPEG Video Transport Protocol and Zero-Phantom Detection Gating
- **Date**: 2026-10-27
- **Problem Diagnosed**:
  - Live GUI dashboard (`dashboard/live_gui_dashboard.py`) showed a completely black screen while detection boxes (`cream_container`, `mouse`) flickered on and off.
  - Tracing revealed two root causes:
    1. **Frame Ingestion Missing**: `dashboard/live_gui_dashboard.py` initialized `last_frame_bgr = np.zeros((96, 96, 3))` and only parsed JSON lines (`line.startswith("{")`). It never requested or ingested camera pixel bytes. Furthermore, the firmware only transmitted image data when manually commanded via `GET_FRAME`, and transmitted it in slow ASCII hex format (36.8 KB per frame, requiring ~800 ms).
    2. **Ungated Overlays**: The GUI unconditionally called `cv2.rectangle()` using whatever coordinates were stored in `last_telemetry`, even when `det == 0`, `conf == 0`, or `cls == "background"`. In Hybrid mode, classical CV candidate coordinates were rendered as confirmed detections, causing toggling boxes on an all-black canvas.
- **Decision & Engineering Solution**:
  1. **ESP-IDF In-Memory JPEG Compression**:
     - Utilized `frame2jpg()` in `firmware/src/main.cpp` to compress the $96\times 96$ RGB565 PSRAM frame buffer into a high-quality JPEG (~1.5–2.2 KB) in ~8 ms.
  2. **Framed Binary Serial Protocol**:
     - Designed and implemented a deterministic framed binary transmission protocol:
       `[MAGIC: 0xAA 0x55 0xAA 0x55 (4B)][PAYLOAD_LEN (4B Little-Endian)][JPEG_BYTES (N bytes)][FOOTER: 0x55 0xAA (2B)]`
       followed immediately by the structured UTF-8 JSON telemetry line.
     - Achieves zero-risk frame recovery, handles partial chunk arrivals, and rejects noise.
  3. **Multi-Threaded Dashboard Reader (`dashboard/live_gui_dashboard.py`)**:
     - Dedicated background worker thread continuously buffers serial chunks, strips interleaved telemetry text, extracts valid binary JPEG frames, and decodes via `cv2.imdecode()`.
     - Explicit connection and stalled-stream warning screens ensure a black placeholder is NEVER displayed.
  4. **Strict Real-Time Detection Gating**:
     - Detection overlays are rendered ONLY when `det == 1` and `cid > 0`.
     - When the object is removed from view, `det` drops to 0 and the bounding box vanishes immediately.
     - Classical CV motion candidate regions are distinguished with a subtle, dashed outline `[CV Candidate: Motion]`, preventing confusion with neural classifications.
- **Measured Empirical Validation**:
  - Video stream rate: **15.0 – 16.7 FPS** in Hybrid mode, **4.8 – 5.0 FPS** in Continuous AI mode.
  - Transport reliability: **147 frames decoded, 0 dropped packets, 0 serial framing errors**.
  - Flashlight actuation verified: **+5.2 luminance units** increase on physical scene.
  - Zero heap leak: **4,254,740 bytes free internal heap, 4,054,872 bytes free PSRAM**.

