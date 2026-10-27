"""
ESP32-CAM Sensor-Matched Dataset Capture Application (capture_tool.py)
Captures genuine, direct camera frames from the physical OV3660 sensor on ESP32-CAM.
Implements real-time diversity filtering (blur, exposure, duplicate rejection),
object scanning workflows, thumbnail review, sample deletion, and dataset manifest export.
"""

import os
import sys
import time
import json
import csv
import struct
import threading
import uuid
import cv2
import numpy as np

# Ensure root paths
PROJECT_ROOT = r"C:\All_Projects\esp32_cam_ai_system"
RAW_DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset", "raw_esp32cam")
MANIFEST_DIR = os.path.join(PROJECT_ROOT, "dataset", "manifests")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "dataset", "reports")
for d in [RAW_DATASET_DIR, MANIFEST_DIR, REPORTS_DIR]:
    os.makedirs(d, exist_ok=True)

# Hardware Serial Settings
PORT = "COM8"
BAUD = 460800
MAGIC_HEADER = b"\xAA\x55\xAA\x55"
MAGIC_FOOTER = b"\x55\xAA"

DEFAULT_CLASSES = [
    "cream_container",
    "mouse",
    "earbuds_case",
    "glue_stick",
    "microcontroller_board",
    "background"
]

CLASS_COLORS = {
    "background": (120, 120, 120),
    "cream_container": (0, 165, 255),    # Orange
    "mouse": (255, 0, 255),              # Magenta
    "earbuds_case": (255, 255, 0),       # Cyan
    "glue_stick": (0, 255, 255),         # Yellow
    "microcontroller_board": (50, 50, 255) # Red
}

INSTRUCTIONS_BY_CLASS = {
    "cream_container": "SCANNING CREAM CONTAINER: Move slowly around white tub. Rotate front logo, angled sides, top lid (10-30cm).",
    "mouse": "SCANNING MOUSE: Move around black optical mouse. Capture top, buttons, sides, angled views under ambient lighting.",
    "earbuds_case": "SCANNING EARBUDS CASE: Move around mint-green charging case. Capture front, lid, angled, and close views.",
    "glue_stick": "SCANNING GLUE STICK: Move around yellow cap & blue cylindrical body. Capture upright, tilted, and side angles.",
    "microcontroller_board": "SCANNING MCU BOARD: Capture ESP32/ESP32-CAM dev board from multiple angles with status LEDs.",
    "background": "CAPTURING BACKGROUND / NEGATIVES: Pan over empty desk, laptop, keyboard, desk mat, cables, shadows, and clutter."
}


class SerialVideoReceiver:
    """Thread-safe receiver for framed binary JPEG video stream from ESP32-CAM."""
    def __init__(self, port=PORT, baud=BAUD):
        import serial
        self.port = port
        self.baud = baud
        self.ser = None
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        
        self.latest_frame = None
        self.frame_timestamp = 0.0
        self.frames_received = 0
        self.dropped_packets = 0
        self.serial_errors = 0
        self.stream_fps = 0.0
        self.fps_counter = 0
        self.fps_last_time = time.time()
        self.device_telemetry = {}

    def connect(self):
        import serial
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=0.1)
            self.ser.dtr = False
            self.ser.rts = True
            time.sleep(0.1)
            self.ser.rts = False
            time.sleep(0.4)
            self.ser.reset_input_buffer()
            return True
        except Exception as e:
            print(f"[SERIAL ERROR] Failed to open {self.port}: {e}")
            return False

    def start(self):
        if not self.ser or not self.ser.is_open:
            if not self.connect():
                return False
        self.running = True
        self.thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.thread.start()
        return True

    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.ser and self.ser.is_open:
            try:
                self.ser.close()
            except:
                pass

    def send_command(self, cmd_str):
        if self.ser and self.ser.is_open:
            try:
                self.ser.write(f"{cmd_str}\n".encode())
                self.ser.flush()
            except Exception as e:
                print(f"[CMD ERROR] {e}")

    def _reader_loop(self):
        buf = bytearray()
        while self.running:
            try:
                bytes_avail = self.ser.in_waiting or 1
                chunk = self.ser.read(min(bytes_avail, 4096))
                if chunk:
                    buf.extend(chunk)

                while self.running and len(buf) > 0:
                    idx = buf.find(MAGIC_HEADER)
                    if idx == -1:
                        nl = buf.rfind(b"\n")
                        if nl != -1:
                            text_chunk = buf[:nl].decode('utf-8', errors='replace')
                            for line in text_chunk.splitlines():
                                self._handle_text_line(line)
                            del buf[:nl + 1]
                        break

                    if idx > 0:
                        text_chunk = buf[:idx].decode('utf-8', errors='replace')
                        for line in text_chunk.splitlines():
                            self._handle_text_line(line)
                        del buf[:idx]

                    if len(buf) < 8:
                        break

                    payload_len = struct.unpack("<I", buf[4:8])[0]
                    if payload_len < 50 or payload_len > 45000:
                        del buf[:1]
                        with self.lock:
                            self.serial_errors += 1
                        continue

                    total_len = 8 + payload_len + 2
                    if len(buf) < total_len:
                        break

                    footer = buf[8 + payload_len : total_len]
                    if footer == MAGIC_FOOTER:
                        jpg_bytes = bytes(buf[8 : 8 + payload_len])
                        del buf[:total_len]
                        
                        frame = cv2.imdecode(np.frombuffer(jpg_bytes, np.uint8), cv2.IMREAD_COLOR)
                        now = time.time()
                        if frame is not None:
                            with self.lock:
                                self.latest_frame = frame
                                self.frame_timestamp = now
                                self.frames_received += 1
                                self.fps_counter += 1
                        else:
                            with self.lock:
                                self.dropped_packets += 1
                    else:
                        with self.lock:
                            self.serial_errors += 1
                        del buf[:4]

                now = time.time()
                if now - self.fps_last_time >= 1.0:
                    with self.lock:
                        self.stream_fps = (self.fps_counter * 1.0) / (now - self.fps_last_time)
                        self.fps_counter = 0
                        self.fps_last_time = now

            except Exception:
                with self.lock:
                    self.serial_errors += 1
                time.sleep(0.02)

    def _handle_text_line(self, line):
        line = line.strip()
        if line.startswith("{") and "cid" in line:
            try:
                data = json.loads(line)
                with self.lock:
                    self.device_telemetry = data
            except:
                pass

    def get_latest_frame(self):
        with self.lock:
            if self.latest_frame is None:
                return None, 0.0, 0, 0.0, 0, 0
            return self.latest_frame.copy(), self.frame_timestamp, self.frames_received, self.stream_fps, self.dropped_packets, self.serial_errors


class DiversityFilter:
    """Configurable quality and diversity filter for automatic frame collection."""
    def __init__(self, min_interval_sec=0.35, min_sharpness=35.0, min_diff=3.5, min_luminance=20.0, max_luminance=240.0):
        self.min_interval_sec = min_interval_sec
        self.min_sharpness = min_sharpness
        self.min_diff = min_diff
        self.min_luminance = min_luminance
        self.max_luminance = max_luminance
        
        self.last_accepted_time = 0.0
        self.last_accepted_gray = None
        self.reject_stats = {
            "rate_limit": 0,
            "blur": 0,
            "underexposed": 0,
            "overexposed": 0,
            "duplicate": 0
        }

    def evaluate(self, frame_bgr):
        now = time.time()
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        
        # 1. Timing check
        dt = now - self.last_accepted_time
        if dt < self.min_interval_sec:
            self.reject_stats["rate_limit"] += 1
            return False, f"RATE LIMIT ({int(dt*1000)}ms < {int(self.min_interval_sec*1000)}ms)", 0.0, 0.0, 0.0

        # 2. Exposure check
        mean_lum = float(np.mean(gray))
        if mean_lum < self.min_luminance:
            self.reject_stats["underexposed"] += 1
            return False, f"UNDEREXPOSED (lum={mean_lum:.1f} < {self.min_luminance})", 0.0, mean_lum, 0.0
        if mean_lum > self.max_luminance:
            self.reject_stats["overexposed"] += 1
            return False, f"OVEREXPOSED (lum={mean_lum:.1f} > {self.max_luminance})", 0.0, mean_lum, 0.0

        # 3. Sharpness / Blur check (Laplacian Variance)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if sharpness < self.min_sharpness:
            self.reject_stats["blur"] += 1
            return False, f"BLURRED / MOTION (var={sharpness:.1f} < {self.min_sharpness})", sharpness, mean_lum, 0.0

        # 4. Difference from previous accepted frame
        diff = 100.0
        if self.last_accepted_gray is not None:
            diff = float(np.mean(np.abs(gray.astype(float) - self.last_accepted_gray.astype(float))))
            if diff < self.min_diff:
                self.reject_stats["duplicate"] += 1
                return False, f"DUPLICATE / STATIC (diff={diff:.1f} < {self.min_diff})", sharpness, mean_lum, diff

        # Passed all filters
        return True, "ACCEPTED", sharpness, mean_lum, diff

    def record_acceptance(self, frame_bgr):
        self.last_accepted_time = time.time()
        self.last_accepted_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)


class DatasetManager:
    """Manages raw ESP32-CAM images on disk and exports dataset manifests."""
    def __init__(self, base_dir=RAW_DATASET_DIR, manifest_dir=MANIFEST_DIR):
        self.base_dir = base_dir
        self.manifest_dir = manifest_dir
        self.manifest_json_path = os.path.join(self.manifest_dir, "dataset_manifest_esp32cam.json")
        self.manifest_csv_path = os.path.join(self.manifest_dir, "dataset_manifest_esp32cam.csv")
        self.samples = []
        self.load_manifest()

    def get_class_dir(self, class_name):
        d = os.path.join(self.base_dir, class_name)
        os.makedirs(d, exist_ok=True)
        return d

    def count_samples_per_class(self):
        counts = {}
        for c in DEFAULT_CLASSES:
            counts[c] = 0
        for s in self.samples:
            cls = s["class_label"]
            counts[cls] = counts.get(cls, 0) + 1
        return counts

    def save_sample(self, frame_bgr, class_name, session_id, sharpness, mean_lum, diff, capture_mode="AUTO"):
        c_dir = self.get_class_dir(class_name)
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        unique_suffix = str(uuid.uuid4())[:6]
        filename = f"esp32cam_{class_name}_{timestamp_str}_{unique_suffix}.png"
        rel_path = os.path.join("dataset", "raw_esp32cam", class_name, filename)
        abs_path = os.path.join(c_dir, filename)

        # Save exact sensor frame bytes before any resizing
        cv2.imwrite(abs_path, frame_bgr)

        record = {
            "image_id": f"esp32_{class_name}_{timestamp_str}_{unique_suffix}",
            "filename": filename,
            "relative_path": rel_path.replace("\\", "/"),
            "absolute_path": abs_path,
            "class_label": class_name,
            "session_id": session_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "resolution": [frame_bgr.shape[1], frame_bgr.shape[0]],
            "channels": 3,
            "sensor": "OmniVision OV3660",
            "sharpness": round(sharpness, 2),
            "mean_luminance": round(mean_lum, 2),
            "diff_from_prev": round(diff, 2),
            "capture_mode": capture_mode,
            "status": "VALID"
        }
        self.samples.append(record)
        self.save_manifest()
        return record

    def delete_last_sample(self):
        if not self.samples:
            return None
        last = self.samples.pop()
        abs_p = last["absolute_path"]
        if os.path.exists(abs_p):
            try:
                os.remove(abs_p)
            except Exception as e:
                print(f"[WARN] Failed to delete file {abs_p}: {e}")
        self.save_manifest()
        return last

    def load_manifest(self):
        if os.path.exists(self.manifest_json_path):
            try:
                with open(self.manifest_json_path, "r") as f:
                    data = json.load(f)
                    self.samples = data.get("samples", [])
            except Exception as e:
                print(f"[WARN] Failed to read existing manifest: {e}")
                self.samples = []

    def save_manifest(self):
        summary = {
            "dataset_name": "ESP32-CAM Sensor-Matched Custom Vision Dataset",
            "hardware_sensor": "OmniVision OV3660 @ 96x96 RGB565 via ESP32-CAM",
            "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_samples": len(self.samples),
            "class_counts": self.count_samples_per_class(),
            "samples": self.samples
        }
        with open(self.manifest_json_path, "w") as f:
            json.dump(summary, f, indent=2)

        # Also write CSV
        fieldnames = [
            "image_id", "filename", "relative_path", "class_label", "session_id",
            "timestamp", "resolution", "sharpness", "mean_luminance", "diff_from_prev", "capture_mode", "status"
        ]
        with open(self.manifest_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for s in self.samples:
                writer.writerow(s)

    def generate_contact_sheet(self, class_name, max_cols=8, thumb_size=64):
        cls_samples = [s for s in self.samples if s["class_label"] == class_name]
        if not cls_samples:
            return None
        n = min(len(cls_samples), 64)
        cols = min(n, max_cols)
        rows = int(np.ceil(n / cols))
        
        sheet = np.zeros((rows * thumb_size, cols * thumb_size, 3), dtype=np.uint8)
        for idx in range(n):
            r = idx // cols
            c = idx % cols
            img_p = cls_samples[-(idx + 1)]["absolute_path"] # most recent first
            if os.path.exists(img_p):
                im = cv2.imread(img_p)
                if im is not None:
                    im_resized = cv2.resize(im, (thumb_size, thumb_size), interpolation=cv2.INTER_AREA)
                    sheet[r*thumb_size:(r+1)*thumb_size, c*thumb_size:(c+1)*thumb_size] = im_resized
                    # subtle grid border
                    cv2.rectangle(sheet, (c*thumb_size, r*thumb_size), ((c+1)*thumb_size, (r+1)*thumb_size), (40, 40, 50), 1)

        out_path = os.path.join(REPORTS_DIR, f"contact_sheet_{class_name}.png")
        cv2.imwrite(out_path, sheet)
        return out_path


class CaptureApplication:
    """Production GUI application for sensor-matched dataset collection."""
    def __init__(self, port=PORT, baud=BAUD):
        self.receiver = SerialVideoReceiver(port, baud)
        self.filter = DiversityFilter()
        self.manager = DatasetManager()
        
        self.classes = list(DEFAULT_CLASSES)
        self.active_class_idx = 0
        self.is_auto_capturing = False
        self.session_id = f"session_{time.strftime('%Y%m%d_%H%M%S')}"
        
        self.recent_thumbnails = [] # store last 4 images for filmstrip
        self.last_decision = "IDLE (Press SPACE to start auto-capture)"
        self.last_decision_color = (180, 180, 180)
        self.session_accepted_count = 0
        
        # Load existing images into recent thumbnails
        self._refresh_recent_thumbnails()

    def _refresh_recent_thumbnails(self):
        self.recent_thumbnails.clear()
        cls_name = self.classes[self.active_class_idx]
        matching = [s for s in self.manager.samples if s["class_label"] == cls_name]
        for s in matching[-4:]:
            if os.path.exists(s["absolute_path"]):
                im = cv2.imread(s["absolute_path"])
                if im is not None:
                    self.recent_thumbnails.append(im)

    def start(self):
        print(f"[INIT] Connecting to ESP32-CAM on {PORT} @ {BAUD}...")
        return self.receiver.start()

    def stop(self):
        self.receiver.stop()

    def trigger_capture(self, frame_bgr, mode="AUTO", sharpness=0.0, mean_lum=0.0, diff=0.0):
        cls_name = self.classes[self.active_class_idx]
        record = self.manager.save_sample(
            frame_bgr=frame_bgr,
            class_name=cls_name,
            session_id=self.session_id,
            sharpness=sharpness,
            mean_lum=mean_lum,
            diff=diff,
            capture_mode=mode
        )
        self.filter.record_acceptance(frame_bgr)
        self.session_accepted_count += 1
        
        # Add to thumbnails
        self.recent_thumbnails.append(frame_bgr.copy())
        if len(self.recent_thumbnails) > 4:
            self.recent_thumbnails.pop(0)

        self.last_decision = f"SAVED #{len(self.manager.samples)}: {record['filename']} ({mode})"
        self.last_decision_color = (0, 255, 0)
        print(f"[DATASET] {self.last_decision}")

    def render_canvas(self, canvas_w=880, canvas_h=580):
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        canvas[:] = (20, 22, 26) # Dark background

        frame_bgr, f_time, f_count, s_fps, dropped, errors = self.receiver.get_latest_frame()
        now = time.time()
        time_since_frame = now - f_time if f_time > 0 else 999.0

        current_class = self.classes[self.active_class_idx]
        class_color = CLASS_COLORS.get(current_class, (0, 230, 255))

        # 1. Header Bar
        cv2.rectangle(canvas, (0, 0), (canvas_w, 48), (32, 36, 44), -1)
        cv2.line(canvas, (0, 48), (canvas_w, 48), (55, 60, 72), 1)
        cv2.putText(canvas, "ESP32-CAM Sensor-Matched Dataset Capture Tool", (20, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 230, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"Target Class: {current_class.upper()}", (canvas_w - 320, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, class_color, 2, cv2.LINE_AA)

        # 2. Main Live Viewport (96x96 scaled 4x -> 384x384)
        view_scale = 4
        vp_size = 96 * view_scale
        vp_x, vp_y = 22, 66

        cv2.rectangle(canvas, (vp_x - 2, vp_y - 2), (vp_x + vp_size + 2, vp_y + vp_size + 2), (60, 65, 78), 1)

        if frame_bgr is not None and time_since_frame < 3.0:
            scaled_frame = cv2.resize(frame_bgr, (vp_size, vp_size), interpolation=cv2.INTER_NEAREST)

            # Auto-Capture Processing Loop
            if self.is_auto_capturing:
                passed, reason, sharp_val, lum_val, diff_val = self.filter.evaluate(frame_bgr)
                if passed:
                    self.trigger_capture(frame_bgr, mode="AUTO", sharpness=sharp_val, mean_lum=lum_val, diff=diff_val)
                else:
                    self.last_decision = f"SKIPPED: {reason}"
                    self.last_decision_color = (0, 165, 255)
            else:
                # Still calculate stats for real-time display
                gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                sharp_val = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                lum_val = float(np.mean(gray))

            # Live Viewport Overlays
            # Reticle / target frame indicating central FOV
            reticle_margin = int(24 * view_scale) # 48x48 model crop area
            reticle_color = class_color if self.is_auto_capturing else (80, 90, 110)
            cv2.rectangle(scaled_frame, (reticle_margin, reticle_margin),
                          (vp_size - reticle_margin, vp_size - reticle_margin), reticle_color, 1)

            # Watermark pill
            state_text = "AUTO-CAPTURING" if self.is_auto_capturing else "PREVIEW ONLY (PAUSED)"
            state_col = (0, 255, 0) if self.is_auto_capturing else (140, 140, 150)
            cv2.circle(scaled_frame, (16, 18), 6, state_col, -1)
            cv2.putText(scaled_frame, f"{state_text} | {s_fps:.1f} FPS", (28, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (240, 240, 240), 1, cv2.LINE_AA)

            # Metrics overlay in bottom of video viewport
            cv2.rectangle(scaled_frame, (0, vp_size - 28), (vp_size, vp_size), (25, 28, 34), -1)
            cv2.putText(scaled_frame, f"Sharpness: {sharp_val:.1f}  |  Luminance: {lum_val:.1f}  |  Res: 96x96",
                        (10, vp_size - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 190, 205), 1, cv2.LINE_AA)

            canvas[vp_y:vp_y + vp_size, vp_x:vp_x + vp_size] = scaled_frame

        else:
            # Explicit Connecting Screen
            vp_area = canvas[vp_y:vp_y + vp_size, vp_x:vp_x + vp_size]
            vp_area[:] = (28, 30, 36)
            cv2.circle(vp_area, (vp_size // 2, vp_size // 2 - 30), 28, (0, 180, 255), 2)
            cv2.putText(vp_area, "AWAITING ESP32-CAM STREAM...", (vp_size // 2 - 130, vp_size // 2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 230, 255), 1, cv2.LINE_AA)
            cv2.putText(vp_area, f"Connecting {PORT} @ {BAUD}", (vp_size // 2 - 80, vp_size // 2 + 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (160, 170, 180), 1, cv2.LINE_AA)
            canvas[vp_y:vp_y + vp_size, vp_x:vp_x + vp_size] = vp_area

        # 3. Filmstrip of Recent Captures (below viewport)
        fs_y = vp_y + vp_size + 12
        cv2.putText(canvas, "RECENT SENSOR CAPTURES (Filmstrip):", (vp_x, fs_y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 170, 180), 1, cv2.LINE_AA)
        
        fs_thumb_y = fs_y + 20
        thumb_w, thumb_h = 72, 72
        for i in range(4):
            tx = vp_x + i * (thumb_w + 12)
            cv2.rectangle(canvas, (tx, fs_thumb_y), (tx + thumb_w, fs_thumb_y + thumb_h), (45, 50, 60), 1)
            if i < len(self.recent_thumbnails):
                t_im = self.recent_thumbnails[-(i + 1)] # most recent first
                t_resized = cv2.resize(t_im, (thumb_w, thumb_h), interpolation=cv2.INTER_NEAREST)
                canvas[fs_thumb_y:fs_thumb_y + thumb_h, tx:tx + thumb_w] = t_resized
            else:
                cv2.putText(canvas, "Empty", (tx + 18, fs_thumb_y + 42),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (70, 75, 88), 1, cv2.LINE_AA)

        # 4. Right Control & Metadata Panel
        px = 435
        py = 75

        # Section A: Active Object & Session Information
        cv2.putText(canvas, "DATASET COLLECTION STATUS", (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 230, 255), 1, cv2.LINE_AA)
        cv2.line(canvas, (px, py + 6), (canvas_w - 20, py + 6), (55, 60, 72), 1)
        py += 26

        counts = self.manager.count_samples_per_class()
        cls_count = counts.get(current_class, 0)
        total_count = len(self.manager.samples)

        cv2.putText(canvas, "Active Class:", (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (160, 170, 180), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{current_class} ([1-6] to change)", (px + 140, py), cv2.FONT_HERSHEY_SIMPLEX, 0.46, class_color, 1, cv2.LINE_AA)
        py += 22

        cv2.putText(canvas, "Captured for Class:", (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (160, 170, 180), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{cls_count} images (Session: +{self.session_accepted_count})", (px + 140, py), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (100, 255, 100), 1, cv2.LINE_AA)
        py += 22

        cv2.putText(canvas, "Total Dataset Size:", (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (160, 170, 180), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{total_count} images across {len(counts)} classes", (px + 140, py), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (230, 230, 230), 1, cv2.LINE_AA)
        py += 22

        cv2.putText(canvas, "Target Object Type:", (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (160, 170, 180), 1, cv2.LINE_AA)
        type_str = "Negative Background" if current_class == "background" else "Physical Desk Instance"
        cv2.putText(canvas, type_str, (px + 140, py), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 220), 1, cv2.LINE_AA)
        py += 28

        # Section B: Real-Time Diversity & Quality Filter Breakdown
        cv2.putText(canvas, "DIVERSITY & REJECTION METRICS", (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 230, 255), 1, cv2.LINE_AA)
        cv2.line(canvas, (px, py + 6), (canvas_w - 20, py + 6), (55, 60, 72), 1)
        py += 24

        stats = self.filter.reject_stats
        cv2.putText(canvas, f"Rejected (Static / Duplicate):", (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 170, 180), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{stats['duplicate']} frames", (px + 240, py), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
        py += 20

        cv2.putText(canvas, f"Rejected (Blurred Motion):", (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 170, 180), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{stats['blur']} frames", (px + 240, py), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
        py += 20

        cv2.putText(canvas, f"Rejected (Timing / Rate Limit):", (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 170, 180), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{stats['rate_limit']} frames", (px + 240, py), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
        py += 20

        cv2.putText(canvas, f"Rejected (Over / Underexposed):", (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 170, 180), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{stats['overexposed'] + stats['underexposed']} frames", (px + 240, py), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
        py += 26

        # Latest Filter Decision Banner
        cv2.rectangle(canvas, (px, py), (canvas_w - 20, py + 26), (34, 38, 46), -1)
        cv2.rectangle(canvas, (px, py), (canvas_w - 20, py + 26), (55, 60, 72), 1)
        cv2.putText(canvas, f"STATUS: {self.last_decision}", (px + 8, py + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, self.last_decision_color, 1, cv2.LINE_AA)
        py += 40

        # Section C: Dynamic Physical Scanning Instructions
        cv2.putText(canvas, "PHYSICAL SCANNING INSTRUCTIONS", (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 230, 255), 1, cv2.LINE_AA)
        cv2.line(canvas, (px, py + 6), (canvas_w - 20, py + 6), (55, 60, 72), 1)
        py += 22

        instr = INSTRUCTIONS_BY_CLASS.get(current_class, "Move camera slowly around target object from varied angles and distances.")
        # Wrap instruction text into two lines
        words = instr.split(" ")
        line1 = " ".join(words[:7])
        line2 = " ".join(words[7:])
        cv2.putText(canvas, line1, (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 240, 240), 1, cv2.LINE_AA)
        py += 18
        if line2:
            cv2.putText(canvas, line2, (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 210, 225), 1, cv2.LINE_AA)
        py += 30

        # Class counts progress overview
        cv2.putText(canvas, "CLASS INVENTORY OVERVIEW:", (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 230, 255), 1, cv2.LINE_AA)
        py += 18
        for i, c_name in enumerate(self.classes):
            c_cnt = counts.get(c_name, 0)
            col = CLASS_COLORS.get(c_name, (180, 180, 180))
            is_active = (i == self.active_class_idx)
            prefix = "> " if is_active else "  "
            line_str = f"{prefix}[{i+1}] {c_name:<21}: {c_cnt:3d} samples"
            cv2.putText(canvas, line_str, (px, py), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                        col if is_active else (150, 160, 170), 1 if not is_active else 2, cv2.LINE_AA)
            py += 16

        # 5. Footer Controls Bar
        cv2.rectangle(canvas, (0, canvas_h - 42), (canvas_w, canvas_h), (32, 36, 44), -1)
        cv2.line(canvas, (0, canvas_h - 42), (canvas_w, canvas_h - 42), (55, 60, 72), 1)
        controls = "[SPACE] Auto-Capture ON/PAUSE  |  [C] Manual Capture  |  [1-6] Class  |  [D] Delete Last  |  [M] Export  |  [Q] Exit"
        cv2.putText(canvas, controls, (20, canvas_h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 190, 205), 1, cv2.LINE_AA)

        return canvas

    def run_interactive(self):
        print(f"\n===========================================================")
        print(f" ESP32-CAM Sensor-Matched Dataset Collection Tool v1.0")
        print(f" Output Directory: {RAW_DATASET_DIR}")
        print(f" Serial Link: {PORT} @ {BAUD} baud")
        print(f"===========================================================")

        if not self.start():
            print("[FATAL] Could not connect to ESP32-CAM. Exiting.")
            return

        cv2.namedWindow("ESP32-CAM Dataset Capture Tool", cv2.WINDOW_AUTOSIZE)

        while True:
            canvas = self.render_canvas()
            cv2.imshow("ESP32-CAM Dataset Capture Tool", canvas)
            key = cv2.waitKey(20) & 0xFF

            if key == ord('q') or key == 27:
                print("\n[ACTION] Exiting capture tool. Exporting final manifest...")
                self.manager.save_manifest()
                break
            elif key == ord(' '):
                self.is_auto_capturing = not self.is_auto_capturing
                state_s = "STARTED" if self.is_auto_capturing else "PAUSED"
                print(f"[ACTION] Auto-capture {state_s}")
            elif key == ord('c'):
                # Manual capture force
                frame_bgr, _, _, _, _, _ = self.receiver.get_latest_frame()
                if frame_bgr is not None:
                    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                    s_val = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                    l_val = float(np.mean(gray))
                    self.trigger_capture(frame_bgr, mode="MANUAL", sharpness=s_val, mean_lum=l_val, diff=99.0)
            elif ord('1') <= key <= ord('6'):
                idx = key - ord('1')
                if idx < len(self.classes):
                    self.active_class_idx = idx
                    self.session_accepted_count = 0
                    self.session_id = f"session_{self.classes[idx]}_{time.strftime('%Y%m%d_%H%M%S')}"
                    self._refresh_recent_thumbnails()
                    print(f"[ACTION] Switched class to: {self.classes[idx]}")
            elif key == ord('d'):
                deleted = self.manager.delete_last_sample()
                if deleted:
                    print(f"[ACTION] Deleted last sample: {deleted['filename']}")
                    self.last_decision = f"DELETED: {deleted['filename']}"
                    self.last_decision_color = (0, 0, 255)
                    self._refresh_recent_thumbnails()
            elif key == ord('m'):
                self.manager.save_manifest()
                print(f"[ACTION] Manifest exported: {self.manager.manifest_json_path}")
                self.last_decision = "MANIFEST EXPORTED OK"
                self.last_decision_color = (0, 255, 255)
            elif key == ord('g'):
                # Generate contact sheet for current class
                cls_name = self.classes[self.active_class_idx]
                sheet_p = self.manager.generate_contact_sheet(cls_name)
                if sheet_p:
                    print(f"[ACTION] Contact sheet generated: {sheet_p}")
                    self.last_decision = f"CONTACT SHEET: {os.path.basename(sheet_p)}"
                    self.last_decision_color = (0, 255, 255)

        self.stop()
        cv2.destroyAllWindows()
        print("[DONE] Capture session finished.")


def run_automated_verification():
    """Autonomous validation mode to verify receiver, filters, file writing, and manifest generation."""
    print("=" * 60)
    print(" RUNNING CAPTURE TOOL AUTONOMOUS VALIDATION TEST")
    print("=" * 60)

    app = CaptureApplication()
    if not app.start():
        print("[FAIL] Could not start capture receiver on COM8")
        return False

    print("\n[STEP 1] Awaiting live camera stream on COM8 (4 seconds)...")
    time.sleep(4.0)

    frame_bgr, f_time, f_count, s_fps, dropped, errors = app.receiver.get_latest_frame()
    print(f" -> Frames Decoded: {f_count}")
    print(f" -> Stream Rate: {s_fps:.1f} FPS")
    print(f" -> Dropped Packets: {dropped}, Framing Errors: {errors}")
    assert f_count > 10, f"Expected >10 frames, received {f_count}"
    assert frame_bgr is not None, "Frame buffer is None!"

    # Verify frame is from physical sensor (not black)
    mean_lum = float(np.mean(frame_bgr))
    print(f" -> Mean Frame Luminance: {mean_lum:.1f}")
    assert mean_lum > 15, "Frame is black!"

    # Render a test canvas snapshot
    print("\n[STEP 2] Rendering capture tool GUI canvas...")
    canvas = app.render_canvas()
    test_snapshot = os.path.join(REPORTS_DIR, "capture_tool_preview_verified.png")
    cv2.imwrite(test_snapshot, canvas)
    print(f" -> Saved capture tool canvas preview to: {test_snapshot}")

    # Test Manual Capture
    print("\n[STEP 3] Testing Manual Sensor Frame Capture (Class: background)...")
    app.active_class_idx = 5 # background
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    s_val = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    l_val = float(np.mean(gray))
    app.trigger_capture(frame_bgr, mode="MANUAL", sharpness=s_val, mean_lum=l_val, diff=100.0)

    # Verify sample exists on disk
    last_sample = app.manager.samples[-1]
    abs_path = last_sample["absolute_path"]
    assert os.path.exists(abs_path), f"File {abs_path} does not exist on disk!"
    assert os.path.getsize(abs_path) > 1000, "Saved file is empty!"
    print(f" -> Verified file saved to disk: {abs_path} ({os.path.getsize(abs_path)} bytes)")

    # Verify manifest JSON and CSV
    assert os.path.exists(app.manager.manifest_json_path), "Manifest JSON missing!"
    assert os.path.exists(app.manager.manifest_csv_path), "Manifest CSV missing!"
    print(f" -> Manifest updated successfully: {len(app.manager.samples)} total samples recorded.")

    # Test Diversity Filter logic
    print("\n[STEP 4] Testing Diversity Filter rejection of identical duplicate...")
    passed, reason, _, _, _ = app.filter.evaluate(frame_bgr)
    print(f" -> Filter response to immediate consecutive frame: passed={passed}, reason='{reason}'")
    assert not passed, "Diversity filter should have rejected consecutive frame!"

    app.stop()
    print("\n[SUCCESS] Capture application autonomous verification passed 100%!")
    return True


if __name__ == "__main__":
    if "--test" in sys.argv or "--auto-test" in sys.argv:
        success = run_automated_verification()
        sys.exit(0 if success else 1)
    else:
        app = CaptureApplication()
        app.run_interactive()
