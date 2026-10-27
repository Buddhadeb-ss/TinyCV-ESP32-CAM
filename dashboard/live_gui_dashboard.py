"""
ESP32-CAM Edge-AI Live Vision Dashboard & Neural Inspector v2.2
Connects to ESP32-CAM via Serial COM8 @ 460800 baud.
Receives framed binary JPEG video stream synchronized with on-device INT8 neural telemetry.
Displays real-time camera feed with verified bounding box overlays, latency meters, and system health.
"""

import cv2
import numpy as np
import serial
import json
import time
import threading
import struct
import os

PORT = "COM8"
BAUD = 460800
MAGIC_HEADER = b"\xAA\x55\xAA\x55"
MAGIC_FOOTER = b"\x55\xAA"

CLASS_COLORS = {
    "background": (120, 120, 120),
    "cream_container": (0, 165, 255),    # Vibrant Orange
    "mouse": (255, 0, 255),              # Magenta / Pink
    "earbuds_case": (255, 255, 0),       # Cyan / Aqua
    "glue_stick": (0, 255, 255),         # Bright Yellow
    "microcontroller_board": (50, 50, 255) # Bright Red
}

CLASS_DISPLAY_NAMES = {
    "background": "Background (None)",
    "cream_container": "Cream Container",
    "mouse": "Computer Mouse",
    "earbuds_case": "Earbuds Case",
    "glue_stick": "Glue Stick",
    "microcontroller_board": "MCU / ESP32 Board"
}


class SerialVideoReceiver:
    """Robust background reader for framed binary video and structured telemetry."""
    def __init__(self, port=PORT, baud=BAUD):
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
        
        self.latest_telemetry = {
            "f": 0,
            "m": 0,
            "ai": 0,
            "cid": 0,
            "cls": "background",
            "conf": 0,
            "det": 0,
            "bx": 24, "by": 24, "bw": 48, "bh": 48,
            "prop": "none",
            "mot": 0.0,
            "lat": 0,
            "fps": 0.0,
            "hp": 0,
            "ps": 0,
            "time": 0.0
        }
        self.active_detection = None # (cls, conf, [bx, by, bw, bh], timestamp)
        self.stream_fps = 0.0
        self.fps_counter = 0
        self.fps_last_time = time.time()

    def connect(self):
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=0.1)
            # Pulse RTS for clean reset
            self.ser.dtr = False
            self.ser.rts = True
            time.sleep(0.1)
            self.ser.rts = False
            time.sleep(0.4)
            self.ser.reset_input_buffer()
            return True
        except Exception as e:
            print(f"[ERROR] Failed to open {self.port}: {e}")
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
                print(f"[HOST CMD SENT] {cmd_str}")
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

                # Process buffer
                while self.running and len(buf) > 0:
                    idx = buf.find(MAGIC_HEADER)
                    if idx == -1:
                        # No magic header currently in buffer.
                        # Parse any complete newline-terminated JSON/log lines
                        nl = buf.rfind(b"\n")
                        if nl != -1:
                            text_chunk = buf[:nl].decode('utf-8', errors='replace')
                            for line in text_chunk.splitlines():
                                self._handle_text_line(line)
                            del buf[:nl + 1]
                        break

                    # Found MAGIC_HEADER at idx.
                    # Any lines before idx are telemetry/text
                    if idx > 0:
                        text_chunk = buf[:idx].decode('utf-8', errors='replace')
                        for line in text_chunk.splitlines():
                            self._handle_text_line(line)
                        del buf[:idx]

                    # Now MAGIC_HEADER is at buf[0]
                    if len(buf) < 8:
                        break  # Need header + length

                    payload_len = struct.unpack("<I", buf[4:8])[0]
                    if payload_len < 50 or payload_len > 45000:
                        # Corrupted length header, advance 1 byte to resync
                        del buf[:1]
                        with self.lock:
                            self.serial_errors += 1
                        continue

                    total_len = 8 + payload_len + 2
                    if len(buf) < total_len:
                        break  # Wait for remaining packet payload

                    # Verify footer
                    footer = buf[8 + payload_len : total_len]
                    if footer == MAGIC_FOOTER:
                        jpg_bytes = bytes(buf[8 : 8 + payload_len])
                        del buf[:total_len]
                        
                        # Decode JPEG
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
                        # Footer mismatch
                        with self.lock:
                            self.serial_errors += 1
                        del buf[:4]

                # Update stream FPS calculation
                now = time.time()
                if now - self.fps_last_time >= 1.0:
                    with self.lock:
                        self.stream_fps = (self.fps_counter * 1.0) / (now - self.fps_last_time)
                        self.fps_counter = 0
                        self.fps_last_time = now

            except Exception as e:
                with self.lock:
                    self.serial_errors += 1
                time.sleep(0.02)

    def _handle_text_line(self, line):
        line = line.strip()
        if not line:
            return
        if line.startswith("{") and "cid" in line:
            try:
                data = json.loads(line)
                now = time.time()
                with self.lock:
                    self.latest_telemetry["f"] = data.get("f", 0)
                    self.latest_telemetry["m"] = data.get("m", 0)
                    self.latest_telemetry["ai"] = data.get("ai", 0)
                    self.latest_telemetry["cid"] = data.get("cid", 0)
                    self.latest_telemetry["cls"] = data.get("cls", "background")
                    self.latest_telemetry["conf"] = data.get("conf", 0)
                    self.latest_telemetry["det"] = data.get("det", 0)
                    self.latest_telemetry["bx"] = data.get("bx", 24)
                    self.latest_telemetry["by"] = data.get("by", 24)
                    self.latest_telemetry["bw"] = data.get("bw", 48)
                    self.latest_telemetry["bh"] = data.get("bh", 48)
                    self.latest_telemetry["prop"] = data.get("prop", "none")
                    self.latest_telemetry["mot"] = data.get("mot", 0.0)
                    self.latest_telemetry["lat"] = data.get("lat", 0)
                    self.latest_telemetry["fps"] = data.get("fps", 0.0)
                    self.latest_telemetry["hp"] = data.get("hp", 0)
                    self.latest_telemetry["ps"] = data.get("ps", 0)
                    self.latest_telemetry["time"] = now

                    # Update active verified detection ONLY when det == 1 and cid > 0
                    if data.get("det", 0) == 1 and data.get("cid", 0) > 0:
                        self.active_detection = {
                            "cls": data.get("cls"),
                            "conf": data.get("conf"),
                            "box": [data.get("bx", 24), data.get("by", 24), data.get("bw", 48), data.get("bh", 48)],
                            "timestamp": now
                        }
                    else:
                        # Clear active detection immediately when no verified object is present
                        self.active_detection = None
            except:
                pass
        elif line.startswith("["):
            print(f" [ESP32] {line}")

    def get_state(self):
        with self.lock:
            frame = None if self.latest_frame is None else self.latest_frame.copy()
            tel = dict(self.latest_telemetry)
            act_det = None if self.active_detection is None else dict(self.active_detection)
            f_time = self.frame_timestamp
            f_count = self.frames_received
            s_fps = self.stream_fps
            dropped = self.dropped_packets
            errors = self.serial_errors
        return frame, tel, act_det, f_time, f_count, s_fps, dropped, errors


def render_dashboard_frame(receiver, flash_on, thresh_val, canvas_w=840, canvas_h=520):
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[:] = (20, 22, 26) # Sleek dark background

    # Header bar
    cv2.rectangle(canvas, (0, 0), (canvas_w, 48), (32, 36, 44), -1)
    cv2.line(canvas, (0, 48), (canvas_w, 48), (55, 60, 72), 1)
    cv2.putText(canvas, "ESP32-CAM TinyML Custom Object Recognition", (20, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 230, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "OV3660 240x240 | 240MHz Xtensa | TFLM INT8", (canvas_w - 340, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (160, 175, 190), 1, cv2.LINE_AA)

    frame, tel, act_det, f_time, f_count, s_fps, dropped, errors = receiver.get_state()
    now = time.time()
    time_since_frame = now - f_time if f_time > 0 else 999.0

    # Video Viewport Setup: 384x384
    vp_size = 384
    vp_x, vp_y = 22, 68

    cv2.rectangle(canvas, (vp_x - 2, vp_y - 2), (vp_x + vp_size + 2, vp_y + vp_size + 2), (55, 60, 72), 1)

    if frame is not None and time_since_frame < 3.0:
        fh, fw = frame.shape[:2]
        view_scale = float(vp_size) / float(fw)
        scaled_frame = cv2.resize(frame, (vp_size, vp_size), interpolation=cv2.INTER_LINEAR)

        # Draw verified neural detection overlay ONLY when active and fresh (< 0.5s)
        draw_detection = False
        if act_det and (now - act_det["timestamp"] < 0.5):
            draw_detection = True
            det_cls = act_det["cls"]
            det_conf = act_det["conf"]
            bx, by, bw, bh = act_det["box"]
            color = CLASS_COLORS.get(det_cls, (0, 255, 0))

            dx = int(bx * view_scale)
            dy = int(by * view_scale)
            dw = int(bw * view_scale)
            dh = int(bh * view_scale)

            # Solid detection box
            cv2.rectangle(scaled_frame, (dx, dy), (dx + dw, dy + dh), color, 2)
            
            # Label banner
            lbl = f"{CLASS_DISPLAY_NAMES.get(det_cls, det_cls.upper())} {det_conf}%"
            (lbl_w, lbl_h), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            label_y = max(lbl_h + 8, dy - 6)
            cv2.rectangle(scaled_frame, (dx, label_y - lbl_h - 4), (dx + lbl_w + 8, label_y + 2), color, -1)
            cv2.putText(scaled_frame, lbl, (dx + 4, label_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

        # Distinguish classical CV candidate proposal (thin outline) when neural net has not confirmed an object
        if not draw_detection and tel.get("ai", 0) == 0 and tel.get("prop", "none") != "none" and tel.get("mot", 0) > 0.8:
            bx = int(tel.get("bx", 24) * view_scale)
            by = int(tel.get("by", 24) * view_scale)
            bw = int(tel.get("bw", 48) * view_scale)
            bh = int(tel.get("bh", 48) * view_scale)
            cv2.rectangle(scaled_frame, (bx, by), (bx + bw, by + bh), (160, 140, 80), 1)
            cv2.putText(scaled_frame, "[CV CANDIDATE: MOTION]", (bx + 4, max(16, by - 4)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (160, 140, 80), 1, cv2.LINE_AA)

        # Watermark
        cv2.circle(scaled_frame, (14, 16), 5, (0, 255, 0), -1)
        cv2.putText(scaled_frame, f"LIVE OV3660 | #{tel.get('f', f_count)}", (26, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 240, 240), 1, cv2.LINE_AA)

        canvas[vp_y:vp_y + vp_size, vp_x:vp_x + vp_size] = scaled_frame

    elif frame is not None and time_since_frame >= 3.0:
        # Stream stalled overlay
        dimmed = cv2.resize(frame, (vp_size, vp_size), interpolation=cv2.INTER_NEAREST)
        dimmed = (dimmed * 0.25).astype(np.uint8)
        cv2.rectangle(dimmed, (20, vp_size // 2 - 30), (vp_size - 20, vp_size // 2 + 30), (0, 0, 180), -1)
        cv2.putText(dimmed, "CAMERA STREAM STALLED", (40, vp_size // 2 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(dimmed, "Reconnecting serial link on COM8...", (40, vp_size // 2 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1, cv2.LINE_AA)
        canvas[vp_y:vp_y + vp_size, vp_x:vp_x + vp_size] = dimmed

    else:
        # Connecting screen (never black placeholder)
        vp_area = canvas[vp_y:vp_y + vp_size, vp_x:vp_x + vp_size]
        vp_area[:] = (28, 30, 36)
        cv2.circle(vp_area, (vp_size // 2, vp_size // 2 - 35), 26, (0, 180, 255), 2)
        cv2.putText(vp_area, "CONNECTING TO CAMERA...", (vp_size // 2 - 120, vp_size // 2 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.54, (0, 230, 255), 1, cv2.LINE_AA)
        cv2.putText(vp_area, "Awaiting OV3660 stream on COM8 @ 460800", (vp_size // 2 - 145, vp_size // 2 + 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 160, 175), 1, cv2.LINE_AA)
        canvas[vp_y:vp_y + vp_size, vp_x:vp_x + vp_size] = vp_area

    # Telemetry Panel Setup (Clean columns with zero overlap)
    tx_lbl = 430
    tx_val = 600
    ty = 75

    # Section 1: Edge AI Inference Metrics
    cv2.putText(canvas, "EDGE-AI INFERENCE METRICS", (tx_lbl, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 230, 255), 1, cv2.LINE_AA)
    cv2.line(canvas, (tx_lbl, ty + 6), (canvas_w - 20, ty + 6), (55, 60, 72), 1)
    ty += 28

    m_num = tel.get("m", 0)
    mode_str = ["HYBRID (Motion ROI + AI)", "CONTINUOUS_AI (Every Frame)", "PURE_CV (High-Speed CV)"][m_num] if m_num < 3 else "UNKNOWN"
    cv2.putText(canvas, "Operating Mode:", (tx_lbl, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (160, 170, 180), 1, cv2.LINE_AA)
    cv2.putText(canvas, mode_str, (tx_val - 20, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 220), 1, cv2.LINE_AA)
    ty += 24

    dev_fps = tel.get("fps", 0.0)
    lat = tel.get("lat", 0)
    cv2.putText(canvas, "Host Stream Rate:", (tx_lbl, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (160, 170, 180), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{s_fps:.1f} FPS", (tx_val, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (100, 255, 100), 1, cv2.LINE_AA)
    ty += 24

    cv2.putText(canvas, "MCU Throughput:", (tx_lbl, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (160, 170, 180), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{dev_fps:.1f} FPS", (tx_val, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (100, 255, 100), 1, cv2.LINE_AA)
    ty += 24

    cv2.putText(canvas, "Neural Latency:", (tx_lbl, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (160, 170, 180), 1, cv2.LINE_AA)
    lat_col = (100, 255, 100) if lat > 0 else (120, 120, 120)
    cv2.putText(canvas, f"{lat} ms (DeskObjectNet)", (tx_val, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.44, lat_col, 1, cv2.LINE_AA)
    ty += 28

    # Section 2: Target Recognition Status
    cv2.putText(canvas, "TARGET RECOGNITION STATUS", (tx_lbl, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 230, 255), 1, cv2.LINE_AA)
    cv2.line(canvas, (tx_lbl, ty + 6), (canvas_w - 20, ty + 6), (55, 60, 72), 1)
    ty += 28

    det_flag = tel.get("det", 0) == 1
    cls_name = tel.get("cls", "background")
    conf = tel.get("conf", 0)
    color = CLASS_COLORS.get(cls_name, (120, 120, 120))

    if det_flag and (now - tel.get("time", 0) < 0.6):
        status_text = f"CONFIRMED: {CLASS_DISPLAY_NAMES.get(cls_name, cls_name).upper()}"
        status_col = color
    elif tel.get("ai", 0) == 1:
        status_text = "EVALUATED: BACKGROUND / LOW CONFIDENCE"
        status_col = (140, 140, 140)
    else:
        status_text = "SCANNING (Awaiting Motion Trigger)"
        status_col = (130, 130, 130)

    cv2.putText(canvas, status_text, (tx_lbl, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.44, status_col, 1, cv2.LINE_AA)
    ty += 22

    cv2.putText(canvas, "Classification:", (tx_lbl, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (160, 170, 180), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{CLASS_DISPLAY_NAMES.get(cls_name, cls_name)}", (tx_val, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 1, cv2.LINE_AA)
    ty += 22

    cv2.putText(canvas, "Confidence Score:", (tx_lbl, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (160, 170, 180), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{conf}% (Thresh: {thresh_val}%)", (tx_val, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (230, 230, 230), 1, cv2.LINE_AA)
    ty += 12

    # Confidence Progress Bar
    bar_w = 360
    bar_fill = int((conf / 100.0) * bar_w)
    cv2.rectangle(canvas, (tx_lbl, ty + 6), (tx_lbl + bar_w, ty + 20), (42, 45, 52), -1)
    if bar_fill > 0:
        cv2.rectangle(canvas, (tx_lbl, ty + 6), (tx_lbl + bar_fill, ty + 20), color, -1)
    cv2.rectangle(canvas, (tx_lbl, ty + 6), (tx_lbl + bar_w, ty + 20), (65, 70, 82), 1)
    th_x = tx_lbl + int((thresh_val / 100.0) * bar_w)
    cv2.line(canvas, (th_x, ty + 3), (th_x, ty + 23), (0, 255, 255), 2)
    ty += 34

    # Section 3: Hardware Diagnostics & Transport Health
    cv2.putText(canvas, "SYSTEM & TRANSPORT HEALTH", (tx_lbl, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 230, 255), 1, cv2.LINE_AA)
    cv2.line(canvas, (tx_lbl, ty + 6), (canvas_w - 20, ty + 6), (55, 60, 72), 1)
    ty += 26

    hp = tel.get("hp", 0)
    ps = tel.get("ps", 0)
    cv2.putText(canvas, "Internal Heap Free:", (tx_lbl, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 170, 180), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{hp:,} B", (tx_val, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 220, 255), 1, cv2.LINE_AA)
    ty += 20

    cv2.putText(canvas, "External PSRAM Free:", (tx_lbl, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 170, 180), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{ps:,} B", (tx_val, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 220, 255), 1, cv2.LINE_AA)
    ty += 20

    cv2.putText(canvas, "Frames Received:", (tx_lbl, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 170, 180), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{f_count} frames", (tx_val, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 220, 255), 1, cv2.LINE_AA)
    ty += 20

    cv2.putText(canvas, "Dropped / Errors:", (tx_lbl, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 170, 180), 1, cv2.LINE_AA)
    err_col = (100, 255, 100) if (dropped == 0 and errors == 0) else (0, 165, 255)
    cv2.putText(canvas, f"{dropped} dropped | {errors} errors", (tx_val, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.42, err_col, 1, cv2.LINE_AA)
    ty += 20

    flash_str = "ON" if flash_on else "OFF"
    cv2.putText(canvas, "Flashlight LED:", (tx_lbl, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 170, 180), 1, cv2.LINE_AA)
    cv2.putText(canvas, flash_str, (tx_val, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255) if flash_on else (160, 160, 160), 1, cv2.LINE_AA)

    # Footer Bar with Controls
    cv2.rectangle(canvas, (0, canvas_h - 42), (canvas_w, canvas_h), (32, 36, 44), -1)
    cv2.line(canvas, (0, canvas_h - 42), (canvas_w, canvas_h - 42), (55, 60, 72), 1)
    controls = "[M] Cycle Mode  |  [F] Toggle Flash  |  [+/-] Threshold  |  [S] Save Snapshot  |  [Q] Exit"
    cv2.putText(canvas, controls, (20, canvas_h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (180, 190, 205), 1, cv2.LINE_AA)

    return canvas


def run_gui():
    print(f"===========================================================")
    print(f" ESP32-CAM TinyML Edge Vision Dashboard v2.2")
    print(f" Hardware: AI-Thinker ESP32-CAM | Model: DeskObjectNet INT8")
    print(f" Connecting to {PORT} @ {BAUD} baud...")
    print(f"===========================================================")

    receiver = SerialVideoReceiver(PORT, BAUD)
    if not receiver.start():
        print(f"[FATAL] Could not connect to ESP32-CAM on {PORT}!")
        return

    # Modes
    modes = ["MODE=HYBRID", "MODE=AI", "MODE=CV"]
    mode_names = ["HYBRID (Motion ROI + AI)", "CONTINUOUS_AI (Every Frame)", "PURE_CV (High-Speed CV)"]
    current_mode_idx = 0
    
    flash_on = False
    thresh_val = 45
    
    print("Dashboard initialized successfully.")
    print("Keyboard Controls:")
    print("  [M] Cycle Operating Mode (Hybrid / Continuous AI / Pure CV)")
    print("  [F] Toggle Onboard Flashlight LED")
    print("  [+] Increase Confidence Threshold")
    print("  [-] Decrease Confidence Threshold")
    print("  [S] Save Full Annotated Dashboard Snapshot")
    print("  [Q] Exit Application")

    os.makedirs(r"C:\All_Projects\esp32_cam_ai_system\screenshots", exist_ok=True)
    cv2.namedWindow("ESP32-CAM Edge AI Dashboard", cv2.WINDOW_AUTOSIZE)

    while True:
        canvas = render_dashboard_frame(receiver, flash_on, thresh_val)
        cv2.imshow("ESP32-CAM Edge AI Dashboard", canvas)
        key = cv2.waitKey(20) & 0xFF

        if key == ord('q') or key == 27:
            print("\nExiting dashboard...")
            break
        elif key == ord('m'):
            current_mode_idx = (current_mode_idx + 1) % len(modes)
            cmd = modes[current_mode_idx]
            receiver.send_command(cmd)
            print(f"[ACTION] Switched mode to {mode_names[current_mode_idx]}")
        elif key == ord('f'):
            flash_on = not flash_on
            receiver.send_command("FLASH=1" if flash_on else "FLASH=0")
            print(f"[ACTION] Flashlight {'ON' if flash_on else 'OFF'}")
        elif key == ord('+') or key == ord('='):
            thresh_val = min(90, thresh_val + 5)
            receiver.send_command(f"THRESH={thresh_val}")
            print(f"[ACTION] Set threshold to {thresh_val}%")
        elif key == ord('-') or key == ord('_'):
            thresh_val = max(15, thresh_val - 5)
            receiver.send_command(f"THRESH={thresh_val}")
            print(f"[ACTION] Set threshold to {thresh_val}%")
        elif key == ord('s'):
            snap_time = time.strftime("%Y%m%d_%H%M%S")
            snap_path = os.path.join(r"C:\All_Projects\esp32_cam_ai_system\screenshots", f"dashboard_live_{snap_time}.png")
            cv2.imwrite(snap_path, canvas)
            print(f"[ACTION] Genuine dashboard screenshot saved: {snap_path}")

    receiver.stop()
    cv2.destroyAllWindows()
    print("Dashboard closed cleanly.")


if __name__ == "__main__":
    run_gui()
