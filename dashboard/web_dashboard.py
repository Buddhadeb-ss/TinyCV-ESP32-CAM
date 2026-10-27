"""
TinyCV-ESP32-CAM: Web-Based Live Edge-AI Vision Dashboard (dashboard/web_dashboard.py)
Serves a responsive real-time web interface at http://127.0.0.1:5000.
Renders live camera frames with synchronized on-device DeskObjectNet INT8 detection overlays,
confidence gauges, mode switching, flashlight toggle, and hardware health metrics.
"""

import os
import sys
import time
import json
import threading
import cv2
import numpy as np
from flask import Flask, Response, render_template_string, request, jsonify

# Include dashboard directory
DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(DASHBOARD_DIR)

from live_gui_dashboard import SerialVideoReceiver, CLASS_COLORS, CLASS_DISPLAY_NAMES, PORT, BAUD

app = Flask(__name__)
receiver = SerialVideoReceiver(PORT, BAUD)

# UI State
ui_state = {
    "mode": "HYBRID",
    "thresh": 45,
    "flash": False,
    "last_snapshot_path": ""
}
ui_lock = threading.Lock()

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TinyCV-ESP32-CAM Live Edge-AI Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-base: #0d0f14;
            --bg-card: #151821;
            --bg-elevated: #1e2230;
            --border: #2b3245;
            --accent: #00e5ff;
            --accent-green: #00e676;
            --accent-amber: #ffd600;
            --accent-red: #ff1744;
            --text-main: #eceff4;
            --text-muted: #8892b0;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'JetBrains Mono', monospace;
        }

        body {
            background-color: var(--bg-base);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 24px;
        }

        header {
            width: 100%;
            max-width: 1100px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding-bottom: 20px;
            border-bottom: 1px solid var(--border);
            margin-bottom: 24px;
        }

        .header-title h1 {
            font-size: 24px;
            font-weight: 800;
            letter-spacing: -0.5px;
            color: #ffffff;
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .header-title p {
            font-size: 13px;
            color: var(--text-muted);
            margin-top: 4px;
        }

        .live-badge {
            background: rgba(0, 230, 118, 0.15);
            border: 1px solid var(--accent-green);
            color: var(--accent-green);
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 700;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .pulse-dot {
            width: 8px;
            height: 8px;
            background: var(--accent-green);
            border-radius: 50%;
            box-shadow: 0 0 8px var(--accent-green);
            animation: pulse 1.5s infinite;
        }

        @keyframes pulse {
            0% { transform: scale(0.95); opacity: 0.8; }
            50% { transform: scale(1.3); opacity: 1; }
            100% { transform: scale(0.95); opacity: 0.8; }
        }

        .main-grid {
            width: 100%;
            max-width: 1100px;
            display: grid;
            grid-template-columns: 500px 1fr;
            gap: 24px;
        }

        .video-card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 16px;
            display: flex;
            flex-direction: column;
            align-items: center;
        }

        .video-container {
            width: 468px;
            height: 468px;
            background: #000000;
            border-radius: 8px;
            overflow: hidden;
            display: flex;
            align-items: center;
            justify-content: center;
            border: 2px solid var(--border);
            position: relative;
        }

        .video-container img {
            width: 100%;
            height: 100%;
            object-fit: contain;
            image-rendering: pixelated;
        }

        .video-meta {
            width: 100%;
            display: flex;
            justify-content: space-between;
            margin-top: 12px;
            font-size: 12px;
            color: var(--text-muted);
        }

        .sidebar {
            display: flex;
            flex-direction: column;
            gap: 18px;
        }

        .metric-card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 18px;
        }

        .card-title {
            font-size: 12px;
            font-weight: 700;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 12px;
            display: flex;
            justify-content: space-between;
        }

        .detection-banner {
            background: var(--bg-elevated);
            border-radius: 8px;
            padding: 14px 16px;
            display: flex;
            flex-direction: column;
            gap: 8px;
            border-left: 4px solid var(--accent);
        }

        .detection-name {
            font-size: 20px;
            font-weight: 700;
            color: #ffffff;
        }

        .conf-bar-wrapper {
            width: 100%;
            height: 10px;
            background: rgba(255, 255, 255, 0.08);
            border-radius: 5px;
            overflow: hidden;
            position: relative;
        }

        .conf-bar-fill {
            height: 100%;
            background: linear-gradient(90deg, #00e5ff, #00e676);
            width: 0%;
            transition: width 0.15s ease-out;
        }

        .metrics-row {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 12px;
        }

        .metric-box {
            background: var(--bg-elevated);
            padding: 12px 14px;
            border-radius: 8px;
        }

        .metric-val {
            font-size: 18px;
            font-weight: 700;
            color: #ffffff;
            margin-top: 4px;
        }

        .metric-sub {
            font-size: 11px;
            color: var(--text-muted);
        }

        .controls-group {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 10px;
        }

        .btn {
            background: var(--bg-elevated);
            color: var(--text-main);
            border: 1px solid var(--border);
            padding: 10px 16px;
            border-radius: 6px;
            font-size: 13px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.15s;
        }

        .btn:hover {
            border-color: var(--accent);
            color: #ffffff;
            transform: translateY(-1px);
        }

        .btn.active {
            background: rgba(0, 229, 255, 0.15);
            border-color: var(--accent);
            color: var(--accent);
        }

        .btn-flash {
            background: rgba(255, 214, 0, 0.1);
            border-color: rgba(255, 214, 0, 0.3);
            color: var(--accent-amber);
        }

        .btn-flash.on {
            background: var(--accent-amber);
            color: #000000;
            font-weight: 800;
        }

        .slider-row {
            display: flex;
            align-items: center;
            gap: 16px;
            margin-top: 12px;
        }

        .slider-row input[type="range"] {
            flex: 1;
            accent-color: var(--accent);
        }

        .slider-val {
            font-size: 14px;
            font-weight: 700;
            color: var(--accent);
            min-width: 45px;
        }

        .status-pill {
            font-size: 11px;
            font-weight: 600;
            padding: 2px 8px;
            border-radius: 4px;
        }
        .status-ok { background: rgba(0, 230, 118, 0.2); color: var(--accent-green); }
        .status-warn { background: rgba(255, 214, 0, 0.2); color: var(--accent-amber); }
    </style>
</head>
<body>

    <header>
        <div class="header-title">
            <h1>TinyCV-ESP32-CAM <span>AI Vision Dashboard</span></h1>
            <p>Model: DeskObjectNet INT8 (14.8 KB) &bull; Platform: ESP32-D0WD-V3 @ 240 MHz &bull; Sensor: OV3660</p>
        </div>
        <div class="live-badge">
            <div class="pulse-dot"></div>
            <span>LIVE COM8</span>
        </div>
    </header>

    <div class="main-grid">
        <div class="video-card">
            <div class="video-container">
                <img id="live-stream" src="/video_feed" alt="Live Camera Stream">
            </div>
            <div class="video-meta">
                <span id="stream-fps">Stream Rate: -- FPS</span>
                <span id="mcu-fps">MCU Throughput: -- FPS</span>
                <span id="dropped-stat">0 dropped / 0 err</span>
            </div>
        </div>

        <div class="sidebar">
            <div class="metric-card">
                <div class="card-title">
                    <span>Object Recognition Status</span>
                    <span id="eval-status" class="status-pill status-ok">ACTIVE</span>
                </div>
                <div id="detection-banner" class="detection-banner">
                    <div class="detection-name" id="detected-class">Background (None)</div>
                    <div class="conf-bar-wrapper">
                        <div class="conf-bar-fill" id="conf-fill"></div>
                    </div>
                    <div style="display:flex; justify-content:space-between; font-size:12px; color:var(--text-muted);">
                        <span id="conf-text">Confidence: 0%</span>
                        <span id="thresh-indicator">Threshold: 45%</span>
                    </div>
                </div>
            </div>

            <div class="metric-card">
                <div class="card-title">Edge-AI Inference & Silicon Health</div>
                <div class="metrics-row">
                    <div class="metric-box">
                        <div class="metric-sub">Neural Latency</div>
                        <div class="metric-val" id="lat-val">-- ms</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-sub">Free Heap Memory</div>
                        <div class="metric-val" id="heap-val">-- KB</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-sub">External PSRAM</div>
                        <div class="metric-val" id="psram-val">-- KB</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-sub">Frame Counter</div>
                        <div class="metric-val" id="frame-counter">#0</div>
                    </div>
                </div>
            </div>

            <div class="metric-card">
                <div class="card-title">Hardware & Operating Controls</div>
                <div style="font-size:12px; color:var(--text-muted); margin-bottom:6px;">Inference Operating Mode:</div>
                <div class="controls-group">
                    <button class="btn active" id="btn-mode-hybrid" onclick="setMode('HYBRID')">HYBRID (Motion + AI)</button>
                    <button class="btn" id="btn-mode-ai" onclick="setMode('AI')">CONTINUOUS AI</button>
                    <button class="btn" id="btn-mode-cv" onclick="setMode('CV')">PURE CV (Peak FPS)</button>
                </div>

                <div style="font-size:12px; color:var(--text-muted); margin-top:14px; margin-bottom:6px;">Detection Confidence Threshold:</div>
                <div class="slider-row">
                    <input type="range" id="thresh-slider" min="15" max="85" value="45" oninput="updateThresh(this.value)">
                    <span class="slider-val" id="thresh-label">45%</span>
                </div>

                <div style="font-size:12px; color:var(--text-muted); margin-top:14px; margin-bottom:6px;">Hardware Actuation & Tools:</div>
                <div class="controls-group">
                    <button class="btn btn-flash" id="btn-flash" onclick="toggleFlash()">FLASH LED: OFF</button>
                    <button class="btn" onclick="saveSnapshot()">SAVE SNAPSHOT</button>
                </div>
            </div>
        </div>
    </div>

    <script>
        let currentMode = "HYBRID";
        let flashState = false;

        function setMode(mode) {
            currentMode = mode;
            ['hybrid', 'ai', 'cv'].forEach(m => {
                document.getElementById('btn-mode-' + m).classList.remove('active');
            });
            document.getElementById('btn-mode-' + mode.toLowerCase()).classList.add('active');
            fetch('/api/command', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({cmd: 'MODE=' + mode})
            });
        }

        function toggleFlash() {
            flashState = !flashState;
            const btn = document.getElementById('btn-flash');
            if (flashState) {
                btn.classList.add('on');
                btn.textContent = 'FLASH LED: ON';
            } else {
                btn.classList.remove('on');
                btn.textContent = 'FLASH LED: OFF';
            }
            fetch('/api/command', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({cmd: flashState ? 'FLASH=1' : 'FLASH=0'})
            });
        }

        function updateThresh(val) {
            document.getElementById('thresh-label').textContent = val + '%';
            document.getElementById('thresh-indicator').textContent = 'Threshold: ' + val + '%';
            fetch('/api/command', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({cmd: 'THRESH=' + val})
            });
        }

        function saveSnapshot() {
            fetch('/api/snapshot', {method: 'POST'})
                .then(r => r.json())
                .then(data => {
                    alert('Snapshot saved to:\\n' + data.path);
                });
        }

        // Live telemetry polling
        setInterval(() => {
            fetch('/api/telemetry')
                .then(r => r.json())
                .then(d => {
                    document.getElementById('stream-fps').textContent = 'Stream Rate: ' + d.stream_fps.toFixed(1) + ' FPS';
                    document.getElementById('mcu-fps').textContent = 'MCU Throughput: ' + d.mcu_fps.toFixed(1) + ' FPS';
                    document.getElementById('dropped-stat').textContent = d.dropped + ' dropped / ' + d.errors + ' err';

                    document.getElementById('detected-class').textContent = d.display_class;
                    document.getElementById('conf-text').textContent = 'Confidence: ' + d.conf + '%';
                    document.getElementById('conf-fill').style.width = d.conf + '%';

                    const banner = document.getElementById('detection-banner');
                    if (d.object_detected) {
                        banner.style.borderLeftColor = '#00e676';
                        document.getElementById('eval-status').textContent = 'DETECTED';
                        document.getElementById('eval-status').className = 'status-pill status-ok';
                    } else {
                        banner.style.borderLeftColor = '#555e75';
                        document.getElementById('eval-status').textContent = d.active_ai ? 'BACKGROUND' : 'STANDBY';
                        document.getElementById('eval-status').className = 'status-pill status-warn';
                    }

                    document.getElementById('lat-val').textContent = d.latency + ' ms';
                    document.getElementById('heap-val').textContent = (d.heap / 1024).toFixed(0) + ' KB';
                    document.getElementById('psram-val').textContent = (d.psram / 1024).toFixed(0) + ' KB';
                    document.getElementById('frame-counter').textContent = '#' + d.frame_num;
                })
                .catch(() => {});
        }, 150);
    </script>
</body>
</html>
"""


class RealTimeObjectTracker:
    """
    Predictive Multi-Cue Color-Spatial & Contour Object Tracker:
    - Segments physical target objects in 240x240 resolution using HSV perceptual color bands
      matched to target objects (cream container, earbuds case, glue stick, mouse, dev board).
    - Fuses color segmentation with adaptive foreground saliency and contour analysis.
    - Uses momentum and exponential smoothing (alpha = 0.35) for fluid real-time tracking across the camera FOV.
    - Handles motion persistence and smooth fade-out.
    """
    def __init__(self):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=60, varThreshold=30, detectShadows=False)
        self.smooth_box = None  # [x, y, w, h] in frame coordinates
        self.active_class = "background"
        self.frames_since_detect = 999
        self.alpha = 0.35  # Smoothing responsiveness factor
        self.velocity = [0.0, 0.0, 0.0, 0.0]

    def _extract_color_mask(self, frame_bgr, target_class):
        """Extracts high-confidence color segmentation masks tailored to target objects."""
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        mask = None

        if target_class == "earbuds_case":
            # Mint Cyan / Turquoise: H in [70, 115], S > 40, V > 50
            lower = np.array([70, 40, 50], dtype=np.uint8)
            upper = np.array([115, 255, 255], dtype=np.uint8)
            mask = cv2.inRange(hsv, lower, upper)

        elif target_class == "cream_container":
            # Vivid Orange rim: H in [6, 26], S > 75, V > 70
            lower_orange = np.array([6, 75, 70], dtype=np.uint8)
            upper_orange = np.array([26, 255, 255], dtype=np.uint8)
            m_orange = cv2.inRange(hsv, lower_orange, upper_orange)
            # White lid center: S < 60, V > 165
            lower_white = np.array([0, 0, 165], dtype=np.uint8)
            upper_white = np.array([180, 60, 255], dtype=np.uint8)
            m_white = cv2.inRange(hsv, lower_white, upper_white)
            mask = cv2.bitwise_or(m_orange, m_white)

        elif target_class == "glue_stick":
            # Yellow cap: H in [18, 38], S > 90, V > 90
            lower_y = np.array([18, 90, 90], dtype=np.uint8)
            upper_y = np.array([38, 255, 255], dtype=np.uint8)
            m_y = cv2.inRange(hsv, lower_y, upper_y)
            # Blue tube: H in [98, 132], S > 70, V > 60
            lower_b = np.array([98, 70, 60], dtype=np.uint8)
            upper_b = np.array([132, 255, 255], dtype=np.uint8)
            m_b = cv2.inRange(hsv, lower_b, upper_b)
            mask = cv2.bitwise_or(m_y, m_b)

        elif target_class == "mouse":
            # Low brightness silhouette: V < 85, or high gradient boundary
            lower_m = np.array([0, 0, 0], dtype=np.uint8)
            upper_m = np.array([180, 120, 85], dtype=np.uint8)
            mask = cv2.inRange(hsv, lower_m, upper_m)

        elif target_class == "microcontroller_board":
            # Metallic shield + circuit texture (high gradient with moderate saturation)
            lower_p = np.array([0, 0, 90], dtype=np.uint8)
            upper_p = np.array([180, 120, 240], dtype=np.uint8)
            mask = cv2.inRange(hsv, lower_p, upper_p)

        if mask is not None:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    def update(self, frame_bgr, predicted_class, confidence, is_detected):
        h, w = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 1. Motion foreground mask
        fg_mask = self.bg_subtractor.apply(blurred)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

        # 2. Gradient / contrast saliency
        grad_x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = cv2.magnitude(grad_x, grad_y)
        _, sal_mask = cv2.threshold(grad_mag, 25, 255, cv2.THRESH_BINARY)
        sal_mask = sal_mask.astype(np.uint8)

        # 3. Class-specific predictive color mask
        color_mask = self._extract_color_mask(frame_bgr, predicted_class)

        # Saliency combination
        combined_saliency = cv2.bitwise_or(fg_mask, sal_mask)
        # Suppress border artifacts
        border_pad = 6
        combined_saliency[:border_pad, :] = 0
        combined_saliency[-border_pad:, :] = 0
        combined_saliency[:, :border_pad] = 0
        combined_saliency[:, -border_pad:] = 0

        best_box = None
        best_area = 0

        # Prioritize predictive color mask if target class has distinct signature
        if color_mask is not None:
            color_mask[:border_pad, :] = 0
            color_mask[-border_pad:, :] = 0
            color_mask[:, :border_pad] = 0
            color_mask[:, -border_pad:] = 0
            contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                # Minimum footprint in 240x240 image is ~250 pixels
                if area > 250 and area > best_area:
                    best_area = area
                    x, y, bw, bh = cv2.boundingRect(cnt)
                    # Add generous contextual padding around color signature
                    pad = 12
                    x = max(0, x - pad)
                    y = max(0, y - pad)
                    bw = min(w - x, bw + pad * 2)
                    bh = min(h - y, bh + pad * 2)
                    best_box = [x, y, bw, bh]

        # If color mask didn't isolate an object, use general motion/saliency contours
        if best_box is None:
            contours, _ = cv2.findContours(combined_saliency, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 400 and area > best_area:
                    best_area = area
                    x, y, bw, bh = cv2.boundingRect(cnt)
                    pad = 8
                    x = max(0, x - pad)
                    y = max(0, y - pad)
                    bw = min(w - x, bw + pad * 2)
                    bh = min(h - y, bh + pad * 2)
                    best_box = [x, y, bw, bh]

        # Process neural inference evaluation
        if is_detected and predicted_class != "background":
            self.frames_since_detect = 0
            self.active_class = predicted_class

            if best_box is not None:
                target_box = best_box
            elif self.smooth_box is not None:
                target_box = self.smooth_box
            else:
                # Default centered region in 240x240: 160x160 box centered
                target_box = [40, 40, 160, 160]

            # Predictive smoothing filter with momentum
            if self.smooth_box is None:
                self.smooth_box = [float(v) for v in target_box]
                self.velocity = [0.0, 0.0, 0.0, 0.0]
            else:
                for i in range(4):
                    diff = target_box[i] - self.smooth_box[i]
                    self.velocity[i] = 0.3 * diff + 0.7 * self.velocity[i]
                    self.smooth_box[i] = self.alpha * target_box[i] + (1.0 - self.alpha) * (self.smooth_box[i] + self.velocity[i] * 0.5)

            box_int = [int(round(v)) for v in self.smooth_box]
            return box_int, self.active_class, confidence, True

        else:
            self.frames_since_detect += 1
            # 6-frame persistence (~0.4s) to glide over occasional frame drops
            if self.frames_since_detect < 6 and self.smooth_box is not None:
                # Coast using residual velocity
                for i in range(4):
                    self.smooth_box[i] += self.velocity[i] * 0.3
                    self.velocity[i] *= 0.8
                box_int = [int(round(v)) for v in self.smooth_box]
                return box_int, self.active_class, confidence, True
            else:
                self.smooth_box = None
                return None, "background", 0, False


tracker = RealTimeObjectTracker()


def draw_sleek_bounding_box(img, box, class_name, conf, scale=2):
    """Draws a responsive, high-visibility tracking box with corner brackets and badges."""
    bx = int(box[0] * scale)
    by = int(box[1] * scale)
    bw = int(box[2] * scale)
    bh = int(box[3] * scale)

    color = CLASS_COLORS.get(class_name, (0, 230, 255))

    # Outer border
    cv2.rectangle(img, (bx, by), (bx + bw, by + bh), color, 2)

    # Corner brackets for tech aesthetic
    bracket_len = min(28, min(bw, bh) // 4)
    thick = 4
    # Top-Left
    cv2.line(img, (bx, by), (bx + bracket_len, by), color, thick)
    cv2.line(img, (bx, by), (bx, by + bracket_len), color, thick)
    # Top-Right
    cv2.line(img, (bx + bw, by), (bx + bw - bracket_len, by), color, thick)
    cv2.line(img, (bx + bw, by), (bx + bw, by + bracket_len), color, thick)
    # Bottom-Left
    cv2.line(img, (bx, by + bh), (bx + bracket_len, by + bh), color, thick)
    cv2.line(img, (bx, by + bh), (bx, by + bh - bracket_len), color, thick)
    # Bottom-Right
    cv2.line(img, (bx + bw, by + bh), (bx + bw - bracket_len, by + bh), color, thick)
    cv2.line(img, (bx + bw, by + bh), (bx + bw, by + bh - bracket_len), color, thick)

    # Class Label & Confidence Tag
    label = f"{CLASS_DISPLAY_NAMES.get(class_name, class_name)} [{conf}%]"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
    tag_y = max(th + 6, by - 6)
    cv2.rectangle(img, (bx, tag_y - th - 6), (bx + tw + 10, tag_y + 2), color, -1)
    cv2.putText(img, label, (bx + 5, tag_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (10, 15, 20), 2)


def generate_annotated_mjpeg():
    """Generates continuous MJPEG multipart stream with real-time detection bounding boxes."""
    while True:
        frame, tel, act_det, f_time, f_count, s_fps, dropped, errors = receiver.get_state()
        if frame is not None:
            # Scale dynamically to target 480x480 container
            h, w = frame.shape[:2]
            target_dim = 480
            scale = target_dim / float(w)
            display = cv2.resize(frame, (target_dim, target_dim), interpolation=cv2.INTER_LINEAR)

            cls_name = tel.get("cls", "background")
            conf = tel.get("conf", 0)
            is_detected = tel.get("det", 0) == 1

            # Update real-time predictive color-spatial tracker
            tracked_box, active_cls, active_conf, has_box = tracker.update(frame, cls_name, conf, is_detected)

            if has_box and tracked_box is not None:
                draw_sleek_bounding_box(display, tracked_box, active_cls, active_conf, scale=scale)
            else:
                # Faint Full-FOV scan reticle
                cv2.rectangle(display, (12, 12), (target_dim - 12, target_dim - 12), (40, 50, 70), 1)

            ret, jpeg = cv2.imencode('.jpg', display, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        time.sleep(0.03)



@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/video_feed')
def video_feed():
    return Response(generate_annotated_mjpeg(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/telemetry')
def get_telemetry():
    frame, tel, act_det, f_time, f_count, s_fps, dropped, errors = receiver.get_state()
    cls_name = tel.get("cls", "background")
    conf = tel.get("conf", 0)
    det = tel.get("det", 0) == 1

    display_name = CLASS_DISPLAY_NAMES.get(cls_name, cls_name)
    if not det:
        display_name = "Background (None)"

    return jsonify({
        "frame_num": tel.get("f", f_count),
        "stream_fps": float(s_fps),
        "mcu_fps": float(tel.get("fps", 0.0)),
        "dropped": dropped,
        "errors": errors,
        "raw_class": cls_name,
        "display_class": display_name,
        "conf": conf,
        "object_detected": det,
        "active_ai": tel.get("ai", 0) == 1,
        "latency": tel.get("lat", 0),
        "heap": tel.get("hp", 0),
        "psram": tel.get("ps", 0)
    })


@app.route('/api/command', methods=['POST'])
def send_cmd():
    data = request.get_json() or {}
    cmd = data.get("cmd", "")
    if cmd:
        receiver.send_command(cmd)
        return jsonify({"status": "ok", "cmd": cmd})
    return jsonify({"error": "missing_cmd"}), 400


@app.route('/api/snapshot', methods=['POST'])
def save_snapshot():
    frame, tel, act_det, f_time, f_count, s_fps, dropped, errors = receiver.get_state()
    if frame is not None:
        screenshots_dir = r"C:\All_Projects\esp32_cam_ai_system\screenshots"
        os.makedirs(screenshots_dir, exist_ok=True)
        filename = f"snapshot_{time.strftime('%Y%m%d_%H%M%S')}.png"
        path = os.path.join(screenshots_dir, filename)
        cv2.imwrite(path, frame)
        return jsonify({"status": "ok", "path": path})
    return jsonify({"error": "no_frame"}), 500


def main():
    print("=" * 60)
    print(" TinyCV-ESP32-CAM: Web-Based Live Edge-AI Vision Dashboard")
    print("=" * 60)
    print(f"Connecting to ESP32-CAM on {PORT} @ {BAUD} baud...")
    if not receiver.start():
        print(f"[ERROR] Could not open {PORT}. Make sure other tools are closed.")
        sys.exit(1)

    print("\nStarting Web Dashboard server on http://127.0.0.1:5000 ...")
    app.run(host="127.0.0.1", port=5000, threaded=True, debug=False)


if __name__ == "__main__":
    main()
