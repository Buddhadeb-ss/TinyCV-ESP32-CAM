"""
ESP32-CAM Web-Based Interactive Capture & Training Studio (tools/web_capture_app.py)
Provides a high-performance web interface at http://127.0.0.1:5000 for sensor-matched
dataset collection, real-time MJPEG live streaming, diversity filtering, and 1-click model training.
Eliminates all Windows session-0 / OpenCV GUI display limitations.
"""

import os
import sys
import time
import json
import uuid
import struct
import threading
import cv2
import numpy as np
from flask import Flask, Response, render_template_string, request, jsonify, send_file

# Add project root
PROJECT_ROOT = r"C:\All_Projects\esp32_cam_ai_system"
sys.path.append(PROJECT_ROOT)

from capture_tool import (
    SerialVideoReceiver, DiversityFilter, DatasetManager,
    DEFAULT_CLASSES, CLASS_COLORS, INSTRUCTIONS_BY_CLASS,
    PORT, BAUD, RAW_DATASET_DIR, MANIFEST_DIR, REPORTS_DIR
)

app = Flask(__name__)

# Global singletons
receiver = SerialVideoReceiver(PORT, BAUD)
diversity_filter = DiversityFilter()
dataset_manager = DatasetManager()

state = {
    "active_class": "cream_container",
    "is_auto_capturing": False,
    "session_id": f"session_{time.strftime('%Y%m%d_%H%M%S')}",
    "session_accepted": 0,
    "last_decision": "IDLE - Click Start Auto-Capture or Snap Frame",
    "last_decision_type": "info", # info, success, warning, error
    "flash_on": False
}
state_lock = threading.Lock()


def generate_mjpeg_stream():
    """Generates continuous MJPEG multipart stream for web browser."""
    while True:
        frame_bgr, f_time, f_count, s_fps, dropped, errors = receiver.get_latest_frame()
        if frame_bgr is not None:
            now = time.time()
            if now - f_time < 3.0:
                # Upscale 96x96 -> 384x384 for clean browser display
                display_frame = cv2.resize(frame_bgr, (384, 384), interpolation=cv2.INTER_NEAREST)

                # Overlay center crop reticle (48x48 model input area)
                cv2.rectangle(display_frame, (96, 96), (288, 288), (0, 230, 255), 1)

                # Process auto-capture if enabled
                with state_lock:
                    if state["is_auto_capturing"]:
                        passed, reason, s_val, l_val, d_val = diversity_filter.evaluate(frame_bgr)
                        if passed:
                            record = dataset_manager.save_sample(
                                frame_bgr=frame_bgr,
                                class_name=state["active_class"],
                                session_id=state["session_id"],
                                sharpness=s_val,
                                mean_lum=l_val,
                                diff=d_val,
                                capture_mode="AUTO"
                            )
                            diversity_filter.record_acceptance(frame_bgr)
                            state["session_accepted"] += 1
                            state["last_decision"] = f"SAVED #{len(dataset_manager.samples)}: {record['filename']}"
                            state["last_decision_type"] = "success"
                        else:
                            state["last_decision"] = f"SKIPPED: {reason}"
                            state["last_decision_type"] = "warning"

                # Encode to JPEG for browser
                ret, jpeg_bytes = cv2.imencode('.jpg', display_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + jpeg_bytes.tobytes() + b'\r\n')
            else:
                time.sleep(0.05)
        else:
            time.sleep(0.05)


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>ESP32-CAM Sensor-Matched Studio</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #0d0f14;
      --card-bg: #161922;
      --card-border: #262b38;
      --primary: #00d2ff;
      --primary-hover: #3ad9ff;
      --accent: #ff9900;
      --success: #00e676;
      --danger: #ff1744;
      --warning: #ffb300;
      --text: #e2e8f0;
      --text-muted: #8b9bb4;
      --font: 'Outfit', sans-serif;
      --mono: 'JetBrains Mono', monospace;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: var(--bg);
      color: var(--text);
      font-family: var(--font);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }
    header {
      background: #12151d;
      border-bottom: 1px solid var(--card-border);
      padding: 16px 28px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .brand { display: flex; align-items: center; gap: 12px; }
    .brand-icon {
      width: 32px; height: 32px; border-radius: 8px;
      background: linear-gradient(135deg, var(--primary), #0077ff);
      display: flex; align-items: center; justify-content: center;
      font-weight: 700; color: #000;
    }
    .brand h1 { font-size: 1.15rem; font-weight: 600; letter-spacing: -0.01em; }
    .badge {
      font-family: var(--mono); font-size: 0.72rem; padding: 4px 10px;
      border-radius: 20px; background: rgba(0, 210, 255, 0.12); color: var(--primary);
      border: 1px solid rgba(0, 210, 255, 0.3);
    }
    main {
      flex: 1; padding: 24px; display: grid;
      grid-template-columns: 440px 1fr; gap: 24px;
      max-width: 1400px; margin: 0 auto; width: 100%;
    }
    .card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 14px;
      padding: 20px;
      position: relative;
    }
    .card-title {
      font-size: 0.85rem; font-weight: 600; text-transform: uppercase;
      letter-spacing: 0.05em; color: var(--primary); margin-bottom: 14px;
      display: flex; align-items: center; justify-content: space-between;
    }
    .video-viewport {
      width: 100%; height: 400px; background: #08090d;
      border-radius: 10px; overflow: hidden; position: relative;
      border: 1px solid var(--card-border); display: flex;
      align-items: center; justify-content: center;
    }
    .video-viewport img { width: 100%; height: 100%; object-fit: contain; }
    .stream-overlay {
      position: absolute; top: 12px; left: 12px;
      background: rgba(13, 15, 20, 0.85); backdrop-filter: blur(8px);
      padding: 6px 12px; border-radius: 20px; font-family: var(--mono);
      font-size: 0.75rem; border: 1px solid rgba(255,255,255,0.1);
      display: flex; align-items: center; gap: 8px;
    }
    .status-dot {
      width: 8px; height: 8px; border-radius: 50%; background: var(--success);
      box-shadow: 0 0 8px var(--success);
    }
    .controls-grid {
      display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; margin-top: 16px;
    }
    button {
      font-family: var(--font); font-size: 0.88rem; font-weight: 600;
      padding: 12px 16px; border-radius: 10px; border: none; cursor: pointer;
      transition: all 0.15s ease; display: flex; align-items: center; justify-content: center; gap: 8px;
    }
    .btn-primary { background: var(--primary); color: #000; }
    .btn-primary:hover { background: var(--primary-hover); transform: translateY(-1px); }
    .btn-success { background: var(--success); color: #000; }
    .btn-danger { background: rgba(255, 23, 68, 0.15); color: var(--danger); border: 1px solid var(--danger); }
    .btn-danger:hover { background: var(--danger); color: #fff; }
    .btn-secondary { background: #222736; color: var(--text); border: 1px solid var(--card-border); }
    .btn-secondary:hover { background: #2b3244; }
    .class-selector-grid {
      display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; margin-bottom: 16px;
    }
    .class-btn {
      padding: 14px 12px; border-radius: 10px; text-align: left;
      background: #1c212e; border: 1px solid var(--card-border);
      color: var(--text); cursor: pointer; transition: all 0.15s ease;
      display: flex; flex-direction: column; gap: 4px;
    }
    .class-btn:hover { border-color: rgba(0, 210, 255, 0.5); transform: translateY(-1px); }
    .class-btn.active {
      border-color: var(--primary);
      background: rgba(0, 210, 255, 0.08);
      box-shadow: 0 0 16px rgba(0, 210, 255, 0.15);
    }
    .class-btn-name { font-weight: 600; font-size: 0.92rem; }
    .class-btn-count { font-family: var(--mono); font-size: 0.75rem; color: var(--text-muted); }
    .instruction-box {
      background: rgba(0, 210, 255, 0.05); border: 1px solid rgba(0, 210, 255, 0.2);
      border-radius: 10px; padding: 14px 18px; margin-bottom: 18px; font-size: 0.9rem;
      line-height: 1.45;
    }
    .instruction-box strong { color: var(--primary); display: block; margin-bottom: 4px; }
    .filmstrip {
      display: flex; gap: 12px; overflow-x: auto; padding: 10px 0; margin-top: 14px;
    }
    .filmstrip-thumb {
      width: 76px; height: 76px; border-radius: 8px; border: 1px solid var(--card-border);
      object-fit: cover; background: #08090d; flex-shrink: 0;
    }
    .metrics-table {
      width: 100%; border-collapse: collapse; font-size: 0.84rem; margin-top: 10px;
    }
    .metrics-table td { padding: 6px 0; border-bottom: 1px solid rgba(255,255,255,0.05); }
    .metrics-table td:last-child { text-align: right; font-family: var(--mono); color: var(--primary); }
    .status-alert {
      padding: 10px 14px; border-radius: 8px; font-size: 0.82rem; font-family: var(--mono);
      margin-top: 14px; display: flex; align-items: center; gap: 8px;
    }
    .status-alert.info { background: #1a2233; color: var(--primary); border: 1px solid rgba(0, 210, 255, 0.3); }
    .status-alert.success { background: #132b1f; color: var(--success); border: 1px solid rgba(0, 230, 118, 0.3); }
    .status-alert.warning { background: #2f2513; color: var(--warning); border: 1px solid rgba(255, 179, 0, 0.3); }
    .action-banner {
      background: linear-gradient(135deg, rgba(0, 210, 255, 0.12), rgba(0, 119, 255, 0.12));
      border: 1px solid rgba(0, 210, 255, 0.35); border-radius: 12px;
      padding: 18px; margin-top: 20px; display: flex; justify-content: space-between; align-items: center;
    }
  </style>
</head>
<body>

  <header>
    <div class="brand">
      <div class="brand-icon">AI</div>
      <div>
        <h1>ESP32-CAM Sensor-Matched Studio</h1>
        <p style="font-size: 0.75rem; color: var(--text-muted);">OmniVision OV3660 96x96 Direct Sensor Pipeline</p>
      </div>
    </div>
    <div style="display: flex; gap: 12px; align-items: center;">
      <span class="badge" id="portBadge">COM8 @ 460800 Baud</span>
      <span class="badge" id="fpsBadge">15.0 FPS</span>
    </div>
  </header>

  <main>
    <!-- Left Column: Live Sensor Feed -->
    <div class="card">
      <div class="card-title">
        <span>Hardware Camera Feed</span>
        <span class="badge" id="resBadge">96x96 RGB565</span>
      </div>
      
      <div class="video-viewport">
        <img src="/video_feed" alt="ESP32-CAM Live Feed">
        <div class="stream-overlay">
          <div class="status-dot" id="streamDot"></div>
          <span id="captureModeText">PREVIEW ONLY</span>
        </div>
      </div>

      <div class="controls-grid">
        <button id="autoBtn" class="btn-primary" onclick="toggleAutoCapture()">
          ▶ Start Auto-Capture
        </button>
        <button class="btn-secondary" onclick="manualCapture()">
          📷 Snap Single Frame
        </button>
        <button class="btn-secondary" onclick="toggleFlash()">
          💡 Toggle Flash LED
        </button>
        <button class="btn-danger" onclick="deleteLastSample()">
          🗑️ Delete Last Sample
        </button>
      </div>

      <div class="status-alert info" id="statusAlert">
        STATUS: Awaiting scan actions...
      </div>

      <div style="margin-top: 16px;">
        <span style="font-size: 0.78rem; text-transform: uppercase; color: var(--text-muted); font-weight: 600;">Recent Captures</span>
        <div class="filmstrip" id="filmstripContainer">
          <!-- Dynamically populated -->
        </div>
      </div>
    </div>

    <!-- Right Column: Classes, Instructions & Training Trigger -->
    <div style="display: flex; flex-direction: column; gap: 20px;">
      
      <div class="card">
        <div class="card-title">
          <span>Target Object Selection</span>
          <span style="font-size: 0.75rem; color: var(--text-muted);" id="datasetTotalText">0 Total Samples</span>
        </div>

        <div class="class-selector-grid" id="classSelectorGrid">
          <!-- Dynamically populated buttons -->
        </div>

        <div class="instruction-box" id="instructionBox">
          <strong>Physical Scanning Recommendation:</strong>
          <span id="instructionText">Select an object to see scanning directions.</span>
        </div>

        <div style="margin-top: 10px;">
          <div class="card-title" style="margin-bottom: 8px;">
            <span>Real-Time Diversity Filter Status</span>
          </div>
          <table class="metrics-table">
            <tr><td>Temporal Rate Limit Discards:</td><td id="statRate">0</td></tr>
            <tr><td>Motion Blur Discards:</td><td id="statBlur">0</td></tr>
            <tr><td>Duplicate / Static Discards:</td><td id="statDup">0</td></tr>
            <tr><td>Exposure (Over/Under) Discards:</td><td id="statExp">0</td></tr>
          </table>
        </div>
      </div>

      <!-- Action Banner: Launch Training -->
      <div class="action-banner">
        <div>
          <h3 style="font-size: 1.05rem; font-weight: 600; color: #fff;">Done Scanning?</h3>
          <p style="font-size: 0.82rem; color: var(--text-muted);">Partition zero-leakage splits, train DeskObjectNet INT8, and flash device.</p>
        </div>
        <button class="btn-success" onclick="startAutomatedTraining()">
          ⚡ Train & Deploy Model
        </button>
      </div>

    </div>
  </main>

  <script>
    let activeClass = "cream_container";
    let isAuto = false;

    const classNames = [
      "cream_container",
      "mouse",
      "earbuds_case",
      "glue_stick",
      "microcontroller_board",
      "background"
    ];

    const classDisplayNames = {
      "cream_container": "Cream Container",
      "mouse": "Computer Mouse",
      "earbuds_case": "Earbuds Case",
      "glue_stick": "Glue Stick",
      "microcontroller_board": "ESP32 MCU Board",
      "background": "Negative Background"
    };

    function updateUI() {
      fetch('/api/status')
        .then(res => res.json())
        .then(data => {
          activeClass = data.active_class;
          isAuto = data.is_auto_capturing;

          // Badges & stream
          document.getElementById('fpsBadge').textContent = data.stream_fps.toFixed(1) + " FPS";
          document.getElementById('datasetTotalText').textContent = data.total_samples + " Total Samples";

          const autoBtn = document.getElementById('autoBtn');
          const captureModeText = document.getElementById('captureModeText');
          const streamDot = document.getElementById('streamDot');

          if (isAuto) {
            autoBtn.className = "btn-secondary";
            autoBtn.textContent = "⏸ Pause Auto-Capture";
            captureModeText.textContent = "AUTO-CAPTURING (" + activeClass.toUpperCase() + ")";
            streamDot.style.background = "var(--success)";
          } else {
            autoBtn.className = "btn-primary";
            autoBtn.textContent = "▶ Start Auto-Capture";
            captureModeText.textContent = "PREVIEW ONLY (PAUSED)";
            streamDot.style.background = "#888";
          }

          // Instruction & Status
          document.getElementById('instructionText').textContent = data.instruction;
          const alertBox = document.getElementById('statusAlert');
          alertBox.className = "status-alert " + data.last_decision_type;
          alertBox.textContent = data.last_decision;

          // Diversity stats
          document.getElementById('statRate').textContent = data.reject_stats.rate_limit;
          document.getElementById('statBlur').textContent = data.reject_stats.blur;
          document.getElementById('statDup').textContent = data.reject_stats.duplicate;
          document.getElementById('statExp').textContent = data.reject_stats.overexposed + data.reject_stats.underexposed;

          // Render class buttons
          renderClassButtons(data.class_counts);

          // Render filmstrip
          renderFilmstrip(data.recent_thumbnails);
        })
        .catch(err => console.error(err));
    }

    function renderClassButtons(counts) {
      const container = document.getElementById('classSelectorGrid');
      container.innerHTML = "";
      classNames.forEach((c) => {
        const btn = document.createElement('div');
        const count = counts[c] || 0;
        const isActive = (c === activeClass);
        btn.className = "class-btn" + (isActive ? " active" : "");
        btn.innerHTML = `
          <span class="class-btn-name">${classDisplayNames[c]}</span>
          <span class="class-btn-count">${count} captured</span>
        `;
        btn.onclick = () => selectClass(c);
        container.appendChild(btn);
      });
    }

    function renderFilmstrip(thumbs) {
      const container = document.getElementById('filmstripContainer');
      container.innerHTML = "";
      if (!thumbs || thumbs.length === 0) {
        container.innerHTML = '<span style="font-size:0.75rem; color:#666;">No captures yet for this session</span>';
        return;
      }
      thumbs.forEach(url => {
        const img = document.createElement('img');
        img.className = "filmstrip-thumb";
        img.src = url;
        container.appendChild(img);
      });
    }

    function selectClass(className) {
      fetch('/api/set_class', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({class_name: className})
      }).then(() => updateUI());
    }

    function toggleAutoCapture() {
      fetch('/api/toggle_auto', {method: 'POST'}).then(() => updateUI());
    }

    function manualCapture() {
      fetch('/api/manual_capture', {method: 'POST'}).then(() => updateUI());
    }

    function toggleFlash() {
      fetch('/api/toggle_flash', {method: 'POST'}).then(() => updateUI());
    }

    function deleteLastSample() {
      if (confirm("Delete the most recently captured image?")) {
        fetch('/api/delete_last', {method: 'POST'}).then(() => updateUI());
      }
    }

    function startAutomatedTraining() {
      if (confirm("Begin dataset splitting, INT8 training, quantization, and ESP32-CAM deployment?")) {
        alert("Training pipeline triggered! Check terminal / IDE context for execution logs.");
        fetch('/api/start_training', {method: 'POST'});
      }
    }

    // Poll status every 500ms
    setInterval(updateUI, 500);
    updateUI();
  </script>
</body>
</html>
"""


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/video_feed')
def video_feed():
    return Response(generate_mjpeg_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/status')
def api_status():
    frame, f_time, f_count, s_fps, dropped, errors = receiver.get_latest_frame()
    counts = dataset_manager.count_samples_per_class()

    # Get recent 6 image URLs for active class
    recent_thumbs = []
    with state_lock:
        active_c = state["active_class"]
        matching = [s for s in dataset_manager.samples if s["class_label"] == active_c]
        for s in matching[-6:]:
            recent_thumbs.append(f"/api/image/{s['class_label']}/{s['filename']}")

        return jsonify({
            "active_class": state["active_class"],
            "is_auto_capturing": state["is_auto_capturing"],
            "session_accepted": state["session_accepted"],
            "last_decision": state["last_decision"],
            "last_decision_type": state["last_decision_type"],
            "stream_fps": s_fps,
            "total_samples": len(dataset_manager.samples),
            "class_counts": counts,
            "instruction": INSTRUCTIONS_BY_CLASS.get(active_c, "Move camera around target object."),
            "reject_stats": diversity_filter.reject_stats,
            "recent_thumbnails": recent_thumbs
        })


@app.route('/api/set_class', methods=['POST'])
def api_set_class():
    data = request.json or {}
    c = data.get("class_name")
    if c in DEFAULT_CLASSES:
        with state_lock:
            state["active_class"] = c
            state["session_accepted"] = 0
            state["session_id"] = f"session_{c}_{time.strftime('%Y%m%d_%H%M%S')}"
            state["last_decision"] = f"Switched to {c.upper()}"
            state["last_decision_type"] = "info"
        return jsonify({"status": "ok"})
    return jsonify({"status": "error", "message": "Invalid class"}), 400


@app.route('/api/toggle_auto', methods=['POST'])
def api_toggle_auto():
    with state_lock:
        state["is_auto_capturing"] = not state["is_auto_capturing"]
        state["last_decision"] = "Auto-Capture STARTED" if state["is_auto_capturing"] else "Auto-Capture PAUSED"
        state["last_decision_type"] = "success" if state["is_auto_capturing"] else "info"
    return jsonify({"status": "ok", "is_auto": state["is_auto_capturing"]})


@app.route('/api/manual_capture', methods=['POST'])
def api_manual_capture():
    frame, _, _, _, _, _ = receiver.get_latest_frame()
    if frame is not None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        s_val = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        l_val = float(np.mean(gray))
        with state_lock:
            rec = dataset_manager.save_sample(
                frame_bgr=frame,
                class_name=state["active_class"],
                session_id=state["session_id"],
                sharpness=s_val,
                mean_lum=l_val,
                diff=99.0,
                capture_mode="MANUAL"
            )
            state["session_accepted"] += 1
            state["last_decision"] = f"MANUALLY SAVED #{len(dataset_manager.samples)}: {rec['filename']}"
            state["last_decision_type"] = "success"
        diversity_filter.record_acceptance(frame)
        return jsonify({"status": "ok", "filename": rec['filename']})
    return jsonify({"status": "error", "message": "No frame available"}), 500


@app.route('/api/delete_last', methods=['POST'])
def api_delete_last():
    deleted = dataset_manager.delete_last_sample()
    if deleted:
        with state_lock:
            state["last_decision"] = f"DELETED: {deleted['filename']}"
            state["last_decision_type"] = "error"
        return jsonify({"status": "ok", "deleted": deleted['filename']})
    return jsonify({"status": "error", "message": "No sample to delete"}), 400


@app.route('/api/toggle_flash', methods=['POST'])
def api_toggle_flash():
    with state_lock:
        state["flash_on"] = not state["flash_on"]
        receiver.send_command("FLASH=1" if state["flash_on"] else "FLASH=0")
        state["last_decision"] = f"Flashlight {'ON' if state['flash_on'] else 'OFF'}"
        state["last_decision_type"] = "info"
    return jsonify({"status": "ok", "flash": state["flash_on"]})


@app.route('/api/image/<class_name>/<filename>')
def serve_image(class_name, filename):
    p = os.path.join(RAW_DATASET_DIR, class_name, filename)
    if os.path.exists(p):
        return send_file(p, mimetype='image/png')
    return "Not found", 404


@app.route('/api/start_training', methods=['POST'])
def api_start_training():
    # Save manifest
    dataset_manager.save_manifest()
    # Launch training pipeline in thread
    def run_training_pipeline():
        print("[TRAINING TRIGGERED VIA WEB STUDIO]")
        import subprocess
        subprocess.run(["python", os.path.join(PROJECT_ROOT, "tools", "build_sensor_dataset_splits.py")])
        subprocess.run(["python", os.path.join(PROJECT_ROOT, "training", "scripts", "train_sensor_matched_model.py")])
        subprocess.run(["python", os.path.join(PROJECT_ROOT, "tools", "calibrate_thresholds.py")])
        print("[TRAINING PIPELINE COMPLETE]")

    t = threading.Thread(target=run_training_pipeline, daemon=True)
    t.start()
    return jsonify({"status": "ok", "message": "Training pipeline started"})


def main():
    print(f"===========================================================")
    print(f" ESP32-CAM Web Capture Studio")
    print(f" Connect to ESP32-CAM on {PORT} @ {BAUD} baud...")
    print(f" Open your browser at: http://127.0.0.1:5000")
    print(f"===========================================================")
    
    if not receiver.start():
        print(f"[FATAL] Failed to connect to ESP32-CAM on {PORT}!")
        sys.exit(1)

    try:
        app.run(host='0.0.0.0', port=5000, threaded=True, debug=False)
    finally:
        receiver.stop()


if __name__ == "__main__":
    main()
