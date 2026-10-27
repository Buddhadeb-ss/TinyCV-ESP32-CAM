import time
import serial
import json
import os
import cv2
import numpy as np

PORT = "COM8"
BAUD = 460800
SCREENSHOTS_DIR = r"C:\All_Projects\esp32_cam_ai_system\screenshots"
BENCHMARKS_DIR = r"C:\All_Projects\esp32_cam_ai_system\benchmarks\raw"
for d in [SCREENSHOTS_DIR, BENCHMARKS_DIR]:
    os.makedirs(d, exist_ok=True)

print(f"Connecting to physical ESP32-CAM on {PORT} at {BAUD} baud...")
s = serial.Serial()
s.port = PORT
s.baudrate = BAUD
s.timeout = 2.0
s.dtr = False
s.rts = False
s.open()

# Clean reset sequence
time.sleep(0.3)
s.dtr = False
s.rts = False
time.sleep(0.5)

print("Listening for boot banner and telemetry...")
start_time = time.time()
banner_lines = []
telemetry_samples = []

while time.time() - start_time < 5.0:
    line = s.readline().decode('ascii', errors='replace').strip()
    if line:
        print(" [ESP32]", line)
        banner_lines.append(line)
        if line.startswith("{") and "cid" in line:
            break

# Switch to Continuous AI Mode to measure steady-state neural inference
print("\n--- Sending Command: MODE=AI (Continuous AI on center FOV) ---")
s.write(b"MODE=AI\n")
s.flush()
time.sleep(0.2)

# Collect 40 telemetry frames
print("\nCollecting 40 live on-device inference frames...")
latencies = []
fps_list = []
free_heaps = []
free_psram = []
detections = []

frame_count = 0
while frame_count < 40:
    line = s.readline().decode('ascii', errors='replace').strip()
    if not line:
        continue
    if line.startswith("{") and "lat" in line:
        try:
            data = json.loads(line)
            lat = data.get("lat", 0)
            fps = data.get("fps", 0.0)
            cls_name = data.get("cls", "unknown")
            conf = data.get("conf", 0)
            hp = data.get("hp", 0)
            ps = data.get("ps", 0)
            
            latencies.append(lat)
            if fps > 0: fps_list.append(fps)
            free_heaps.append(hp)
            free_psram.append(ps)
            detections.append(data)
            
            frame_count += 1
            print(f" Frame {frame_count:2d}: Top Class={cls_name:<18} Conf={conf}% Latency={lat:2d}ms FPS={fps:.1f} FreeHeap={hp} PSRAM={ps}")
        except Exception as e:
            print(" Parse error:", e, "Line:", line)
    else:
        print(" [ESP32 MSG]", line)

# Request a live camera frame
print("\n--- Requesting Live Camera Frame (GET_FRAME) ---")
s.write(b"GET_FRAME\n")
s.flush()

raw_hex = ""
reading_frame = False
expected_len = 0
t0 = time.time()

while time.time() - t0 < 4.0:
    line = s.readline().decode('ascii', errors='replace').strip()
    if "[FRAME_START:" in line:
        reading_frame = True
        expected_len = int(line.split(":")[1].replace("]", ""))
        print(f" Receiving frame buffer of length {expected_len} bytes...")
    elif "[FRAME_END]" in line:
        reading_frame = False
        break
    elif reading_frame:
        raw_hex += line

s.close()
print("Serial port closed.")

# Process and save telemetry summary
summary = {
    "test_name": "Physical ESP32-CAM Live Hardware Verification",
    "port": PORT,
    "baud": BAUD,
    "samples_recorded": len(latencies),
    "mean_inference_latency_ms": round(float(np.mean(latencies)), 2),
    "min_inference_latency_ms": int(np.min(latencies)),
    "max_inference_latency_ms": int(np.max(latencies)),
    "p95_inference_latency_ms": round(float(np.percentile(latencies, 95)), 2),
    "mean_fps": round(float(np.mean(fps_list)), 2) if fps_list else 0.0,
    "min_free_heap": int(np.min(free_heaps)),
    "max_free_heap": int(np.max(free_heaps)),
    "free_psram_bytes": int(np.mean(free_psram)),
    "heap_leak_bytes": int(free_heaps[0] - free_heaps[-1]) if free_heaps else 0
}

json_path = os.path.join(BENCHMARKS_DIR, "live_hardware_benchmark.json")
with open(json_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved live hardware benchmark: {json_path}")
print(json.dumps(summary, indent=2))

# Decode live frame if received
if raw_hex:
    try:
        raw_bytes = bytes.fromhex(raw_hex)
        print(f"\nDecoded {len(raw_bytes)} bytes from hex stream.")
        if len(raw_bytes) == 96 * 96 * 2: # RGB565
            # Convert RGB565 to RGB888
            arr16 = np.frombuffer(raw_bytes, dtype=np.uint16)
            # ESP32 is little-endian or big-endian from camera FIFO
            # In camera FIFO, high byte is first
            r = (((arr16 >> 8) & 0xF8)).astype(np.uint8)
            g = (((arr16 >> 3) & 0xFC)).astype(np.uint8)
            b = (((arr16 << 3) & 0xF8)).astype(np.uint8)
            rgb_img = np.dstack((r.reshape((96, 96)), g.reshape((96, 96)), b.reshape((96, 96))))
            bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
            
            # Save raw and 4x upscaled preview
            raw_path = os.path.join(SCREENSHOTS_DIR, "live_camera_capture_raw.png")
            cv2.imwrite(raw_path, bgr_img)
            
            preview_img = cv2.resize(bgr_img, (384, 384), interpolation=cv2.INTER_NEAREST)
            # Draw telemetry on preview
            cv2.putText(preview_img, f"LIVE ESP32-CAM: {latencies[-1]}ms, {fps_list[-1]:.1f} FPS", 
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            cv2.putText(preview_img, f"Pred: {detections[-1]['cls']} ({detections[-1]['conf']}%)", 
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            preview_path = os.path.join(SCREENSHOTS_DIR, "live_camera_capture_preview.png")
            cv2.imwrite(preview_path, preview_img)
            print(f"Saved live camera capture: {raw_path} and {preview_path}")
    except Exception as e:
        print("Frame decode error:", e)
