import serial
import json
import time

def run_test():
    port = 'COM8'
    baud = 115200
    print(f"Opening {port} at {baud}...")
    s = serial.Serial(port, baud, timeout=1.0)
    time.sleep(0.5)

    def send_cmd(cmd):
        print(f"\n>>> SENDING: {cmd}")
        s.write((cmd + "\n").encode())
        s.flush()
        time.sleep(0.1)

    # Clear input buffer
    s.reset_input_buffer()

    # Test 1: Ping & Status
    send_cmd("PING")
    for _ in range(5):
        line = s.readline().decode('utf-8', errors='replace').strip()
        if line:
            print("RECV:", line)
            if "PONG" in line:
                break

    send_cmd("STATUS")
    for _ in range(5):
        line = s.readline().decode('utf-8', errors='replace').strip()
        if line:
            print("RECV:", line)
            if "status" in line:
                break

    # Test 2: Flash LED control
    print("\n--- Testing Flash LED ---")
    send_cmd("FLASH=1")
    time.sleep(1.0)
    send_cmd("FLASH=0")
    time.sleep(0.5)

    # Test 3: Continuous AI Mode (Continuous Neural Network Inference)
    print("\n--- Testing Continuous AI Mode (Every Frame Runs INT8 Neural Network) ---")
    send_cmd("MODE=AI")
    s.reset_input_buffer()
    
    ai_latencies = []
    ai_frames = []
    start_time = time.time()
    
    while len(ai_frames) < 15 and (time.time() - start_time) < 15:
        line = s.readline().decode('utf-8', errors='replace').strip()
        if line.startswith('{') and line.endswith('}'):
            try:
                data = json.loads(line)
                if data.get("ai") == 1:
                    ai_frames.append(data)
                    ai_latencies.append(data.get("lat", 0))
                    print(f"  [AI Frame {data['f']}] Latency: {data['lat']}ms | Person: {data['p']}% | Non-Person: {data['np']}% | FPS: {data['fps']} | Heap: {data['hp']}")
            except Exception:
                pass

    # Test 4: Pure CV Mode (High-speed frame differencing)
    print("\n--- Testing Pure CV Mode (Peak FPS Classical Vision) ---")
    send_cmd("MODE=CV")
    s.reset_input_buffer()

    cv_fps_list = []
    cv_frames = []
    start_time = time.time()
    while len(cv_frames) < 30 and (time.time() - start_time) < 5:
        line = s.readline().decode('utf-8', errors='replace').strip()
        if line.startswith('{') and line.endswith('}'):
            try:
                data = json.loads(line)
                cv_frames.append(data)
                if data.get("fps", 0) > 0:
                    cv_fps_list.append(data["fps"])
                if len(cv_frames) % 5 == 0:
                    print(f"  [CV Frame {data['f']}] Motion: {data['mot']}% | Lum: {data['lum']} | Centroid: ({data['cx']}, {data['cy']}) | FPS: {data['fps']}")
            except Exception:
                pass

    # Test 5: Restore Hybrid Mode
    print("\n--- Restoring Hybrid Mode ---")
    send_cmd("MODE=HYBRID")
    time.sleep(1.0)

    s.close()

    print("\n================ BENCHMARK SUMMARY ================")
    if ai_latencies:
        avg_lat = sum(ai_latencies) / len(ai_latencies)
        min_lat = min(ai_latencies)
        max_lat = max(ai_latencies)
        print(f"Neural Network Inference Latency (INT8 MobileNet):")
        print(f"  Samples: {len(ai_latencies)}")
        print(f"  Average: {avg_lat:.1f} ms")
        print(f"  Min:     {min_lat} ms")
        print(f"  Max:     {max_lat} ms")
    if cv_fps_list:
        max_fps = max(cv_fps_list)
        avg_fps = sum(cv_fps_list) / len(cv_fps_list)
        print(f"Classical CV Pipeline Throughput:")
        print(f"  Average FPS: {avg_fps:.1f}")
        print(f"  Peak FPS:    {max_fps:.1f}")
    print("====================================================")

if __name__ == '__main__':
    run_test()
