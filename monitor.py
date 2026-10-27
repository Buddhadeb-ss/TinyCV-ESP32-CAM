import serial
import time
import sys

def main():
    port = 'COM8'
    baud = 115200
    duration = 12
    if len(sys.argv) > 1:
        try:
            duration = float(sys.argv[1])
        except ValueError:
            pass

    print(f"Connecting to {port} at {baud} baud...")
    try:
        s = serial.Serial(port, baud, timeout=0.5)
        # Pulse RTS to reset
        s.dtr = False
        s.rts = True
        time.sleep(0.1)
        s.rts = False
        time.sleep(0.3)

        start = time.time()
        while time.time() - start < duration:
            line = s.readline()
            if line:
                try:
                    text = line.decode('utf-8', errors='replace').rstrip()
                    if text:
                        print(text.encode('ascii', errors='backslashreplace').decode('ascii'))
                except Exception as ex:
                    print("Decode error:", ex)
        s.close()
        print("Done listening.")
    except Exception as e:
        print("Serial error:", e)

if __name__ == '__main__':
    main()
