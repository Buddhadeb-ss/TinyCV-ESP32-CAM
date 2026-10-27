# ESP32-CAM Hardware & Environment Inspection Report

**Date & Time**: 2026-10-27 | **System**: Windows 11 (Host) | **Workspace**: `c:\All_Projects\esp32_cam_ai_system`

---

## 1. Physical Hardware Discovery & Identification

| Hardware Parameter | Detected Value | Verification Method |
|---|---|---|
| **Module / Variant** | AI-Thinker ESP32-CAM (with ESP32-CAM-MB base) | Physical pinout, DVP camera bus, CH340 USB bridge |
| **SoC / Silicon** | **ESP32-D0WD-V3** (Revision v3.1) | `esptool.exe flash_id` & `espefuse.exe summary` |
| **CPU Architecture** | Dual-core Xtensa 32-bit LX6 @ 240 MHz | eFuse rated frequency: rated 240MHz, low freq: false |
| **Internal SRAM** | 520 KB SRAM (~328 KB usable heap) | ESP32-D0WD-V3 architectural specification |
| **External PSRAM** | **4 MB SPI PSRAM** (64 Mbit AP_3v3) | Standard on AI-Thinker ESP32-CAM modules |
| **SPI Flash Memory** | **4 MB (32 Mbit)**, Manufacturer `0x68`, Device `0x4016` | `esptool.exe` flash ID probe at 3.3V |
| **Flash Voltage (VDD_SDIO)** | 3.3V (strapped via GPIO12) | eFuse reading: `XPD_SDIO_TIEH = 1.8V R/W (0b0)` -> 3.3V |
| **MAC Address** | `a4:f0:0f:76:0a:4c` | Block0 eFuse CRC verified (0x80 OK) |
| **Camera Sensor** | **OmniVision OV2640** (2 Megapixel) | Standard DVP 8-bit camera module with 24-pin FPC |
| **Supported Resolutions** | 96x96, QQVGA (160x120), QCIF (176x144), HQVGA (240x176), QVGA (320x240), CIF (352x288), VGA (640x480), SVGA (800x600), XGA (1024x768), SXGA (1280x1024), UXGA (1600x1200) | OV2640 sensor hardware specification |
| **Camera Pin Mapping** | D0: 5, D1: 18, D2: 19, D3: 21, D4: 36, D5: 39, D6: 34, D7: 35, XCLK: 0, PCLK: 22, VSYNC: 25, HREF: 23, SIOD: 26, SIOC: 27, PWDN: 32, RESET: -1 | AI-Thinker ESP32-CAM pin configuration |
| **On-board LEDs** | Flashlight LED: GPIO 4 (active high) | Red Status LED: GPIO 33 (active low) | Hardware schematic |

---

## 2. Serial Interface, Port & Auto-Reset

| Parameter | Configuration | Observations & Status |
|---|---|---|
| **USB-to-Serial Adapter** | WCH CH340 USB-to-UART Bridge | `USB\VID_1A86&PID_7523` |
| **COM Port** | **COM8** | Verified active and responsive |
| **Serial Boot Baud** | 115,200 baud | Bootloader banner received cleanly (`ets Jul 29 2019`) |
| **Upload Baud Rate** | **460,800 baud** (tested) | Full 4MB flash read in 101.9s at 329.3 kbit/s |
| **Auto-Reset Mechanism** | Hardware DTR/RTS auto-reset circuit | **Confirmed working!** Flashing and resetting do NOT require pressing physical BOOT/IO0 or RST buttons. |

---

## 3. Host Development Environment

| Tool / Resource | Version / Path |
|---|---|
| **Host CPU** | 13th Gen Intel Core i5-13450HX (10 cores, 16 threads) |
| **Host RAM / GPU** | 16 GB DDR5 / NVIDIA GeForce RTX 4050 Laptop GPU |
| **Python** | Python 3.13.15 with `pyserial`, `numpy`, `pillow`, `opencv-python`, `torch` |
| **PlatformIO** | PlatformIO Core v6.2.0 (`espressif32` platform, `framework-arduinoespressif32`) |
| **ESP-IDF** | ESP-IDF v6.0.2 at `C:\esp\v6.0.2\esp-idf` with tools at `C:\Espressif\tools` |
| **esptool** | esptool v5.3.1 at `C:\Espressif\tools\python\v6.0.2\venv\Scripts\esptool.exe` |

---

## 4. Existing Firmware & Data Preservation

Before any write operations, a complete bit-for-bit backup of the original 4MB flash was extracted:
- **Backup File**: `backup/firmware_backup_4MB.bin` (4,194,304 bytes)
- **Original Partition Table**:
  - `nvs`: 0x9000 (20 KB)
  - `otadata`: 0xe000 (8 KB)
  - `app0`: 0x10000 (3 MB - `huge_app` configuration)
  - `spiffs`: 0x310000 (896 KB)
  - `coredump`: 0x3f0000 (64 KB)
