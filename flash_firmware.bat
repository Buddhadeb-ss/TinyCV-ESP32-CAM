@echo off
echo ============================================================
echo Flashing ESP32-CAM Edge-AI Vision System Firmware
echo Target Port: COM8 | Baud Rate: 460800
echo ============================================================

set ESPTOOL="C:\Espressif\tools\python\v6.0.2\venv\Scripts\esptool.exe"
if not exist %ESPTOOL% (
    set ESPTOOL=python -m esptool
)

%ESPTOOL% --chip esp32 --port COM8 --baud 460800 --before default_reset --after hard_reset write_flash -z --flash_mode dio --flash_freq 40m --flash_size 4MB 0x1000 binaries\bootloader.bin 0x8000 binaries\partitions.bin 0xe000 binaries\boot_app0.bin 0x10000 binaries\firmware.bin

echo.
echo Flashing completed!
pause
