param(
    [string]$Port = "COM8",
    [int]$Baud = 460800
)

$esptool = "C:\Espressif\tools\python\v6.0.2\venv\Scripts\esptool.exe"
if (-not (Test-Path $esptool)) {
    $esptool = "esptool.py"
}

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " Flashing ESP32-CAM Edge-AI Vision System Firmware" -ForegroundColor Cyan
Write-Host " Target Port: $Port | Baud: $Baud" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

& $esptool --chip esp32 --port $Port --baud $Baud --before default_reset --after hard_reset write_flash -z --flash_mode dio --flash_freq 40m --flash_size 4MB 0x1000 "$PSScriptRoot\binaries\bootloader.bin" 0x8000 "$PSScriptRoot\binaries\partitions.bin" 0xe000 "$PSScriptRoot\binaries\boot_app0.bin" 0x10000 "$PSScriptRoot\binaries\firmware.bin"

if ($LASTEXITCODE -eq 0) {
    Write-Host "`nFirmware flashed successfully!" -ForegroundColor Green
} else {
    Write-Host "`nFlashing failed with exit code $LASTEXITCODE" -ForegroundColor Red
}
