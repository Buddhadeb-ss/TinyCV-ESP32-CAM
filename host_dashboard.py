"""
ESP32-CAM Edge-AI Host Monitoring & Control Dashboard v2.0
Displays live on-device multi-object recognition telemetry, bounding box proposals,
confidence metrics, motion tracking, memory health, and interactive control.
"""

import sys
import time
import json
import serial
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.text import Text

PORT = "COM8"
BAUD = 460800

def create_dashboard_layout():
    layout = Layout()
    layout.split(
        Layout(name="header", size=3),
        Layout(name="main", ratio=1),
        Layout(name="footer", size=3)
    )
    layout["main"].split_row(
        Layout(name="telemetry", ratio=1),
        Layout(name="detection", ratio=1)
    )
    return layout

def main():
    console = Console()
    console.print(f"[bold cyan]Connecting to ESP32-CAM on {PORT} at {BAUD} baud...[/bold cyan]")

    try:
        ser = serial.Serial(PORT, BAUD, timeout=0.5)
        ser.dtr = False
        ser.rts = True
        time.sleep(0.1)
        ser.rts = False
        time.sleep(0.3)
        ser.reset_input_buffer()
    except Exception as e:
        console.print(f"[bold red]Failed to open {PORT}: {e}[/bold red]")
        sys.exit(1)

    layout = create_dashboard_layout()
    layout["header"].update(Panel("[bold green]ESP32-CAM Desk Object Recognition Dashboard v2.0[/bold green] | Xtensa LX6 @ 240MHz | DeskObjectNet INT8 (6 Classes)", style="white on blue"))
    layout["footer"].update(Panel("[bold yellow]Status: Streaming On-Device Inference Telemetry | Press Ctrl+C to exit[/bold yellow]", style="white on black"))

    detection_events = []
    start_time = time.time()

    try:
        with Live(layout, refresh_per_second=10, console=console):
            while True:
                line = ser.readline().decode('utf-8', errors='replace').strip()
                if not line:
                    continue

                if line.startswith("{") and line.endswith("}") and "cid" in line:
                    try:
                        data = json.loads(line)
                        frame = data.get("f", 0)
                        mode_num = data.get("m", 0)
                        mode_str = ["HYBRID (ROI Saliency + AI)", "CONTINUOUS_AI (Every Frame)", "PURE_CV (High FPS)"][mode_num] if mode_num < 3 else "UNKNOWN"
                        
                        cid = data.get("cid", 0)
                        cls_name = data.get("cls", "background")
                        conf = data.get("conf", 0)
                        is_det = data.get("det", 0) == 1
                        
                        bx = data.get("bx", 0)
                        by = data.get("by", 0)
                        bw = data.get("bw", 48)
                        bh = data.get("bh", 48)
                        prop = data.get("prop", "none")
                        mot = data.get("mot", 0.0)
                        lat = data.get("lat", 0)
                        fps = data.get("fps", 0.0)
                        hp = data.get("hp", 0)
                        ps = data.get("ps", 0)

                        if is_det:
                            t_str = time.strftime("%H:%M:%S")
                            detection_events.insert(0, f"[{t_str}] Frame #{frame}: [bold green]{cls_name.upper()}[/bold green] ({conf}%) @ [{bx},{by},{bw},{bh}]")
                            if len(detection_events) > 8:
                                detection_events.pop()

                        # Telemetry Panel
                        tel_table = Table.grid(padding=(0, 2))
                        tel_table.add_column(style="bold yellow")
                        tel_table.add_column(style="white")

                        tel_table.add_row("Camera Frame:", f"#{frame}")
                        tel_table.add_row("Operating Mode:", f"[bold cyan]{mode_str}[/bold cyan]")
                        tel_table.add_row("Throughput:", f"[bold green]{fps:.1f} FPS[/bold green]")
                        tel_table.add_row("On-Device Latency:", f"[bold magenta]{lat} ms[/bold magenta]" if lat > 0 else "[dim]0 ms (CV Fast Path)[/dim]")
                        tel_table.add_row("Proposal Region:", f"X={bx}, Y={by}, W={bw}, H={bh} (Type: {prop})")
                        tel_table.add_row("Motion Activity:", f"{mot:.1f}%")
                        tel_table.add_row("Free Heap RAM:", f"{hp:,} bytes")
                        tel_table.add_row("Free PSRAM:", f"{ps:,} bytes (External SPI)")
                        tel_table.add_row("Execution Engine:", "[bold green]100% On-Device Xtensa LX6 (No Cloud/PC Inference)[/bold green]")

                        layout["telemetry"].update(Panel(tel_table, title="[bold]Embedded Telemetry[/bold]", border_style="cyan"))

                        # Detection Panel
                        det_table = Table.grid(padding=(0, 2))
                        det_table.add_column(style="bold white", width=18)
                        det_table.add_column(style="white")

                        status_color = "green" if is_det else "yellow"
                        status_text = f"[bold {status_color}]{cls_name.upper()} ({conf}%)[/bold {status_color}]" if is_det else "[dim]Scanning scene...[/dim]"
                        det_table.add_row("Top Recognized:", status_text)
                        det_table.add_row("Confidence Bar:", f"[{'=' * (conf // 5)}{' ' * (20 - conf // 5)}] {conf}%")
                        det_table.add_row("", "")
                        det_table.add_row("[bold]Recent Detections:[/bold]", "")

                        if not detection_events:
                            det_table.add_row("", "[dim]No thresholded objects detected yet[/dim]")
                        else:
                            for event in detection_events:
                                det_table.add_row("", event)

                        layout["detection"].update(Panel(det_table, title="[bold]Real-Time Object Recognition[/bold]", border_style="green"))

                    except Exception:
                        pass

    except KeyboardInterrupt:
        console.print("[yellow]Exiting dashboard...[/yellow]")
    finally:
        ser.close()
        console.print("[green]Serial connection closed.[/green]")

if __name__ == "__main__":
    main()
