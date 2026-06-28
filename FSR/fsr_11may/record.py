#!/usr/bin/env python3
"""
record.py — Real-time session recording for FSR406 × 6  (fsr_11may)

Streams M lines from ESP32 at 100 Hz and saves to CSV.
Optionally displays live force estimates if calibration_coeffs.json is present.

CSV format:
  timestamp_ms, V1_mv, V2_mv, V3_mv, V4_mv, V5_mv, V6_mv, event

Usage:
    conda activate massage
    cd /home/primpunn/experiment/FSR/fsr_11may

    # Record only (no live forces)
    python record.py -o sessions/session_001.csv

    # Record + live force display
    python record.py -o sessions/session_001.csv --live

    # Add event marker while recording: type marker text and press Enter
    # Press Ctrl+C to stop recording.
"""

import argparse
import csv
import json
import sys
import threading
import time
from pathlib import Path

import serial
import serial.tools.list_ports
import numpy as np

from config import N_SENSORS, SENSOR_NAMES, BAUD, SESSIONS_DIR
from fsr_pipeline import IntegralEngine, find_latest_calibration_path


# ── Serial reader ──────────────────────────────────────────────────────────────

def _detect_port():
    for p in serial.tools.list_ports.comports():
        if any(k in p.description.lower() for k in ("cp210", "ch340", "uart", "usb")):
            return p.device
    return None


class Recorder:
    def __init__(self, port, output_path, cal=None):
        self._ser     = serial.Serial(port, BAUD, timeout=0.1)
        time.sleep(2)
        self._ser.reset_input_buffer()

        self._out     = Path(output_path)
        self._out.parent.mkdir(parents=True, exist_ok=True)

        self._engine  = IntegralEngine(cal["sensors"]) if cal else None
        self._lock    = threading.Lock()
        self._event   = ""          # set from main thread via set_event()
        self._running = False
        self._rows_written = 0

    def set_event(self, text):
        with self._lock:
            self._event = text

    def _pop_event(self):
        with self._lock:
            ev = self._event
            self._event = ""
        return ev

    def run(self, live_display=False):
        self._ser.write(b"START\n")
        self._running = True

        with open(self._out, "w", newline="", buffering=1) as fh:
            writer = csv.writer(fh)
            header = ["timestamp_ms"] + [f"V{i+1}_mv" for i in range(N_SENSORS)] + ["event"]
            writer.writerow(header)

            t_start  = None
            seen_ts: set = set()

            while self._running:
                try:
                    raw = self._ser.readline().decode("ascii", errors="ignore").strip()
                except Exception:
                    continue

                if not raw.startswith("M,"):
                    continue

                parts = raw.split(",")
                if len(parts) != N_SENSORS + 2:
                    continue

                try:
                    ts_hw = float(parts[1])
                    mv    = [float(p) for p in parts[2:]]
                except ValueError:
                    continue

                if ts_hw in seen_ts:
                    continue
                seen_ts.add(ts_hw)

                if t_start is None:
                    t_start = ts_hw

                ts_rel = ts_hw - t_start   # ms from session start
                ev     = self._pop_event()
                row    = [int(ts_rel)] + [round(v, 1) for v in mv] + [ev]
                writer.writerow(row)
                self._rows_written += 1

                if live_display and self._engine and self._rows_written % 10 == 0:
                    forces = self._engine.update(ts_hw, mv)
                    totals = self._engine.total_by_group(forces)
                    parts_disp = [f"{n}={forces.get(n, 0):5.1f}N"
                                  for n in sorted(forces)]
                    totals_disp = "  ".join(
                        f"{g}={v:.1f}N" for g, v in totals.items())
                    print(f"\r  {'  '.join(parts_disp)}  │  {totals_disp}  "
                          f"[{self._rows_written} rows]", end="", flush=True)

        self._ser.write(b"STOP\n")
        self._ser.close()

    def stop(self):
        self._running = False


def _event_input_thread(recorder):
    """Reads event markers from stdin while recording runs in another thread."""
    print("\n  Type an event marker and press Enter to annotate the current row.")
    print("  Press Ctrl+C to stop recording.\n")
    while True:
        try:
            text = input()
            recorder.set_event(text.strip())
            print(f"  [event: {text.strip()!r}]")
        except (EOFError, KeyboardInterrupt):
            break


def main():
    ap = argparse.ArgumentParser(description="FSR session recorder (fsr_11may)")
    ap.add_argument("-o", "--output", default=None,
                    help="Output CSV path (default: sessions/YYYY-MM-DD_HH-MM-SS.csv)")
    ap.add_argument("--port",   default=None)
    ap.add_argument("--live",   action="store_true",
                    help="Display live force estimates (requires a saved calibration)")
    ap.add_argument("--cal",    default=None,
                    help="Path to calibration_coeffs.json (default: latest in calibrations/)")
    ap.add_argument("--no-cal", action="store_true",
                    help="Skip loading calibration, record raw voltages only")
    args = ap.parse_args()

    port = args.port or _detect_port()
    if port is None:
        sys.exit("Cannot detect ESP32 port. Use --port /dev/ttyUSBx")

    if args.output is None:
        ts   = time.strftime("%Y-%m-%d_%H-%M-%S")
        args.output = str(Path(SESSIONS_DIR) / f"{ts}.csv")

    cal = None
    if not args.no_cal:
        cal_path = Path(args.cal) if args.cal else find_latest_calibration_path()
        if cal_path and cal_path.exists():
            try:
                cal = json.loads(cal_path.read_text())
                print(f"  Calibration: {cal_path.parent.name}  "
                      f"({len(cal['sensors'])} sensors)")
            except Exception as e:
                print(f"  Warning: could not load calibration ({e}). Recording raw only.")
        else:
            print("  No calibration found — recording raw voltages only.")

    recorder = Recorder(port, args.output, cal)

    print(f"  Recording to {args.output}")
    print("  Ctrl+C to stop.")

    # Start event-input in background
    inp_thread = threading.Thread(target=_event_input_thread,
                                  args=(recorder,), daemon=True)
    inp_thread.start()

    try:
        recorder.run(live_display=args.live and cal is not None)
    except KeyboardInterrupt:
        recorder.stop()
        print(f"\n  Stopped. {recorder._rows_written} rows saved → {args.output}")


if __name__ == "__main__":
    main()
