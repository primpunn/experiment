#!/usr/bin/env python3
"""
session_logger.py — Interactive serial terminal with log saving.

- ALL lines from ESP32 are saved to session.txt (needed for Phase 4 fitting)
- Only important lines are printed to screen (so constant 10 Hz LOG lines
  don't flood the display and block your typing)

Lines shown on screen:
    # ...          ESP32 comments and status messages
    S, ...         baseline / capture stats (PASS/FAIL)
    D,...,*_BL_*   baseline sample readings
    D,...,*_CAP_*  capture window readings (after you send C)

Lines saved to file only (not shown on screen):
    D,...,*_LOG_*  continuous 10 Hz loading log (too noisy for screen)
    M,...          monitor mode readings

Usage:
    python session_logger.py                           auto-detect port
    python session_logger.py /dev/ttyUSB0
    python session_logger.py /dev/ttyUSB0 session.txt

Commands to type and press Enter:
    A1 .. A6    select sensor
    B           Phase 2: baseline
    L           Phase 3: start loading
    C           Phase 3: capture (send after 10 s creep wait)
    R           Phase 3: next rep
    X           stop
    G           Phase 6: monitor all channels
    quit        exit this script
"""

import sys
import threading
import time
import serial
import serial.tools.list_ports

BAUD = 115200


def detect_port():
    for p in serial.tools.list_ports.comports():
        if any(k in p.description.lower() for k in ('cp210', 'ch340', 'uart', 'usb')):
            return p.device
    return None


def is_important(line: str) -> bool:
    """Return True if this line should be shown on screen."""
    if line.startswith('#'):
        return True
    if line.startswith('S,'):
        return True
    if line.startswith('D,') and ('_BL_' in line or '_CAP_' in line):
        return True
    # LOG lines (D,...,CH1_LOG_R1) and M lines → file only
    return False


def reader_thread(ser, logfile_path, stop_event):
    with open(logfile_path, 'a', buffering=1) as fh:
        while not stop_event.is_set():
            try:
                raw = ser.readline()
                if not raw:
                    continue
                line = raw.decode('ascii', errors='replace').rstrip('\r\n')
                if not line:
                    continue

                # Always save everything to file
                fh.write(line + '\n')
                fh.flush()

                # Only print important lines to screen
                if is_important(line):
                    print(f'\r{line}')

            except Exception:
                if not stop_event.is_set():
                    time.sleep(0.05)


def main():
    port    = sys.argv[1] if len(sys.argv) > 1 else detect_port()
    logfile = sys.argv[2] if len(sys.argv) > 2 else 'session.txt'

    if port is None:
        sys.exit("ERROR: Cannot auto-detect ESP32 port.\n"
                 "Run:  python session_logger.py /dev/ttyUSB0")

    print(f"Port    : {port} @ {BAUD} baud")
    print(f"Log     : {logfile}  (ALL lines saved here)")
    print(f"Screen  : only # comments, S stats, BL and CAP readings")
    print("──────────────────────────────────────────────────────")
    print("Type a command and press Enter:")
    print("  A1-A6  B  L  C  R  X  G  quit")
    print("──────────────────────────────────────────────────────\n")

    ser = serial.Serial(port, BAUD, timeout=0.1)
    time.sleep(2)
    ser.reset_input_buffer()

    stop_event = threading.Event()
    t = threading.Thread(
        target=reader_thread,
        args=(ser, logfile, stop_event),
        daemon=True,
    )
    t.start()

    try:
        while True:
            cmd = input('> ')
            cmd = cmd.strip()
            if not cmd:
                continue
            if cmd.lower() == 'quit':
                print("Saving and exiting ...")
                break
            ser.write((cmd + '\n').encode())
    except (KeyboardInterrupt, EOFError):
        print()

    stop_event.set()
    ser.close()
    print(f"\nDone. Session saved → {logfile}")
    print(f"Next:  python fsr406_6ch_phase4.py fit {logfile}")


if __name__ == '__main__':
    main()
