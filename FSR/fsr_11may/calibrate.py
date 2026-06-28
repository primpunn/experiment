#!/usr/bin/env python3
"""
calibrate.py — FSR406 × 6  Log-Linear Calibration  (fsr_11may protocol)

Requires: fsr_11may.ino flashed to ESP32  (115200 baud, START/STOP/D commands)

Usage:
    conda activate massage
    cd /home/primpunn/experiment/FSR/fsr_11may
    python calibrate.py                     # auto-detect ESP32 port
    python calibrate.py /dev/ttyUSB0        # specify port

Menu:
  1. New calibration session  (Phase 0 → 1 → 2 → fit)
  2. Validate calibration     (Phase 3)
  3. Recheck baseline drift   (pre-session check)
  q. Quit
"""

import collections
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np

try:
    import serial
    import serial.tools.list_ports
except ImportError:
    sys.exit("Run: pip install pyserial")

from config import (
    N_SENSORS, SENSOR_NAMES, SENSOR_GROUPS,
    V_CC_MV, BAUD, FS,
    LOAD_G, UNLOAD_G, N_REPS,
    CREEP_WAIT_S, POST_UNLOAD_WAIT_S,
    CAPTURE_N, CAPTURE_INTERVAL_S,
    BASELINE_N_READINGS, BASELINE_INTERVAL_S, BASELINE_SD_MAX_PCT,
    WINDOW_SEC,
    VAL_LOADS_G, RMSE_MAX_N, PCT_ERROR_MAX, RMSE_TOTAL_MAX_N,
    CALIBRATIONS_DIR,
)
from fsr_pipeline import (
    subtract_baseline, fit_log_linear, eval_log_linear,
    CalibBuffer, ForceEngine,
    find_latest_calibration_path, new_calibration_dir,
)


# ── Serial stream ──────────────────────────────────────────────────────────────

class SerialStream:
    """Background thread reading M / DATA lines from ESP32."""

    def __init__(self, port):
        print(f"  Opening {port} @ {BAUD} baud ...", end="", flush=True)
        self._ser = serial.Serial(port, BAUD, timeout=0.1)
        time.sleep(2)
        self._ser.reset_input_buffer()
        print(" ready.")

        self._lock     = threading.Lock()
        self._latest_ts: float | None = None
        self._latest_mv: list | None  = None
        self._running  = False
        self._thread: threading.Thread | None = None
        self._callbacks: list = []   # called with (ts_ms, mv_list) on each M line

    def add_callback(self, fn):
        """Register fn(ts_ms, mv_list) called for every incoming M line."""
        self._callbacks.append(fn)

    def start(self):
        self._running = True
        self._ser.write(b"START\n")
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._ser.write(b"STOP\n")
        self._running = False

    def _loop(self):
        while self._running:
            try:
                raw = self._ser.readline().decode("ascii", errors="ignore").strip()
                if raw.startswith("M,"):
                    parts = raw.split(",")
                    if len(parts) == N_SENSORS + 2:
                        ts  = float(parts[1])
                        mv  = [float(p) for p in parts[2:]]
                        with self._lock:
                            self._latest_ts = ts
                            self._latest_mv = mv
                        for cb in self._callbacks:
                            cb(ts, mv)
            except Exception:
                pass

    def read_latest(self):
        with self._lock:
            return self._latest_ts, (list(self._latest_mv) if self._latest_mv else None)

    def collect_seconds(self, duration):
        """Blocking: collect samples for `duration` seconds. Returns [(ts, mv_list)]."""
        samples = []
        seen_ts: set = set()
        end = time.time() + duration
        while time.time() < end:
            ts, mv = self.read_latest()
            if ts is not None and ts not in seen_ts:
                samples.append((ts, list(mv)))
                seen_ts.add(ts)
            time.sleep(0.005)
        return samples

    def single_read(self):
        """Request one DATA line. Returns mv_list or None on timeout."""
        self._ser.reset_input_buffer()
        self._ser.write(b"D\n")
        deadline = time.time() + 2.0
        while time.time() < deadline:
            raw = self._ser.readline().decode("ascii", errors="ignore").strip()
            if raw.startswith("DATA,"):
                parts = raw.split(",")
                if len(parts) == N_SENSORS + 2:
                    try:
                        return [float(p) for p in parts[2:]]
                    except ValueError:
                        pass
        return None

    def close(self):
        self.stop()
        time.sleep(0.2)
        self._ser.close()


# ── UI helpers ─────────────────────────────────────────────────────────────────

def _sep(ch="─", n=60):
    print(ch * n)

def _wait(prompt="Press Enter to continue"):
    input(f"\n  >>> {prompt} <<<\n  ")

def _countdown(seconds, msg=""):
    for s in range(seconds, 0, -1):
        print(f"\r  {msg} {s:3d}s ...", end="", flush=True)
        time.sleep(1)
    print()


def _detect_port():
    for p in serial.tools.list_ports.comports():
        if any(k in p.description.lower() for k in ("cp210", "ch340", "uart", "usb")):
            return p.device
    return None


# ── Phase 0: environment checklist ────────────────────────────────────────────

def phase0_checklist():
    _sep("═")
    print("  PHASE 0 — ENVIRONMENT PREPARATION  (Schofield Protocol)")
    _sep()
    print("  Before calibration, confirm all of the following:")
    items = [
        "Housing + aluminium plate + strap assembled correctly for each sensor",
        "Array attached to volunteer's skin at the target location (calf / foot)",
        "Strap tension same as during real use",
        "Patient in the actual measurement posture (e.g., supine, leg extended)",
    ]
    for i, item in enumerate(items, 1):
        print(f"   {i}. {item}")
    print()
    _wait("Confirm all items above, then press Enter to proceed")

    # print()
    # _countdown(600, "Warm-up (sensor equilibrating to skin ~34-37°C)")
    # print("  Warm-up complete. Proceed to Phase 1 (baseline).")


# ── Phase 1: baseline ─────────────────────────────────────────────────────────

def phase1_baseline(stream, sensor_indices):
    """
    Record baseline V0 for the selected sensors.
    Takes BASELINE_N_READINGS readings with BASELINE_INTERVAL_S seconds between each.

    Returns {sensor_name: V0_mv} for sensors that passed.
    """
    _sep("═")
    print("  PHASE 1 — BASELINE  (Zero Reference)")
    _sep()
    print(f"  Sensors: {[SENSOR_NAMES[i] for i in sensor_indices]}")
    print(f"  {BASELINE_N_READINGS} readings, {BASELINE_INTERVAL_S} s apart")
    print("  No extra load on sensors (only strap preload).")
    print("  Press Enter before each reading (wait at least 30 s between readings).")

    readings = {SENSOR_NAMES[i]: [] for i in sensor_indices}

    for k in range(BASELINE_N_READINGS):
        _wait(f"Take reading {k+1}/{BASELINE_N_READINGS}"
              + ("  (≥30 s since last)" if k > 0 else ""))

        # Average 5 rapid snapshots to reduce noise
        mv_sum = [0.0] * N_SENSORS
        n_ok   = 0
        for _ in range(5):
            mv = stream.single_read()
            if mv:
                for j in range(N_SENSORS):
                    mv_sum[j] += mv[j]
                n_ok += 1
            time.sleep(0.2)

        if n_ok == 0:
            print("  ERROR: no response from ESP32")
            continue

        mv_avg = [v / n_ok for v in mv_sum]
        for i in sensor_indices:
            readings[SENSOR_NAMES[i]].append(mv_avg[i])
        readout = "  ".join(f"{SENSOR_NAMES[i]}={mv_avg[i]:.1f}mV"
                            for i in sensor_indices)
        print(f"  {readout}")

    _sep()
    thresh_mv = BASELINE_SD_MAX_PCT / 100.0 * V_CC_MV
    results   = {}
    v0_all    = {}   # V0 for every sensor with enough readings, pass or fail

    for i in sensor_indices:
        name = SENSOR_NAMES[i]
        arr  = np.array(readings[name])
        if len(arr) < 3:
            print(f"  {name}: too few readings — SKIP")
            continue
        V0     = float(arr.mean())
        sd     = float(arr.std())
        sd_pct = sd / V_CC_MV * 100.0
        ok     = sd <= thresh_mv
        status = "PASS" if ok else "FAIL"
        print(f"  {name}: V0={V0:.1f} mV  SD={sd:.1f} mV ({sd_pct:.2f}%)  [{status}]")
        if not ok:
            print(f"    FAIL: SD > {BASELINE_SD_MAX_PCT}% × {V_CC_MV:.0f} mV = {thresh_mv:.0f} mV")
            print("    Check strap tension, housing seating, no extra preload.")
        v0_all[name] = round(V0, 2)
        if ok:
            results[name] = round(V0, 2)

    failed = [SENSOR_NAMES[i] for i in sensor_indices
              if SENSOR_NAMES[i] not in results]
    if failed:
        ans = input(f"\n  {failed} failed. Retry baseline for these? [Y/n] > ").strip().lower()
        if ans in ("", "y", "yes"):
            retry_idx = [SENSOR_NAMES.index(n) for n in failed]
            results.update(phase1_baseline(stream, retry_idx))
        else:
            # Accept the measured V0 despite high SD and continue
            for name in failed:
                if name in v0_all:
                    print(f"  Warning: {name} SD out of spec — using V0={v0_all[name]:.1f} mV anyway.")
                    results[name] = v0_all[name]

    return results


# ── Phase 2: loading protocol ─────────────────────────────────────────────────

def phase2_loading(stream, sensor_idx, V0_mv):
    """
    3-rep loading/unloading protocol for one sensor.
    Returns (V_eff_list, F_list) — training data for log-linear fitting.
    """
    name      = SENSOR_NAMES[sensor_idx]
    force_seq = [(g, "L") for g in LOAD_G] + [(g, "U") for g in UNLOAD_G]

    # CalibBuffer feeds on every incoming M line
    cbuf = CalibBuffer(sensor_idx, V0_mv, WINDOW_SEC)
    stream.add_callback(cbuf.push)

    # Warm up the integral buffer (0.5 s)
    stream.start()
    time.sleep(0.6)

    _sep("═")
    print(f"  PHASE 2 — LOADING PROTOCOL  ({name})")
    _sep()
    print(f"  V0 = {V0_mv:.1f} mV")
    print(f"  Applicator: 38×38 mm silicone island on {name}")
    print(f"  Loading:   {LOAD_G} g")
    print(f"  Unloading: {UNLOAD_G} g")
    print(f"  3 reps — at each level:")
    print(f"    1. Adjust weight to target")
    print(f"    2. Auto-countdown {CREEP_WAIT_S} s for creep")
    print(f"    3. Script captures {CAPTURE_N} readings automatically")

    V_data: list = []
    F_data: list = []

    for rep in range(1, N_REPS + 1):
        _sep("·")
        print(f"  Rep {rep}/{N_REPS}")
        _wait(f"Ready to start Rep {rep}?")

        for f_g, direction in force_seq:
            f_N  = f_g / 1000.0 * 9.81   # grams → Newtons for model
            tag  = "Loading" if direction == "L" else "Unloading"

            if f_g == 0:
                input(f"\n  [{tag}] 0 g — Remove ALL weights, press Enter > ")
                _countdown(POST_UNLOAD_WAIT_S, "Resting")
            else:
                input(f"\n  [{tag}] {f_g} g ({f_N:.2f} N) — Place weight, press Enter > ")
                _countdown(CREEP_WAIT_S, "Creep wait")

            # Collect CAPTURE_N samples spaced CAPTURE_INTERVAL_S apart
            v_effs = []
            for s in range(CAPTURE_N):
                v_eff, _ = cbuf.snapshot()
                v_effs.append(v_eff)
                if s < CAPTURE_N - 1:
                    time.sleep(CAPTURE_INTERVAL_S)

            v_mean = float(np.mean(v_effs))
            V_data.append(v_mean)
            F_data.append(f_N)
            print(f"  {f_g}g ({f_N:.2f}N)  V_eff={v_mean*1000:.1f}mV")

        print(f"\n  Rep {rep} done.  ({len(V_data)} training points so far)")

    stream.stop()
    # Remove callback to avoid interfering with later sensors
    stream._callbacks = [cb for cb in stream._callbacks if cb is not cbuf.push]

    return V_data, F_data


# ── Curve fitting ──────────────────────────────────────────────────────────────

def fit_all_sensors(datasets, existing_cal=None):
    """
    Fit log-linear model (Linde et al. 2021, Eq. 1) for each sensor.
    datasets: {name: {"V0_mv": float, "V": list, "F": list}}
    Merges with existing_cal so previously calibrated sensors are preserved.
    Saves to a new timestamped folder under calibrations/.
    Returns (cal dict, cal_folder Path).
    """
    _sep("═")
    print("  CURVE FITTING  (Log-Linear, Linde et al. 2021)")
    _sep()
    print(f"  {'Sensor':6}  {'n':>4}  {'RMSE(N)':>8}  {'R²':>8}  {'c1':>10}  {'c0':>8}  Status")
    _sep()

    cal = {"model": "log_linear", "sensors": {}}
    if existing_cal:
        cal["sensors"] = dict(existing_cal.get("sensors", {}))

    for name, d in datasets.items():
        V = np.array(d["V"], dtype=float)
        F = np.array(d["F"], dtype=float)

        if len(V) < 3:
            print(f"  {name:6}  — too few points ({len(V)}), skipping")
            continue

        c1, c0, rmse, r2 = fit_log_linear(V, F)
        ok = rmse < RMSE_MAX_N and not np.isnan(r2)
        status = "OK" if ok else "RMSE>3N"
        print(f"  {name:6}  {len(V):>4}  {rmse:>8.3f}  {r2:>8.5f}  {c1:>10.4f}  {c0:>8.4f}  {status}")
        if r2 < 0.99:
            print(f"  {'':6}  ⚠  R²={r2:.4f} < 0.99 — consider re-running calibration")

        cal["sensors"][name] = {
            "ch":    SENSOR_NAMES.index(name),
            "V0_mv": round(d["V0_mv"], 2),
            "c1":    round(c1, 8),
            "c0":    round(c0, 8),
            "fit":   {"rmse_N": round(rmse, 4), "r2": round(r2, 6), "n": len(V)},
        }

    cal_folder = new_calibration_dir()
    (cal_folder / "calibration_coeffs.json").write_text(json.dumps(cal, indent=2))
    print(f"\n  Saved → {cal_folder / 'calibration_coeffs.json'}")
    return cal, cal_folder


# ── Phase 3: validation ────────────────────────────────────────────────────────

def phase3_validate(stream, cal, cal_folder=None):
    """
    Per-sensor and aggregate validation.
    Pass criteria: RMSE < 3 N and %error < 10% per sensor; RMSE_total < 5 N.
    """
    _sep("═")
    print("  PHASE 3 — VALIDATION")
    _sep()
    print(f"  Validation weights: {VAL_LOADS_G} g  (not in calibration set)")
    print(f"  Pass: RMSE < {RMSE_MAX_N} N  and  %error < {PCT_ERROR_MAX}%  per sensor")
    print(f"        RMSE_total < {RMSE_TOTAL_MAX_N} N  for full array")

    if "sensors" not in cal or not cal["sensors"]:
        print("  No calibrated sensors found. Run option 1 first.")
        return

    engine  = ForceEngine(cal["sensors"])
    results = {}

    stream.add_callback(lambda ts, mv: engine.update(ts, mv))
    stream.start()
    time.sleep(0.6)

    # 3a: per-sensor
    for name in sorted(cal["sensors"], key=lambda n: cal["sensors"][n]["ch"]):
        _sep("·")
        print(f"  Sensor: {name}  — place 38×38 mm applicator on this sensor's island only")
        _wait(f"Ready to validate {name}?")

        errs, pcts = [], []
        for f_g in VAL_LOADS_G:
            f_true = f_g / 1000.0 * 9.81   # grams → Newtons
            input(f"\n  Apply {f_g} g ({f_true:.2f} N). Press Enter when on > ")
            _countdown(CREEP_WAIT_S, "Creep wait")

            readings = stream.collect_seconds(5.0)
            if not readings:
                print("  WARNING: no data received")
                continue

            f_preds = []
            for ts, mv in readings[-CAPTURE_N:]:
                forces = engine.update(ts, mv)
                f_preds.append(forces.get(name, 0.0))

            f_pred = float(np.mean(f_preds))
            err    = abs(f_pred - f_true)
            pct    = err / f_true * 100.0 if f_true > 0 else 0.0
            errs.append(err)
            pcts.append(pct)
            print(f"  F_pred={f_pred:.2f}N  F_true={f_true:.2f}N  "
                  f"err={err:.2f}N  ({pct:.1f}%)")
            input("  Remove weight. Press Enter > ")

        if errs:
            rmse    = float(np.sqrt(np.mean(np.array(errs) ** 2)))
            max_pct = float(max(pcts))
            ok      = rmse < RMSE_MAX_N and max_pct < PCT_ERROR_MAX
            status  = "PASS" if ok else "FAIL"
            results[name] = {"rmse": rmse, "max_pct": max_pct, "pass": ok}
            print(f"\n  {name}: RMSE={rmse:.2f}N  max%={max_pct:.1f}%  [{status}]")

    # 3b: aggregate
    _sep("·")
    print("  Aggregate — distribute weight across full array")
    agg_errs = []
    for f_g in VAL_LOADS_G:
        f_true = f_g / 1000.0 * 9.81
        input(f"\n  Apply {f_g} g total ({f_true:.2f} N) spread over array. Enter > ")
        _countdown(CREEP_WAIT_S, "Stabilising")

        readings = stream.collect_seconds(5.0)
        if not readings:
            continue

        ftots = []
        for ts, mv in readings[-CAPTURE_N:]:
            forces = engine.update(ts, mv)
            ftots.append(sum(forces.values()))

        ftot = float(np.mean(ftots))
        err  = abs(ftot - f_true)
        agg_errs.append(err)
        print(f"  F_total={ftot:.2f}N  F_true={f_true}N  err={err:.2f}N")
        input("  Remove weight. Press Enter > ")

    if agg_errs:
        rmse_tot = float(np.sqrt(np.mean(np.array(agg_errs) ** 2)))
        ok_tot   = rmse_tot < RMSE_TOTAL_MAX_N
        status   = "PASS" if ok_tot else "FAIL"
        results["_aggregate"] = {"rmse": rmse_tot, "pass": ok_tot}
        print(f"\n  Aggregate RMSE={rmse_tot:.2f}N  [{status}]")

    # Summary
    _sep()
    print("  Summary:")
    for name, r in results.items():
        flag = "PASS" if r["pass"] else "FAIL"
        if name == "_aggregate":
            print(f"    Aggregate: RMSE={r['rmse']:.2f}N  [{flag}]")
        else:
            print(f"    {name:6}: RMSE={r['rmse']:.2f}N  max%={r['max_pct']:.1f}%  [{flag}]")

    folder = cal_folder or find_latest_calibration_path().parent
    out    = folder / "validation_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\n  Saved → {out}")

    stream.stop()


# ── Baseline drift recheck ─────────────────────────────────────────────────────

def recheck_baseline(stream, cal):
    """Compare current no-load readings to saved V0. Flag sensors with >2% drift."""
    _sep("═")
    print("  BASELINE DRIFT RECHECK")
    _sep()
    print("  Patient lying still, NO extra load on any sensor.")
    _wait("Patient in position?")

    print("  Collecting 10 s of data ...")
    stream.start()
    samples = stream.collect_seconds(10.0)
    stream.stop()

    if not samples:
        print("  ERROR: no data received.")
        return

    vs_arr = np.array([mv for _, mv in samples])  # (N_samples, 6)
    means  = vs_arr.mean(axis=0)                  # mV per channel

    thresh_mv = 2.0 / 100.0 * V_CC_MV
    needs_recal = False

    _sep()
    print(f"  {'Sensor':6}  {'Saved V0(mV)':>13}  {'Now(mV)':>9}  "
          f"{'Drift(mV)':>10}  {'Drift%':>7}  Status")
    _sep()

    for name, sensor in sorted(cal["sensors"].items(),
                                key=lambda kv: kv[1]["ch"]):
        ch      = sensor["ch"]
        orig    = sensor["V0_mv"]
        current = float(means[ch])
        drift   = abs(current - orig)
        pct     = drift / V_CC_MV * 100.0
        ok      = drift <= thresh_mv
        if not ok:
            needs_recal = True
        status  = "OK" if ok else "RECALIBRATE"
        print(f"  {name:6}  {orig:>13.1f}  {current:>9.1f}  "
              f"{drift:>10.1f}  {pct:>6.2f}%  {status}")

    print()
    if needs_recal:
        print("  ACTION: Drift > 2% on one or more sensors.")
        print("  Run full recalibration (option 1) for those sensors.")
    else:
        print("  All sensors within ±2% tolerance — proceed with session.")


# ── Main menu ──────────────────────────────────────────────────────────────────

def main():
    port = sys.argv[1] if len(sys.argv) > 1 else _detect_port()
    if port is None:
        sys.exit("Cannot detect ESP32 port.\n"
                 "Run:  python calibrate.py /dev/ttyUSB0")

    stream = SerialStream(port)

    # Quick connection check
    print("  Testing connection ...", end="", flush=True)
    mv = stream.single_read()
    if mv is None:
        stream.close()
        sys.exit("\n  ERROR: no response from ESP32.\n"
                 "  Ensure fsr_11may.ino is flashed and Arduino Serial Monitor is closed.")
    print(f" OK  ({[f'{v:.0f}' for v in mv]} mV)\n")

    current_cal_path = find_latest_calibration_path()   # tracks most recently used

    while True:
        _sep("═")
        print("  FSR406 × 6  Log-Linear Calibration  (fsr_11may)")
        _sep("═")
        if current_cal_path:
            print(f"  Active calibration: {current_cal_path.parent.name}")
        else:
            print("  Active calibration: none")
        _sep()
        print("  1. New calibration session   (Phase 0 → 1 → 2 → fit)")
        print("  2. Validate calibration      (Phase 3)")
        print("  3. Recheck baseline drift")
        print("  q. Quit")
        _sep()
        choice = input("  Choose > ").strip().lower()

        if choice == "1":
            print("\n  Which sensors to calibrate?")
            print("  Enter numbers separated by spaces  (e.g. 1 2 3)  or Enter for all 6")
            ans = input("  > ").strip()
            if ans:
                indices = [int(x) - 1 for x in ans.split()
                           if x.isdigit() and 1 <= int(x) <= N_SENSORS]
            else:
                indices = list(range(N_SENSORS))
            if not indices:
                print("  No valid sensors selected.")
                continue

            phase0_checklist()

            # Phase 1: baseline (using single D readings, no streaming)
            V0_dict = phase1_baseline(stream, indices)
            if not V0_dict:
                print("  No valid readings collected. Check ESP32 connection and retry.")
                continue

            # Phase 2: loading per sensor
            datasets = {}
            for idx in indices:
                name = SENSOR_NAMES[idx]
                if name not in V0_dict:
                    print(f"\n  {name}: no valid baseline — skipping loading phase.")
                    continue
                _sep("═")
                print(f"  ══  {name}  ({list(V0_dict.keys()).index(name)+1}"
                      f" of {len(V0_dict)})  ══")
                _wait(f"Baseline done for {name}. "
                      "Attach applicator to this sensor, then press Enter.")

                V_list, F_list = phase2_loading(stream, idx, V0_dict[name])
                datasets[name] = {
                    "V0_mv": V0_dict[name],
                    "V": V_list,
                    "F": F_list,
                }
                print(f"\n  {name}: {len(F_list)} calibration points collected.")

            if not datasets:
                print("  No data collected.")
                continue

            # Load existing calibration to preserve sensors not updated this session
            existing = None
            if current_cal_path and current_cal_path.exists():
                try:
                    existing = json.loads(current_cal_path.read_text())
                except Exception:
                    pass

            _wait("All sensors done. Press Enter to run curve fitting.")
            _, cal_folder   = fit_all_sensors(datasets, existing)
            current_cal_path = cal_folder / "calibration_coeffs.json"

        elif choice == "2":
            if current_cal_path is None:
                print("  No calibration found. Run option 1 first.")
                continue
            print(f"  Using calibration: {current_cal_path.parent.name}")
            cal = json.loads(current_cal_path.read_text())
            phase3_validate(stream, cal, cal_folder=current_cal_path.parent)

        elif choice == "3":
            if current_cal_path is None:
                print("  No calibration found. Run option 1 first.")
                continue
            print(f"  Using calibration: {current_cal_path.parent.name}")
            cal = json.loads(current_cal_path.read_text())
            recheck_baseline(stream, cal)

        elif choice in ("q", "quit", "exit"):
            break
        else:
            print("  Invalid choice — type 1, 2, 3, or q")

    stream.close()
    print("\n  Bye.")


if __name__ == "__main__":
    main()
