#!/usr/bin/env python3
"""
calibrate.py — FSR406 × 6  Step-by-step calibration (Phases 2–7)

Requires: fsr406_6ch_calibration.ino flashed to ESP32
          (the R command added in the latest version)

Usage:
    conda activate massage
    cd /home/primpunn/experiment/FSR/fsr406_6ch_calibration
    python calibrate.py
    python calibrate.py /dev/ttyUSB0      # if auto-detect fails
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

try:
    import serial
    import serial.tools.list_ports
except ImportError:
    sys.exit("Run: pip install pyserial")

# ── Constants ─────────────────────────────────────────────────────────────────

N_SENSORS    = 6
NAMES        = [f"FSR{i+1}" for i in range(N_SENSORS)]
BAUD         = 115200
V_CC_MV      = 3300.0

N_BASELINE        = 10     # readings per baseline session
N_CAPTURE         = 10     # readings per force level (500 ms each = 5 s window)
CAPTURE_INTERVAL  = 0.5    # seconds between capture readings

LOAD_N    = [0, 5, 10, 20, 30, 40, 50, 60, 80]
UNLOAD_N  = [60, 50, 40, 30, 20, 10, 5, 0]
ALL_FORCES   = LOAD_N + UNLOAD_N          # 17 levels per rep (80 appears once)
DIRECTIONS   = ['L'] * len(LOAD_N) + ['U'] * len(UNLOAD_N)
N_REPS       = 3

VAL_FORCES        = [8, 15, 25, 45, 70]  # N — not in calibration set
BASELINE_SD_MAX   = 2.0   # % of full scale
RMSE_SINGLE_MAX   = 3.0   # N
PCT_SINGLE_MAX    = 10.0  # %
RMSE_TOTAL_MAX    = 5.0   # N

CAL_FILE = Path("calibration_6ch.json")

# ── Serial helpers ────────────────────────────────────────────────────────────

def detect_port():
    for p in serial.tools.list_ports.comports():
        if any(k in p.description.lower() for k in ('cp210', 'ch340', 'uart', 'usb')):
            return p.device
    return None


class ESP32:
    def __init__(self, port):
        print(f"  Opening port {port} ...", end='', flush=True)
        self.ser = serial.Serial(port, BAUD, timeout=1)
        print(" OK")
        print("  Waiting 2 s for ESP32 to boot ...", end='', flush=True)
        time.sleep(2)
        self.ser.reset_input_buffer()
        print(" done.")

    def read_once(self):
        """Send R, return [v0_mv … v5_mv] as floats, or None on timeout."""
        self.ser.reset_input_buffer()
        self.ser.write(b'D\n')
        deadline = time.time() + 2.0
        while time.time() < deadline:
            raw = self.ser.readline().decode('ascii', errors='ignore').strip()
            if raw.startswith('DATA,'):
                parts = raw.split(',')
                if len(parts) == 7:
                    try:
                        return [float(p) for p in parts[1:]]
                    except ValueError:
                        pass
        return None

    def read_average(self, n=N_CAPTURE, interval=CAPTURE_INTERVAL):
        """n readings spaced interval seconds apart. Returns mean per channel."""
        samples = []
        for i in range(n):
            v = self.read_once()
            if v is not None:
                samples.append(v)
            if i < n - 1:
                time.sleep(interval)
        if not samples:
            return None
        return list(np.mean(samples, axis=0))

    def close(self):
        self.ser.close()


# ── UI helpers ────────────────────────────────────────────────────────────────

def line(ch='─', n=58):
    print(ch * n)

def wait(prompt="Press Enter to continue"):
    input(f"\n  >>> {prompt} <<<\n  ")

def countdown(seconds, msg=""):
    for s in range(seconds, 0, -1):
        print(f"\r  {msg} {s:2d}s ...", end='', flush=True)
        time.sleep(1)
    print()


# ── Curve fit (power law) ─────────────────────────────────────────────────────

def _power_law(V, a, b):
    return a * np.power(np.clip(V, 1e-9, None), b)

def fit_power_law(Veff_V, F_N):
    """F = a * Veff_V^b. Returns (a, b, r2) or (None, None, None)."""
    mask = (Veff_V > 0) & (F_N > 0)
    if mask.sum() < 3:
        return None, None, None
    try:
        popt, _ = curve_fit(_power_law, Veff_V[mask], F_N[mask],
                            p0=[1.0, 1.5], maxfev=5000)
        a, b   = popt
        F_pred = _power_law(Veff_V[mask], a, b)
        ss_res = np.sum((F_N[mask] - F_pred) ** 2)
        ss_tot = np.sum((F_N[mask] - F_N[mask].mean()) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        return float(a), float(b), r2
    except RuntimeError:
        return None, None, None

def predict_force(coef, v_mv, v0_mv):
    """Predict F given raw voltage v_mv and baseline v0_mv."""
    veff_v = max((v_mv - v0_mv) / 1000.0, 0.0)
    if veff_v <= 0:
        return 0.0
    return float(_power_law(np.array([veff_v]), coef['a'], coef['b'])[0])


# ── Phase 2: Baseline ─────────────────────────────────────────────────────────

def phase2_baseline(esp, sensor_idx):
    """
    Take N_BASELINE readings for one sensor.
    User presses Enter before each reading (controls the 30 s gap themselves).
    Returns V0_mv (float) or None if failed.
    """
    name = NAMES[sensor_idx]
    ch   = sensor_idx

    line('═')
    print(f"  PHASE 2 — BASELINE  ({name})")
    line()
    print(f"  Sensor attached on patient's leg. No extra force applied.")
    print(f"  You will take {N_BASELINE} readings, each at least 30 seconds apart.")
    print(f"  Press Enter before each reading when you are ready.")

    readings = []

    for i in range(N_BASELINE):
        wait(f"Take reading {i+1}/{N_BASELINE}  (wait ≥30 s since last reading)")

        print(f"  Reading ...", end='', flush=True)
        vs = esp.read_once()
        if vs is None:
            print(" FAILED — no response from ESP32. Check firmware + port.")
            continue

        v_mv = vs[ch]
        readings.append(v_mv)
        print(f" {name} = {v_mv:.1f} mV")

    if len(readings) < 5:
        print(f"\n  ERROR: only {len(readings)} valid readings. Cannot compute baseline.")
        return None

    arr    = np.array(readings)
    V0     = float(arr.mean())
    sd     = float(arr.std())
    sd_pct = sd / V_CC_MV * 100.0
    thresh = BASELINE_SD_MAX / 100.0 * V_CC_MV

    line()
    print(f"  Baseline result for {name}:")
    print(f"    V₀  = {V0:.1f} mV")
    print(f"    SD  = {sd:.1f} mV  ({sd_pct:.2f}%)")
    print(f"    Threshold: SD < {BASELINE_SD_MAX:.0f}% × {V_CC_MV:.0f} mV = {thresh:.0f} mV")

    if sd_pct < BASELINE_SD_MAX:
        print(f"    → PASS ✓")
        return V0
    else:
        print(f"    → FAIL ✗  (SD too high)")
        print(f"    Check: strap tension, housing seating, no extra preload.")
        ans = input("\n  Retry baseline for this sensor? [Y/n] > ").strip().lower()
        if ans in ('', 'y', 'yes'):
            return phase2_baseline(esp, sensor_idx)
        return None


# ── Phase 3: Loading protocol ─────────────────────────────────────────────────

def phase3_loading(esp, sensor_idx, V0_mv):
    """
    3-rep loading/unloading for one sensor.
    User presses Enter when each weight is placed and 10 s have passed.
    Returns (F_list, Veff_V_list).
    """
    name = NAMES[sensor_idx]
    ch   = sensor_idx

    line('═')
    print(f"  PHASE 3 — LOADING PROTOCOL  ({name})")
    line()
    print(f"  V₀ = {V0_mv:.1f} mV")
    print(f"  Place 38×38 mm applicator plate on {name}'s island.")
    print(f"  Loading:   {LOAD_N} N")
    print(f"  Unloading: {UNLOAD_N} N")
    print(f"  Reps: {N_REPS}")
    print()
    print(f"  For each force level:")
    print(f"    1. Place (or remove) weights to reach the target")
    print(f"    2. Wait 10 seconds for creep to settle")
    print(f"    3. Press Enter  → script records {N_CAPTURE} readings automatically")

    all_F    = []
    all_Veff = []

    for rep in range(1, N_REPS + 1):
        line('·')
        print(f"  Rep {rep}/{N_REPS}")
        wait(f"Ready to start Rep {rep}?")

        for f_N, direction in zip(ALL_FORCES, DIRECTIONS):
            weight_g = f_N * 1000.0 / 9.81
            tag      = "Loading" if direction == 'L' else "Unloading"

            print(f"\n  [{tag}]  Target: {f_N} N  ({weight_g:.0f} g)")
            if f_N == 0:
                input("  → Remove ALL weights, then press Enter > ")
            else:
                input(f"  → Place {weight_g:.0f} g on applicator, wait 10 s, then press Enter > ")

            print(f"  Collecting {N_CAPTURE} readings ({N_CAPTURE * CAPTURE_INTERVAL:.0f} s) ...",
                  end='', flush=True)
            means = esp.read_average(N_CAPTURE, CAPTURE_INTERVAL)

            if means is None:
                print(" FAILED — skipping this level.")
                continue

            v_mv   = means[ch]
            veff_v = max((v_mv - V0_mv) / 1000.0, 0.0)
            print(f"  V={v_mv:.1f} mV   Veff={veff_v*1000:.1f} mV")

            all_F.append(float(f_N))
            all_Veff.append(veff_v)

        print(f"\n  Rep {rep} done.")

    return all_F, all_Veff


# ── Phase 4: Curve fitting ────────────────────────────────────────────────────

def phase4_fit(new_sensor_data):
    """
    Fit power-law F = a·Veff^b per sensor.
    Merges with existing calibration_6ch.json so previously calibrated
    sensors are not lost.

    new_sensor_data: { name: {'V0_mv': float, 'F': list, 'Veff_V': list} }
    """
    line('═')
    print("  PHASE 4 — CURVE FITTING  (F = a · Veff^b)")
    line()

    # Load existing calibration to preserve sensors not calibrated this session
    existing = {}
    if CAL_FILE.exists():
        try:
            existing = json.loads(CAL_FILE.read_text()).get("sensors", {})
        except Exception:
            pass

    calibration = {"model": "power_law", "sensors": dict(existing)}

    for name, d in new_sensor_data.items():
        F    = np.array(d['F'],      dtype=float)
        Veff = np.array(d['Veff_V'], dtype=float)
        V0   = d['V0_mv']

        a, b, r2 = fit_power_law(Veff, F)

        if a is None:
            print(f"  {name}: FIT FAILED — not enough non-zero points")
            continue

        calibration["sensors"][name] = {
            "ch":    NAMES.index(name),
            "V0_mv": round(V0, 2),
            "a":     round(a, 6),
            "b":     round(b, 6),
            "r2":    round(r2, 5),
        }
        status = "OK" if r2 >= 0.99 else "⚠  R² < 0.99 — consider more calibration points"
        print(f"  {name}: a={a:.4f}  b={b:.4f}  R²={r2:.4f}  V₀={V0:.1f} mV  [{status}]")

    CAL_FILE.write_text(json.dumps(calibration, indent=2))
    print(f"\n  Saved → {CAL_FILE}")
    return calibration


# ── Phase 5: Validation ───────────────────────────────────────────────────────

def phase5_validate(esp):
    if not CAL_FILE.exists():
        print("  No calibration file found. Run option 1 first.")
        return
    cal = json.loads(CAL_FILE.read_text())

    line('═')
    print("  PHASE 5 — VALIDATION")
    line()
    print(f"  Validation forces: {VAL_FORCES} N  (not used in calibration)")
    print(f"  Pass criteria: RMSE < {RMSE_SINGLE_MAX} N per sensor,  RMSE_total < {RMSE_TOTAL_MAX} N")

    results = {}

    # 5a: per sensor
    for name, coef in sorted(cal["sensors"].items(), key=lambda kv: kv[1]["ch"]):
        ch = coef["ch"]
        line('·')
        print(f"  Sensor: {name}  — place applicator on this sensor only")
        wait(f"Ready to validate {name}?")

        errs, pcts = [], []
        for f_true in VAL_FORCES:
            weight_g = f_true * 1000.0 / 9.81
            input(f"\n  Apply {f_true} N ({weight_g:.0f} g). Press Enter when weight is on > ")
            countdown(10, "Waiting for creep")

            means = esp.read_average()
            if means is None:
                print("  WARNING: no reading received.")
                continue

            f_pred = predict_force(coef, means[ch], coef["V0_mv"])
            err    = abs(f_pred - f_true)
            pct    = err / f_true * 100.0 if f_true > 0 else 0.0
            errs.append(err)
            pcts.append(pct)
            print(f"  F_pred={f_pred:.2f} N   F_true={f_true} N   "
                  f"err={err:.2f} N  ({pct:.1f}%)")

            input("  Remove weight. Press Enter when done > ")

        if errs:
            rmse    = float(np.sqrt(np.mean(np.array(errs) ** 2)))
            max_pct = float(max(pcts))
            ok      = rmse < RMSE_SINGLE_MAX and max_pct < PCT_SINGLE_MAX
            results[name] = {"rmse": rmse, "max_pct": max_pct, "pass": ok}
            print(f"\n  {name}: RMSE={rmse:.2f} N   max%={max_pct:.1f}%   "
                  f"[{'PASS ✓' if ok else 'FAIL ✗'}]")

    # 5b: aggregate
    line('·')
    print("  Aggregate — distribute total weight across ALL sensors")
    agg_errs = []
    for f_true in VAL_FORCES:
        weight_g = f_true * 1000.0 / 9.81
        input(f"\n  Apply {f_true} N total ({weight_g:.0f} g) spread over whole array. "
              f"Press Enter when on > ")
        countdown(10, "Waiting for creep")

        means = esp.read_average()
        if means is None:
            continue
        ftot = sum(
            predict_force(c, means[c["ch"]], c["V0_mv"])
            for c in cal["sensors"].values()
        )
        err = abs(ftot - f_true)
        agg_errs.append(err)
        print(f"  F_total={ftot:.2f} N   F_true={f_true} N   err={err:.2f} N")
        input("  Remove weight. Press Enter > ")

    if agg_errs:
        rmse_tot = float(np.sqrt(np.mean(np.array(agg_errs) ** 2)))
        ok_tot   = rmse_tot < RMSE_TOTAL_MAX
        results["_aggregate"] = {"rmse": rmse_tot, "pass": ok_tot}
        print(f"\n  Aggregate RMSE={rmse_tot:.2f} N  "
              f"[{'PASS ✓' if ok_tot else 'FAIL ✗'}]")

    Path("validation_results.json").write_text(json.dumps(results, indent=2))
    print("  Saved → validation_results.json")


# ── Phase 6: Monitor ──────────────────────────────────────────────────────────

def phase6_monitor(esp):
    if not CAL_FILE.exists():
        print("  No calibration file found. Run option 1 first.")
        return
    cal   = json.loads(CAL_FILE.read_text())
    names = sorted(cal["sensors"].keys(), key=lambda n: cal["sensors"][n]["ch"])

    line('═')
    print("  PHASE 6 — REAL-TIME MONITOR  (Ctrl+C to stop)")
    line()

    try:
        while True:
            vs = esp.read_once()
            if vs is None:
                continue
            parts = []
            ftot  = 0.0
            for name in names:
                coef  = cal["sensors"][name]
                f     = predict_force(coef, vs[coef["ch"]], coef["V0_mv"])
                ftot += f
                parts.append(f"{name}={f:5.1f}N")
            print("\r  " + "  ".join(parts) + f"  │  Total={ftot:6.1f}N", end='', flush=True)
    except KeyboardInterrupt:
        print("\n  Stopped.")


# ── Phase 7: Recheck baseline ─────────────────────────────────────────────────

def phase7_recheck(esp):
    if not CAL_FILE.exists():
        print("  No calibration file found. Run option 1 first.")
        return
    cal = json.loads(CAL_FILE.read_text())

    line('═')
    print("  PHASE 7 — BASELINE DRIFT CHECK")
    line()
    print("  Patient lying still, NO extra load on any sensor.")
    wait("Patient in position, no load?")

    print("  Taking 10 readings ...")
    samples = []
    for i in range(10):
        vs = esp.read_once()
        if vs:
            samples.append(vs)
        time.sleep(0.5)

    if not samples:
        print("  ERROR: no data received.")
        return

    current = np.array(samples).mean(axis=0)
    thresh  = BASELINE_SD_MAX / 100.0 * V_CC_MV

    line()
    print(f"  {'Sensor':6}  {'Saved V₀(mV)':>13}  {'Now(mV)':>9}  "
          f"{'Drift(mV)':>10}  {'Drift%':>7}  Status")
    print(f"  {'──────':6}  {'─────────────':>13}  {'───────':>9}  "
          f"{'──────────':>10}  {'──────':>7}  ──────")

    needs_recal = False
    for name, coef in sorted(cal["sensors"].items(), key=lambda kv: kv[1]["ch"]):
        ch    = coef["ch"]
        orig  = coef["V0_mv"]
        cur   = float(current[ch])
        drift = abs(cur - orig)
        pct   = drift / V_CC_MV * 100.0
        ok    = drift < thresh
        if not ok:
            needs_recal = True
        status = "OK" if ok else "RECALIBRATE"
        print(f"  {name:6}  {orig:>13.1f}  {cur:>9.1f}  "
              f"{drift:>10.1f}  {pct:>6.2f}%  {status}")

    print()
    if needs_recal:
        print("  → Some sensors drifted > 2%.  Re-run option 1 for those sensors.")
    else:
        print("  → All OK. Proceed to option 3 (monitor).")


# ── Main menu ─────────────────────────────────────────────────────────────────

def main():
    port = sys.argv[1] if len(sys.argv) > 1 else detect_port()
    if port is None:
        sys.exit("Cannot detect ESP32 port.\n"
                 "Run:  python calibrate.py /dev/ttyUSB0")

    esp = ESP32(port)

    # Quick connection test
    print("  Testing connection ...", end='', flush=True)
    test = esp.read_once()
    if test is None:
        esp.close()
        sys.exit("\n  ERROR: ESP32 did not respond to R command.\n"
                 "  Make sure fsr406_6ch_calibration.ino is flashed and Arduino Serial Monitor is CLOSED.")
    print(f" OK  (voltages: {[f'{v:.0f}' for v in test]} mV)\n")

    while True:
        line('═')
        print("  FSR406 × 6 — Calibration Menu")
        line('═')
        print("  1. New calibration session   (Phase 2 + 3 + 4)")
        print("  2. Validate calibration       (Phase 5)")
        print("  3. Monitor live forces        (Phase 6)")
        print("  4. Recheck baseline drift     (Phase 7)")
        print("  q. Quit")
        line()
        choice = input("  Choose > ").strip().lower()

        if choice == '1':
            # Choose which sensors
            print("\n  Which sensors to calibrate?")
            print("  Enter numbers separated by space  (e.g. 1 2 3)  or Enter for ALL 6")
            ans = input("  > ").strip()
            if ans:
                indices = [int(x) - 1 for x in ans.split()
                           if x.isdigit() and 1 <= int(x) <= N_SENSORS]
            else:
                indices = list(range(N_SENSORS))

            if not indices:
                print("  No valid sensors selected.")
                continue

            new_data = {}
            for idx in indices:
                name = NAMES[idx]
                line('═')
                print(f"  ══  {name}  ({indices.index(idx)+1} of {len(indices)})  ══")

                V0 = phase2_baseline(esp, idx)
                if V0 is None:
                    print(f"  Skipping {name} loading phase.")
                    continue

                wait(f"Baseline done. Press Enter to start Phase 3 loading for {name}")
                F_list, Veff_list = phase3_loading(esp, idx, V0)
                new_data[name] = {"V0_mv": V0, "F": F_list, "Veff_V": Veff_list}
                print(f"\n  {name} data collected ({len(F_list)} points).")

            if new_data:
                wait("All sensors done. Press Enter to run curve fitting (Phase 4)")
                phase4_fit(new_data)
            else:
                print("  No data collected.")

        elif choice == '2':
            phase5_validate(esp)

        elif choice == '3':
            phase6_monitor(esp)

        elif choice == '4':
            phase7_recheck(esp)

        elif choice in ('q', 'quit', 'exit'):
            break
        else:
            print("  Invalid choice — type 1, 2, 3, 4, or q")

    esp.close()
    print("\n  Bye.")


if __name__ == '__main__':
    main()
