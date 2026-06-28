#!/usr/bin/env python3
"""
fsr406_6ch_phase4.py — Phases 4-7 for FSR406 × 6 Array

Model (Hall Option B, per sensor):
  F(t) = a0 + a1·V + a2·V² + a3·V³ + a4·V⁴
             + b1·I + b2·I² + b3·I³ + b4·I⁴
  V = voltage in Volts (v_mv / 1000)
  I = linearly-weighted moving integral of V over past WINDOW_S seconds (V·s)

Subcommands
-----------
  fit       Phase 4  Parse session log, fit Hall model per sensor, save calibration_6ch.json
  validate  Phase 5  Live serial: guided per-sensor + aggregate validation
  monitor   Phase 6  Live serial: real-time F_total display
  recheck   Phase 7  Live serial: baseline drift check (no-load recheck)

Usage
-----
  conda activate massage
  cd /home/primpunn/experiment/FSR/fsr406_6ch_calibration

  # Save log while running ESP32 (all 6 sensors in one session):
  python -m serial.tools.miniterm --raw /dev/ttyUSB0 115200 | tee session.txt

  python fsr406_6ch_phase4.py fit session.txt
  python fsr406_6ch_phase4.py fit session.txt --window 1.5   # try longer window if R²<0.99
  python fsr406_6ch_phase4.py fit session.txt --sensors 1 3  # fit only FSR1 and FSR3

  python fsr406_6ch_phase4.py validate --port /dev/ttyUSB0
  python fsr406_6ch_phase4.py monitor  --port /dev/ttyUSB0
  python fsr406_6ch_phase4.py recheck  --port /dev/ttyUSB0

Force sequence used in Phase 3 (edit if yours differs):
  Loading:    0 → 5 → 10 → 20 → 30 → 40 → 50 → 60 → 80 N
  Unloading: 80 → 60 → 50 → 40 → 30 → 20 → 10 →  5 →  0 N
  → 3 reps per sensor → 51 capture samples per sensor

Validation levels (Phase 5, not in calibration sequence):
  8, 15, 25, 45, 70 N
"""

import argparse
import collections
import json
import re
import sys
import threading
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

try:
    import serial
    import serial.tools.list_ports
except ImportError:
    sys.exit("pyserial not found.  pip install pyserial")

# ─── Protocol constants ────────────────────────────────────────────────────────

N_SENSORS       = 6
SENSOR_NAMES    = [f"FSR{i+1}" for i in range(N_SENSORS)]
BAUD_RATE       = 115200
LOG_INTERVAL_MS = 100          # 10 Hz (must match firmware LOG_INTERVAL_MS)
DEFAULT_WINDOW  = 1.0          # integral window in seconds
V_CC_MV         = 3300

# Phase 3 force sequence (edit if protocol differs)
FORCE_SEQ_N = [
    0, 5, 10, 20, 30, 40, 50, 60, 80,   # loading  (9 levels)
    60, 50, 40, 30, 20, 10,  5,  0,     # unloading (8 levels)
]
DIRECTION = ["loading"] * 9 + ["unloading"] * 8
SEQ_LEN   = len(FORCE_SEQ_N)

# Phase 5 validation (must NOT overlap with FORCE_SEQ_N non-zero values)
VALIDATION_LEVELS_N = [8, 15, 25, 45, 70]

# Acceptance thresholds
RMSE_SINGLE_MAX  = 3.0   # N
PCT_SINGLE_MAX   = 10.0  # %
RMSE_TOTAL_MAX   = 5.0   # N
BASELINE_DRIFT   = 0.02  # 2% of full scale → recalibrate

# ─── Log parsing ───────────────────────────────────────────────────────────────

_CAP_RE  = re.compile(r'^CH(\d)_CAP_R(\d+)_(\d+)$')   # CH<ch>_CAP_R<rep>_<sample>
_BL_RE   = re.compile(r'^S,BASELINE,CH(\d)_result,([\d.]+),([\d.]+),(\d+),(PASS|FAIL)')

def parse_log(path: str, window_s: float, sensor_filter=None):
    """
    Parse a session log file.

    Returns
    -------
    baselines : dict  {ch_1based: {'mean_mv': float, 'sd_mv': float, 'pass': bool}}
    datasets  : dict  {ch_1based: (X np.ndarray, y np.ndarray, meta list)}
    """
    baselines = {}
    rows      = []   # (ts_ms, ch_1based, v_V, tag)

    with open(path, encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.strip()

            # Baseline summary line
            m = _BL_RE.match(line)
            if m:
                ch       = int(m.group(1))
                mean_mv  = float(m.group(2))
                sd_mv    = float(m.group(3))
                passed   = m.group(5) == "PASS"
                baselines[ch] = {"mean_mv": mean_mv, "sd_mv": sd_mv, "pass": passed}
                continue

            # Data line: D,ts_ms,ch,raw_adc,v_mv,tag
            if line.startswith("D,"):
                parts = line.split(",")
                if len(parts) < 6:
                    continue
                try:
                    ts_ms = int(parts[1])
                    ch    = int(parts[2])
                    v_V   = float(parts[4]) / 1000.0   # mV → Volts
                    tag   = parts[5].strip()
                    rows.append((ts_ms, ch, v_V, tag))
                except (ValueError, IndexError):
                    continue

    if not rows:
        return baselines, {}

    rows.sort(key=lambda r: r[0])
    ts_all  = np.array([r[0] for r in rows], dtype=np.float64)
    ch_all  = np.array([r[1] for r in rows], dtype=int)
    v_all   = np.array([r[2] for r in rows], dtype=np.float64)
    tag_all = [r[3] for r in rows]

    datasets = {}
    channels = range(1, N_SENSORS + 1)
    for ch in channels:
        if sensor_filter and ch not in sensor_filter:
            continue
        if ch not in baselines:
            continue

        mask_ch  = (ch_all == ch)
        ts_ch    = ts_all[mask_ch]
        v_ch     = v_all[mask_ch]
        tag_ch   = [tag_all[i] for i in np.where(mask_ch)[0]]
        v0_mv    = baselines[ch]["mean_mv"]

        X, y, meta = _build_dataset(ts_ch, v_ch, tag_ch, v0_mv, window_s)
        if X is not None and len(X) >= 9:
            datasets[ch] = (X, y, meta)
        else:
            n = 0 if X is None else len(X)
            print(f"  FSR{ch}: only {n} CAP samples (need ≥9) — check log")

    return baselines, datasets


def _find_cap_items(ts_all, tags):
    """Map CAP_R<rep>_<sample> tag sequences to force levels."""
    rep_window = collections.defaultdict(int)
    items      = []

    for i, tag in enumerate(tags):
        m = _CAP_RE.match(tag)
        if not m:
            continue
        rep    = int(m.group(2))
        sample = int(m.group(3))   # 1-based within C-command window

        if sample == 1:
            rep_window[rep] += 1

        win_idx = rep_window[rep] - 1
        if win_idx >= SEQ_LEN:
            continue

        items.append({
            "ts_idx":     i,
            "ts_ms":      float(ts_all[i]),
            "force_N":    float(FORCE_SEQ_N[win_idx]),
            "direction":  DIRECTION[win_idx],
            "rep":        rep,
            "cap_window": win_idx,
            "sample":     sample,
        })
    return items


def _compute_integral(ts_all, v_all, idx, window_s):
    """Linearly-weighted integral of V(t) over the past window_s seconds."""
    t_cur = ts_all[idx]
    t_min = t_cur - window_s * 1000.0
    in_win = np.where((ts_all >= t_min) & (ts_all <= t_cur))[0]
    if len(in_win) < 2:
        return 0.0
    ts_w = ts_all[in_win]
    v_w  = v_all[in_win]
    span = t_cur - t_min
    if span == 0:
        return 0.0
    w       = (ts_w - t_min) / span
    dt_s    = np.diff(ts_w) / 1000.0
    wv_trap = 0.5 * (w[:-1] * v_w[:-1] + w[1:] * v_w[1:])
    return float(np.sum(wv_trap * dt_s))


def _build_dataset(ts_ch, v_ch, tags, v0_mv, window_s):
    cap_items = _find_cap_items(ts_ch, tags)
    if not cap_items:
        return None, None, None

    rows_X, rows_y, meta = [], [], []
    for c in cap_items:
        i   = c["ts_idx"]
        v_V = v_ch[i]
        I   = _compute_integral(ts_ch, v_ch, i, window_s)
        F   = c["force_N"]
        rows_X.append([1.0, v_V, v_V**2, v_V**3, v_V**4,
                            I,   I**2,   I**3,   I**4])
        rows_y.append(F)
        meta.append(c)

    return np.array(rows_X), np.array(rows_y), meta


# ─── Ridge regression ──────────────────────────────────────────────────────────

def ridge_fit(X, y, alpha):
    n = X.shape[1]
    return np.linalg.solve(X.T @ X + alpha * np.eye(n), X.T @ y)


def cv_alpha(X, y, k=5):
    alphas = np.logspace(-4, 4, 17)
    n      = len(y)
    fold   = max(1, n // k)
    best_a, best_e = alphas[0], np.inf
    for alpha in alphas:
        errs = []
        for fi in range(k):
            vs = fi * fold
            ve = min(vs + fold, n)
            val = np.arange(vs, ve)
            tr  = np.concatenate([np.arange(0, vs), np.arange(ve, n)])
            if len(tr) < X.shape[1]:
                continue
            w    = ridge_fit(X[tr], y[tr], alpha)
            errs.append(float(np.mean((X[val] @ w - y[val])**2)))
        if errs:
            e = float(np.mean(errs))
            if e < best_e:
                best_e, best_a = e, alpha
    return float(best_a)


def eval_model(coefs, v_V, I):
    """Evaluate Hall model. coefs = [a0,a1,a2,a3,a4,b1,b2,b3,b4]."""
    a0,a1,a2,a3,a4,b1,b2,b3,b4 = coefs
    F = (a0 + v_V*(a1 + v_V*(a2 + v_V*(a3 + v_V*a4)))
            + I*(b1 + I*(b2 + I*(b3 + I*b4))))
    return max(0.0, float(F))


# ─── Phase 4: fit ──────────────────────────────────────────────────────────────

def cmd_fit(args):
    log_path = Path(args.logfile)
    if not log_path.exists():
        sys.exit(f"File not found: {log_path}")

    window_s = args.window
    sensor_filter = set(args.sensors) if args.sensors else None

    print(f"\nParsing {log_path}  (window={window_s:.1f} s) ...")
    baselines, datasets = parse_log(str(log_path), window_s, sensor_filter)

    if not baselines:
        sys.exit("No S,BASELINE lines found. Run Phase 2 (B command) before Phase 3.")
    if not datasets:
        sys.exit("No usable CAP samples found. Check Phase 3 log has CH*_CAP_R*_* lines.")

    n_sensors = len(datasets)
    ncols     = min(3, n_sensors)
    nrows     = (n_sensors + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 5*nrows), squeeze=False)
    fig.suptitle("FSR406 × 6 — Hall Model Fit (Option B)", fontsize=13)

    calibration = {"window_s": window_s, "sensors": {}}
    SEP = "═" * 62

    print(f"\n{SEP}")
    print(f"  {'Sensor':6}  {'V₀(mV)':>8}  {'RMSE(N)':>8}  {'R²':>8}  {'α':>8}  n")
    print(f"  {'──────':6}  {'────────':>8}  {'────────':>8}  {'────────':>8}  {'────────':>8}  ─")

    plot_idx = 0
    for ch in sorted(datasets.keys()):
        name     = SENSOR_NAMES[ch - 1]
        X, y, meta = datasets[ch]
        v0_mv    = baselines[ch]["mean_mv"]

        # Ridge regression
        if args.alpha is not None:
            alpha = args.alpha
        else:
            alpha = cv_alpha(X, y)

        w       = ridge_fit(X, y, alpha)
        y_pred  = X @ w
        res     = y_pred - y
        rmse    = float(np.sqrt(np.mean(res**2)))
        ss_res  = float(np.sum(res**2))
        ss_tot  = float(np.sum((y - y.mean())**2))
        r2      = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

        coefs = list(w)   # [a0,a1,a2,a3,a4,b1,b2,b3,b4]

        calibration["sensors"][name] = {
            "ch": ch,
            "V0_mv": round(v0_mv, 2),
            "coefficients": {
                "a0": coefs[0], "a1": coefs[1], "a2": coefs[2],
                "a3": coefs[3], "a4": coefs[4],
                "b1": coefs[5], "b2": coefs[6], "b3": coefs[7], "b4": coefs[8],
            },
            "fit": {
                "rmse_N": round(rmse, 4),
                "r2":     round(r2, 6),
                "n":      len(y),
                "alpha":  round(alpha, 6),
            },
        }

        print(f"  {name:6}  {v0_mv:>8.1f}  {rmse:>8.3f}  {r2:>8.5f}  {alpha:>8.4g}  {len(y)}")
        if r2 < 0.99:
            print(f"  {'':6}  ⚠  R²<0.99 — retry with  --window {window_s+0.5:.1f}")
        if rmse > RMSE_SINGLE_MAX:
            print(f"  {'':6}  ⚠  RMSE>{RMSE_SINGLE_MAX} N — check calibration data quality")

        # Plot
        row, col = divmod(plot_idx, ncols)
        ax = axes[row][col]
        dirs   = np.array([m["direction"] for m in meta])
        forces = np.array([m["force_N"]   for m in meta])
        for d, c, mk in [("loading","tab:blue","o"), ("unloading","tab:orange","s")]:
            mask = dirs == d
            if not mask.any():
                continue
            fu = np.unique(forces[mask])
            pp = [np.mean(y_pred[mask & (forces == fi)]) for fi in fu]
            ax.scatter(fu, pp, color=c, marker=mk, s=50, label=d, zorder=5)
        lim = [0, max(forces)*1.06]
        ax.plot(lim, lim, "k--", lw=1.2, label="ideal")
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel("F true (N)"); ax.set_ylabel("F pred (N)")
        ax.set_title(f"{name}  RMSE={rmse:.2f}N  R²={r2:.4f}", fontsize=9)
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
        plot_idx += 1

    # Hide unused subplots
    for k in range(plot_idx, nrows * ncols):
        r, c = divmod(k, ncols)
        axes[r][c].set_visible(False)

    print(f"{SEP}\n")

    # Print per-sensor define block for optional ESP32 embed
    _print_defines(calibration)

    # Save JSON
    out_path = log_path.parent / "calibration_6ch.json"
    out_path.write_text(json.dumps(calibration, indent=2))
    print(f"Calibration saved → {out_path}\n")

    # Save figure
    fig.tight_layout()
    fig_path = log_path.parent / "calibration_curves_6ch.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"Curves saved → {fig_path}")
    plt.show()


def _print_defines(calibration):
    """Print C array #define block to paste into firmware (optional)."""
    print("  ── Optional: paste into firmware (only needed for on-device Phase 5) ──")

    fields = ["V0_mv"]
    coef_names = ["a0","a1","a2","a3","a4","b1","b2","b3","b4"]
    sensors_sorted = sorted(calibration["sensors"].items(),
                            key=lambda kv: kv[1]["ch"])

    v0s = [f"{s['V0_mv']:.0f}" for _, s in sensors_sorted]
    print(f"  static const uint32_t CALIB_V0_MV[6] = {{ {', '.join(v0s)} }};")

    for cn in coef_names:
        vals = [f"{s['coefficients'][cn]:+.7f}f" for _, s in sensors_sorted]
        cname = f"CALIB_{cn.upper()}"
        print(f"  static const float {cname}[6] = {{ {', '.join(vals)} }};")
    print()


# ─── Serial reader ─────────────────────────────────────────────────────────────

class SerialReader:
    """Reads M lines from ESP32 G-mode: M,ts_ms,v0_mv,...,v5_mv."""

    def __init__(self, port: str):
        self._ser     = serial.Serial(port, BAUD_RATE, timeout=1)
        time.sleep(2)
        self._ser.reset_input_buffer()
        self._latest  = None   # [v_mv × 6] as floats (Volts)
        self._ts_ms   = None
        self._lock    = threading.Lock()
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while self._running:
            try:
                raw = self._ser.readline().decode("ascii", errors="ignore").strip()
                if raw.startswith("M,"):
                    parts = raw.split(",")
                    if len(parts) == N_SENSORS + 2:
                        ts  = float(parts[1])
                        vs  = [float(p) / 1000.0 for p in parts[2:]]   # mV → V
                        with self._lock:
                            self._ts_ms  = ts
                            self._latest = vs
                elif raw.startswith("#"):
                    print(f"  ESP32: {raw[1:].strip()}")
            except Exception:
                pass

    def read(self):
        with self._lock:
            return (self._ts_ms, list(self._latest)) if self._latest else (None, None)

    def collect_seconds(self, duration: float):
        """Collect samples for `duration` seconds. Returns list of (ts, [v×6])."""
        samples = []
        end = time.time() + duration
        seen_ts = set()
        while time.time() < end:
            ts, vs = self.read()
            if ts is not None and ts not in seen_ts:
                samples.append((ts, vs))
                seen_ts.add(ts)
            time.sleep(LOG_INTERVAL_MS / 1000 / 2)
        return samples

    def send(self, cmd: str):
        self._ser.write((cmd + "\n").encode())

    def close(self):
        self._running = False
        self._ser.close()


# ─── Integral engine (Python-side, for G-mode output) ──────────────────────────

class IntegralEngine:
    """Maintains per-channel circular buffer; computes Hall-model F_pred."""

    def __init__(self, calibration: dict, window_s: float):
        self._cal  = calibration
        self._ws   = window_s
        self._bufs = {}   # ch_1based → deque of (ts_ms, v_V)

        for name, sensor in calibration["sensors"].items():
            ch = sensor["ch"]
            self._bufs[ch] = collections.deque()

    def update(self, ts_ms: float, voltages_V: list) -> dict:
        """
        Push new readings and return {name: F_N} for all calibrated sensors.
        voltages_V: list of 6 voltages in Volts (index 0 = FSR1).
        """
        forces = {}
        for name, sensor in self._cal["sensors"].items():
            ch    = sensor["ch"]
            v_V   = voltages_V[ch - 1]
            v0_V  = sensor["V0_mv"] / 1000.0
            coefs = sensor["coefficients"]
            c9    = [coefs[k] for k in ("a0","a1","a2","a3","a4","b1","b2","b3","b4")]

            buf = self._bufs[ch]
            buf.append((ts_ms, v_V))
            # Trim old samples outside window
            cutoff = ts_ms - self._ws * 1000.0
            while buf and buf[0][0] < cutoff:
                buf.popleft()

            I = self._integral(buf, ts_ms)
            v_eff = max(v_V - v0_V, 0.0)

            # Use v_eff in the polynomial to zero-reference
            F = eval_model(c9, v_eff, I)
            forces[name] = F
        return forces

    @staticmethod
    def _integral(buf, t_cur):
        if len(buf) < 2:
            return 0.0
        t_min = buf[0][0]
        span  = t_cur - t_min
        if span == 0:
            return 0.0
        total = 0.0
        items = list(buf)
        for i in range(len(items) - 1):
            t0, v0 = items[i]
            t1, v1 = items[i+1]
            w0 = (t0 - t_min) / span
            w1 = (t1 - t_min) / span
            dt = (t1 - t0) / 1000.0
            total += 0.5 * (w0*v0 + w1*v1) * dt
        return total


def _load_calibration(path_str: str | None) -> dict:
    p = Path(path_str) if path_str else Path("calibration_6ch.json")
    if not p.exists():
        sys.exit(f"Calibration file not found: {p}\n"
                 "Run: python fsr406_6ch_phase4.py fit <session.txt>")
    return json.loads(p.read_text())


def _detect_port():
    for p in serial.tools.list_ports.comports():
        if any(k in p.description.lower() for k in ("cp210","ch340","uart","usb")):
            return p.device
    return None


def _open_serial(port_arg: str | None) -> SerialReader:
    port = port_arg or _detect_port()
    if port is None:
        sys.exit("Cannot detect ESP32 port. Specify with --port /dev/ttyUSBx")
    print(f"  Opening {port} @ {BAUD_RATE} baud ...")
    return SerialReader(port)


# ─── Phase 5: validate ────────────────────────────────────────────────────────

def cmd_validate(args):
    print("\n" + "="*62)
    print("PHASE 5 — VALIDATION")
    print("="*62)

    cal    = _load_calibration(args.calibration)
    window = cal.get("window_s", DEFAULT_WINDOW)
    reader = _open_serial(args.port)
    engine = IntegralEngine(cal, window)

    try:
        reader.send("G")
        time.sleep(0.5)

        print(f"\n  Validation force levels: {VALIDATION_LEVELS_N} N")
        print(f"  Per-sensor: RMSE < {RMSE_SINGLE_MAX} N, %err < {PCT_SINGLE_MAX}%")
        print(f"  Aggregate:  RMSE < {RMSE_TOTAL_MAX} N\n")

        # 5a: per-sensor
        results = {}
        for name, sensor in sorted(cal["sensors"].items(), key=lambda kv: kv[1]["ch"]):
            print(f"  ─── {name} ─── place applicator on this sensor's island")
            input(f"  Press ENTER when ready for {name} validation > ")

            errs, pct_errs = [], []
            for f_true in VALIDATION_LEVELS_N:
                weight_g = f_true * 1000 / 9.81
                print(f"\n    {f_true} N ({weight_g:.0f} g) — place weight now")
                _countdown(10, "Waiting for creep")

                samples = reader.collect_seconds(5.0)
                if not samples:
                    print("    WARNING: no M lines received — is ESP32 in G mode?")
                    continue

                # Use most recent 10 readings
                recent = samples[-10:] if len(samples) >= 10 else samples
                f_preds = []
                for ts, vs in recent:
                    forces = engine.update(ts, vs)
                    f_preds.append(forces.get(name, 0.0))
                f_pred = float(np.mean(f_preds))
                err    = abs(f_pred - f_true)
                pct    = err / f_true * 100 if f_true > 0 else 0.0
                errs.append(err)
                pct_errs.append(pct)
                print(f"    F_pred={f_pred:.2f}N  F_true={f_true}N  "
                      f"err={err:.2f}N  ({pct:.1f}%)")

                print(f"    Remove weight, resting 5 s...", end="", flush=True)
                _countdown(5, "")

            if not errs:
                continue
            rmse   = float(np.sqrt(np.mean(np.array(errs)**2)))
            max_pct = float(max(pct_errs))
            ok     = rmse < RMSE_SINGLE_MAX and max_pct < PCT_SINGLE_MAX
            results[name] = {"rmse": rmse, "max_pct": max_pct, "pass": ok}
            status = "PASS" if ok else "FAIL"
            print(f"\n  {name}: RMSE={rmse:.2f}N  max_pct={max_pct:.1f}%  [{status}]")

        # 5b: aggregate
        print("\n  ─── Aggregate validation — distribute weight across ALL sensors ───")
        agg_errs = []
        for f_true in VALIDATION_LEVELS_N:
            weight_g = f_true * 1000 / 9.81
            print(f"\n  {f_true} N total ({weight_g:.0f} g) — spread evenly across array")
            _countdown(10, "Stabilizing")

            samples = reader.collect_seconds(5.0)
            if not samples:
                continue

            recent  = samples[-10:]
            ftots   = []
            for ts, vs in recent:
                forces = engine.update(ts, vs)
                ftots.append(sum(forces.values()))
            ftot   = float(np.mean(ftots))
            err    = abs(ftot - f_true)
            agg_errs.append(err)
            print(f"  F_total_pred={ftot:.2f}N  F_true={f_true}N  err={err:.2f}N")
            _countdown(5, "")

        if agg_errs:
            rmse_tot = float(np.sqrt(np.mean(np.array(agg_errs)**2)))
            ok_tot   = rmse_tot < RMSE_TOTAL_MAX
            status   = "PASS" if ok_tot else "FAIL"
            print(f"\n  Aggregate RMSE={rmse_tot:.2f}N  [{status}]")
            results["_aggregate"] = {"rmse": rmse_tot, "pass": ok_tot}

        # Summary
        print("\n  ─── Summary ───")
        for name, r in results.items():
            flag = "PASS" if r["pass"] else "FAIL"
            if name == "_aggregate":
                print(f"  Aggregate: RMSE={r['rmse']:.2f}N  [{flag}]")
            else:
                print(f"  {name:6}: RMSE={r['rmse']:.2f}N  max%={r['max_pct']:.1f}%  [{flag}]")

        # Save
        out = Path("validation_results.json")
        out.write_text(json.dumps(results, indent=2))
        print(f"\n  Saved → {out}")

    finally:
        reader.send("X")
        reader.close()


# ─── Phase 6: monitor ─────────────────────────────────────────────────────────

def cmd_monitor(args):
    print("\n" + "="*62)
    print("PHASE 6 — REAL-TIME MONITOR  (Ctrl+C to stop)")
    print("="*62)

    cal    = _load_calibration(args.calibration)
    window = cal.get("window_s", DEFAULT_WINDOW)
    reader = _open_serial(args.port)
    engine = IntegralEngine(cal, window)
    names  = sorted(cal["sensors"].keys(),
                    key=lambda n: cal["sensors"][n]["ch"])

    try:
        reader.send("G")
        time.sleep(0.5)
        print("  Reading from ESP32 G mode ...\n")

        while True:
            ts, vs = reader.read()
            if ts is None or vs is None:
                time.sleep(0.02)
                continue
            forces = engine.update(ts, vs)
            parts  = [f"{n}={forces.get(n, 0):5.1f}N" for n in names]
            ftot   = sum(forces.values())
            line   = "  " + "  ".join(parts) + f"  │  F_total={ftot:6.1f}N"
            print(f"\r{line}", end="", flush=True)
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n  Stopped.")
    finally:
        reader.send("X")
        reader.close()


# ─── Phase 7: recheck ─────────────────────────────────────────────────────────

def cmd_recheck(args):
    print("\n" + "="*62)
    print("PHASE 7 — RE-CALIBRATION BASELINE CHECK")
    print("="*62)
    print("  Patient lying still, NO extra load on any sensor.")
    input("  Press ENTER when ready > ")

    cal    = _load_calibration(args.calibration)
    reader = _open_serial(args.port)

    try:
        reader.send("G")
        time.sleep(0.5)
        print("  Collecting 10 s of baseline data ...")
        samples = reader.collect_seconds(10.0)
        reader.send("X")

        if not samples:
            print("  ERROR: no M lines received — is ESP32 in G mode?")
            return

        vs_arr = np.array([vs for _, vs in samples])    # (N, 6)
        means  = vs_arr.mean(axis=0) * 1000.0           # Volts → mV

        print("\n  ─── Baseline drift report ───")
        print(f"  {'Sensor':6}  {'Orig V₀(mV)':>12}  {'Current(mV)':>12}  "
              f"{'Drift(mV)':>10}  {'Drift%':>7}  Status")
        print(f"  {'──────':6}  {'──────────':>12}  {'──────────':>12}  "
              f"{'──────────':>10}  {'──────':>7}  ──────")

        needs_recal = False
        report      = {}
        for name, sensor in sorted(cal["sensors"].items(),
                                   key=lambda kv: kv[1]["ch"]):
            ch       = sensor["ch"]
            orig     = sensor["V0_mv"]
            current  = float(means[ch - 1])
            drift    = abs(current - orig)
            drift_pct = drift / V_CC_MV
            ok       = drift_pct < BASELINE_DRIFT
            if not ok:
                needs_recal = True
            status   = "OK" if ok else "RECALIBRATE"
            report[name] = {
                "original_V0_mv": orig,
                "current_mv":     round(current, 2),
                "drift_mv":       round(drift, 2),
                "drift_pct":      round(drift_pct * 100, 2),
                "needs_recal":    not ok,
            }
            print(f"  {name:6}  {orig:>12.1f}  {current:>12.1f}  "
                  f"{drift:>10.1f}  {drift_pct*100:>6.2f}%  {status}")

        print()
        if needs_recal:
            print("  ACTION: One or more sensors drifted >2% of full scale.")
            print("  Run full recalibration: Phase 2 (B) → Phase 3 (L/C/R) → fit")
        else:
            print("  All sensors within ±2% tolerance — no recalibration needed.")
            print("  Tip: update V0_mv in calibration_6ch.json with current values")
            print("       if you want tighter zero-referencing for this session.")

        out = Path(f"recheck_{time.strftime('%Y%m%d_%H%M%S')}.json")
        out.write_text(json.dumps(report, indent=2))
        print(f"\n  Saved → {out}")

    finally:
        reader.close()


# ─── Utility ───────────────────────────────────────────────────────────────────

def _countdown(seconds: int, msg: str):
    for remaining in range(seconds, 0, -1):
        print(f"\r  {msg} {remaining:3d}s ...", end="", flush=True)
        time.sleep(1)
    print()


# ─── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="FSR406 × 6 — Phases 4-7",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    sub = ap.add_subparsers(dest="cmd")

    # fit
    p4 = sub.add_parser("fit", help="Phase 4: fit Hall model per sensor")
    p4.add_argument("logfile",
                    help="Session log file (from: miniterm | tee session.txt)")
    p4.add_argument("--window", type=float, default=DEFAULT_WINDOW,
                    metavar="s", help=f"Integral window in seconds (default: {DEFAULT_WINDOW})")
    p4.add_argument("--alpha", type=float, default=None,
                    help="Ridge α (default: auto 5-fold CV)")
    p4.add_argument("--sensors", nargs="+", type=int, metavar="N",
                    help="Fit only these channels, e.g. --sensors 1 3 5")
    p4.add_argument("--out", metavar="FILE",
                    help="Save figure to FILE instead of displaying")

    # validate
    p5 = sub.add_parser("validate", help="Phase 5: per-sensor + aggregate validation")
    p5.add_argument("--port", default=None, metavar="PORT")
    p5.add_argument("--calibration", default=None, metavar="FILE",
                    help="Calibration JSON (default: calibration_6ch.json)")

    # monitor
    p6 = sub.add_parser("monitor", help="Phase 6: real-time F_total display")
    p6.add_argument("--port", default=None, metavar="PORT")
    p6.add_argument("--calibration", default=None, metavar="FILE")

    # recheck
    p7 = sub.add_parser("recheck", help="Phase 7: baseline drift check")
    p7.add_argument("--port", default=None, metavar="PORT")
    p7.add_argument("--calibration", default=None, metavar="FILE")

    args = ap.parse_args()
    if args.cmd is None:
        ap.print_help()
        sys.exit(0)

    dispatch = {"fit": cmd_fit, "validate": cmd_validate,
                "monitor": cmd_monitor, "recheck": cmd_recheck}
    dispatch[args.cmd](args)


if __name__ == "__main__":
    main()
