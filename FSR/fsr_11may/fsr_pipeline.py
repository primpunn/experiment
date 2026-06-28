"""
Log-linear FSR calibration pipeline.

Model (per sensor, 2 coefficients):
  F (N) = 10^(c1 · V_eff + c0) × 0.00981    [Linde et al. 2021, Eq. 1]
  V_eff = (V_raw − V0) / 1000  in Volts, clamped ≥ 0
  c1 = slope, c0 = intercept of log10(F_g) vs V_eff
  where F_g = F_N / 0.00981  (grams)
"""

import collections
import numpy as np
from config import (
    N_SENSORS, SENSOR_NAMES, SENSOR_GROUPS,
    V_CC_MV,
    CALIBRATIONS_DIR,
)


# ── Signal processing ──────────────────────────────────────────────────────────

def subtract_baseline(v_mv, v0_mv):
    """V_eff in Volts, clamped >= 0."""
    return max((v_mv - v0_mv) / 1000.0, 0.0)


# ── Log-linear model (Linde et al. 2021, Eq. 1) ───────────────────────────────

def fit_log_linear(V_arr, F_N_arr):
    """
    Fit: log10(F_g) = c1 * V_eff + c0  where F_g = F_N / 0.00981.
    OLS on positive-force, positive-voltage points.

    Returns (c1, c0, rmse_N, r2).
    """
    V = np.asarray(V_arr, dtype=float)
    F = np.asarray(F_N_arr, dtype=float)
    mask = (F > 0) & (V > 0)
    if mask.sum() < 2:
        return 0.0, 0.0, float("nan"), float("nan")
    V_fit  = V[mask]
    log_F  = np.log10(F[mask] / 0.00981)
    c1, c0 = np.polyfit(V_fit, log_F, 1)
    F_pred = np.power(10.0, c1 * V_fit + c0) * 0.00981
    res    = F_pred - F[mask]
    rmse   = float(np.sqrt(np.mean(res ** 2)))
    ss_tot = float(np.sum((F[mask] - F[mask].mean()) ** 2))
    r2     = float(1.0 - np.sum(res ** 2) / ss_tot) if ss_tot > 0 else float("nan")
    return float(c1), float(c0), rmse, r2


def eval_log_linear(v_eff_V, c1, c0):
    """F (N) = 10^(c1 * V_eff + c0) * 0.00981; returns 0.0 if v_eff_V <= 0."""
    if v_eff_V <= 0:
        return 0.0
    return max(0.0, float(10.0 ** (c1 * v_eff_V + c0) * 0.00981))


# ── Real-time force engine ─────────────────────────────────────────────────────

class ForceEngine:
    """
    Computes per-sensor force from raw voltage using the log-linear model.
    """

    def __init__(self, cal, window_sec=None):
        """
        cal : dict  {sensor_name: {"ch": int, "V0_mv": float, "c1": float, "c0": float}}
        window_sec is accepted for API compatibility but not used.
        """
        self._cal = cal

    def update(self, ts_ms, v_mv_list):
        """
        Returns {sensor_name: F_N} for all calibrated sensors.
        ts_ms is accepted for API compatibility but not used.
        """
        forces = {}
        for name, sensor in self._cal.items():
            ch    = sensor["ch"]
            v_eff = subtract_baseline(v_mv_list[ch], sensor["V0_mv"])
            forces[name] = eval_log_linear(v_eff, sensor["c1"], sensor["c0"])
        return forces

    def total_by_group(self, forces):
        """Compute F_total per sensor group from a forces dict."""
        totals = {}
        for gname, indices in SENSOR_GROUPS.items():
            names = [SENSOR_NAMES[i] for i in indices]
            totals[gname] = sum(forces.get(n, 0.0) for n in names if n in self._cal)
        return totals


IntegralEngine = ForceEngine  # backward-compat alias


# ── Calibration buffer ─────────────────────────────────────────────────────────

class CalibBuffer:
    """
    Tracks latest V_eff for one sensor during calibration.
    Thread-safe; push() is called from the background reader thread.
    """

    def __init__(self, ch, V0_mv, window_sec=None):
        import threading
        self._ch    = ch
        self._v0    = V0_mv
        self._v_eff = 0.0
        self._lock  = threading.Lock()

    def push(self, ts_ms, v_mv_list):
        v_eff = subtract_baseline(v_mv_list[self._ch], self._v0)
        with self._lock:
            self._v_eff = v_eff

    def snapshot(self):
        """Return (latest_v_eff, 0.0) — second value kept for API compatibility."""
        with self._lock:
            return self._v_eff, 0.0


# ── Calibration folder helpers ─────────────────────────────────────────────────

def find_latest_calibration_path(cal_dir=None):
    """
    Return Path to calibration_coeffs.json in the most recently modified
    subfolder of CALIBRATIONS_DIR.  Returns None if none found.
    """
    from pathlib import Path
    base = Path(cal_dir or CALIBRATIONS_DIR)
    if not base.exists():
        return None
    candidates = [
        d / "calibration_coeffs.json"
        for d in base.iterdir()
        if d.is_dir() and (d / "calibration_coeffs.json").exists()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def new_calibration_dir(cal_dir=None):
    """Create and return a new timestamped calibration folder."""
    import time
    from pathlib import Path
    ts   = time.strftime("%Y-%m-%d_%H-%M-%S")
    path = Path(cal_dir or CALIBRATIONS_DIR) / ts
    path.mkdir(parents=True, exist_ok=True)
    return path


# ── Batch preprocessing ────────────────────────────────────────────────────────

def process_session_csv(csv_path, cal, output_path=None):
    """
    Apply log-linear model to a recorded session CSV.

    CSV columns: timestamp_ms, V1_mv … V6_mv [,event]
    Appends columns: F_FSR1_N … F_FSR6_N, F_calf_N, F_foot_N
    """
    import csv as csv_mod
    from pathlib import Path

    csv_path = str(csv_path)

    with open(csv_path, newline="") as fh:
        reader = csv_mod.DictReader(fh)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    if not rows:
        raise ValueError(f"Empty CSV: {csv_path}")

    col_mv   = [f"V{i+1}_mv" for i in range(N_SENSORS)]
    f_arrays = {}

    for name, sensor in cal.items():
        ch    = sensor["ch"]
        v0_mv = sensor["V0_mv"]
        c1    = sensor["c1"]
        c0    = sensor["c0"]

        v_mv  = np.array([float(r[col_mv[ch]]) for r in rows])
        v_eff = np.maximum((v_mv - v0_mv) / 1000.0, 0.0)
        F     = np.where(v_eff > 0, np.power(10.0, c1 * v_eff + c0) * 0.00981, 0.0)
        f_arrays[f"F_{name}_N"] = np.maximum(F, 0.0)

    for gname, indices in SENSOR_GROUPS.items():
        names = [SENSOR_NAMES[i] for i in indices]
        cols  = [f"F_{n}_N" for n in names if n in cal]
        if cols:
            f_arrays[f"F_{gname}_N"] = sum(f_arrays[c] for c in cols)

    out_fields = list(fieldnames) + list(f_arrays.keys())

    if output_path is None:
        p = Path(csv_path)
        output_path = str(p.with_name(p.stem + "_forces.csv"))

    with open(output_path, "w", newline="") as fh:
        writer = csv_mod.DictWriter(fh, fieldnames=out_fields)
        writer.writeheader()
        for i, row in enumerate(rows):
            out_row = dict(row)
            for col, arr in f_arrays.items():
                out_row[col] = round(float(arr[i]), 4)
            writer.writerow(out_row)

    return str(output_path)
