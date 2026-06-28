#!/usr/bin/env python3
"""
fsr406_phase4_calibration.py — Phase 4: Hall model (Option B)

Model
-----
  F(t) = a0 + a1·V + a2·V² + a3·V³ + a4·V⁴
              + b1·I + b2·I² + b3·I³ + b4·I⁴

  V  = voltage at time t (Volts = v_mv / 1000)
  I  = linearly-weighted moving integral of V over the past WINDOW_S seconds
         I(t) = ∫[t-W, t]  w(τ)·V(τ) dτ
       where  w(τ) rises linearly from 0 at τ=t-W to 1 at τ=t

Fitting
-------
  Uses Ridge regression to handle the collinearity between V and I features
  that is unavoidable with static-only calibration data.
  Ridge alpha is chosen automatically via 5-fold cross-validation.

Input log format (from fsr406_calibration.ino, Phase 3)
--------------------------------------------------------
  D,<ts_ms>,<raw_adc>,<v_mv>,<tag>    — raw sample (LOG or CAP)
  S,BASELINE,result,<mean_mv>,...      — Phase 2 baseline
  S,LOADING,R<rep>_cap,<mean_mv>,...  — stable-window summary (not used for fitting;
                                        actual CAP_R*_* D-lines are used instead)

Usage
-----
  python fsr406_phase4_calibration.py phase3_log.txt
  python fsr406_phase4_calibration.py phase3_log.txt --v0 1234
  python fsr406_phase4_calibration.py phase3_log.txt --window 1.5
  python fsr406_phase4_calibration.py phase3_log.txt --alpha 0.01
  python fsr406_phase4_calibration.py phase3_log.txt --out calib.png

How to save the serial log
--------------------------
  Arduino IDE: Serial Monitor → right-click → Save as → phase3_log.txt
  CLI:  python -m serial.tools.miniterm --raw /dev/ttyUSB0 115200 | tee phase3_log.txt
"""

import re
import sys
import argparse
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Force sequence — one rep (edit if yours differs) ─────────────────
FORCE_SEQ_N = [
    0, 5, 10, 20, 30, 40, 50, 60, 80,   # loading  (9 captures)
    60, 50, 40, 30, 20, 10,  5,  0,     # unloading (8 captures)
]
DIRECTION = ['loading'] * 9 + ['unloading'] * 8
SEQ_LEN   = len(FORCE_SEQ_N)

WINDOW_S  = 1.0    # default integral window (seconds); override with --window


# ═══════════════════════════════════════════════════════════════════════
# Parsing
# ═══════════════════════════════════════════════════════════════════════

def parse_log(path):
    """
    Returns (baseline_mv_or_None, ts_ms_array, v_V_array, tags_list).
    All D-lines (LOG + CAP) are included, sorted by timestamp.
    """
    baseline_mv = None
    rows = []

    with open(path, encoding='utf-8', errors='replace') as fh:
        for raw in fh:
            line = raw.strip()

            if line.startswith('S,BASELINE'):
                parts = line.split(',')
                if len(parts) >= 4:
                    try:
                        baseline_mv = float(parts[3])
                    except ValueError:
                        pass

            elif line.startswith('D,'):
                parts = line.split(',')
                if len(parts) < 5:
                    continue
                try:
                    ts  = int(parts[1])
                    v_V = float(parts[3]) / 1000.0   # mV → Volts
                    tag = parts[4].strip()
                    rows.append((ts, v_V, tag))
                except (ValueError, IndexError):
                    continue

    if not rows:
        return baseline_mv, np.empty(0), np.empty(0), []

    rows.sort(key=lambda r: r[0])
    ts_all = np.array([r[0] for r in rows], dtype=np.float64)
    v_all  = np.array([r[1] for r in rows], dtype=np.float64)
    tags   = [r[2] for r in rows]
    return baseline_mv, ts_all, v_all, tags


# ═══════════════════════════════════════════════════════════════════════
# Assign forces to CAP samples
# ═══════════════════════════════════════════════════════════════════════

_CAP_RE = re.compile(r'^CAP_R(\d+)_(\d+)$')

def find_cap_items(ts_all, tags):
    """
    Scans tags for CAP_R<rep>_<sample> patterns and assigns
    force_N / direction from FORCE_SEQ_N by counting C-command windows per rep.

    Returns list of dicts:
      { ts_idx, ts_ms, force_N, direction, rep, cap_window, sample }
    """
    rep_window = defaultdict(int)   # rep → number of C commands issued so far
    prev_cap_window = defaultdict(lambda: -1)

    items = []
    for i, tag in enumerate(tags):
        m = _CAP_RE.match(tag)
        if not m:
            continue
        rep    = int(m.group(1))
        sample = int(m.group(2))   # 1-based within this C-command window

        # Detect start of a new C-command window (sample resets to 1)
        if sample == 1:
            rep_window[rep] += 1

        win_idx = rep_window[rep] - 1   # 0-based force-sequence index
        if win_idx >= SEQ_LEN:
            continue

        items.append({
            'ts_idx':     i,
            'ts_ms':      float(ts_all[i]),
            'force_N':    float(FORCE_SEQ_N[win_idx]),
            'direction':  DIRECTION[win_idx],
            'rep':        rep,
            'cap_window': win_idx,
            'sample':     sample,
        })

    return items


# ═══════════════════════════════════════════════════════════════════════
# Moving integral
# ═══════════════════════════════════════════════════════════════════════

def compute_integral(ts_all, v_all, idx, window_s):
    """
    Linearly-weighted integral of V(t) over [ts_all[idx] - window_s*1000, ts_all[idx]].
    Weight w(τ) = (τ - t_min) / (t_cur - t_min) rises from 0 to 1.
    Uses trapezoidal rule with actual sample timestamps.
    Returns I in Volt·seconds.
    """
    t_cur = ts_all[idx]
    t_min = t_cur - window_s * 1000.0

    # Indices within the window (inclusive)
    in_win = np.where((ts_all >= t_min) & (ts_all <= t_cur))[0]
    if len(in_win) < 2:
        return 0.0

    ts_w = ts_all[in_win]
    v_w  = v_all[in_win]

    # Linear weight: 0 at t_min, 1 at t_cur
    span = t_cur - t_min
    if span == 0:
        return 0.0
    w = (ts_w - t_min) / span   # in [0, 1]

    # Trapezoidal integration of w(t)·V(t)
    dt_s    = np.diff(ts_w) / 1000.0          # seconds
    wv_trap = 0.5 * (w[:-1] * v_w[:-1] + w[1:] * v_w[1:])
    return float(np.sum(wv_trap * dt_s))


# ═══════════════════════════════════════════════════════════════════════
# Feature matrix
# ═══════════════════════════════════════════════════════════════════════

def build_dataset(ts_all, v_all, tags, v0_mv, window_s):
    """
    Builds (X, y, meta) from CAP samples.
      X[i] = [1, V, V², V³, V⁴, I, I², I³, I⁴]   (9 features)
      y[i] = force in Newtons
    """
    cap_items = find_cap_items(ts_all, tags)
    if not cap_items:
        return None, None, None

    rows_X, rows_y, meta = [], [], []
    for c in cap_items:
        i   = c['ts_idx']
        v_V = v_all[i]
        I   = compute_integral(ts_all, v_all, i, window_s)
        F   = c['force_N']

        rows_X.append([1.0,
                        v_V,   v_V**2,  v_V**3,  v_V**4,
                        I,     I**2,    I**3,     I**4])
        rows_y.append(F)
        meta.append(c)

    return np.array(rows_X), np.array(rows_y), meta


# ═══════════════════════════════════════════════════════════════════════
# Ridge regression
# ═══════════════════════════════════════════════════════════════════════

def ridge_fit(X, y, alpha):
    """Closed-form Ridge: w = (XᵀX + αI)⁻¹ Xᵀy"""
    n = X.shape[1]
    return np.linalg.solve(X.T @ X + alpha * np.eye(n), X.T @ y)


def cv_alpha(X, y, k=5):
    """5-fold CV over log-spaced alphas; returns best alpha."""
    alphas = np.logspace(-4, 4, 17)
    n      = len(y)
    fold   = max(1, n // k)
    best_a, best_e = alphas[0], np.inf

    for alpha in alphas:
        errs = []
        for fold_i in range(k):
            v_start = fold_i * fold
            v_end   = min(v_start + fold, n)
            val_idx = np.arange(v_start, v_end)
            tr_idx  = np.concatenate([np.arange(0, v_start),
                                      np.arange(v_end, n)])
            if len(tr_idx) < X.shape[1]:
                continue
            w    = ridge_fit(X[tr_idx], y[tr_idx], alpha)
            pred = X[val_idx] @ w
            errs.append(np.mean((pred - y[val_idx])**2))
        if errs:
            e = float(np.mean(errs))
            if e < best_e:
                best_e, best_a = e, alpha

    return float(best_a)


def metrics(y_true, y_pred):
    res    = y_pred - y_true
    rmse   = float(np.sqrt(np.mean(res**2)))
    ss_res = float(np.sum(res**2))
    ss_tot = float(np.sum((y_true - y_true.mean())**2))
    r2     = float(1 - ss_res / ss_tot) if ss_tot > 0 else float('nan')
    return rmse, r2


# ═══════════════════════════════════════════════════════════════════════
# Plots
# ═══════════════════════════════════════════════════════════════════════

def plot_results(y_true, y_pred, meta, rmse, r2, save_path=None):
    dirs   = np.array([m['direction'] for m in meta])
    forces = np.array([m['force_N']   for m in meta])

    fig = plt.figure(figsize=(13, 9))
    gs  = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.3)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    # ── Predicted vs True (main) ─────────────────────────────────────
    for d, c, mk in [('loading', 'tab:blue', 'o'), ('unloading', 'tab:orange', 's')]:
        mask = (dirs == d)
        if not mask.any():
            continue
        f_u = np.unique(forces[mask])
        p_u = [np.mean(y_pred[mask & (forces == fi)]) for fi in f_u]
        ax1.scatter(f_u, p_u, color=c, marker=mk, s=70, zorder=5,
                    label=f'{d} (rep-averaged)')

    lim = [0, forces.max() * 1.06]
    ax1.plot(lim, lim, 'k--', lw=1.5, label='ideal')
    ax1.set_xlim(lim); ax1.set_ylim(lim)
    ax1.set_xlabel('F true (N)', fontsize=11)
    ax1.set_ylabel('F predicted (N)', fontsize=11)
    ax1.set_title(
        f'Option B — 4th-order Poly + Moving Integral  '
        f'|  RMSE = {rmse:.2f} N  |  R² = {r2:.4f}',
        fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # ── Residuals ─────────────────────────────────────────────────────
    ax2.axhline(0, color='k', lw=1)
    ax2.fill_between(lim, -3, 3, alpha=0.08, color='green', label='±3 N band')
    ax2.scatter(y_true, y_pred - y_true, c='tab:purple', s=20, alpha=0.7)
    ax2.set_xlim(lim)
    ax2.set_xlabel('F true (N)', fontsize=10)
    ax2.set_ylabel('Residual (N)', fontsize=10)
    ax2.set_title('Residuals', fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # ── Hysteresis check ──────────────────────────────────────────────
    for d, c, mk in [('loading', 'tab:blue', 'o'), ('unloading', 'tab:orange', 's')]:
        mask = (dirs == d)
        if not mask.any():
            continue
        f_u = np.unique(forces[mask])
        t_u = [np.mean(y_true[mask & (forces == fi)]) for fi in f_u]
        p_u = [np.mean(y_pred[mask & (forces == fi)]) for fi in f_u]
        ax3.scatter(t_u, p_u, color=c, marker=mk, s=50, label=d.capitalize(), zorder=5)

    ax3.plot(lim, lim, 'k--', lw=1)
    ax3.set_xlim(lim); ax3.set_ylim(lim)
    ax3.set_xlabel('F true (N)', fontsize=10)
    ax3.set_ylabel('F pred (N)', fontsize=10)
    ax3.set_title('Hysteresis (loading vs unloading)', fontsize=10)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Figure saved → {save_path}')
    else:
        plt.show()


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description='FSR406 Phase 4 — Option B: 4th-order polynomial + moving integral',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    ap.add_argument('logfile',
                    help='Text file with Serial Monitor output from Phase 3')
    ap.add_argument('--v0', type=float, default=None, metavar='mV',
                    help='Override V₀ baseline (mV); default: read from log')
    ap.add_argument('--window', type=float, default=WINDOW_S, metavar='s',
                    help=f'Integral window in seconds (default: {WINDOW_S})')
    ap.add_argument('--alpha', type=float, default=None,
                    help='Ridge α (default: auto 5-fold CV)')
    ap.add_argument('--out', metavar='FILE',
                    help='Save figure to file instead of displaying')
    args = ap.parse_args()

    window_s = args.window

    # ── Parse ────────────────────────────────────────────────────────
    log_v0, ts_all, v_all, tags = parse_log(args.logfile)

    if ts_all.size == 0:
        sys.exit('ERROR: no D lines found. Check that you saved the full serial output.')

    n_cap = sum(1 for t in tags if _CAP_RE.match(t))
    print(f'Parsed {ts_all.size} D lines  ({n_cap} CAP samples)')

    # ── Baseline ─────────────────────────────────────────────────────
    if args.v0 is not None:
        v0_mv = args.v0
        print(f'V₀ = {v0_mv:.1f} mV  (--v0 argument)')
    elif log_v0 is not None:
        v0_mv = log_v0
        print(f'V₀ = {v0_mv:.1f} mV  (from S,BASELINE in log)')
    else:
        sys.exit('ERROR: no baseline found in log. Pass --v0 <mV> on command line.')

    # ── Build dataset ────────────────────────────────────────────────
    print(f'Building features with window = {window_s:.1f} s  ...')
    X, y, meta = build_dataset(ts_all, v_all, tags, v0_mv, window_s)

    if X is None or len(X) < 9:
        n = 0 if X is None else len(X)
        sys.exit(f'ERROR: only {n} usable CAP samples (need ≥ 9).\n'
                 'Ensure Phase 3 log contains CAP_R*_* D lines and S,LOADING lines.')

    print(f'Dataset: {len(y)} samples from {len(y)//10} capture windows')

    # ── Ridge alpha ──────────────────────────────────────────────────
    if args.alpha is not None:
        alpha = args.alpha
        print(f'Ridge α = {alpha:.4g}  (manual)')
    else:
        print('Selecting Ridge α via 5-fold CV ...', end=' ', flush=True)
        alpha = cv_alpha(X, y)
        print(f'{alpha:.4g}')

    # ── Fit ──────────────────────────────────────────────────────────
    w      = ridge_fit(X, y, alpha)
    y_pred = X @ w
    rmse, r2 = metrics(y, y_pred)

    a0, a1, a2, a3, a4 = w[0], w[1], w[2], w[3], w[4]
    b1, b2, b3, b4     = w[5], w[6], w[7], w[8]

    # ── Print results ─────────────────────────────────────────────────
    SEP = '═' * 60
    print()
    print(SEP)
    print('  FSR406 Phase 4 — Hall Model (Option B) Result')
    print(SEP)
    print(f'  V₀       =  {v0_mv:.1f} mV')
    print(f'  Window   =  {window_s:.2f} s')
    print(f'  Ridge α  =  {alpha:.4g}')
    print(f'  Samples  =  {len(y)}')
    print(f'  RMSE     =  {rmse:.3f} N     (target < 3 N)')
    print(f'  R²       =  {r2:.5f}   (target > 0.99)')
    print()
    print('  Polynomial terms  (V in Volts):')
    for name, val in [('a0', a0), ('a1', a1), ('a2', a2), ('a3', a3), ('a4', a4)]:
        print(f'    {name} = {val:+.7f}')
    print()
    print('  Integral terms  (I in V·s):')
    for name, val in [('b1', b1), ('b2', b2), ('b3', b3), ('b4', b4)]:
        print(f'    {name} = {val:+.7f}')
    print()
    print('  ── Paste into fsr406_calibration.ino ───────────────')
    print(f'  #define CALIB_V0_MV      {int(round(v0_mv))}')
    print(f'  #define CALIB_OB_A0      {a0:+.7f}f')
    print(f'  #define CALIB_OB_A1      {a1:+.7f}f')
    print(f'  #define CALIB_OB_A2      {a2:+.7f}f')
    print(f'  #define CALIB_OB_A3      {a3:+.7f}f')
    print(f'  #define CALIB_OB_A4      {a4:+.7f}f')
    print(f'  #define CALIB_OB_B1      {b1:+.7f}f')
    print(f'  #define CALIB_OB_B2      {b2:+.7f}f')
    print(f'  #define CALIB_OB_B3      {b3:+.7f}f')
    print(f'  #define CALIB_OB_B4      {b4:+.7f}f')
    print(SEP)
    print()

    if r2 < 0.99:
        print('  ⚠  R² < 0.99 — try  --window 1.5  or  --window 2.0')
    if rmse > 3.0:
        print('  ⚠  RMSE > 3 N — check calibration data quality and V₀')
    if r2 < 0.99 or rmse > 3.0:
        print()

    # ── Per-level summary ─────────────────────────────────────────────
    dirs   = np.array([m['direction'] for m in meta])
    forces = np.array([m['force_N']   for m in meta])

    print(f'  {"F_true(N)":>10}  {"F_pred(N)":>10}  {"err(N)":>7}  '
          f'{"err%":>6}  {"dir":>10}  n')
    print('  ' + '─' * 58)
    for d in ['loading', 'unloading']:
        for fi in sorted(np.unique(forces)):
            mask = (dirs == d) & (forces == fi)
            if not mask.any():
                continue
            fp   = float(np.mean(y_pred[mask]))
            err  = fp - fi
            pct  = abs(err) / fi * 100 if fi > 0 else float('nan')
            flag = '  ← !' if not np.isnan(pct) and pct > 10 else ''
            print(f'  {fi:>10.1f}  {fp:>10.3f}  {err:>7.3f}  '
                  f'{pct:>5.1f}%  {d:>10}  {int(mask.sum())}{flag}')
    print()

    # ── Plot ──────────────────────────────────────────────────────────
    plot_results(y, y_pred, meta, rmse, r2, save_path=args.out)


if __name__ == '__main__':
    main()
