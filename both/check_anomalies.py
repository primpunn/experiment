"""
Scan all saved sessions for 4 types of pose anomalies in arm marker poses.

Analysed independently on:
  (T) Translation  — x, y, z  of T[:3, 3]
  (R) Rotation     — rx, ry, rz of rotvec(T[:3,:3]), continuity-corrected

Artifact types:
  1. Spike       — value jumps 1–5 frames then returns
  2. Heavy noise — high std sustained >= 10 consecutive frames
  3. Rect distort— flat plateau deviated from baseline 6–30 frames (step up/down)
  4. Slow drift  — gradual monotonic shift over >= 40 frames

Per-frame source labelling:
  T        — artifact in translation only
  R        — artifact in rotation only
  T+R      — artifact in both
  NaN      — marker not detected
  OK       — no artifact detected

Only analyses valid (non-NaN) runs.
"""

import numpy as np
from pathlib import Path
from dataclasses import dataclass
from scipy.spatial.transform import Rotation

# ── Translation thresholds ────────────────────────────────────────────────────
BASELINE_WINDOW    = 15     # frames — rolling median window for baseline
SPIKE_THRESH       = 0.05   # m — deviation from baseline
SPIKE_MIN_LEN      = 1
SPIKE_MAX_LEN      = 5

NOISE_WINDOW       = 7      # frames — rolling std window
NOISE_STD_THRESH   = 0.03   # m — rolling std threshold
NOISE_MIN_LEN      = 10     # frames — minimum sustained duration

RECT_THRESH        = 0.05   # m — deviation from baseline
RECT_MIN_LEN       = 6      # frames
RECT_MAX_LEN       = 40     # frames
RECT_FLAT_RATIO    = 0.40   # plateau std < this × mean_dev

DRIFT_MIN_RUN      = 40     # frames
DRIFT_SLOPE_THRESH = 0.001  # m/frame
DRIFT_TOTAL_THRESH = 0.05   # m

# ── Rotation thresholds (same detector logic, different units: radians) ───────
ROT_SPIKE_THRESH   = 0.10   # rad (~5.7°) — deviation from baseline
ROT_NOISE_THRESH   = 0.05   # rad (~2.9°) — rolling std threshold
ROT_RECT_THRESH    = 0.10   # rad
ROT_DRIFT_SLOPE    = 0.002  # rad/frame
ROT_DRIFT_TOTAL    = 0.10   # rad

MARKERS  = ["right_wrist", "right_elbow", "left_wrist", "left_elbow"]
DATA_ROOT = Path(__file__).parent / "saved_data"


# ── Rolling statistics ────────────────────────────────────────────────────────

def rolling_median(x: np.ndarray, w: int) -> np.ndarray:
    h, n = w // 2, len(x)
    return np.array([np.median(x[max(0, i-h): min(n, i+h+1)]) for i in range(n)])


def rolling_std(x: np.ndarray, w: int) -> np.ndarray:
    h, n = w // 2, len(x)
    return np.array([x[max(0, i-h): min(n, i+h+1)].std(ddof=0) for i in range(n)])


def find_runs(mask: np.ndarray):
    runs, in_run = [], False
    for i, v in enumerate(mask):
        if v and not in_run:
            start, in_run = i, True
        elif not v and in_run:
            runs.append((start, i - 1)); in_run = False
    if in_run:
        runs.append((start, len(mask) - 1))
    return runs


# ── Generic detectors (work on any 1-D signal with given thresholds) ─────────

def detect_spikes(x: np.ndarray, thresh: float,
                  min_len: int, max_len: int) -> np.ndarray:
    """Return bool mask of spike frames."""
    mask = np.zeros(len(x), dtype=bool)
    if len(x) < 3:
        return mask
    baseline = rolling_median(x, BASELINE_WINDOW)
    residual = np.abs(x - baseline)
    for s, e in find_runs(residual > thresh):
        if min_len <= (e - s + 1) <= max_len:
            mask[s: e + 1] = True
    return mask


def detect_heavy_noise(x: np.ndarray, std_thresh: float,
                       min_len: int) -> np.ndarray:
    """Return bool mask of noise frames."""
    mask = np.zeros(len(x), dtype=bool)
    if len(x) < NOISE_WINDOW:
        return mask
    rstd = rolling_std(x, NOISE_WINDOW)
    for s, e in find_runs(rstd > std_thresh):
        if (e - s + 1) >= min_len:
            mask[s: e + 1] = True
    return mask


def detect_rect(x: np.ndarray, thresh: float,
                min_len: int, max_len: int) -> np.ndarray:
    """Return bool mask of rectangular-distortion frames."""
    mask = np.zeros(len(x), dtype=bool)
    if len(x) < min_len + 2:
        return mask
    baseline = rolling_median(x, BASELINE_WINDOW)
    residual = x - baseline
    abs_res  = np.abs(residual)
    for s, e in find_runs(abs_res > thresh):
        length = e - s + 1
        if not (min_len <= length <= max_len):
            continue
        plateau  = x[s: e + 1]
        mean_dev = abs_res[s: e + 1].mean()
        if plateau.std(ddof=0) < RECT_FLAT_RATIO * mean_dev:
            mask[s: e + 1] = True
    return mask


def detect_drift(x: np.ndarray, slope_thresh: float,
                 total_thresh: float) -> np.ndarray:
    """Return bool mask marking the whole run as drift (or empty)."""
    mask = np.zeros(len(x), dtype=bool)
    if len(x) < DRIFT_MIN_RUN:
        return mask
    t = np.arange(len(x), dtype=float)
    slope, intercept = np.polyfit(t, x, 1)
    total = abs(slope) * len(x)
    if abs(slope) > slope_thresh and total > total_thresh:
        detrended_std = (x - (slope * t + intercept)).std()
        if detrended_std < total * 0.6:
            mask[:] = True
    return mask


# ── Rotation vector extraction (continuity-corrected) ────────────────────────

def rotmats_to_rotvecs(R_list: list) -> np.ndarray:
    """
    Convert list of 3×3 rotation matrices to (N, 3) rotation vectors.
    Enforces quaternion continuity to avoid sign-flip discontinuities.
    """
    quats = []
    q_prev = None
    for R in R_list:
        q = Rotation.from_matrix(R).as_quat()   # [x, y, z, w]
        if q_prev is not None and np.dot(q, q_prev) < 0:
            q = -q                               # flip to same hemisphere
        quats.append(q)
        q_prev = q
    rotvecs = Rotation.from_quat(np.array(quats)).as_rotvec()  # (N, 3) radians
    return rotvecs


# ── Per-axis artifact union across all axes ───────────────────────────────────

def artifact_mask_for_run(signals: np.ndarray,
                          spike_thresh: float,
                          noise_thresh: float,
                          rect_thresh: float,
                          drift_slope: float,
                          drift_total: float):
    """
    signals : (L, 3) — x/y/z or rx/ry/rz for one valid run.
    Returns four bool arrays of length L: spike, noise, rect, drift.
    (Each is the union across all 3 axes.)
    """
    L = signals.shape[0]
    m_spike = np.zeros(L, dtype=bool)
    m_noise = np.zeros(L, dtype=bool)
    m_rect  = np.zeros(L, dtype=bool)
    m_drift = np.zeros(L, dtype=bool)
    for ax in range(3):
        col = signals[:, ax]
        m_spike |= detect_spikes(col, spike_thresh, SPIKE_MIN_LEN, SPIKE_MAX_LEN)
        m_noise |= detect_heavy_noise(col, noise_thresh, NOISE_MIN_LEN)
        m_rect  |= detect_rect(col, rect_thresh, RECT_MIN_LEN, RECT_MAX_LEN)
        m_drift |= detect_drift(col, drift_slope, drift_total)
    return m_spike, m_noise, m_rect, m_drift


# ── Main analysis ─────────────────────────────────────────────────────────────

@dataclass
class AnomalyCount:
    spike: int = 0
    noise: int = 0
    rect:  int = 0
    drift: int = 0

    def __add__(self, other):
        return AnomalyCount(self.spike + other.spike,
                            self.noise + other.noise,
                            self.rect  + other.rect,
                            self.drift + other.drift)

    def total(self):
        return self.spike + self.noise + self.rect + self.drift


sessions = sorted([d for d in DATA_ROOT.iterdir()
                   if d.is_dir() and any(d.glob("frame_*"))])

n_filled = sum(1 for s in sessions for f in s.glob("frame_*/pose_*_filled.txt"))
print(f"Reading: pose_<joint>.txt  (RAW, pre-interpolation)")
print(f"_filled.txt files present: {n_filled}"
      f"{'  ← WARNING: some sessions already interpolated' if n_filled else '  (none — confirmed raw)'}")
print(f"\nSessions: {len(sessions)}   Markers: {len(MARKERS)}")
print(f"\nTranslation thresholds — spike:{SPIKE_THRESH*100:.0f}cm/{SPIKE_MAX_LEN}fr  "
      f"noise_std:{NOISE_STD_THRESH*100:.0f}cm/{NOISE_MIN_LEN}fr  "
      f"rect:{RECT_THRESH*100:.0f}cm/{RECT_MIN_LEN}-{RECT_MAX_LEN}fr  "
      f"drift:{DRIFT_SLOPE_THRESH*100:.1f}cm/fr/{DRIFT_MIN_RUN}fr")
print(f"Rotation  thresholds — spike:{np.degrees(ROT_SPIKE_THRESH):.1f}°/{SPIKE_MAX_LEN}fr  "
      f"noise_std:{np.degrees(ROT_NOISE_THRESH):.1f}°/{NOISE_MIN_LEN}fr  "
      f"rect:{np.degrees(ROT_RECT_THRESH):.1f}°/{RECT_MIN_LEN}-{RECT_MAX_LEN}fr  "
      f"drift:{np.degrees(ROT_DRIFT_SLOPE):.2f}°/fr/{DRIFT_MIN_RUN}fr\n")

grand_T = AnomalyCount()
grand_R = AnomalyCount()
session_results = []

for session in sessions:
    frame_dirs = sorted(session.glob("frame_*"),
                        key=lambda d: int(d.name.split("_")[1]))
    n_frames = len(frame_dirs)
    sess_T = AnomalyCount()
    sess_R = AnomalyCount()
    marker_rows = []

    for marker in MARKERS:
        # ── Load raw poses ────────────────────────────────────────────────────
        raw = []
        for fd in frame_dirs:
            try:
                T = np.loadtxt(fd / f"pose_{marker}.txt")
                raw.append(None if np.any(np.isnan(T)) else T)
            except Exception:
                raw.append(None)

        n_nan = sum(1 for t in raw if t is None)

        # ── Build valid runs ──────────────────────────────────────────────────
        runs = []
        run_idx = []
        for i, v in enumerate(raw):
            if v is not None:
                run_idx.append(i)
            else:
                if run_idx:
                    runs.append(run_idx)
                run_idx = []
        if run_idx:
            runs.append(run_idx)

        mc_T = AnomalyCount()
        mc_R = AnomalyCount()

        for run in runs:
            trans_xyz = np.array([raw[i][:3, 3]    for i in run])  # (L, 3)
            rot_mats  = [raw[i][:3, :3] for i in run]
            rot_rvec  = rotmats_to_rotvecs(rot_mats)               # (L, 3)

            # Translation artifacts
            s, n, r, d = artifact_mask_for_run(
                trans_xyz,
                SPIKE_THRESH, NOISE_STD_THRESH, RECT_THRESH,
                DRIFT_SLOPE_THRESH, DRIFT_TOTAL_THRESH)
            mc_T.spike += int(s.any()); mc_T.noise += int(n.any())
            mc_T.rect  += int(r.any()); mc_T.drift += int(d.any())

            # Rotation artifacts
            s, n, r, d = artifact_mask_for_run(
                rot_rvec,
                ROT_SPIKE_THRESH, ROT_NOISE_THRESH, ROT_RECT_THRESH,
                ROT_DRIFT_SLOPE, ROT_DRIFT_TOTAL)
            mc_R.spike += int(s.any()); mc_R.noise += int(n.any())
            mc_R.rect  += int(r.any()); mc_R.drift += int(d.any())

        marker_rows.append((marker, mc_T, mc_R, n_nan))
        sess_T = sess_T + mc_T
        sess_R = sess_R + mc_R

    session_results.append((session.name, n_frames, sess_T, sess_R, marker_rows))
    grand_T = grand_T + sess_T
    grand_R = grand_R + sess_R


# ── Print session reports ─────────────────────────────────────────────────────

COL = 35
HDR = f"  {'':.<{COL}} {'NaN%':>6}  {'Spike':>6} {'Noise':>6} {'Rect':>6} {'Drift':>6}  {'Total':>6}"
SEP = f"  {'':-<{COL}} {'------':>6}  {'------':>6} {'------':>6} {'------':>6} {'------':>6}  {'------':>6}"

for sess_name, n_frames, sess_T, sess_R, marker_rows in session_results:
    print(f"{'='*78}")
    print(f"Session: {sess_name}  ({n_frames} frames)")
    sess_nan = sum(n for _, _, _, n in marker_rows)

    for label, counts_list in [("TRANSLATION", [(m, mc_T, n) for m, mc_T, _, n in marker_rows]),
                                ("ROTATION",    [(m, mc_R, n) for m, _, mc_R, n in marker_rows])]:
        print(f"\n  ── {label} ──")
        print(HDR); print(SEP)
        total_c = AnomalyCount()
        for marker, mc, n_nan in counts_list:
            nan_pct = 100 * n_nan / n_frames if n_frames else 0
            print(f"  {marker:<{COL}} {nan_pct:>5.1f}%  "
                  f"{mc.spike:>6} {mc.noise:>6} {mc.rect:>6} {mc.drift:>6}  "
                  f"{mc.total():>6}")
            total_c = total_c + mc
        avg_nan = 100 * sess_nan / (n_frames * len(MARKERS)) if n_frames else 0
        print(f"  {'SESSION TOTAL':<{COL}} {avg_nan:>5.1f}%  "
              f"{total_c.spike:>6} {total_c.noise:>6} {total_c.rect:>6} "
              f"{total_c.drift:>6}  {total_c.total():>6}")
    print()


# ── Grand totals ──────────────────────────────────────────────────────────────
print(f"{'='*78}")
print(f"GRAND TOTAL (all sessions, all markers, all runs)")
for label, gc in [("TRANSLATION", grand_T), ("ROTATION", grand_R)]:
    print(f"\n  ── {label} ──")
    print(HDR); print(SEP)
    print(f"  {'ALL':<{COL}} {'':>6}  "
          f"{gc.spike:>6} {gc.noise:>6} {gc.rect:>6} {gc.drift:>6}  "
          f"{gc.total():>6}")

print(f"""
Notes:
  NaN%      = frames where marker was NOT detected
  Spike     = 1–{SPIKE_MAX_LEN} frame deviation then returns  (count = number of spike events)
  Noise     = sustained high std > {NOISE_MIN_LEN} consecutive frames
  Rect      = flat plateau deviation lasting {RECT_MIN_LEN}–{RECT_MAX_LEN} frames
  Drift     = gradual monotonic shift over >= {DRIFT_MIN_RUN} frames
  (T) units = metres;  (R) units = radians (rotvec components)
  Source labelling per valid run:
    T only  — artifact in translation but NOT rotation
    R only  — artifact in rotation but NOT translation
    T+R     — artifact detected in both independently
""")


# ── Per-frame source table for 2026-05-11 ────────────────────────────────────
TARGET_FRAMES = sorted([
    58, 60, 66, 72, 74, 78, 79, 81, 82, 84, 85, 87, 89,
    93, 95, 96, 100, 107, 109, 113, 122, 123, 126,
    141, 142, 143, 144, 145, 151, 154, 155, 165, 168,
    171, 174, 178, 179, 181, 193
])

TARGET_SESSION = next(
    (s for s in sessions if s.name == "2026-05-11"), None)

if TARGET_SESSION:
    print(f"{'='*78}")
    print(f"PER-FRAME SOURCE TABLE — Session 2026-05-11")
    print(f"  T = translation artifact   R = rotation artifact")
    print(f"  Source: T / R / T+R / NaN / OK\n")

    frame_dirs = sorted(TARGET_SESSION.glob("frame_*"),
                        key=lambda d: int(d.name.split("_")[1]))
    frame_indices = [int(d.name.split("_")[1]) for d in frame_dirs]
    N = len(frame_dirs)

    # Build per-frame per-marker source maps
    # source_map[marker][frame_idx] = "T" | "R" | "T+R" | "NaN" | "OK"
    source_map = {m: {} for m in MARKERS}

    for marker in MARKERS:
        raw = []
        for fd in frame_dirs:
            try:
                T = np.loadtxt(fd / f"pose_{marker}.txt")
                raw.append(None if np.any(np.isnan(T)) else T)
            except Exception:
                raw.append(None)

        # per-frame artifact flags
        t_spike = np.zeros(N, dtype=bool)
        t_noise = np.zeros(N, dtype=bool)
        t_rect  = np.zeros(N, dtype=bool)
        t_drift = np.zeros(N, dtype=bool)
        r_spike = np.zeros(N, dtype=bool)
        r_noise = np.zeros(N, dtype=bool)
        r_rect  = np.zeros(N, dtype=bool)
        r_drift = np.zeros(N, dtype=bool)

        # build valid runs
        runs = []
        run_idx = []
        for i, v in enumerate(raw):
            if v is not None:
                run_idx.append(i)
            else:
                if run_idx: runs.append(run_idx); run_idx = []
        if run_idx: runs.append(run_idx)

        for run in runs:
            trans_xyz = np.array([raw[i][:3, 3]    for i in run])
            rot_mats  = [raw[i][:3, :3] for i in run]
            rot_rvec  = rotmats_to_rotvecs(rot_mats)

            # Translation
            ms, mn, mr, md = artifact_mask_for_run(
                trans_xyz,
                SPIKE_THRESH, NOISE_STD_THRESH, RECT_THRESH,
                DRIFT_SLOPE_THRESH, DRIFT_TOTAL_THRESH)
            for k, gi in enumerate(run):
                t_spike[gi] |= ms[k]; t_noise[gi] |= mn[k]
                t_rect[gi]  |= mr[k]; t_drift[gi]  |= md[k]

            # Rotation
            ms, mn, mr, md = artifact_mask_for_run(
                rot_rvec,
                ROT_SPIKE_THRESH, ROT_NOISE_THRESH, ROT_RECT_THRESH,
                ROT_DRIFT_SLOPE, ROT_DRIFT_TOTAL)
            for k, gi in enumerate(run):
                r_spike[gi] |= ms[k]; r_noise[gi] |= mn[k]
                r_rect[gi]  |= mr[k]; r_drift[gi]  |= md[k]

        for gi, fi in enumerate(frame_indices):
            is_nan = raw[gi] is None
            t_any  = t_spike[gi] or t_noise[gi] or t_rect[gi] or t_drift[gi]
            r_any  = r_spike[gi] or r_noise[gi] or r_rect[gi] or r_drift[gi]

            if is_nan:
                cell = "NaN"
            elif t_any and r_any:
                # build detail label
                t_parts = ([f"Sp" if t_spike[gi] else ""] +
                           [f"No" if t_noise[gi] else ""] +
                           [f"Re" if t_rect[gi]  else ""] +
                           [f"Dr" if t_drift[gi] else ""])
                r_parts = ([f"Sp" if r_spike[gi] else ""] +
                           [f"No" if r_noise[gi] else ""] +
                           [f"Re" if r_rect[gi]  else ""] +
                           [f"Dr" if r_drift[gi] else ""])
                tl = "+".join(p for p in t_parts if p)
                rl = "+".join(p for p in r_parts if p)
                cell = f"T+R ({tl} / {rl})"
            elif t_any:
                t_parts = ([f"Sp" if t_spike[gi] else ""] +
                           [f"No" if t_noise[gi] else ""] +
                           [f"Re" if t_rect[gi]  else ""] +
                           [f"Dr" if t_drift[gi] else ""])
                cell = "T (" + "+".join(p for p in t_parts if p) + ")"
            elif r_any:
                r_parts = ([f"Sp" if r_spike[gi] else ""] +
                           [f"No" if r_noise[gi] else ""] +
                           [f"Re" if r_rect[gi]  else ""] +
                           [f"Dr" if r_drift[gi] else ""])
                cell = "R (" + "+".join(p for p in r_parts if p) + ")"
            else:
                cell = "OK"

            source_map[marker][fi] = cell

    # Print table
    W = 22
    print(f"  {'Frame':>5}  {'RW (right_wrist)':<{W}} {'RE (right_elbow)':<{W}} "
          f"{'LW (left_wrist)':<{W}} {'LE (left_elbow)':<{W}}")
    print(f"  {'-----':>5}  {'':-<{W}} {'':-<{W}} {'':-<{W}} {'':-<{W}}")

    for fi in TARGET_FRAMES:
        row = [source_map[m].get(fi, "?") for m in MARKERS]
        print(f"  {fi:>5}  {row[0]:<{W}} {row[1]:<{W}} {row[2]:<{W}} {row[3]:<{W}}")

    print(f"""
  Legend (detail codes):
    Sp = Spike     No = Noise     Re = Rect     Dr = Drift
    T  = translation source only
    R  = rotation source only
    T+R (Sp+No / Sp) = T has Spike+Noise, R has Spike
""")
