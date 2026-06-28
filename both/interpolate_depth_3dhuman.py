#!/usr/bin/env python3
"""
Post-process a saved session: depth-assisted artifact detection and interpolation.

Enhanced pipeline over interpolate_modify.py with L515 depth-assisted Kalman.

Step 0 (NEW): Depth pre-fill
    For frames where ArUco RGB detection failed (pose is None), attempt to
    estimate the 3D world position of each arm marker by searching the L515
    world-frame point cloud (pointcloud.npy) near the linearly-interpolated
    expected position.
    Reference: Zhao et al. (2012) SCA — "Combining Marker-based Mocap and
    RGB-D Camera for Acquiring High-fidelity Hand Motion Data."

Stages 1-4: Paper 1 artifact detection (unchanged from interpolate_modify.py).
  Stage 1: Single peak    isolated 1-5 frame jumps
  Stage 2: Heavy noise    sustained high-amplitude fluctuation
  Stage 3: Step change    rectangular plateau from mislabelled marker
  Stage 4: Slow drift     gradual accumulation (FFNN loose predictor)

Depth-assisted Kalman smoother (replaces B&L for Stages 1-2):
    Four measurement noise levels (R) per joint per frame:
      RGB valid (ArUco detected, not artifact) → R = R_RGB²        (low)
      RGB absent/artifact, depth available     → R = R_depth²      (medium)
      Kinematic estimate (elbow from wrist)    → R = R_kinematic²  (high)
      All unavailable                          → predict-only (no update)
    Equation: K_t = P̂_t Ĥ_t^T (Ĥ_t P̂_t Ĥ_t^T + R_t)^{-1}
              m_t = m̂_t + K_t(z_t - Ĥ_t m̂_t)
    R_t is a non-uniform diagonal matrix reflecting per-dimension quality.

Kinematic constraint (3D human model extension):
    For elbow frames with no RGB or depth observation, a position estimate is
    derived from the wrist position and a session-calibrated forearm length:
      e_kinematic = wrist_pos + L_forearm * (e_pred - wrist_pos) / ||e_pred - wrist_pos||
    e_pred is the Kalman-predicted elbow projected onto the sphere of radius
    L_forearm centred on the wrist.  L_forearm is calibrated per-session as
    the median wrist-elbow distance over frames where both are simultaneously
    visible (fallback: KIN_FOREARM_DEFAULT).

Interpolation (translation):
  Stages 1-2  Depth-assisted Burke & Lasenby Kalman Smoothing
              fallback to linear when <BL_MIN_COMPLETE complete frames
  Stage 3     Linear bridging over detected step region
  Stage 4     Replace with FFNN prediction P_NN(n)

Interpolation (rotation — Haarbach et al. 2018 survey, Eqs. 14-18):
  Stage 1     SLERP  — C0, 2 anchors
  Stage 2     SQUAD  — C1, 4 keyframes
  Stages 3-4  CuBsp  — C2, cumulative B-spline

Output: pose_<joint>_processed.txt alongside originals.

Usage:
    conda activate massage
    python interpolate_depth_3dhuman.py <session_dir>
    python interpolate_depth_3dhuman.py <session_dir> --no-bl --no-ffnn --no-depth --dry_run
"""

import argparse
import os
import sys
import warnings

import numpy as np
from scipy.ndimage import binary_closing, binary_opening, grey_closing, median_filter
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation, Slerp
from sklearn.neural_network import MLPRegressor

ARM_JOINTS = ['right_wrist', 'right_elbow', 'left_wrist', 'left_elbow']

# ── Stage 1 — Single Peak (Paper 1 §3.4.2) ───────────────────────────────────
PEAK_MEDIAN_WINDOW = 19     # median filter window; must be >> max peak length
PEAK_K1            = 3.0    # sigma multiplier for D_HP threshold  (threshold1)
PEAK_AMP_RATIO_T   = 5.0    # ampRatio threshold                   (threshold2)
PEAK_MAX_SIZE      = 5      # max peak duration in frames          (maxSize)

# ── Stage 2 — Heavy Noise (Paper 1 §3.4.3) ───────────────────────────────────
NOISE_SG_POLY  = 5          # Savitzky-Golay polynomial order   L
NOISE_SG_WIN   = 13         # Savitzky-Golay window (odd)       M
NOISE_K2       = 2.0        # sigma multiplier for noise threshold
NOISE_MIN_LEN  = 20         # minimum noise segment in frames   minLen

# ── Stage 3 — Step Change (Paper 1 §3.4.4) ───────────────────────────────────
# Paper 1 §3.4.4: "Default parameters of Savitzky-Golay are the same as for
# heavy noise L=5, M=13" — Stage 3 reuses NOISE_SG_* vars.
STEP_K3       = 3.0         # sigma multiplier for derivative threshold
STEP_MIN_LEN  = 20          # minimum step segment in frames
STEP_MAX_DIST = 200         # max frame-distance between complementary spikes

# ── Stage 4 — Slow Drift (Paper 1 §3.4.5) ────────────────────────────────────
DRIFT_SG_POLY     = 7       # SG polynomial for residual smoothing
DRIFT_SG_WIN      = 11      # SG window for residual smoothing (odd)
DRIFT_K_UPPER     = 3.0     # kU   — upper hysteresis threshold
DRIFT_K_LOWER     = 0.5     # k_hv — lower expansion threshold
DRIFT_MIN_LEN     = 30      # tau_hv — min drift frames (30@30fps ~ 1 s)
DRIFT_MA_WINDOW   = 200     # moving-average window for FFNN self-input
FFNN_K            = 1       # context size  (k past sibling values)
FFNN_L            = 2       # polynomial degree for FFNN input features
FFNN_REPLICATIONS = 5       # M-fold output replication -> averaged prediction

# ── Burke & Lasenby (Paper 3) ─────────────────────────────────────────────────
BL_MIN_COMPLETE  = 10       # minimum complete frames to attempt B&L
BL_MIN_JOINTS_TRAIN = 2     # minimum valid joints per frame for SVD training (Fix 1)
BL_ENERGY_THRESH = 0.99     # gamma — SVD energy fraction to retain
BL_Q_SIGMA       = 0.05     # process noise sigma  (m) — increased to track faster arm motion
BL_R_SIGMA       = 0.03     # RGB measurement noise sigma  (m)

# ── Depth assistance — Zhao et al. (2012), §5.2 depth term ───────────────────
# L515 depth is less accurate than RGB ArUco → R_depth >> R_RGB so the
# smoother down-weights depth observations relative to clean RGB ones.
BL_R_DEPTH_SIGMA    = 0.04  # depth-only measurement noise sigma (m)
DEPTH_SEARCH_RADIUS = 0.12  # max L515 point distance from expected pos (m)
DEPTH_MIN_POINTS    = 3     # min L515 points in radius to accept estimate

INNOVATION_WINDOW   = 20    # frames of innovation history for adaptive Q

# ── Kinematic constraints — 3D human model ────────────────────────────────────
KIN_FOREARM_PAIRS    = [('right_elbow', 'right_wrist'),
                        ('left_elbow',  'left_wrist')]
KIN_FOREARM_DEFAULT  = 0.27   # fallback forearm length (m) if calibration fails
KIN_MIN_CALIB_FRAMES = 10     # min co-visible frames needed to calibrate bone length
BL_R_KINEMATIC_SIGMA = 0.12   # kinematic measurement noise sigma (m)


# =============================================================================
# Shared math helpers (kept from interpolate_poses.py)
# =============================================================================

def mat_to_rt(T: np.ndarray):
    return Rotation.from_matrix(T[:3, :3]), T[:3, 3]


def rt_to_mat(R: Rotation, t: np.ndarray) -> np.ndarray:
    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = R.as_matrix()
    M[:3, 3] = t
    return M


def interpolate_gap(T_before: np.ndarray, T_after: np.ndarray,
                    n_fill: int) -> list:
    """SLERP rotation + linear translation between two 4x4 transforms."""
    R_b, t_b = mat_to_rt(T_before)
    R_a, t_a = mat_to_rt(T_after)
    slerp = Slerp([0.0, 1.0], Rotation.concatenate([R_b, R_a]))
    result = []
    for k in range(1, n_fill + 1):
        alpha = k / (n_fill + 1)
        result.append(rt_to_mat(slerp(alpha), (1 - alpha) * t_b + alpha * t_a))
    return result


# =============================================================================
# Quaternion primitives for SQUAD and CuBsp
# (Haarbach et al. 2018 survey — Eqs. 7, 12-18)
# Convention throughout: q = [w, x, y, z]
# =============================================================================

def _quat_from_matrix(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> unit quaternion [w, x, y, z]."""
    # scipy stores as [x, y, z, w]; reorder on the way out
    return Rotation.from_matrix(R).as_quat()[[3, 0, 1, 2]]


def _matrix_from_quat(q: np.ndarray) -> np.ndarray:
    """Unit quaternion [w, x, y, z] -> 3x3 rotation matrix."""
    return Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()


def _quat_mul(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Hamilton product.  (Eq. 7 in survey)"""
    pw, pv = p[0], p[1:]
    qw, qv = q[0], q[1:]
    return np.array([pw * qw - np.dot(pv, qv),
                     *(pw * qv + qw * pv + np.cross(pv, qv))])


def _quat_conj(q: np.ndarray) -> np.ndarray:
    """Conjugate = inverse for unit quaternions."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _quat_exp(omega: np.ndarray) -> np.ndarray:
    """su(2) -> SU(2): angular velocity vector -> unit quaternion.  (Eq. 12)"""
    theta = np.linalg.norm(omega)
    if theta < 1e-10:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return np.array([np.cos(theta / 2.0),
                     *(np.sin(theta / 2.0) * omega / theta)])


def _quat_log(q: np.ndarray) -> np.ndarray:
    """SU(2) -> su(2): unit quaternion -> angular velocity vector.  (Eq. 13)"""
    qw  = float(np.clip(q[0], -1.0, 1.0))
    qv  = q[1:]
    qvn = np.linalg.norm(qv)
    if qvn < 1e-10:
        return np.zeros(3)
    return 2.0 * np.arccos(qw) * qv / qvn


def _quat_same_hemisphere(q0: np.ndarray, q1: np.ndarray) -> np.ndarray:
    """Return q1 (or -q1) so that dot(q0, q1) >= 0 — avoids long-way-round path."""
    return q1 if np.dot(q0, q1) >= 0.0 else -q1


def _slerp_quat(q0: np.ndarray, q1: np.ndarray, u: float) -> np.ndarray:
    """SLERP in pure quaternion algebra.  (Eq. 14 in survey)"""
    q1 = _quat_same_hemisphere(q0, q1)
    return _quat_mul(q0, _quat_exp(u * _quat_log(_quat_mul(_quat_conj(q0), q1))))


# ── SQUAD  (Shoemake 1987, Eqs. 15-16 in survey) ─────────────────────────────

def _squad_inner_point(q_prev: np.ndarray, q_i: np.ndarray,
                       q_next: np.ndarray) -> np.ndarray:
    """Compute inner control point s_i for SQUAD.  (Eq. 16 in survey)"""
    q_next = _quat_same_hemisphere(q_i, q_next)
    q_prev = _quat_same_hemisphere(q_i, q_prev)
    log1 = _quat_log(_quat_mul(_quat_conj(q_i), q_next))
    log2 = _quat_log(_quat_mul(_quat_conj(q_i), q_prev))
    return _quat_mul(q_i, _quat_exp(-(log1 + log2) / 4.0))


def _squad(q_i: np.ndarray, s_i: np.ndarray,
           s_next: np.ndarray, q_next: np.ndarray, u: float) -> np.ndarray:
    """SQUAD: C1 quaternion interpolation.  (Eq. 15 in survey)"""
    return _slerp_quat(
        _slerp_quat(q_i, q_next, u),
        _slerp_quat(s_i, s_next, u),
        2.0 * u * (1.0 - u)
    )


# ── Cumulative B-spline on SU(2)  (Kim et al. 1995, Eqs. 17-18 in survey) ────

# Cumulative basis matrix: Ñ = [u³, u², u, 1] · _CUBSP_C  gives [N0,N1,N2,N3]
_CUBSP_C = (1.0 / 6.0) * np.array([
    [0,  1, -2, 1],
    [0, -3,  3, 0],
    [0,  3,  3, 0],
    [6,  5,  1, 0],
], dtype=np.float64)


def _cubsp_basis(u: float):
    """Return (N1, N2, N3) cumulative B-spline basis values at u.  (Eq. 18)"""
    vec = np.array([u ** 3, u ** 2, u, 1.0])
    N = vec @ _CUBSP_C      # [N0=1, N1, N2, N3]
    return N[1], N[2], N[3]


def _cubsp(q0: np.ndarray, q1: np.ndarray,
           q2: np.ndarray, q3: np.ndarray, u: float) -> np.ndarray:
    """
    Cubic cumulative B-spline on SU(2).  (Eq. 17 in survey)
    q(u) = q0 · exp(N1·log(q0⁻¹q1)) · exp(N2·log(q1⁻¹q2)) · exp(N3·log(q2⁻¹q3))
    q0 = context before gap, q1 = anchor before, q2 = anchor after, q3 = context after.
    """
    q1 = _quat_same_hemisphere(q0, q1)
    q2 = _quat_same_hemisphere(q1, q2)
    q3 = _quat_same_hemisphere(q2, q3)
    N1, N2, N3 = _cubsp_basis(u)
    w1 = _quat_log(_quat_mul(_quat_conj(q0), q1))
    w2 = _quat_log(_quat_mul(_quat_conj(q1), q2))
    w3 = _quat_log(_quat_mul(_quat_conj(q2), q3))
    return _quat_mul(
        _quat_mul(
            _quat_mul(q0, _quat_exp(N1 * w1)),
            _quat_exp(N2 * w2)
        ),
        _quat_exp(N3 * w3)
    )


# =============================================================================
# Data I/O
# =============================================================================

def load_poses(session_dir: str, joint: str) -> list:
    frames = sorted(
        [d for d in os.listdir(session_dir) if d.startswith('frame_')],
        key=lambda x: int(x.split('_')[1])
    )
    poses = []
    for frame in frames:
        path = os.path.join(session_dir, frame, f'pose_{joint}.txt')
        if os.path.exists(path):
            T = np.loadtxt(path)
            poses.append(None if np.any(np.isnan(T)) else T)
        else:
            poses.append(None)
    return poses


def extract_translations(poses: list) -> list:
    return [p[:3, 3].copy() if p is not None else None for p in poses]


def get_valid_runs(seq: list) -> list:
    """Return [(start, end_exclusive), ...] for contiguous non-None runs."""
    runs, i, n = [], 0, len(seq)
    while i < n:
        if seq[i] is not None:
            j = i
            while j < n and seq[j] is not None:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


def bool_runs(mask: np.ndarray) -> list:
    """Return [(start, end_exclusive), ...] for True runs in a boolean array."""
    runs, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


# =============================================================================
# Stage 1 — Single Peak Detection  (Paper 1 §3.4.2, Eqs. 18-25)
# =============================================================================

def _binary_tophat(binary: np.ndarray, size: int) -> np.ndarray:
    """Binary top-hat = signal - morphological_opening(signal).  (Eq. 17/25)"""
    opened = binary_opening(binary, structure=np.ones(size, dtype=bool))
    return binary & ~opened


def _detect_peaks_1d(x: np.ndarray) -> np.ndarray:
    """Single-peak detection on a 1-D valid run. Returns bool array."""
    n = len(x)
    if n < PEAK_MEDIAN_WINDOW + 2:
        return np.zeros(n, dtype=bool)

    # Eq. 18: median high-pass
    X_HP = x - median_filter(x, size=PEAK_MEDIAN_WINDOW, mode='reflect')

    # Eq. 19: differential (padded back to length n)
    D_HP = np.append(np.diff(X_HP), 0.0)

    # Eq. 20: zero out sub-threshold values
    sigma = D_HP.std(ddof=0) + 1e-12
    D_tilde = np.where(np.abs(D_HP) > PEAK_K1 * sigma, D_HP, 0.0)

    # Eqs. 21-23: amplitude ratio over window W_n of size 2*maxSize+3
    half = (2 * PEAK_MAX_SIZE + 3) // 2
    mov_sum     = np.zeros(n)
    mov_sum_abs = np.zeros(n)
    for i in range(n):
        seg = D_tilde[max(0, i - half): min(n, i + half + 1)]
        mov_sum[i]     = seg.sum()
        mov_sum_abs[i] = np.abs(seg).sum()
    amp_ratio = mov_sum_abs / (np.abs(mov_sum) + 1e-12)

    # Eq. 24: peak candidates
    candidates = amp_ratio > PEAK_AMP_RATIO_T

    # Eq. 25: binary top-hat — keep only isolated peaks, discard clusters
    return _binary_tophat(candidates, size=PEAK_MAX_SIZE)


def detect_single_peaks(trans_joint: list) -> np.ndarray:
    """Per-joint single-peak mask (union across x/y/z axes, valid runs only)."""
    n = len(trans_joint)
    mask = np.zeros(n, dtype=bool)
    for start, end in get_valid_runs(trans_joint):
        if end - start < 3:
            continue
        xyz = np.array([trans_joint[i] for i in range(start, end)])
        for ax in range(3):
            mask[start:end] |= _detect_peaks_1d(xyz[:, ax])
    return mask


# =============================================================================
# Stage 2 — Heavy Noise Detection  (Paper 1 §3.4.3, Eqs. 26-30)
# =============================================================================

def _sg_highpass(x: np.ndarray, win: int, poly: int) -> np.ndarray:
    """x - SG_lowpass(x).  Clamps window to valid odd size >= poly+2."""
    win = min(win, len(x) if len(x) % 2 == 1 else max(1, len(x) - 1))
    min_win = poly + 2 if (poly + 2) % 2 == 1 else poly + 3
    win = max(win, min_win)
    if win > len(x):
        return x - x.mean()
    return x - savgol_filter(x, window_length=win, polyorder=poly)


def _detect_noise_1d(x: np.ndarray) -> np.ndarray:
    """Heavy-noise detection on a 1-D valid run. Returns bool array."""
    n = len(x)
    if n < max(NOISE_SG_WIN, 2 * NOISE_MIN_LEN):
        return np.zeros(n, dtype=bool)

    # Eq. 26: differential first
    D1 = np.append(np.diff(x), 0.0)

    # Eq. 27: SG high-pass applied to D1  (not to position — see §3.4.3)
    D1_HP = _sg_highpass(D1, NOISE_SG_WIN, NOISE_SG_POLY)

    # Eq. 28: float morphological closing of |D1_HP|
    struct_size = max(3, 2 * NOISE_MIN_LEN - 1)
    D1_HP_cleaned = grey_closing(np.abs(D1_HP), size=struct_size)

    # Eq. 29: threshold
    sigma = D1_HP.std(ddof=0) + 1e-12
    raw_noise = D1_HP_cleaned > (NOISE_K2 * sigma)

    # Eq. 30: binary morphological closing to fill holes inside noise bursts
    heavy = binary_closing(raw_noise, structure=np.ones(struct_size, dtype=bool))

    # Reject segments shorter than NOISE_MIN_LEN
    result = np.zeros(n, dtype=bool)
    for s, e in bool_runs(heavy):
        if e - s >= NOISE_MIN_LEN:
            result[s:e] = True
    return result


def detect_heavy_noise(trans_joint: list,
                       exclude_mask: np.ndarray = None) -> np.ndarray:
    """exclude_mask: frames already filled by earlier stages — treated as None
    so detection never runs on interpolated data (same pattern as detect_step_changes)."""
    n = len(trans_joint)
    mask = np.zeros(n, dtype=bool)
    visible = [None if (exclude_mask is not None and exclude_mask[i])
               else trans_joint[i]
               for i in range(n)]
    for start, end in get_valid_runs(visible):
        if end - start < 3:
            continue
        xyz = np.array([visible[i] for i in range(start, end)])
        for ax in range(3):
            mask[start:end] |= _detect_noise_1d(xyz[:, ax])
    return mask


# =============================================================================
# Stage 3 — Step Change Detection  (Paper 1 §3.4.4)
# =============================================================================

def _find_derivative_pairs(D_HP: np.ndarray, threshold: float) -> np.ndarray:
    """
    Scan for complementary spike pairs (opposite sign, within STEP_MAX_DIST).
    Region between each pair is marked as step-change candidate.
    """
    n = len(D_HP)
    result = np.zeros(n, dtype=bool)
    i = 0
    while i < n:
        if np.abs(D_HP[i]) > threshold:
            sign_i = np.sign(D_HP[i])
            for j in range(i + 1, min(n, i + STEP_MAX_DIST + 1)):
                if np.abs(D_HP[j]) > threshold and np.sign(D_HP[j]) != sign_i:
                    if j - i >= STEP_MIN_LEN:
                        result[i: j + 1] = True
                    i = j
                    break
        i += 1
    return result


def _detect_step_1d(x: np.ndarray) -> np.ndarray:
    n = len(x)
    if n < NOISE_SG_WIN + 2:
        return np.zeros(n, dtype=bool)

    # Paper 1 §3.4.4: "first two steps are shared with heavy noise" (Eqs. 26-27)
    D1 = np.append(np.diff(x), 0.0)
    D1_HP = _sg_highpass(D1, NOISE_SG_WIN, NOISE_SG_POLY)

    threshold = STEP_K3 * (D1_HP.std(ddof=0) + 1e-12)
    candidates = _find_derivative_pairs(D1_HP, threshold)

    result = np.zeros(n, dtype=bool)
    for s, e in bool_runs(candidates):
        if e - s >= STEP_MIN_LEN:
            result[s:e] = True
    return result


def detect_step_changes(trans_joint: list,
                        exclude_mask: np.ndarray = None) -> np.ndarray:
    """
    exclude_mask: bool array — frames filled by earlier stages are treated as
    None so the step detector never runs on interpolated data (Paper 1 §4.3).
    """
    n = len(trans_joint)
    mask = np.zeros(n, dtype=bool)
    # Hide interpolated frames from valid-run detection
    visible = [None if (exclude_mask is not None and exclude_mask[i])
               else trans_joint[i]
               for i in range(n)]
    for start, end in get_valid_runs(visible):
        if end - start < 3:
            continue
        xyz = np.array([visible[i] for i in range(start, end)])
        for ax in range(3):
            mask[start:end] |= _detect_step_1d(xyz[:, ax])
    return mask


# =============================================================================
# Depth pre-fill — Zhao et al. (2012) §5.2 (depth term concept)
# =============================================================================

def _expected_position(trans_seq: list, t: int):
    """
    Linear interpolation (or nearest-neighbor) from valid frames around index t.
    Returns None if no valid frames exist anywhere in the sequence.
    """
    n = len(trans_seq)
    t_before = next((k for k in range(t - 1, -1, -1) if trans_seq[k] is not None), None)
    t_after  = next((k for k in range(t + 1, n)       if trans_seq[k] is not None), None)

    if t_before is not None and t_after is not None:
        alpha = (t - t_before) / (t_after - t_before)
        return (1.0 - alpha) * trans_seq[t_before] + alpha * trans_seq[t_after]
    if t_before is not None:
        return trans_seq[t_before].copy()
    if t_after is not None:
        return trans_seq[t_after].copy()
    return None


def preload_pointclouds(session_dir: str, frames: list) -> dict:
    """Load all pointcloud.npy files into memory once. Returns {frame_idx: ndarray|None}."""
    pcs = {}
    for f_idx, frame in enumerate(frames):
        pc_path = os.path.join(session_dir, frame, 'pointcloud.npy')
        if not os.path.exists(pc_path):
            pcs[f_idx] = None
            continue
        try:
            pc = np.load(pc_path)
            pcs[f_idx] = pc if pc.shape[0] > 0 else None
        except Exception:
            pcs[f_idx] = None
    return pcs


def load_depth_positions(frames: list, trans: dict, pointclouds: dict) -> dict:
    """
    For each frame where an arm marker's RGB pose is None, attempt to recover
    a 3D world-frame position from the preloaded L515 point cloud.

    Uses linear interpolation to estimate the expected position (fallback path).
    The Kalman forward pass uses Kalman-predicted positions for depth search instead
    — this function is only used when B&L cannot run (fallback to linear fill).

    Returns:
        depth_xyz : {joint: [np.ndarray(3) | None, ...]}  length == len(frames)
    """
    depth_xyz = {j: [None] * len(frames) for j in ARM_JOINTS}

    for f_idx, frame in enumerate(frames):
        needs_depth = [j for j in ARM_JOINTS if trans[j][f_idx] is None]
        if not needs_depth:
            continue

        pc = pointclouds.get(f_idx)
        if pc is None:
            continue
        pts_world = pc[:, :3].astype(np.float64)   # (N, 3)

        for j in needs_depth:
            expected = _expected_position(trans[j], f_idx)
            if expected is None:
                continue

            dists = np.linalg.norm(pts_world - expected[np.newaxis, :], axis=1)
            mask = dists < DEPTH_SEARCH_RADIUS
            nearby_pts = pts_world[mask]
            if len(nearby_pts) >= DEPTH_MIN_POINTS:
                nearby_bgr = pc[mask, 3:6].astype(np.float64)
                brightness = nearby_bgr.mean(axis=1) + 1.0
                weights    = brightness / brightness.sum()
                depth_xyz[j][f_idx] = (nearby_pts * weights[:, np.newaxis]).sum(axis=0)

    return depth_xyz


# =============================================================================
# Kinematic calibration — session-specific bone length estimation
# =============================================================================

def calibrate_bone_lengths(trans_all: dict, artifact_mask: dict) -> dict:
    """
    Estimate forearm length for each arm from frames where both elbow and wrist
    are simultaneously visible and clean.

    Returns {(elbow_joint, wrist_joint): length_in_metres}.
    Falls back to KIN_FOREARM_DEFAULT when fewer than KIN_MIN_CALIB_FRAMES
    co-visible frames exist.
    """
    n = len(trans_all[ARM_JOINTS[0]])
    lengths = {}
    for elbow_j, wrist_j in KIN_FOREARM_PAIRS:
        dists = []
        for t in range(n):
            e = trans_all[elbow_j][t]
            w = trans_all[wrist_j][t]
            if (e is not None and w is not None
                    and not artifact_mask[elbow_j][t]
                    and not artifact_mask[wrist_j][t]):
                dists.append(np.linalg.norm(e - w))
        if len(dists) >= KIN_MIN_CALIB_FRAMES:
            lengths[(elbow_j, wrist_j)] = float(np.median(dists))
            print(f"  Bone length {elbow_j}←{wrist_j}: "
                  f"{lengths[(elbow_j, wrist_j)]:.3f} m  ({len(dists)} frames)")
        else:
            lengths[(elbow_j, wrist_j)] = KIN_FOREARM_DEFAULT
            print(f"  Bone length {elbow_j}←{wrist_j}: "
                  f"{KIN_FOREARM_DEFAULT:.3f} m  (default — only {len(dists)} frames)")
    return lengths


# =============================================================================
# Burke & Lasenby — Low-Dimensional Kalman Smoothing (Paper 3, Algorithm 1)
# =============================================================================

def burke_lasenby_smooth(trans_all: dict, artifact_mask: dict):
    """
    Joint 12-D Kalman smoother in a PCA subspace learned from complete frames.
    (Original B&L without depth assistance — used when --no-depth is set.)

    trans_all     : {joint: list[xyz|None]}   length N
    artifact_mask : {joint: bool ndarray}     frames flagged as artifact
    Returns       : {joint: list[xyz|None]}   updated translations,
                    or None when fewer than BL_MIN_COMPLETE complete frames.
    """
    joints = ARM_JOINTS
    n = len(trans_all[joints[0]])
    d = 3 * len(joints)   # 12

    # Training set: frames where all joints are valid AND not artifact-flagged
    train_rows = []
    for t in range(n):
        if all(trans_all[j][t] is not None and not artifact_mask[j][t]
               for j in joints):
            train_rows.append(np.concatenate([trans_all[j][t] for j in joints]))
    if len(train_rows) < BL_MIN_COMPLETE:
        return None

    X_train = np.array(train_rows)      # (M, 12)
    m_bar   = X_train.mean(axis=0)      # (12,)
    _, S_vals, Vt = np.linalg.svd(X_train - m_bar, full_matrices=False)

    energy = np.cumsum(S_vals ** 2) / (np.sum(S_vals ** 2) + 1e-12)
    d_hat  = int(np.searchsorted(energy, BL_ENERGY_THRESH)) + 1
    d_hat  = max(1, min(d_hat, len(S_vals)))
    V_hat  = Vt[:d_hat].T               # (12, d_hat)  — projection matrix

    Q = BL_Q_SIGMA ** 2 * np.eye(d_hat)

    # ── Forward Kalman pass (random walk: F = I) ──────────────────────────────
    m_fwd, P_fwd   = [None] * n, [None] * n
    m_pred, P_pred = [None] * n, [None] * n
    m_t = np.zeros(d_hat)
    P_t = np.eye(d_hat)

    for t in range(n):
        m_hat = m_t.copy()
        P_hat = P_t + Q
        m_pred[t], P_pred[t] = m_hat, P_hat

        # Build observation: joints visible and not artifact this frame
        obs_cols, obs_vals = [], []
        for j_idx, j in enumerate(joints):
            if trans_all[j][t] is not None and not artifact_mask[j][t]:
                base = j_idx * 3
                obs_cols.extend([base, base + 1, base + 2])
                obs_vals.extend(trans_all[j][t].tolist())

        if obs_cols:
            n_obs  = len(obs_cols)
            H_full = np.zeros((n_obs, d))
            for r, c in enumerate(obs_cols):
                H_full[r, c] = 1.0
            H_hat = H_full @ V_hat                      # (n_obs, d_hat)
            R_t   = BL_R_SIGMA ** 2 * np.eye(n_obs)    # per-frame size (Point 4 fix)

            z = np.array(obs_vals) - H_full @ m_bar
            S_inn = H_hat @ P_hat @ H_hat.T + R_t
            K     = P_hat @ H_hat.T @ np.linalg.inv(S_inn)
            m_t   = m_hat + K @ (z - H_hat @ m_hat)
            P_t   = (np.eye(d_hat) - K @ H_hat) @ P_hat
        else:
            m_t, P_t = m_hat, P_hat

        m_fwd[t], P_fwd[t] = m_t.copy(), P_t.copy()

    # ── Backward RTS pass ─────────────────────────────────────────────────────
    m_smo, P_smo = [None] * n, [None] * n
    m_smo[n - 1] = m_fwd[n - 1].copy()
    P_smo[n - 1] = P_fwd[n - 1].copy()

    for t in range(n - 2, -1, -1):
        C        = P_fwd[t] @ np.linalg.inv(P_pred[t + 1])
        m_smo[t] = m_fwd[t] + C @ (m_smo[t + 1] - m_pred[t + 1])
        P_smo[t] = P_fwd[t] + C @ (P_smo[t + 1] - P_pred[t + 1]) @ C.T

    # ── Project back, fill artifact / NaN frames ──────────────────────────────
    result = {j: list(trans_all[j]) for j in joints}
    for t in range(n):
        y_t = V_hat @ m_smo[t] + m_bar     # (12,)
        for j_idx, j in enumerate(joints):
            if trans_all[j][t] is None or artifact_mask[j][t]:
                result[j][t] = y_t[j_idx * 3: j_idx * 3 + 3]
    return result


def burke_lasenby_smooth_depth(trans_all: dict, artifact_mask: dict,
                                depth_xyz: dict, pointclouds: dict = None,
                                bone_lengths: dict = None):
    """
    Depth-assisted joint 12-D Kalman smoother.

    Extends burke_lasenby_smooth (Paper 3, Algorithm 1) with a three-level
    measurement noise scheme inspired by Zhao et al. (2012) §5.2:

      RGB valid  (trans not None, not artifact) → R = BL_R_SIGMA²      (low)
      Depth only (trans None or artifact,
                  depth_xyz available)           → R = BL_R_DEPTH_SIGMA² (medium)
      Both fail                                  → predict-only (no update, P grows)

    The non-uniform diagonal R_t matrix lets the smoother weight each joint's
    measurement by its quality independently.  Depth observations anchor the
    Kalman state during gaps and prevent unbounded uncertainty growth, without
    over-trusting the noisier depth data.

    Returns {joint: list[xyz|None]} or None when training data insufficient.
    """
    joints = ARM_JOINTS
    n = len(trans_all[joints[0]])
    d = 3 * len(joints)   # 12

    # Per-joint means for imputing missing joints in partial training rows
    joint_means = {}
    for j in joints:
        vals = [trans_all[j][t] for t in range(n)
                if trans_all[j][t] is not None and not artifact_mask[j][t]]
        joint_means[j] = np.mean(vals, axis=0) if vals else np.zeros(3)

    # SVD subspace: include frames where >= BL_MIN_JOINTS_TRAIN joints are valid.
    # Missing joints are filled with depth_xyz if available, else joint mean (Fix 1).
    train_rows = []
    for t in range(n):
        valid_count = sum(
            1 for j in joints
            if trans_all[j][t] is not None and not artifact_mask[j][t])
        if valid_count < BL_MIN_JOINTS_TRAIN:
            continue
        row = []
        for j in joints:
            if trans_all[j][t] is not None and not artifact_mask[j][t]:
                row.extend(trans_all[j][t].tolist())
            elif depth_xyz[j][t] is not None:
                row.extend(depth_xyz[j][t].tolist())
            else:
                row.extend(joint_means[j].tolist())
        train_rows.append(row)
    if len(train_rows) < BL_MIN_COMPLETE:
        return None

    X_train = np.array(train_rows)
    m_bar   = X_train.mean(axis=0)
    _, S_vals, Vt = np.linalg.svd(X_train - m_bar, full_matrices=False)

    energy = np.cumsum(S_vals ** 2) / (np.sum(S_vals ** 2) + 1e-12)
    d_hat  = int(np.searchsorted(energy, BL_ENERGY_THRESH)) + 1
    d_hat  = max(1, min(d_hat, len(S_vals)))
    V_hat  = Vt[:d_hat].T   # (12, d_hat)

    # Use pre-calibrated bone lengths (from raw data before any filling).
    # Fall back to in-place calibration only when called without pre-calibration.
    if bone_lengths is None:
        bone_lengths = calibrate_bone_lengths(trans_all, artifact_mask)

    Q = BL_Q_SIGMA ** 2 * np.eye(d_hat)

    # ── Forward Kalman pass ───────────────────────────────────────────────────
    m_fwd, P_fwd   = [None] * n, [None] * n
    m_pred, P_pred = [None] * n, [None] * n
    m_t = np.zeros(d_hat)
    P_t = np.eye(d_hat)
    innovations = []    # innovation magnitude history for adaptive Q

    for t in range(n):
        m_hat = m_t.copy()
        P_hat = P_t + Q
        m_pred[t], P_pred[t] = m_hat, P_hat

        obs_cols, obs_vals, obs_r = [], [], []
        for j_idx, j in enumerate(joints):
            base   = j_idx * 3
            rgb_ok = (trans_all[j][t] is not None and not artifact_mask[j][t])

            if rgb_ok:
                # RGB valid — low measurement noise R_RGB
                obs_cols.extend([base, base + 1, base + 2])
                obs_vals.extend(trans_all[j][t].tolist())
                obs_r.extend([BL_R_SIGMA ** 2] * 3)
            elif pointclouds is None and depth_xyz[j][t] is not None:
                # No preloaded clouds — fall back to linear-interp-guided depth_xyz
                obs_cols.extend([base, base + 1, base + 2])
                obs_vals.extend(depth_xyz[j][t].tolist())
                obs_r.extend([BL_R_DEPTH_SIGMA ** 2] * 3)

        # y_hat_world is used by both the Kalman-guided depth search and the
        # kinematic constraint below — compute once here.
        y_hat_world = V_hat @ m_hat + m_bar   # predicted full state in world coords

        # Kalman-guided depth search: use Kalman predicted state m_hat as search
        # centre instead of linear interpolation (Zhao 2012 §5.2, improved).
        if pointclouds is not None:
            pc = pointclouds.get(t)
            if pc is not None:
                pts_world = pc[:, :3].astype(np.float64)
                for j_idx, j in enumerate(joints):
                    base = j_idx * 3
                    if base in obs_cols:
                        continue
                    expected   = y_hat_world[base: base + 3]
                    dists      = np.linalg.norm(
                        pts_world - expected[np.newaxis, :], axis=1)
                    near       = dists < DEPTH_SEARCH_RADIUS
                    nearby_pts = pts_world[near]
                    if len(nearby_pts) >= DEPTH_MIN_POINTS:
                        nearby_bgr = pc[near, 3:6].astype(np.float64)
                        brightness = nearby_bgr.mean(axis=1) + 1.0
                        weights    = brightness / brightness.sum()
                        obs_cols.extend([base, base + 1, base + 2])
                        obs_vals.extend(
                            (nearby_pts * weights[:, np.newaxis]).sum(axis=0).tolist())
                        obs_r.extend([BL_R_DEPTH_SIGMA ** 2] * 3)

        # ── Kinematic constraint: estimate unseen elbows from wrist + bone length ──
        # Project Kalman-predicted elbow onto sphere of radius L_forearm around wrist.
        # Uses RGB > depth > Kalman prior for wrist position (best available).
        for elbow_j, wrist_j in KIN_FOREARM_PAIRS:
            elbow_idx  = joints.index(elbow_j)
            wrist_idx  = joints.index(wrist_j)
            elbow_base = elbow_idx * 3
            wrist_base = wrist_idx * 3

            if elbow_base in obs_cols:
                continue   # elbow already has RGB or depth observation

            # Best available wrist position
            if (trans_all[wrist_j][t] is not None
                    and not artifact_mask[wrist_j][t]):
                wrist_pos = trans_all[wrist_j][t]
            elif depth_xyz[wrist_j][t] is not None:
                wrist_pos = depth_xyz[wrist_j][t]
            else:
                wrist_pos = y_hat_world[wrist_base: wrist_base + 3]

            e_pred = y_hat_world[elbow_base: elbow_base + 3]
            vec    = e_pred - wrist_pos
            dist   = np.linalg.norm(vec)
            if dist < 1e-6:
                continue   # predicted elbow coincides with wrist — skip

            bone_len     = bone_lengths[(elbow_j, wrist_j)]
            e_kinematic  = wrist_pos + bone_len * (vec / dist)
            obs_cols.extend([elbow_base, elbow_base + 1, elbow_base + 2])
            obs_vals.extend(e_kinematic.tolist())
            obs_r.extend([BL_R_KINEMATIC_SIGMA ** 2] * 3)

        if obs_cols:
            n_obs  = len(obs_cols)
            H_full = np.zeros((n_obs, d))
            for r, c in enumerate(obs_cols):
                H_full[r, c] = 1.0
            H_hat = H_full @ V_hat                  # (n_obs, d_hat)
            R_t   = np.diag(obs_r)                  # non-uniform diagonal R

            z = np.array(obs_vals) - H_full @ m_bar

            # Track pre-update innovation magnitude for adaptive Q
            innovations.append(np.linalg.norm(z - H_hat @ m_hat))

            S_inn = H_hat @ P_hat @ H_hat.T + R_t
            K     = P_hat @ H_hat.T @ np.linalg.inv(S_inn)
            m_t   = m_hat + K @ (z - H_hat @ m_hat)
            P_t   = (np.eye(d_hat) - K @ H_hat) @ P_hat

            # Adaptive Q: if recent innovations are large, motion is fast → raise Q;
            # if small, motion is slow → lower Q for smoother output.
            if len(innovations) >= INNOVATION_WINDOW:
                innov_std = np.std(innovations[-INNOVATION_WINDOW:])
                Q = np.clip((innov_std ** 2) * np.eye(d_hat),
                            BL_Q_SIGMA ** 2, (BL_Q_SIGMA * 20) ** 2)
        else:
            m_t, P_t = m_hat, P_hat

        m_fwd[t], P_fwd[t] = m_t.copy(), P_t.copy()

    # ── Backward RTS pass (identical structure to original B&L) ───────────────
    m_smo, P_smo = [None] * n, [None] * n
    m_smo[n - 1] = m_fwd[n - 1].copy()
    P_smo[n - 1] = P_fwd[n - 1].copy()

    for t in range(n - 2, -1, -1):
        C        = P_fwd[t] @ np.linalg.inv(P_pred[t + 1])
        m_smo[t] = m_fwd[t] + C @ (m_smo[t + 1] - m_pred[t + 1])
        P_smo[t] = P_fwd[t] + C @ (P_smo[t + 1] - P_pred[t + 1]) @ C.T

    # ── Project back, fill artifact / NaN frames ──────────────────────────────
    result = {j: list(trans_all[j]) for j in joints}
    for t in range(n):
        y_t = V_hat @ m_smo[t] + m_bar     # (12,)
        for j_idx, j in enumerate(joints):
            if trans_all[j][t] is None or artifact_mask[j][t]:
                result[j][t] = y_t[j_idx * 3: j_idx * 3 + 3]
    return result


# =============================================================================
# Rotation fill dispatcher — SLERP / SQUAD / CuBsp
# =============================================================================

def fill_rotations(poses_joint: list, artifact_mask: np.ndarray,
                   method: str = 'slerp') -> list:
    """
    Fill rotation matrices over artifact intervals.
    Translations must already be set in poses_joint before calling.

    method : 'slerp'  C0 — 2 anchors only        (Stage 1, single peak)
             'squad'  C1 — needs 4 keyframes      (Stage 2, heavy noise)
             'cubsp'  C2 — cumulative B-spline     (Stages 3-4, step/drift)
    Fallback: cubsp/squad -> slerp when context frames unavailable.
    """
    n = len(poses_joint)
    result = [p.copy() if p is not None else None for p in poses_joint]

    for s, e in bool_runs(artifact_mask):
        # ── Mandatory anchors ─────────────────────────────────────────────────
        before_idx = next(
            (k for k in range(s - 1, -1, -1)
             if not artifact_mask[k] and poses_joint[k] is not None), None)
        after_idx = next(
            (k for k in range(e, n)
             if not artifact_mask[k] and poses_joint[k] is not None), None)
        if before_idx is None or after_idx is None:
            continue
        T_before = poses_joint[before_idx]
        T_after  = poses_joint[after_idx]
        q_before = _quat_from_matrix(T_before[:3, :3])
        q_after  = _quat_from_matrix(T_after[:3, :3])

        # ── Context frames for SQUAD / CuBsp ─────────────────────────────────
        T_prev = next(
            (poses_joint[k] for k in range(before_idx - 1, -1, -1)
             if not artifact_mask[k] and poses_joint[k] is not None), None)
        T_next = next(
            (poses_joint[k] for k in range(after_idx + 1, n)
             if not artifact_mask[k] and poses_joint[k] is not None), None)

        # ── Fallback: squad/cubsp -> slerp when context missing ───────────────
        chosen = method
        if chosen in ('squad', 'cubsp') and (T_prev is None or T_next is None):
            chosen = 'slerp'

        # ── Pre-compute control points ONCE (outside inner loop) ─────────────
        if chosen in ('squad', 'cubsp'):
            q_prev = _quat_from_matrix(T_prev[:3, :3])
            q_next = _quat_from_matrix(T_next[:3, :3])
        if chosen == 'squad':
            s_i    = _squad_inner_point(q_prev, q_before, q_after)
            s_next = _squad_inner_point(q_before, q_after, q_next)

        # ── Fill each interior frame ──────────────────────────────────────────
        n_fill = e - s
        for k in range(n_fill):
            u = (k + 1) / (n_fill + 1)   # u in (0,1) exclusive — interior only

            if chosen == 'slerp':
                q_fill = _slerp_quat(q_before, q_after, u)
            elif chosen == 'squad':
                q_fill = _squad(q_before, s_i, s_next, q_after, u)
            else:  # cubsp
                q_fill = _cubsp(q_prev, q_before, q_after, q_next, u)

            R_fill = _matrix_from_quat(q_fill)
            if result[s + k] is None:
                T_new = np.eye(4, dtype=np.float64)
                T_new[:3, :3] = R_fill
                result[s + k] = T_new
            else:
                result[s + k][:3, :3] = R_fill

    return result


# =============================================================================
# SLERP+linear fallback  (also used for Stage 3)
# =============================================================================

def apply_linear_slerp(poses_joint: list, trans_joint: list,
                       artifact_mask: np.ndarray, rot_method: str = 'slerp'):
    """
    Fill artifact runs: linear translation + rotation via rot_method.
    Translation and rotation are filled independently so each can use its
    own paper-specified method.
    """
    n = len(poses_joint)
    new_trans = list(trans_joint)
    new_poses = [p.copy() if p is not None else None for p in poses_joint]

    # ── Translation: linear interpolation ────────────────────────────────────
    for s, e in bool_runs(artifact_mask):
        T_before = next(
            (poses_joint[k] for k in range(s - 1, -1, -1)
             if not artifact_mask[k] and poses_joint[k] is not None), None)
        T_after = next(
            (poses_joint[k] for k in range(e, n)
             if not artifact_mask[k] and poses_joint[k] is not None), None)
        if T_before is None or T_after is None:
            continue
        t_b, t_a = T_before[:3, 3], T_after[:3, 3]
        n_fill = e - s
        for k in range(n_fill):
            alpha = (k + 1) / (n_fill + 1)
            t_k = (1 - alpha) * t_b + alpha * t_a
            new_trans[s + k] = t_k.copy()
            if new_poses[s + k] is None:
                new_poses[s + k] = np.eye(4, dtype=np.float64)
            else:
                new_poses[s + k] = new_poses[s + k].copy()
            new_poses[s + k][:3, 3] = t_k

    # ── Rotation: chosen method ───────────────────────────────────────────────
    new_poses = fill_rotations(new_poses, artifact_mask, method=rot_method)

    # Re-sync translations (fill_rotations only touches [:3,:3])
    for t in range(n):
        if new_trans[t] is not None and new_poses[t] is not None:
            new_poses[t][:3, 3] = new_trans[t]

    return new_poses, new_trans


# =============================================================================
# B&L or fallback dispatcher — original (no depth, kept for --no-depth path)
# =============================================================================

def apply_bl_or_fallback(poses_all: dict, trans_all: dict,
                         artifact_mask: dict, use_bl: bool,
                         rot_method: str = 'slerp'):
    new_poses = {j: list(poses_all[j]) for j in ARM_JOINTS}
    new_trans = {j: list(trans_all[j]) for j in ARM_JOINTS}

    bl_result = burke_lasenby_smooth(trans_all, artifact_mask) if use_bl else None

    for j in ARM_JOINTS:
        mask = artifact_mask[j]
        n_j = len(trans_all[j])
        if bl_result is not None:
            fill_mask = np.array(
                [mask[t] or trans_all[j][t] is None for t in range(n_j)], dtype=bool)

            for t in range(n_j):
                if fill_mask[t] and bl_result[j][t] is not None:
                    new_trans[j][t] = bl_result[j][t]

            new_poses[j] = fill_rotations(poses_all[j], fill_mask, method=rot_method)

            for t in range(n_j):
                if fill_mask[t] and new_trans[j][t] is not None:
                    if new_poses[j][t] is None:
                        T_new = np.eye(4, dtype=np.float64)
                        T_new[:3, 3] = new_trans[j][t]
                        new_poses[j][t] = T_new
                    else:
                        new_poses[j][t][:3, 3] = new_trans[j][t]
        else:
            new_poses[j], new_trans[j] = apply_linear_slerp(
                poses_all[j], trans_all[j], mask, rot_method=rot_method)

    return new_poses, new_trans


# =============================================================================
# Depth-assisted B&L dispatcher (used for Stages 1-2 when depth is enabled)
# =============================================================================

def apply_bl_or_fallback_depth(poses_all: dict, trans_all: dict,
                                artifact_mask: dict, use_bl: bool,
                                depth_xyz: dict, pointclouds: dict = None,
                                bone_lengths: dict = None,
                                rot_method: str = 'slerp'):
    """
    Like apply_bl_or_fallback but routes through burke_lasenby_smooth_depth()
    when use_bl=True, supplying L515 depth estimates as z^depth_t measurements.

    depth_xyz    : {joint: [xyz|None, ...]}  precomputed fallback positions
    pointclouds  : {frame_idx: ndarray|None} preloaded point clouds for Kalman-guided search
    bone_lengths : pre-calibrated from raw data; avoids re-calibrating on filled data
    """
    new_poses = {j: list(poses_all[j]) for j in ARM_JOINTS}
    new_trans = {j: list(trans_all[j]) for j in ARM_JOINTS}

    if use_bl:
        bl_result = burke_lasenby_smooth_depth(
            trans_all, artifact_mask, depth_xyz,
            pointclouds=pointclouds, bone_lengths=bone_lengths)
    else:
        bl_result = None

    for j in ARM_JOINTS:
        mask = artifact_mask[j]
        n_j = len(trans_all[j])
        if bl_result is not None:
            fill_mask = np.array(
                [mask[t] or trans_all[j][t] is None for t in range(n_j)], dtype=bool)

            for t in range(n_j):
                if fill_mask[t] and bl_result[j][t] is not None:
                    new_trans[j][t] = bl_result[j][t]

            new_poses[j] = fill_rotations(poses_all[j], fill_mask, method=rot_method)

            for t in range(n_j):
                if fill_mask[t] and new_trans[j][t] is not None:
                    if new_poses[j][t] is None:
                        T_new = np.eye(4, dtype=np.float64)
                        T_new[:3, 3] = new_trans[j][t]
                        new_poses[j][t] = T_new
                    else:
                        new_poses[j][t][:3, 3] = new_trans[j][t]
        else:
            # Fix 2: inject depth estimates directly for NaN frames
            for t in range(n_j):
                if new_trans[j][t] is None and depth_xyz[j][t] is not None:
                    new_trans[j][t] = depth_xyz[j][t].copy()
                    T_new = np.eye(4, dtype=np.float64)
                    T_new[:3, 3] = depth_xyz[j][t]
                    new_poses[j][t] = T_new

            # Fix 3: extend mask to cover remaining NaN gaps (not just artifact frames)
            extended_mask = np.array(
                [mask[t] or new_trans[j][t] is None for t in range(n_j)], dtype=bool)
            new_poses[j], new_trans[j] = apply_linear_slerp(
                new_poses[j], new_trans[j], extended_mask, rot_method=rot_method)

    return new_poses, new_trans


# =============================================================================
# Stage 4 — FFNN predictor + slow drift  (Paper 1 §3.3.3 & §3.4.5)
# =============================================================================

def _moving_average(arr: np.ndarray, window: int) -> np.ndarray:
    """Causal moving average of (N, 3) with NaN-forward-fill."""
    n = arr.shape[0]
    filled = arr.copy()
    for ax in range(3):
        col = filled[:, ax]
        nans = np.isnan(col)
        if nans.any():
            idx = np.where(~nans, np.arange(n), 0)
            np.maximum.accumulate(idx, out=idx)
            col[nans] = col[idx[nans]]
        filled[:, ax] = col
    ma = np.zeros((n, 3))
    for t in range(n):
        ma[t] = filled[max(0, t - window + 1): t + 1].mean(axis=0)
    return ma


def _build_features(trans_all: dict, target_joint: str,
                    t_indices: list, ma_self: np.ndarray):
    """
    Build FFNN input rows for target_joint (Eq. 10 of Paper 1).
    Features: siblings (current + k past) + polynomial L + MA of self.
    Returns list aligned to t_indices; None where features unavailable.
    """
    siblings = [j for j in ARM_JOINTS if j != target_joint]
    rows = []
    for t in t_indices:
        if t < FFNN_K:
            rows.append(None)
            continue
        base = []
        ok = True
        for sib in siblings:
            for lag in range(FFNN_K + 1):
                v = trans_all[sib][t - lag]
                if v is None:
                    ok = False
                    break
                base.extend(v.tolist())
            if not ok:
                break
        if not ok:
            rows.append(None)
            continue
        poly = [f ** FFNN_L for f in base]
        rows.append(base + poly + ma_self[t].tolist())
    return rows


def train_ffnn(trans_all: dict, target_joint: str,
               clean_mask: np.ndarray, ma_self: np.ndarray):
    """
    Train FFNN_REPLICATIONS MLPRegressors on clean frames.
    Architecture (Paper 1 Fig. 8): input -> 12 sigmoid -> 3 sigmoid -> 3 linear.
    Returns list of regressors, or None if fewer than 30 usable frames.
    """
    n = len(trans_all[ARM_JOINTS[0]])
    train_idx = [t for t in range(n)
                 if clean_mask[t] and trans_all[target_joint][t] is not None]
    if len(train_idx) < 30:
        return None

    feat_rows = _build_features(trans_all, target_joint, train_idx, ma_self)
    X_list, y_list = [], []
    for feat, t in zip(feat_rows, train_idx):
        if feat is not None:
            X_list.append(feat)
            y_list.append(trans_all[target_joint][t])
    if len(X_list) < 30:
        return None

    X_tr = np.array(X_list)
    y_tr = np.array(y_list)

    regressors = []
    for _ in range(FFNN_REPLICATIONS):
        mlp = MLPRegressor(hidden_layer_sizes=(12, 3), activation='logistic',
                           max_iter=500, n_iter_no_change=20)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            mlp.fit(X_tr, y_tr)
        regressors.append(mlp)
    return regressors


def predict_ffnn(regressors, trans_all: dict, target_joint: str,
                 ma_self: np.ndarray) -> list:
    """Predict for all frames. Returns list of xyz arrays or None."""
    n = len(trans_all[ARM_JOINTS[0]])
    feat_rows = _build_features(trans_all, target_joint, list(range(n)), ma_self)
    preds = []
    for feat in feat_rows:
        if feat is None:
            preds.append(None)
        else:
            X = np.array(feat).reshape(1, -1)
            avg = np.mean([m.predict(X)[0] for m in regressors], axis=0)
            preds.append(avg)
    return preds


def detect_slow_drift(trans_all: dict, target_joint: str,
                      clean_mask: np.ndarray, regressors,
                      ma_self: np.ndarray) -> np.ndarray:
    """Hysteresis thresholding on SG-smoothed FFNN residual.  (Paper 1 §3.4.5)"""
    n = len(trans_all[ARM_JOINTS[0]])
    if regressors is None:
        return np.zeros(n, dtype=bool)

    preds = predict_ffnn(regressors, trans_all, target_joint, ma_self)

    res = np.full((n, 3), np.nan)
    for t in range(n):
        if (trans_all[target_joint][t] is not None
                and preds[t] is not None and clean_mask[t]):
            res[t] = trans_all[target_joint][t] - preds[t]

    has_res = ~np.any(np.isnan(res), axis=1)
    mask = np.zeros(n, dtype=bool)

    for ax in range(3):
        R_filled = np.where(has_res, res[:, ax], 0.0)
        valid_count = int(has_res.sum())
        if valid_count < DRIFT_SG_WIN:
            continue
        win = DRIFT_SG_WIN if DRIFT_SG_WIN <= valid_count else max(3, valid_count | 1)
        win = win if win % 2 == 1 else win - 1
        if win < 3:
            continue
        try:
            R_smooth = savgol_filter(R_filled, window_length=win,
                                     polyorder=DRIFT_SG_POLY)
        except ValueError:
            continue

        sigma_R = R_smooth[has_res].std(ddof=0) + 1e-12
        upper_t = DRIFT_K_UPPER * sigma_R
        lower_t = DRIFT_K_LOWER * sigma_R

        i = 0
        while i < n:
            if np.abs(R_smooth[i]) > upper_t:
                left = i
                while left > 0 and np.abs(R_smooth[left - 1]) > lower_t:
                    left -= 1
                right = i + 1
                while right < n and np.abs(R_smooth[right]) > lower_t:
                    right += 1
                if right - left >= DRIFT_MIN_LEN:
                    mask[left:right] = True
                i = right
            else:
                i += 1

    return mask


# =============================================================================
# Main pipeline
# =============================================================================

def run_pipeline(session_dir: str, use_bl: bool, use_ffnn: bool,
                 use_depth: bool, dry_run: bool):
    frames = sorted(
        [d for d in os.listdir(session_dir) if d.startswith('frame_')],
        key=lambda x: int(x.split('_')[1])
    )
    n = len(frames)
    if n == 0:
        print("No frames found.")
        return

    print(f"Session : {session_dir}")
    print(f"Frames  : {n}   B&L: {'on' if use_bl else 'off (--no-bl)'}   "
          f"FFNN: {'on' if use_ffnn else 'off (--no-ffnn)'}   "
          f"Depth: {'on' if use_depth else 'off (--no-depth)'}"
          f"{'   [DRY RUN]' if dry_run else ''}\n")

    poses = {j: load_poses(session_dir, j) for j in ARM_JOINTS}
    trans = {j: extract_translations(poses[j]) for j in ARM_JOINTS}
    nan_before = {j: sum(1 for t in trans[j] if t is None) for j in ARM_JOINTS}

    counts = {j: dict(depth=0, peak=0, noise=0, step=0, drift=0)
              for j in ARM_JOINTS}
    cumulative = {j: np.zeros(n, dtype=bool) for j in ARM_JOINTS}

    # interp_mask tracks every frame whose value was inserted by any earlier
    # stage (originally NaN or artifact-detected). Per Paper 1 §4.3, these
    # frames must be excluded from detection in all subsequent stages.
    interp_mask = {j: np.array([t is None for t in trans[j]], dtype=bool)
                   for j in ARM_JOINTS}

    # ── Step 0: L515 depth pre-fill ───────────────────────────────────────────
    # For frames where RGB ArUco detection failed, attempt to recover a 3D
    # world-frame position from the L515 point cloud.  These depth estimates
    # are passed into the Kalman smoother as z^depth_t with higher R_depth
    # noise rather than skipping the update step entirely.
    if use_depth:
        print("Step 0: Preloading L515 point clouds...")
        pointclouds = preload_pointclouds(session_dir, frames)
        n_pc = sum(1 for v in pointclouds.values() if v is not None)
        print(f"  Loaded {n_pc}/{n} point clouds")
        print("  Computing linear-interp depth fallback positions (for B&L-fallback path)...")
        depth_xyz = load_depth_positions(frames, trans, pointclouds)
        for j in ARM_JOINTS:
            counts[j]['depth'] = sum(
                1 for t in range(n) if depth_xyz[j][t] is not None)
        total_d = sum(counts[j]['depth'] for j in ARM_JOINTS)
        print("  Depth fallback estimates: "
              + "  ".join(f"{j}: {counts[j]['depth']}" for j in ARM_JOINTS))
        if total_d == 0:
            print("  (no pointcloud.npy found or no points within search radius)")
        print()
    else:
        depth_xyz   = {j: [None] * n for j in ARM_JOINTS}
        pointclouds = None

    # Calibrate bone lengths once from the original raw poses before any filling.
    # Passing this to every smoother call prevents re-calibration on filled data.
    print("Calibrating forearm bone lengths from raw data...")
    raw_artifact_mask = {j: np.zeros(n, dtype=bool) for j in ARM_JOINTS}
    bone_lengths = calibrate_bone_lengths(trans, raw_artifact_mask)
    print()

    # ── Stage 1: Single Peaks — rotation: SLERP (C0, 1-5 frames) ────────────
    print("Stage 1: Detecting single peaks...")
    mask_peak = {j: detect_single_peaks(trans[j]) for j in ARM_JOINTS}
    for j in ARM_JOINTS:
        counts[j]['peak'] = int(mask_peak[j].sum())
        cumulative[j]   |= mask_peak[j]
        interp_mask[j]  |= mask_peak[j]
    poses, trans = apply_bl_or_fallback_depth(
        poses, trans, mask_peak, use_bl, depth_xyz,
        pointclouds=pointclouds, bone_lengths=bone_lengths, rot_method='slerp')

    # ── Stage 2: Heavy Noise — rotation: SQUAD (C1, 5-50 frames) ─────────────
    print("Stage 2: Detecting heavy noise...")
    mask_noise = {j: detect_heavy_noise(trans[j], exclude_mask=interp_mask[j])
                  for j in ARM_JOINTS}
    for j in ARM_JOINTS:
        counts[j]['noise'] = int(mask_noise[j].sum())
        cumulative[j]  |= mask_noise[j]
        interp_mask[j] |= mask_noise[j]
    poses, trans = apply_bl_or_fallback_depth(
        poses, trans, mask_noise, use_bl, depth_xyz,
        pointclouds=pointclouds, bone_lengths=bone_lengths, rot_method='squad')

    # ── Stage 3: Step Changes — rotation: CuBsp (C2, >20 frames) ────────────
    # Depth assistance is not applied here: step changes are mislabelling
    # artifacts; depth cannot help distinguish correct from swapped markers.
    print("Stage 3: Detecting step changes...")
    mask_step = {j: detect_step_changes(trans[j], exclude_mask=interp_mask[j])
                 for j in ARM_JOINTS}
    for j in ARM_JOINTS:
        counts[j]['step'] = int(mask_step[j].sum())
        cumulative[j]  |= mask_step[j]
        interp_mask[j] |= mask_step[j]
        poses[j], trans[j] = apply_linear_slerp(poses[j], trans[j], mask_step[j],
                                                 rot_method='cubsp')

    # ── Stage 4: Slow Drift (FFNN) ────────────────────────────────────────────
    if use_ffnn:
        print("Stage 4: Training FFNN predictors and detecting slow drift...")
        # clean_mask = only genuinely measured, never interpolated frames
        clean_mask = {j: ~interp_mask[j] for j in ARM_JOINTS}

        for j in ARM_JOINTS:
            self_arr = np.full((n, 3), np.nan)
            for t in range(n):
                if trans[j][t] is not None:
                    self_arr[t] = trans[j][t]
            ma_self = _moving_average(self_arr, DRIFT_MA_WINDOW)

            regressors = train_ffnn(trans, j, clean_mask[j], ma_self)
            if regressors is None:
                print(f"  {j}: insufficient clean data, skipping Stage 4")
                continue

            drift_j = detect_slow_drift(trans, j, clean_mask[j], regressors, ma_self)
            counts[j]['drift'] = int(drift_j.sum())
            cumulative[j] |= drift_j

            preds = predict_ffnn(regressors, trans, j, ma_self)
            for t in range(n):
                if drift_j[t] and preds[t] is not None:
                    trans[j][t] = preds[t]
                    if poses[j][t] is not None:
                        poses[j][t] = poses[j][t].copy()
                        poses[j][t][:3, 3] = preds[t]
            poses[j] = fill_rotations(poses[j], drift_j, method='cubsp')

    nan_after = {j: sum(1 for t in trans[j] if t is None) for j in ARM_JOINTS}

    # ── Report ────────────────────────────────────────────────────────────────
    COL = 20
    depth_hdr = f"{'Depth':>6}" if use_depth else ""
    sep_len   = 84 if use_depth else 78
    print(f"\n{'Joint':<{COL}} {'NaN bef':>10} {'NaN aft':>10} "
          f"{depth_hdr} {'Peak':>6} {'Noise':>6} {'Step':>6} {'Drift':>6} {'Recov':>6}")
    print('─' * sep_len)
    for j in ARM_JOINTS:
        nb, na = nan_before[j], nan_after[j]
        pct_b = 100 * nb / n if n else 0
        pct_a = 100 * na / n if n else 0
        ac = counts[j]
        depth_col = f"{ac['depth']:>6}" if use_depth else ""
        print(f"{j:<{COL}} {nb:>4}({pct_b:4.1f}%) {na:>4}({pct_a:4.1f}%) "
              f"{depth_col} {ac['peak']:>6} {ac['noise']:>6} {ac['step']:>6} "
              f"{ac['drift']:>6} {nb - na:>6}")
    print()

    # ── Save ──────────────────────────────────────────────────────────────────
    if not dry_run:
        for f_idx, frame in enumerate(frames):
            frame_dir = os.path.join(session_dir, frame)
            for j in ARM_JOINTS:
                out_path = os.path.join(frame_dir, f'pose_{j}_processed.txt')
                T = poses[j][f_idx]
                np.savetxt(out_path,
                           T if T is not None else np.full((4, 4), np.nan))
        print("Wrote pose_<joint>_processed.txt in each frame directory.")
    else:
        print("(Dry run — no files written)")


def main():
    parser = argparse.ArgumentParser(
        description='Depth-assisted 4-stage artifact detection + repair for arm pose files')
    parser.add_argument('session_dir', help='Path to a saved session directory')
    parser.add_argument('--no-bl',    action='store_true',
                        help='Disable Burke & Lasenby; use SLERP/linear for stages 1-2')
    parser.add_argument('--no-ffnn',  action='store_true',
                        help='Disable Stage 4 FFNN slow-drift detection')
    parser.add_argument('--no-depth', action='store_true',
                        help='Disable L515 depth assistance; revert to original B&L behavior')
    parser.add_argument('--dry_run',  action='store_true',
                        help='Detect and report without writing any files')
    args = parser.parse_args()

    session_dir = args.session_dir.rstrip('/')
    if not os.path.isdir(session_dir):
        print(f"Error: {session_dir} is not a directory")
        sys.exit(1)

    run_pipeline(session_dir,
                 use_bl    = not args.no_bl,
                 use_ffnn  = not args.no_ffnn,
                 use_depth = not args.no_depth,
                 dry_run   = args.dry_run)


if __name__ == '__main__':
    main()
