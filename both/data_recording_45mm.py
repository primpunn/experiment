#!/usr/bin/env python3
"""
Dual-camera data collection: Intel RealSense L515 (floor, static) + D435i (head-mounted, 90° CW)

Setup:
  - L515 placed on the floor as a static third-person-view camera.
  - D435i mounted on the human head, physically rotated 90° clockwise.
  - ArUco markers (DICT_4X4_100):
      right wrist  (45 mm): ID  0,  1,  2,  3
      right elbow  (45 mm): ID  4,  5,  6,  7
      left  wrist  (45 mm): ID  8,  9, 12, 15
      left  elbow  (45 mm): ID 18, 19, 22, 23
      floor/world  (95 mm): ID 10 (world origin), 11, 13, 14, 16, 17, 20, 21

Collection steps:
  1. Place floor markers ID10,11,13,14,16,17,20,21 on the floor. ID10 defines the world frame.
  2. Run this script. Both cameras must see ID10 simultaneously during initialization.
  3. After init, record freely. D435i must always see at least one floor marker.
     Press 0-5 in the preview window (or type digit+Enter with --no-preview) to set task phase.

Output per frame  (saved in output_dir/frame_N/):
  color_image.png       L515 color, BGR, 1280×720
  depth_image.png       D435i depth aligned to color, rotated 90° CCW, uint16, 720×1280
  pointcloud.npy        L515 point cloud in world frame, (N,6) float32: [X,Y,Z,B,G,R]
  pose.txt              D435i camera pose in world frame (4×4)
  pose_right_wrist.txt  right wrist in world frame (4×4) — NaN if not detected by either camera
  pose_right_elbow.txt  right elbow in world frame (4×4)
  pose_left_wrist.txt   left  wrist in world frame (4×4)
  pose_left_elbow.txt   left  elbow in world frame (4×4)
  phase.txt             task phase label: 0=idle 1=approach 2=lift 3=press 4=hold 5=release

Transform convention:
  T_A_B is a 4×4 matrix that transforms points FROM frame B TO frame A:
      p_A = T_A_B @ p_B_homogeneous

D435i rotation note:
  Physical 90° CW mounting → images appear rotated 90° CW.
  Correction: rotate images 90° CCW before saving and before ArUco detection.
  Intrinsics are adjusted accordingly (fx↔fy, cx↔cy with axis flip).

Usage:
  conda activate massage
  python data_recording_45mm.py -o ./saved_data --total_frames 1000
  python data_recording_45mm.py -o ./saved_data          # run until Ctrl+C
"""

import os
import time
import argparse
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import sys
import select
sys.path.insert(0, '/home/primpunn/librealsense/build/Release')

import numpy as np
import cv2
import pyrealsense2 as rs

# Display scale: resize preview windows to this height (keeps aspect ratio)
PREVIEW_H = 480


# ── Constants ──────────────────────────────────────────────────────────────────
ARUCO_DICT_ID        = cv2.aruco.DICT_4X4_100
ARM_MARKER_SIZE_M    = 0.045    # 45 mm — arm markers
FLOOR_MARKER_SIZE_M  = 0.095    # 95 mm — floor markers
WORLD_ID        = 10            # ID10 defines the world frame origin
FLOOR_IDS       = [10, 11, 13, 14, 16, 17, 20, 21]   # IDs placed on floor (static)

ARM_JOINTS = ['right_wrist', 'right_elbow', 'left_wrist', 'left_elbow']
ARM_MARKER_GROUPS = {          # 4 markers per joint; first detected one gives the pose
    'right_wrist': [ 0,  1,  2,  3],
    'right_elbow': [ 4,  5,  6,  7],
    'left_wrist':  [ 8,  9, 12, 15],
    'left_elbow':  [18, 19, 22, 23],
}
ARM_IDS = [mid for ids in ARM_MARKER_GROUPS.values() for mid in ids]

L515_COLOR_W, L515_COLOR_H = 1280, 720
L515_DEPTH_W, L515_DEPTH_H = 1024, 768
D435I_W,      D435I_H      = 1280, 720
TARGET_FPS                  = 30
PC_NUM_POINTS               = 8192   # number of points to subsample per frame

# D435i physically rotated 90° CW → correct with 90° CCW
D435I_ROTATE = cv2.ROTATE_90_COUNTERCLOCKWISE

NAN_POSE = np.full((4, 4), np.nan)

MAX_ARM_SPEED_M_PER_FRAME = 0.30 / 15   # 0.3 m/s at ~15 fps = 0.02 m/frame

PHASE_NAMES = {0: 'idle', 1: 'approach', 2: 'lift', 3: 'press', 4: 'hold', 5: 'release'}


# ── Math helpers ───────────────────────────────────────────────────────────────

def rvec_tvec_to_4x4(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """OpenCV Rodrigues rvec + tvec → 4×4 homogeneous matrix T_cam_marker."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3], _ = cv2.Rodrigues(rvec.flatten())
    T[:3, 3] = tvec.flatten()
    return T


def orthonormalize(T: np.ndarray) -> np.ndarray:
    """Re-orthogonalize the rotation part of a 4×4 transform (removes drift from averaging)."""
    U, _, Vt = np.linalg.svd(T[:3, :3])
    T[:3, :3] = U @ Vt
    return T


def average_transforms(Ts: list) -> np.ndarray:
    """Average a list of 4×4 transforms and re-orthogonalize."""
    avg = np.mean(Ts, axis=0)
    return orthonormalize(avg)


# ── Intrinsics helpers ─────────────────────────────────────────────────────────

def rotate_intrinsics_90ccw(intr: rs.intrinsics, orig_w: int, orig_h: int) -> rs.intrinsics:
    """
    Adjust RealSense intrinsics after rotating image 90° CCW.

    For 90° CCW rotation of an (orig_w × orig_h) image:
      - New dimensions: orig_h × orig_w
      - new_fx = fy,   new_fy = fx
      - new_cx = cy,   new_cy = (orig_w - 1) - cx
    """
    r = rs.intrinsics()
    r.width  = orig_h
    r.height = orig_w
    r.fx     = intr.fy
    r.fy     = intr.fx
    r.ppx    = intr.ppy
    r.ppy    = float(orig_w - 1) - intr.ppx
    r.model  = intr.model
    r.coeffs = intr.coeffs
    return r


def intr_to_K_D(intr: rs.intrinsics):
    """Returns (K 3×3, D distortion array) from rs.intrinsics."""
    K = np.array([[intr.fx, 0., intr.ppx],
                  [0., intr.fy, intr.ppy],
                  [0., 0., 1.]], dtype=np.float64)
    D = np.array(intr.coeffs, dtype=np.float64)
    return K, D


# ── ArUco helpers ──────────────────────────────────────────────────────────────

def build_aruco_detector():
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    params = cv2.aruco.DetectorParameters()
    # Wider adaptive threshold range handles motion blur and varying illumination
    params.adaptiveThreshWinSizeMin  = 3
    params.adaptiveThreshWinSizeMax  = 53
    params.adaptiveThreshWinSizeStep = 4
    params.adaptiveThreshConstant    = 3    # lower = more sensitive in low-contrast regions
    # Tolerate more bit errors (useful when markers are partially blurred)
    params.errorCorrectionRate = 0.9
    # Sub-pixel corner refinement for better pose accuracy
    params.cornerRefinementMethod        = cv2.aruco.CORNER_REFINE_SUBPIX
    params.cornerRefinementWinSize       = 5
    params.cornerRefinementMaxIterations = 30
    params.cornerRefinementMinAccuracy   = 0.01
    return cv2.aruco.ArucoDetector(aruco_dict, params)


def _sharpen(gray: np.ndarray) -> np.ndarray:
    """Unsharp mask to recover edges lost to motion blur."""
    blurred = cv2.GaussianBlur(gray, (0, 0), 3)
    return cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)


def _clahe(gray: np.ndarray) -> np.ndarray:
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)


_detector = None  # module-level singleton

def detect_markers(gray: np.ndarray, K: np.ndarray, D: np.ndarray) -> dict:
    """
    Detect ArUco markers in a grayscale image.
    Returns {marker_id (int): T_cam_marker (4×4)}.
    T_cam_marker transforms points FROM marker frame TO camera frame.
    """
    global _detector
    if _detector is None:
        _detector = build_aruco_detector()

    corners, ids, _ = _detector.detectMarkers(gray)
    result = {}
    if ids is None:
        return result
    for i, mid in enumerate(ids.flatten()):
        size = ARM_MARKER_SIZE_M if int(mid) in ARM_IDS else FLOOR_MARKER_SIZE_M
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            [corners[i]], size, K, D)
        result[int(mid)] = rvec_tvec_to_4x4(rvecs[0], tvecs[0])
    return result


# ── Preview helpers ────────────────────────────────────────────────────────────

def draw_preview(bgr_img: np.ndarray,
                 detections: dict,
                 highlight_ids: list,
                 status_lines: list) -> np.ndarray:
    """
    Draw ArUco detections and status overlay on a BGR image.
    highlight_ids: marker IDs to mark as important (drawn in green if seen, red if missing).
    status_lines: list of (text, color) tuples drawn at the top-left.
    Returns a resized copy suitable for display.
    """
    vis = bgr_img.copy()
    h, w = vis.shape[:2]

    # Draw all detected marker borders
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
    params     = cv2.aruco.DetectorParameters()
    detector   = cv2.aruco.ArucoDetector(aruco_dict, params)
    gray       = cv2.cvtColor(vis, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(vis, corners, ids)
        for i, mid in enumerate(ids.flatten()):
            # Label each marker ID above its top-left corner
            cx = int(corners[i][0][:, 0].mean())
            cy = int(corners[i][0][:, 1].min()) - 8
            color = (0, 255, 0) if int(mid) in highlight_ids else (255, 200, 0)
            cv2.putText(vis, f'ID{mid}', (cx, max(cy, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    # Status overlay — semi-transparent dark bar at top
    bar_h = 28 * (len(status_lines) + 1)
    overlay = vis.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.55, vis, 0.45, 0, vis)

    for i, (text, color) in enumerate(status_lines):
        cv2.putText(vis, text, (8, 22 + i * 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Resize to PREVIEW_H
    scale   = PREVIEW_H / h
    new_w   = int(w * scale)
    return cv2.resize(vis, (new_w, PREVIEW_H))


# ── Point cloud helpers ────────────────────────────────────────────────────────

def depth_to_pointcloud_cam(depth_img: np.ndarray,
                            color_bgr: np.ndarray,
                            intr: rs.intrinsics,
                            depth_scale: float,
                            max_depth_m: float = 5.0) -> np.ndarray:
    """
    Back-project L515 aligned depth to 3D in camera frame, with BGR colour.
    depth_img and color_bgr must have the same (H, W).
    Returns (N, 6) float32: [X, Y, Z, B, G, R] in camera frame.
    """
    h, w = depth_img.shape
    u, v = np.meshgrid(np.arange(w, dtype=np.float32),
                       np.arange(h, dtype=np.float32))
    z = depth_img.astype(np.float32) * depth_scale

    valid = (z > 0) & (z < max_depth_m)
    z_v = z[valid]
    x_v = (u[valid] - intr.ppx) * z_v / intr.fx
    y_v = (v[valid] - intr.ppy) * z_v / intr.fy

    bgr = color_bgr[valid].astype(np.float32)          # (N, 3)
    pts = np.concatenate([np.stack([x_v, y_v, z_v], axis=1), bgr], axis=1)

    # Subsample
    if len(pts) > PC_NUM_POINTS:
        idx = np.random.choice(len(pts), PC_NUM_POINTS, replace=False)
        pts = pts[idx]
    return pts.astype(np.float32)


def transform_pointcloud(pts: np.ndarray, T_world_cam: np.ndarray) -> np.ndarray:
    """Transform (N,6) point cloud from camera frame to world frame."""
    xyz_h = np.hstack([pts[:, :3],
                       np.ones((len(pts), 1), dtype=np.float32)])  # (N,4)
    xyz_w = (T_world_cam.astype(np.float32) @ xyz_h.T).T[:, :3]   # (N,3)
    return np.concatenate([xyz_w, pts[:, 3:]], axis=1)             # (N,6)


# ── Main recorder class ────────────────────────────────────────────────────────

class DualCameraRecorder:

    def __init__(self, output_dir: str):
        base_dir = Path(output_dir)
        now = datetime.now()
        date_str = now.strftime('%Y-%m-%d')
        session_dir = base_dir / date_str
        if session_dir.exists():
            # Same date already has data — append HH-MM-SS to distinguish
            session_dir = base_dir / now.strftime('%Y-%m-%d_%H-%M-%S')
        self.output_dir = session_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Session folder: {self.output_dir}")

        self.l515_pipe  = rs.pipeline()
        self.d435i_pipe = rs.pipeline()

        # Set during start_cameras()
        self.l515_color_intr : rs.intrinsics = None
        self.l515_depth_scale: float         = None
        self.l515_align                      = None
        self.l515_K: np.ndarray              = None
        self.l515_D: np.ndarray              = None

        self.d435i_depth_scale: float        = None
        self.d435i_K: np.ndarray             = None   # after rotation correction
        self.d435i_D: np.ndarray             = None
        self.d435i_align                     = None

        # Set during initialize_transforms()
        self.T_world_L515 : np.ndarray       = None   # L515 camera → world
        self.T_world_floor: dict             = {}     # {id: 4×4} floor markers → world

        # Updated each frame
        self.T_world_head : np.ndarray       = None   # last known D435i → world
        self.head_pose_age: int              = 0      # frames since T_world_head last updated

        self.frame_buffer: list  = []
        self.executor            = ThreadPoolExecutor(max_workers=8)
        self.frame_idx           = 0
        self._show_preview       = True   # toggled by --no-preview flag
        self.current_phase: int  = 0
        self._last_valid_arm_pose: dict = {}   # joint -> (frame_idx, T_world_joint)

    # ── Camera setup ──────────────────────────────────────────────────────────

    def start_cameras(self):
        # ── L515 ──────────────────────────────────────────────────
        l515_cfg = rs.config()
        l515_cfg.enable_stream(rs.stream.color, L515_COLOR_W, L515_COLOR_H,
                               rs.format.rgb8, TARGET_FPS)
        l515_cfg.enable_stream(rs.stream.depth, L515_DEPTH_W, L515_DEPTH_H,
                               rs.format.z16, TARGET_FPS)
        l515_prof = self.l515_pipe.start(l515_cfg)

        self.l515_depth_scale = (l515_prof.get_device()
                                 .first_depth_sensor()
                                 .get_depth_scale())
        # Color intrinsics (used after aligning depth to color)
        self.l515_color_intr = (l515_prof
                                .get_stream(rs.stream.color)
                                .as_video_stream_profile()
                                .get_intrinsics())
        # Align L515 depth to color so depth and color share the same resolution
        self.l515_align = rs.align(rs.stream.color)
        self.l515_K, self.l515_D = intr_to_K_D(self.l515_color_intr)

        # ── D435i ─────────────────────────────────────────────────
        d435i_cfg = rs.config()
        d435i_cfg.enable_stream(rs.stream.color, D435I_W, D435I_H,
                                rs.format.bgr8, TARGET_FPS)
        d435i_cfg.enable_stream(rs.stream.depth, D435I_W, D435I_H,
                                rs.format.z16, TARGET_FPS)
        d435i_prof = self.d435i_pipe.start(d435i_cfg)

        self.d435i_depth_scale = (d435i_prof.get_device()
                                  .first_depth_sensor()
                                  .get_depth_scale())
        raw_intr = (d435i_prof
                    .get_stream(rs.stream.color)
                    .as_video_stream_profile()
                    .get_intrinsics())
        # Adjust intrinsics for 90° CCW rotation correction
        intr_rot = rotate_intrinsics_90ccw(raw_intr, D435I_W, D435I_H)
        self.d435i_K, self.d435i_D = intr_to_K_D(intr_rot)

        self.d435i_align = rs.align(rs.stream.color)

        # Warm-up: discard first 60 frames to stabilize auto-exposure
        print("Warming up cameras (2 s)...")
        for _ in range(60):
            self.l515_pipe.wait_for_frames()
            self.d435i_pipe.wait_for_frames()
        print("Cameras ready.\n")

    # ── Initialization ────────────────────────────────────────────────────────

    def initialize_transforms(self):
        """
        Phase 1 (run once before recording):
          Both cameras must see ArUco ID10 simultaneously.
          Live preview windows show each camera's view with marker detection overlay.
          Press 'S' in either window (or ENTER in terminal) to start collecting.

          From L515:
            T_world_L515 = inv(T_L515_ID10)
          From D435i:
            T_world_head  = inv(T_head_ID10)
            T_world_IDk   = T_world_head @ T_head_IDk   for k in FLOOR_IDS

          Averages 60 valid frames for robustness.
        """
        print("=== INITIALIZATION ===")
        print("Preview windows are open. Position cameras so ID10 is visible in BOTH.")
        print("Press 'S' in either preview window  OR  press ENTER here to start.\n")

        cv2.namedWindow('L515 (floor)',  cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow('D435i (head)',  cv2.WINDOW_AUTOSIZE)

        # ── Phase A: live preview until user signals ready ─────────────────────
        collecting = False
        T_world_L515_list   = []
        T_world_floor_lists = {k: [] for k in FLOOR_IDS}
        needed  = 60
        got     = 0
        t_start = None

        import select, sys as _sys

        while True:
            l515_frames  = self.l515_pipe.wait_for_frames()
            d435i_frames = self.d435i_pipe.wait_for_frames()

            # ── L515 ──────────────────────────────────────────────
            l515_aligned = self.l515_align.process(l515_frames)
            l515_color   = np.asanyarray(l515_aligned.get_color_frame().get_data())  # RGB
            l515_bgr     = l515_color[:, :, ::-1].copy()
            l515_gray    = cv2.cvtColor(l515_color, cv2.COLOR_RGB2GRAY)
            l515_det     = detect_markers(l515_gray, self.l515_K, self.l515_D)
            l515_ok      = WORLD_ID in l515_det

            # ── D435i ─────────────────────────────────────────────
            d435i_aligned = self.d435i_align.process(d435i_frames)
            d435i_color   = np.asanyarray(d435i_aligned.get_color_frame().get_data())
            d435i_rot     = cv2.rotate(d435i_color, D435I_ROTATE)
            d435i_gray    = cv2.cvtColor(d435i_rot, cv2.COLOR_BGR2GRAY)
            d_det         = detect_markers(d435i_gray, self.d435i_K, self.d435i_D)
            d435i_ok      = WORLD_ID in d_det

            both_ok = l515_ok and d435i_ok

            # ── Build status lines ────────────────────────────────
            if collecting:
                prog      = f"{got}/{needed}"
                bar_done  = int(20 * got / needed)
                bar       = '[' + '#' * bar_done + '-' * (20 - bar_done) + ']'
                l515_status = [
                    (f"ID10: {'DETECTED' if l515_ok else 'NOT SEEN'}",
                     (0, 255, 0) if l515_ok else (0, 0, 255)),
                    (f"Collecting: {bar} {prog}",
                     (0, 220, 255)),
                ]
                d435i_status = [
                    (f"ID10: {'DETECTED' if d435i_ok else 'NOT SEEN'}",
                     (0, 255, 0) if d435i_ok else (0, 0, 255)),
                    (f"Floor IDs seen: {sorted(k for k in d_det if k in FLOOR_IDS)}",
                     (200, 200, 200)),
                    (f"Collecting: {bar} {prog}",
                     (0, 220, 255)),
                ]
            else:
                hint = "Press 'S' here or ENTER in terminal to start"
                ready_color = (0, 255, 0) if both_ok else (0, 165, 255)
                ready_text  = "READY — both see ID10!" if both_ok else "Waiting — adjust until both see ID10"
                l515_status = [
                    (f"ID10: {'DETECTED' if l515_ok else 'NOT SEEN'}",
                     (0, 255, 0) if l515_ok else (0, 0, 255)),
                    (ready_text, ready_color),
                    (hint, (200, 200, 200)),
                ]
                d435i_status = [
                    (f"ID10: {'DETECTED' if d435i_ok else 'NOT SEEN'}",
                     (0, 255, 0) if d435i_ok else (0, 0, 255)),
                    (f"Floor IDs seen: {sorted(k for k in d_det if k in FLOOR_IDS)}",
                     (200, 200, 200)),
                    (ready_text, ready_color),
                    (hint, (200, 200, 200)),
                ]

            # ── Show windows ──────────────────────────────────────
            l515_preview  = draw_preview(l515_bgr,  l515_det, [WORLD_ID], l515_status)
            d435i_preview = draw_preview(d435i_rot, d_det,    [WORLD_ID], d435i_status)

            cv2.imshow('L515 (floor)', l515_preview)
            cv2.imshow('D435i (head)', d435i_preview)

            key = cv2.waitKey(1) & 0xFF

            # Check for 'S' keypress in window OR non-blocking ENTER in terminal
            if not collecting:
                start_signal = (key == ord('s') or key == ord('S'))
                # Non-blocking check for ENTER in terminal
                if not start_signal:
                    r, _, _ = select.select([_sys.stdin], [], [], 0)
                    if r:
                        _sys.stdin.readline()
                        start_signal = True
                if start_signal:
                    if not both_ok:
                        print("  Warning: not both cameras see ID10 yet — starting anyway.")
                    print("  Starting to collect init frames...")
                    collecting = True
                    t_start    = time.time()

            # ── Collect init frames ───────────────────────────────
            if collecting:
                if l515_ok and d435i_ok:
                    T_L515_ID10   = l515_det[WORLD_ID]
                    T_world_L515 = np.linalg.inv(T_L515_ID10)

                    T_head_ID10   = d_det[WORLD_ID]
                    T_world_head = np.linalg.inv(T_head_ID10)

                    for fid in FLOOR_IDS:
                        if fid in d_det:
                            T_world_floor_lists[fid].append(T_world_head @ d_det[fid])

                    T_world_L515_list.append(T_world_L515)
                    got += 1

                if got >= needed:
                    break

            # 'Q' to quit
            if key == ord('q') or key == ord('Q'):
                cv2.destroyAllWindows()
                raise KeyboardInterrupt("Quit from preview window.")

        cv2.destroyAllWindows()

        # ── Average and orthogonalize ──────────────────────────────────────────
        self.T_world_L515 = average_transforms(T_world_L515_list)

        for fid, lst in T_world_floor_lists.items():
            if lst:
                self.T_world_floor[fid] = average_transforms(lst)

        self.T_world_floor[WORLD_ID] = np.eye(4)   # ID10 is the world frame origin

        # Save static transforms for post-processing
        np.savetxt(str(self.output_dir / 'T_world_L515.txt'), self.T_world_L515)
        for fid, T in self.T_world_floor.items():
            np.savetxt(str(self.output_dir / f'T_world_ID{fid}.txt'), T)

        visible = sorted(self.T_world_floor.keys())
        print(f"\nInitialization complete.")
        print(f"  L515 pose saved.")
        print(f"  Floor markers acquired: {visible}")
        missing = [f for f in FLOOR_IDS if f not in self.T_world_floor]
        if missing:
            print(f"  Not seen during init (will skip): {missing}")

    # ── Per-frame ─────────────────────────────────────────────────────────────

    def _update_head_pose(self, d_det: dict):
        """
        Update T_world_head from any visible floor marker.
        T_world_head = T_world_IDk @ inv(T_head_IDk)

        This removes head-movement artifacts: even if the human turns their
        head, T_world_head tracks the actual head orientation, so all arm
        poses computed as T_world_head @ T_head_arm reflect true arm motion
        in the world frame, not camera-relative motion.
        """
        for fid in FLOOR_IDS:
            if fid in d_det and fid in self.T_world_floor:
                T_head_IDk  = d_det[fid]
                T_world_IDk = self.T_world_floor[fid]
                # p_world = T_world_IDk @ inv(T_head_IDk) @ p_head
                self.T_world_head = T_world_IDk @ np.linalg.inv(T_head_IDk)
                self.head_pose_age = 0
                return

    def record_frame(self):
        l515_frames  = self.l515_pipe.wait_for_frames()
        d435i_frames = self.d435i_pipe.wait_for_frames()

        # ── L515 ──────────────────────────────────────────────────
        # Align depth to color (L515 depth 1024×768 → 1280×720)
        l515_aligned     = self.l515_align.process(l515_frames)
        l515_color_rgb   = np.asanyarray(l515_aligned.get_color_frame().get_data())  # (720,1280,3) RGB
        l515_depth_align = np.asanyarray(l515_aligned.get_depth_frame().get_data())  # (720,1280) uint16

        # color_image.png: RGB → BGR
        color_image = l515_color_rgb[:, :, ::-1].copy()   # BGR

        # Point cloud: back-project aligned depth using color intrinsics
        pts_cam   = depth_to_pointcloud_cam(l515_depth_align, color_image,
                                            self.l515_color_intr, self.l515_depth_scale)
        # Transform to world frame
        pts_world = transform_pointcloud(pts_cam, self.T_world_L515)

        # ── D435i ─────────────────────────────────────────────────
        d435i_aligned = self.d435i_align.process(d435i_frames)
        d435i_color   = np.asanyarray(d435i_aligned.get_color_frame().get_data())  # BGR 1280×720
        d435i_depth   = np.asanyarray(d435i_aligned.get_depth_frame().get_data())  # uint16 1280×720

        # Correct 90° CW physical mounting → rotate 90° CCW
        # depth_image.png: 720×1280 uint16 after rotation
        depth_image = cv2.rotate(d435i_depth, D435I_ROTATE)   # (1280, 720) → (720, 1280)... wait
        # Note: D435I_W=1280, D435I_H=720. Rotating 90° CCW: output shape = (1280, 720)
        # i.e. new height=D435I_W=1280, new width=D435I_H=720

        d435i_color_rot = cv2.rotate(d435i_color, D435I_ROTATE)

        # ArUco detection on rotation-corrected image with adjusted intrinsics
        d435i_gray = cv2.cvtColor(d435i_color_rot, cv2.COLOR_BGR2GRAY)
        d_det      = detect_markers(d435i_gray, self.d435i_K, self.d435i_D)
        # Retry with sharpened image for any arm markers not found (motion-blur recovery)
        missing_arm = [m for ids in ARM_MARKER_GROUPS.values() for m in ids if m not in d_det]
        if missing_arm:
            d_det_sharp = detect_markers(_sharpen(d435i_gray), self.d435i_K, self.d435i_D)
            for m in missing_arm:
                if m in d_det_sharp:
                    d_det[m] = d_det_sharp[m]
        # CLAHE retry for markers still missing (poor-contrast edges of dropout runs)
        still_missing = [m for m in missing_arm if m not in d_det]
        if still_missing:
            d_det_clahe = detect_markers(_clahe(d435i_gray), self.d435i_K, self.d435i_D)
            for m in still_missing:
                if m in d_det_clahe:
                    d_det[m] = d_det_clahe[m]

        # Update head pose from floor markers (removes head-turn artifact)
        self.head_pose_age += 1   # reset to 0 inside _update_head_pose if a marker is seen
        self._update_head_pose(d_det)

        T_wh = self.T_world_head if self.T_world_head is not None else NAN_POSE.copy()

        # L515 arm detection — run every frame for fallback
        l515_gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        l515_det  = detect_markers(l515_gray, self.l515_K, self.l515_D)
        # Retry with sharpened image for any arm markers not found
        missing_l515 = [m for ids in ARM_MARKER_GROUPS.values() for m in ids if m not in l515_det]
        if missing_l515:
            l515_det_sharp = detect_markers(_sharpen(l515_gray), self.l515_K, self.l515_D)
            for m in missing_l515:
                if m in l515_det_sharp:
                    l515_det[m] = l515_det_sharp[m]
        # CLAHE retry for markers still missing
        still_missing_l515 = [m for m in missing_l515 if m not in l515_det]
        if still_missing_l515:
            l515_det_clahe = detect_markers(_clahe(l515_gray), self.l515_K, self.l515_D)
            for m in still_missing_l515:
                if m in l515_det_clahe:
                    l515_det[m] = l515_det_clahe[m]

        # Arm poses in world frame: D435i primary, L515 fallback
        arm_poses = {}
        for joint, marker_ids in ARM_MARKER_GROUPS.items():
            T_joint = NAN_POSE.copy()
            if not np.any(np.isnan(T_wh)):
                for mid in marker_ids:
                    if mid in d_det:
                        T_joint = T_wh @ d_det[mid]
                        break
            if np.any(np.isnan(T_joint)) and self.T_world_L515 is not None:
                for mid in marker_ids:
                    if mid in l515_det:
                        T_joint = self.T_world_L515 @ l515_det[mid]
                        break
            # Outlier rejection: discard pose if it implies physically implausible speed
            if not np.any(np.isnan(T_joint)):
                last = self._last_valid_arm_pose.get(joint)
                if last is not None:
                    dist = np.linalg.norm(T_joint[:3, 3] - last[1][:3, 3])
                    dt = self.frame_idx - last[0]
                    if dist / dt > MAX_ARM_SPEED_M_PER_FRAME * 3:
                        T_joint = NAN_POSE.copy()
            if not np.any(np.isnan(T_joint)):
                self._last_valid_arm_pose[joint] = (self.frame_idx, T_joint)
            arm_poses[joint] = T_joint

        self.frame_buffer.append({
            'idx':           self.frame_idx,
            'color_image':   color_image,       # (720, 1280, 3) uint8 BGR
            'depth_image':   depth_image,       # (1280, 720) uint16  ← after rotation
            'pointcloud':    pts_world,         # (N, 6) float32 world frame
            'pose':          T_wh.copy(),       # (4, 4) D435i in world
            'arm_poses':     arm_poses,
            'phase':         self.current_phase,
            'head_pose_age': self.head_pose_age,  # frames since T_world_head last updated
        })

        # ── Optional live preview during recording ─────────────────────────────
        if self._show_preview:
            head_tracked = not np.any(np.isnan(T_wh))
            arms_d435i   = [j for j, ids in ARM_MARKER_GROUPS.items()
                            if any(mid in d_det for mid in ids)]
            arms_l515    = [j for j in ARM_JOINTS
                            if j not in arms_d435i
                            and not np.any(np.isnan(arm_poses[j]))]

            age = self.head_pose_age
            if not head_tracked:
                head_str   = "Head: NO FLOOR MARKER"
                head_color = (0, 0, 255)       # red
            elif age > 30:
                head_str   = f"Head: STALE ({age} fr) — floor marker needed"
                head_color = (0, 140, 255)     # orange
            else:
                head_str   = f"Head: OK (age {age} fr)"
                head_color = (0, 255, 0)       # green

            l515_status = [
                (f"Frame {self.frame_idx}", (200, 200, 200)),
                (f"Arms (L515 fallback): {arms_l515 if arms_l515 else 'none'}",
                 (0, 200, 255) if arms_l515 else (150, 150, 150)),
            ]
            d435i_status = [
                (head_str, head_color),
                (f"Arms (D435i): {arms_d435i if arms_d435i else 'none'}",
                 (200, 200, 200)),
                (f"Phase [{self.current_phase}] {PHASE_NAMES[self.current_phase]}"
                 "  — press 0-5 to change",
                 (255, 200, 0)),
                ("Press 'Q' to stop", (200, 200, 200)),
            ]

            l515_preview  = draw_preview(color_image,     l515_det, ARM_IDS, l515_status)
            d435i_preview = draw_preview(d435i_color_rot, d_det,    ARM_IDS, d435i_status)

            cv2.imshow('L515 (floor)', l515_preview)
            cv2.imshow('D435i (head)', d435i_preview)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                raise KeyboardInterrupt("Quit from preview window.")
            for digit in range(6):
                if key == ord(str(digit)):
                    self.current_phase = digit
                    print(f"\n  Phase → {digit}: {PHASE_NAMES[digit]}", flush=True)

        else:
            # No preview — non-blocking stdin check for phase change (type digit + Enter)
            r, _, _ = select.select([sys.stdin], [], [], 0)
            if r:
                line = sys.stdin.readline().strip()
                if line.isdigit() and 0 <= int(line) <= 5:
                    self.current_phase = int(line)
                    print(f"\n  Phase → {self.current_phase}: "
                          f"{PHASE_NAMES[self.current_phase]}", flush=True)

        self.frame_idx += 1

    # ── Saving ────────────────────────────────────────────────────────────────

    def _save_frame(self, data: dict):
        frame_dir = self.output_dir / f"frame_{data['idx']}"
        frame_dir.mkdir(exist_ok=True)

        # color_image.png — L515 BGR 1280×720
        cv2.imwrite(str(frame_dir / 'color_image.png'), data['color_image'])

        # depth_image.png — D435i uint16, aligned+rotated (shape: 1280×720 after CCW rotation)
        cv2.imwrite(str(frame_dir / 'depth_image.png'), data['depth_image'])

        # pointcloud.npy — (N, 6) float32 in world frame [X,Y,Z,B,G,R]
        np.save(str(frame_dir / 'pointcloud.npy'), data['pointcloud'])

        # pose.txt — D435i camera (head) pose in world frame (4×4)
        np.savetxt(str(frame_dir / 'pose.txt'), data['pose'])

        # pose_<joint>.txt — each arm joint in world frame (4×4)
        for joint in ARM_JOINTS:
            np.savetxt(str(frame_dir / f'pose_{joint}.txt'), data['arm_poses'][joint])

        # phase.txt — task phase label (integer 0-5)
        with open(str(frame_dir / 'phase.txt'), 'w') as f:
            f.write(f"{data['phase']}\n")

        # head_staleness.txt — frames since T_world_head was last updated from a floor marker
        with open(str(frame_dir / 'head_staleness.txt'), 'w') as f:
            f.write(f"{data['head_pose_age']}\n")

    def flush(self):
        n = len(self.frame_buffer)
        if n == 0:
            return
        print(f"\nSaving {n} frames to disk...")
        futures = [self.executor.submit(self._save_frame, d) for d in self.frame_buffer]
        for fut in futures:
            fut.result()
        self.frame_buffer.clear()
        print("Saved.")

    def stop(self):
        cv2.destroyAllWindows()
        self.flush()
        try:
            self.l515_pipe.stop()
        except RuntimeError:
            pass
        try:
            self.d435i_pipe.stop()
        except RuntimeError:
            pass
        self.executor.shutdown(wait=True)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Dual-camera data collection: L515 (floor) + D435i (head)')
    parser.add_argument('-o', '--output',
                        default='/home/primpunn/experiment/both/saved_data',
                        help='Output directory (default: ./both)')
    parser.add_argument('--total_frames', type=int, default=200,
                        help='Stop after N frames (default: 30)')
    parser.add_argument('--no-preview', action='store_true',
                        help='Disable live preview windows during recording (init always shows)')
    args = parser.parse_args()

    recorder = DualCameraRecorder(args.output)
    recorder._show_preview = not args.no_preview

    try:
        recorder.start_cameras()
        recorder.initialize_transforms()

        print("\n=== RECORDING ===")
        print("Perform the task.")
        print("  Phase keys 0-5 in preview window  (or type digit+Enter with --no-preview)")
        print("  0=idle  1=approach  2=lift  3=press  4=hold  5=release")
        print("Press Ctrl+C or 'Q' to stop.\n")

        t0 = time.time()
        while True:
            recorder.record_frame()
            n = recorder.frame_idx

            if n % 30 == 0:
                elapsed = time.time() - t0
                fps = n / elapsed if elapsed > 0 else 0
                age = recorder.head_pose_age
                if recorder.T_world_head is None:
                    head_msg = "NO HEAD POSE"
                elif age > 30:
                    head_msg = f"STALE {age}fr ← glance at floor marker"
                else:
                    head_msg = f"OK (age {age}fr)"
                print(f"\r  Frame {n:5d} | {fps:4.1f} fps | Head: {head_msg}",
                      end='', flush=True)

            if args.total_frames is not None and n >= args.total_frames:
                print()
                break

    except KeyboardInterrupt:
        print("\nCtrl+C — stopping...")

    finally:
        recorder.stop()

    print(f"\nDone. {recorder.frame_idx} frames saved to: {recorder.output_dir}")


if __name__ == '__main__':
    main()
