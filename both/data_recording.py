#!/usr/bin/env python3
"""
Dual-camera data collection: Intel RealSense L515 (floor, static) + D435i (head-mounted, 90° CW)

Setup:
  - L515 placed on the floor as a static third-person-view camera.
  - D435i mounted on the human head, physically rotated 90° clockwise.
  - ArUco markers (DICT_4X4_100, 3 cm):
      ID0 = right wrist     ID1 = right elbow
      ID2 = left wrist      ID3 = left elbow
      ID4 = world frame     ID5-ID10 = additional floor markers

Collection steps:
  1. Place ID4-ID10 on the floor (arbitrary positions). ID4 defines the world frame.
  2. Run this script. Both cameras must see ID4 simultaneously during initialization.
  3. After init, record freely. D435i must always see at least one of ID4-ID10.

Output per frame  (saved in output_dir/frame_N/):
  color_image.png       L515 color, BGR, 1280×720
  depth_image.png       D435i depth aligned to color, rotated 90° CCW, uint16, 720×1280
  pointcloud.npy        L515 point cloud in world frame, (N,6) float32: [X,Y,Z,B,G,R]
  pose.txt              D435i camera pose in world frame (4×4)
  pose_right_wrist.txt  ID0 in world frame (4×4)  — NaN matrix if not detected
  pose_right_elbow.txt  ID1 in world frame (4×4)
  pose_left_wrist.txt   ID2 in world frame (4×4)
  pose_left_elbow.txt   ID3 in world frame (4×4)

Transform convention:
  T_A_B is a 4×4 matrix that transforms points FROM frame B TO frame A:
      p_A = T_A_B @ p_B_homogeneous

D435i rotation note:
  Physical 90° CW mounting → images appear rotated 90° CW.
  Correction: rotate images 90° CCW before saving and before ArUco detection.
  Intrinsics are adjusted accordingly (fx↔fy, cx↔cy with axis flip).

Usage:
  conda activate massage
  python data_recording.py -o ./saved_data --total_frames 1000
  python data_recording.py -o ./saved_data          # run until Ctrl+C
"""

import os
import sys
import time
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import cv2
import pyrealsense2 as rs

# Display scale: resize preview windows to this height (keeps aspect ratio)
PREVIEW_H = 480


# ── Constants ──────────────────────────────────────────────────────────────────
ARUCO_DICT_ID   = cv2.aruco.DICT_4X4_100
MARKER_SIZE_M   = 0.03          # 3 cm physical size
WORLD_ID        = 4             # ID4 defines the world frame origin
FLOOR_IDS       = list(range(4, 11))       # IDs placed on floor (static)
ARM_IDS         = [0, 1, 2, 3]
ARM_NAMES       = {0: 'right_wrist', 1: 'right_elbow',
                   2: 'left_wrist',  3: 'left_elbow'}

L515_COLOR_W, L515_COLOR_H = 1280, 720
L515_DEPTH_W, L515_DEPTH_H = 1024, 768
D435I_W,      D435I_H      = 1280, 720
TARGET_FPS                  = 30
PC_NUM_POINTS               = 8192   # number of points to subsample per frame

# D435i physically rotated 90° CW → correct with 90° CCW
D435I_ROTATE = cv2.ROTATE_90_COUNTERCLOCKWISE

NAN_POSE = np.full((4, 4), np.nan)


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
    params     = cv2.aruco.DetectorParameters()
    return cv2.aruco.ArucoDetector(aruco_dict, params)


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
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            [corners[i]], MARKER_SIZE_M, K, D)
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
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.l515_pipe  = rs.pipeline()
        self.d435i_pipe = rs.pipeline()

        # Set during start_cameras()
        self.l515_color_intr : rs.intrinsics = None
        self.l515_depth_scale: float         = None
        self.l515_align                      = None

        self.d435i_depth_scale: float        = None
        self.d435i_K: np.ndarray             = None   # after rotation correction
        self.d435i_D: np.ndarray             = None
        self.d435i_align                     = None

        # Set during initialize_transforms()
        self.T_world_L515 : np.ndarray       = None   # L515 camera → world
        self.T_world_floor: dict             = {}     # {id: 4×4} floor markers → world

        # Updated each frame
        self.T_world_head : np.ndarray       = None   # last known D435i → world

        self.frame_buffer: list  = []
        self.executor            = ThreadPoolExecutor(max_workers=8)
        self.frame_idx           = 0
        self._show_preview       = True   # toggled by --no-preview flag

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
          Both cameras must see ArUco ID4 simultaneously.
          Live preview windows show each camera's view with marker detection overlay.
          Press 'S' in either window (or ENTER in terminal) to start collecting.

          From L515:
            T_world_L515 = inv(T_L515_ID4)
          From D435i:
            T_world_head  = inv(T_head_ID4)
            T_world_IDk   = T_world_head @ T_head_IDk   for k in FLOOR_IDS

          Averages 60 valid frames for robustness.
        """
        print("=== INITIALIZATION ===")
        print("Preview windows are open. Position cameras so ID4 is visible in BOTH.")
        print("Press 'S' in either preview window  OR  press ENTER here to start.\n")

        cv2.namedWindow('L515 (floor)',  cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow('D435i (head)',  cv2.WINDOW_AUTOSIZE)

        l515_K, l515_D = intr_to_K_D(self.l515_color_intr)

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
            l515_det     = detect_markers(l515_gray, l515_K, l515_D)
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
                    (f"ID4: {'DETECTED' if l515_ok else 'NOT SEEN'}",
                     (0, 255, 0) if l515_ok else (0, 0, 255)),
                    (f"Collecting: {bar} {prog}",
                     (0, 220, 255)),
                ]
                d435i_status = [
                    (f"ID4: {'DETECTED' if d435i_ok else 'NOT SEEN'}",
                     (0, 255, 0) if d435i_ok else (0, 0, 255)),
                    (f"Floor IDs seen: {sorted(k for k in d_det if k in FLOOR_IDS)}",
                     (200, 200, 200)),
                    (f"Collecting: {bar} {prog}",
                     (0, 220, 255)),
                ]
            else:
                hint = "Press 'S' here or ENTER in terminal to start"
                ready_color = (0, 255, 0) if both_ok else (0, 165, 255)
                ready_text  = "READY — both see ID4!" if both_ok else "Waiting — adjust until both see ID4"
                l515_status = [
                    (f"ID4: {'DETECTED' if l515_ok else 'NOT SEEN'}",
                     (0, 255, 0) if l515_ok else (0, 0, 255)),
                    (ready_text, ready_color),
                    (hint, (200, 200, 200)),
                ]
                d435i_status = [
                    (f"ID4: {'DETECTED' if d435i_ok else 'NOT SEEN'}",
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
                        print("  Warning: not both cameras see ID4 yet — starting anyway.")
                    print("  Starting to collect init frames...")
                    collecting = True
                    t_start    = time.time()

            # ── Collect init frames ───────────────────────────────
            if collecting:
                if l515_ok and d435i_ok:
                    T_L515_ID4   = l515_det[WORLD_ID]
                    T_world_L515 = np.linalg.inv(T_L515_ID4)

                    T_head_ID4   = d_det[WORLD_ID]
                    T_world_head = np.linalg.inv(T_head_ID4)

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

        self.T_world_floor[WORLD_ID] = np.eye(4)   # ID4 == world by definition

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

        # Update head pose from floor markers (removes head-turn artifact)
        self._update_head_pose(d_det)

        T_wh = self.T_world_head if self.T_world_head is not None else NAN_POSE.copy()

        # Arm poses in world frame
        arm_poses = {}
        for aid in ARM_IDS:
            if aid in d_det and not np.any(np.isnan(T_wh)):
                # T_world_arm = T_world_head @ T_head_arm
                arm_poses[aid] = T_wh @ d_det[aid]
            else:
                arm_poses[aid] = NAN_POSE.copy()

        self.frame_buffer.append({
            'idx':         self.frame_idx,
            'color_image': color_image,       # (720, 1280, 3) uint8 BGR
            'depth_image': depth_image,       # (1280, 720) uint16  ← after rotation
            'pointcloud':  pts_world,         # (N, 6) float32 world frame
            'pose':        T_wh.copy(),       # (4, 4) D435i in world
            'arm_poses':   arm_poses,
        })

        # ── Optional live preview during recording ─────────────────────────────
        if self._show_preview:
            head_tracked = not np.any(np.isnan(T_wh))
            arm_seen     = [ARM_NAMES[a] for a in ARM_IDS if a in d_det]

            l515_status = [
                (f"Frame {self.frame_idx}", (200, 200, 200)),
                ('L515 — recording', (0, 255, 0)),
            ]
            d435i_status = [
                (f"Head tracked: {'YES' if head_tracked else 'NO — check floor markers'}",
                 (0, 255, 0) if head_tracked else (0, 0, 255)),
                (f"Arms seen: {arm_seen if arm_seen else 'none'}",
                 (200, 200, 200)),
                ("Press 'Q' to stop recording", (200, 200, 200)),
            ]
            l515_K, l515_D = intr_to_K_D(self.l515_color_intr)
            l515_gray_prev = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            l515_det_prev  = detect_markers(l515_gray_prev, l515_K, l515_D)

            l515_preview  = draw_preview(color_image,   l515_det_prev, [],      l515_status)
            d435i_preview = draw_preview(d435i_color_rot, d_det,       ARM_IDS, d435i_status)

            cv2.imshow('L515 (floor)', l515_preview)
            cv2.imshow('D435i (head)', d435i_preview)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                raise KeyboardInterrupt("Quit from preview window.")

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

        # pose_<arm>.txt — each arm marker in world frame (4×4)
        for aid, name in ARM_NAMES.items():
            np.savetxt(str(frame_dir / f'pose_{name}.txt'), data['arm_poses'][aid])

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
        self.l515_pipe.stop()
        self.d435i_pipe.stop()
        self.executor.shutdown(wait=True)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Dual-camera data collection: L515 (floor) + D435i (head)')
    parser.add_argument('-o', '--output',
                        default='/home/primpunn/Desktop/1st paper/both',
                        help='Output directory (default: ./both)')
    parser.add_argument('--total_frames', type=int, default=30,
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
        print("Perform the task. Press Ctrl+C to stop.\n")

        t0 = time.time()
        while True:
            recorder.record_frame()
            n = recorder.frame_idx

            if n % 30 == 0:
                elapsed = time.time() - t0
                fps = n / elapsed if elapsed > 0 else 0
                head_ok = recorder.T_world_head is not None
                print(f"\r  Frame {n:5d} | {fps:4.1f} fps | "
                      f"Head tracked: {'YES' if head_ok else 'NO ← check floor markers'}",
                      end='', flush=True)

            if args.total_frames is not None and n >= args.total_frames:
                print()
                break

    except KeyboardInterrupt:
        print("\nCtrl+C — stopping...")

    finally:
        recorder.stop()

    print(f"\nDone. {recorder.frame_idx} frames saved to: {args.output}")


if __name__ == '__main__':
    main()
