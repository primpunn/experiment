#!/usr/bin/env python3
"""
Create a video from all RGB frames with processed joint positions overlaid.

Projects pose_<joint>_processed.txt 3D world positions onto each color_image.png
using the L515 camera intrinsics, draws coloured dots + skeleton links, and
encodes the result as an MP4 video for visual sanity-checking.

Usage:
    conda activate massage
    python rgb_overlay_video.py saved_data/2026-05-01_16-30-06
    python rgb_overlay_video.py saved_data/2026-05-01_16-30-06 --fps 15 --out my_check.mp4
"""

import argparse
import os
import numpy as np
import cv2

# ── Intrinsics (calibrated from pointcloud/colour alignment) ──────────────────
FX, FY = 914.0, 914.0
CX, CY = 653.0, 347.0

# ── Joint colours (BGR) and skeleton links ────────────────────────────────────
JOINT_COLOR = {
    'right_wrist':  (0,   80, 230),
    'right_elbow':  (0,  160, 255),
    'left_wrist':   (230, 130,  41),
    'left_elbow':   (70,  200,  41),
}
LINKS = [
    ('right_wrist', 'right_elbow'),
    ('left_wrist',  'left_elbow'),
]
JOINTS = list(JOINT_COLOR.keys())

DOT_RADIUS   = 10
LINK_THICK   = 2
FONT         = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE   = 0.45
FONT_THICK   = 1


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_joint_world(frame_dir: str, joint: str):
    """Return (3,) XYZ in world frame from _processed.txt, or None."""
    path = os.path.join(frame_dir, f'pose_{joint}_processed.txt')
    if not os.path.exists(path):
        return None
    T = np.loadtxt(path)
    if np.any(np.isnan(T)):
        return None
    return T[:3, 3]


def project(xyz_world: np.ndarray, T_L515_world: np.ndarray):
    """Project world point to (u, v) pixel. Returns None if behind camera."""
    p = T_L515_world @ np.array([*xyz_world, 1.0])
    if p[2] < 0.1:
        return None
    u = int(FX * p[0] / p[2] + CX)
    v = int(FY * p[1] / p[2] + CY)
    return u, v


def draw_legend(img: np.ndarray):
    """Draw a small colour legend in the bottom-left corner."""
    H, W = img.shape[:2]
    labels = [
        ('right wrist',  JOINT_COLOR['right_wrist']),
        ('right elbow',  JOINT_COLOR['right_elbow']),
        ('left wrist',   JOINT_COLOR['left_wrist']),
        ('left elbow',   JOINT_COLOR['left_elbow']),
    ]
    x0, y0 = 8, H - 10 - len(labels) * 18
    for i, (label, col) in enumerate(labels):
        y = y0 + i * 18
        cv2.circle(img, (x0 + 7, y + 5), 6, col, -1, cv2.LINE_AA)
        cv2.putText(img, label, (x0 + 17, y + 10),
                    FONT, 0.4, (230, 230, 230), 1, cv2.LINE_AA)


def overlay_frame(color_img: np.ndarray, frame_dir: str,
                  T_L515_world: np.ndarray, frame_idx: int) -> np.ndarray:
    img = color_img.copy()
    H, W = img.shape[:2]

    pts = {}
    for joint in JOINTS:
        xyz = load_joint_world(frame_dir, joint)
        if xyz is None:
            continue
        uv = project(xyz, T_L515_world)
        if uv and 0 <= uv[0] < W and 0 <= uv[1] < H:
            pts[joint] = uv

    # Skeleton links
    for j_a, j_b in LINKS:
        if j_a in pts and j_b in pts:
            cv2.line(img, pts[j_a], pts[j_b],
                     JOINT_COLOR[j_a], LINK_THICK, cv2.LINE_AA)

    # Joint dots + labels
    for joint, uv in pts.items():
        col = JOINT_COLOR[joint]
        cv2.circle(img, uv, DOT_RADIUS, col, -1, cv2.LINE_AA)
        cv2.circle(img, uv, DOT_RADIUS + 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, joint.replace('_', ' '),
                    (uv[0] + 14, uv[1] + 5),
                    FONT, FONT_SCALE, (255, 255, 255), FONT_THICK, cv2.LINE_AA)

    # Frame counter
    cv2.putText(img, f'frame {frame_idx}', (8, 22),
                FONT, 0.65, (200, 200, 200), 2, cv2.LINE_AA)

    draw_legend(img)
    return img


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='RGB overlay video with processed joint positions')
    parser.add_argument('session_dir')
    parser.add_argument('--fps', type=int, default=15,
                        help='Output video FPS (default 15)')
    parser.add_argument('--out', default=None,
                        help='Output file path (default: <session>/rgb_overlay.mp4)')
    args = parser.parse_args()

    session_dir = args.session_dir.rstrip('/')
    if not os.path.isdir(session_dir):
        print(f'Error: {session_dir} not found'); return

    out_path = args.out or os.path.join(session_dir, 'rgb_overlay.mp4')

    T_world_L515 = np.loadtxt(os.path.join(session_dir, 'T_world_L515.txt'))
    T_L515_world = np.linalg.inv(T_world_L515)

    frames = sorted(
        [d for d in os.listdir(session_dir) if d.startswith('frame_')],
        key=lambda x: int(x.split('_')[1])
    )
    n = len(frames)
    print(f'Session : {session_dir}')
    print(f'Frames  : {n}   FPS: {args.fps}')

    # Determine frame size from first available colour image
    sample_img = None
    for f in frames:
        p = os.path.join(session_dir, f, 'color_image.png')
        if os.path.exists(p):
            sample_img = cv2.imread(p)
            break
    if sample_img is None:
        print('No color_image.png found.'); return
    H, W = sample_img.shape[:2]
    print(f'Frame size: {W}x{H}')

    writer = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        args.fps,
        (W, H)
    )
    if not writer.isOpened():
        print('Failed to open video writer.'); return

    for i, frame_name in enumerate(frames):
        frame_idx = int(frame_name.split('_')[1])
        frame_dir = os.path.join(session_dir, frame_name)
        img_path  = os.path.join(frame_dir, 'color_image.png')

        if not os.path.exists(img_path):
            print(f'  Missing {img_path}, skipping')
            continue

        color_img = cv2.imread(img_path)
        out_frame = overlay_frame(color_img, frame_dir, T_L515_world, frame_idx)
        writer.write(out_frame)

        if i % 50 == 0:
            print(f'  {i+1}/{n} frames...', end='\r', flush=True)

    writer.release()
    size_mb = os.path.getsize(out_path) / 1e6
    print(f'\nVideo saved: {out_path}  ({size_mb:.1f} MB)')


if __name__ == '__main__':
    main()
