#!/usr/bin/env python3
"""
Project 3D arm joint positions onto L515 colour images for visual sanity-check.
Saves overlay PNGs to <session>/overlay_check/.

Usage:
    conda activate massage
    python overlay_joints.py saved_data/2026-05-01_16-30-06
    python overlay_joints.py saved_data/2026-05-01_16-30-06 --frames 0 50 100 200 300 400
"""
import argparse, os
import numpy as np
import cv2

# Calibrated from colour/pointcloud alignment
FX, FY = 914.0, 914.0
CX, CY = 653.0, 347.0

JOINTS = {
    'right_wrist':  (0,   80, 230),   # red-ish (BGR)
    'right_elbow':  (0,  130, 230),   # orange-ish
    'left_wrist':   (230, 130,  41),  # blue
    'left_elbow':   (70, 180,  41),   # green
}
LINKS = [
    ('right_wrist', 'right_elbow'),
    ('left_wrist',  'left_elbow'),
]


def load_joint_world(frame_dir: str, joint: str):
    """Returns (3,) XYZ or None if NaN/missing."""
    for suffix in ('_processed', ''):
        p = os.path.join(frame_dir, f'pose_{joint}{suffix}.txt')
        if os.path.exists(p):
            T = np.loadtxt(p)
            if not np.any(np.isnan(T)):
                return T[:3, 3]
    return None


def project(xyz_world, T_L515_world):
    """Project a world-frame point to pixel (u, v). Returns None if behind camera."""
    p = T_L515_world @ np.array([*xyz_world, 1.0])
    if p[2] < 0.1:
        return None
    u = int(FX * p[0] / p[2] + CX)
    v = int(FY * p[1] / p[2] + CY)
    return u, v


def overlay_frame(session_dir, frame_idx, T_L515_world):
    frame_dir = os.path.join(session_dir, f'frame_{frame_idx}')
    img_path  = os.path.join(frame_dir, 'color_image.png')
    if not os.path.exists(img_path):
        return None

    img = cv2.imread(img_path)
    H, W = img.shape[:2]

    pts = {}
    for joint in JOINTS:
        xyz = load_joint_world(frame_dir, joint)
        if xyz is not None:
            uv = project(xyz, T_L515_world)
            if uv and 0 <= uv[0] < W and 0 <= uv[1] < H:
                pts[joint] = uv

    # Draw skeleton links
    for j_a, j_b in LINKS:
        if j_a in pts and j_b in pts:
            col = JOINTS[j_a]
            cv2.line(img, pts[j_a], pts[j_b], col, 2, cv2.LINE_AA)

    # Draw joints
    for joint, uv in pts.items():
        col = JOINTS[joint]
        cv2.circle(img, uv, 10, col, -1, cv2.LINE_AA)
        cv2.circle(img, uv, 11, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, joint.replace('_', ' '), (uv[0]+13, uv[1]+4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(img, f'frame {frame_idx}', (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('session_dir')
    parser.add_argument('--frames', type=int, nargs='+', default=None,
                        help='Frame indices to render (default: 10 evenly spaced)')
    args = parser.parse_args()

    session_dir = args.session_dir.rstrip('/')
    out_dir = os.path.join(session_dir, 'overlay_check')
    os.makedirs(out_dir, exist_ok=True)

    T_world_L515 = np.loadtxt(os.path.join(session_dir, 'T_world_L515.txt'))
    T_L515_world = np.linalg.inv(T_world_L515)

    all_frames = sorted(
        [int(d.split('_')[1]) for d in os.listdir(session_dir) if d.startswith('frame_')]
    )
    if args.frames:
        frame_list = args.frames
    else:
        step = max(1, len(all_frames) // 10)
        frame_list = all_frames[::step]

    saved = []
    for fi in frame_list:
        img = overlay_frame(session_dir, fi, T_L515_world)
        if img is None:
            print(f'  frame_{fi}: color image missing, skip')
            continue
        out_path = os.path.join(out_dir, f'frame_{fi:04d}.png')
        cv2.imwrite(out_path, img)
        saved.append(out_path)
        print(f'  Saved {out_path}')

    print(f'\nDone — {len(saved)} overlays in {out_dir}')


if __name__ == '__main__':
    main()
