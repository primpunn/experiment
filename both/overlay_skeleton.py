"""
Overlay contact-point keypoints onto the source video frames.

Focuses on positions relevant to calf-stretching motion capture:
  - Therapist elbows + wrists (arm configuration and contact with patient)
  - Patient ankles (body part being lifted/pressed)

SAM-3D-Body's pred_keypoints_3d (mhr70 layout) is root-relative -- it has to be
shifted by pred_cam_t (the per-frame camera translation) before it represents
real camera-space 3D coordinates. Confirmed empirically: kp + cam_t projects
back inside each frame's saved person bbox.

Since no real camera intrinsics were estimated (FOV estimator was disabled for
this run), this uses a fixed weak-perspective projection:
    u = fx * X/Z + cx   (fx≈1000, cx=W/2)
    v = fy * Y/Z + cy   (fy≈1000, cy=H/2)

Usage:
  python overlay_skeleton.py \
      --input ./output/body_params_per_frame.json \
      --frames ./frames \
      --output ./output/overlay_video.mp4 \
      --fps 30
"""

import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

# Therapist: elbow (arm config) + wrist (contact with patient)
THERAPIST_JOINTS = {
    "T_L_elbow": 7,
    "T_R_elbow": 8,
    "T_L_wrist": 62,
    "T_R_wrist": 41,
}

# Patient: ankles — body part being manipulated
PATIENT_JOINTS = {
    "P_L_ankle": 13,
    "P_R_ankle": 14,
}

LEFT_CHAIN   = ["T_L_elbow", "T_L_wrist"]
RIGHT_CHAIN  = ["T_R_elbow", "T_R_wrist"]

LEFT_COLOR_BGR    = (255, 80,  0)    # blue  — therapist left arm
RIGHT_COLOR_BGR   = (0,   80, 255)   # red   — therapist right arm
PATIENT_COLOR_BGR = (0,  220,  80)   # green — patient ankles


def project_weak_perspective(points_3d: np.ndarray, fx: float, fy: float,
                              cx: float, cy: float) -> np.ndarray:
    """points_3d: (N, 3) camera-space XYZ -> (N, 2) pixel coords."""
    z = points_3d[:, 2]
    u = fx * points_3d[:, 0] / z + cx
    v = fy * points_3d[:, 1] / z + cy
    return np.stack([u, v], axis=1)


def project_joints(kp: np.ndarray, cam_t: np.ndarray, joint_map: dict,
                    fx: float, fy: float, cx: float, cy: float) -> dict:
    """Shift root-relative keypoints into camera space and project to 2D pixels."""
    cam_xyz = kp + cam_t
    joints_2d = {}
    for name, idx in joint_map.items():
        if idx < len(cam_xyz):
            uv = project_weak_perspective(cam_xyz[idx:idx + 1], fx, fy, cx, cy)[0]
            joints_2d[name] = uv
    return joints_2d


def draw_chain(frame, joints_2d: dict, chain: list, color: tuple, radius: int = 6) -> None:
    pts = [joints_2d[name] for name in chain if name in joints_2d]
    for pt in pts:
        x, y = int(round(pt[0])), int(round(pt[1]))
        cv2.circle(frame, (x, y), radius, color, -1, lineType=cv2.LINE_AA)
        cv2.putText(frame, [n for n, p in joints_2d.items() if (p == pt).all()][0],
                    (x + 8, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    for (x1, y1), (x2, y2) in zip(pts[:-1], pts[1:]):
        cv2.line(frame, (int(round(x1)), int(round(y1))),
                  (int(round(x2)), int(round(y2))), color, 3, lineType=cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="./output/body_params_per_frame.json")
    parser.add_argument("--frames", default="./frames")
    parser.add_argument("--output", default="./output/overlay_video.mp4")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--fx", type=float, default=1000.0)
    parser.add_argument("--fy", type=float, default=1000.0)
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"[ERROR] Input not found: {args.input}")
        sys.exit(1)

    with open(args.input) as f:
        body_params = json.load(f)

    frame_paths = sorted(glob.glob(os.path.join(args.frames, "frame_*.jpg")))
    if not frame_paths:
        print(f"[ERROR] No frames found in {args.frames}")
        sys.exit(1)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    writer = None
    n_overlaid = 0
    for frame_idx, frame_path in enumerate(frame_paths):
        frame = cv2.imread(frame_path)
        if frame is None:
            continue
        h, w = frame.shape[:2]
        cx, cy = w / 2.0, h / 2.0

        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(args.output, fourcc, args.fps, (w, h))

        params = body_params.get(f"frame_{frame_idx}")
        if params is not None:
            # Support both new nested format and old flat format
            if "therapist" in params or "patient" in params:
                t_data = params.get("therapist", {})
                p_data = params.get("patient", {})
            else:
                t_data = params   # old flat format → treat as therapist
                p_data = {}

            # --- Therapist ---
            if t_data.get("pred_keypoints_3d"):
                kp = np.array(t_data["pred_keypoints_3d"], dtype=np.float64)
                cam_t = np.array(t_data["pred_cam_t"], dtype=np.float64)
                j2d = project_joints(kp, cam_t, THERAPIST_JOINTS, args.fx, args.fy, cx, cy)
                draw_chain(frame, j2d, LEFT_CHAIN,  LEFT_COLOR_BGR)
                draw_chain(frame, j2d, RIGHT_CHAIN, RIGHT_COLOR_BGR)

            # --- Patient ankles ---
            if p_data.get("pred_keypoints_3d"):
                pkp = np.array(p_data["pred_keypoints_3d"], dtype=np.float64)
                pcam_t = np.array(p_data["pred_cam_t"], dtype=np.float64)
                pj2d = project_joints(pkp, pcam_t, PATIENT_JOINTS, args.fx, args.fy, cx, cy)
                for name, uv in pj2d.items():
                    x, y = int(round(uv[0])), int(round(uv[1]))
                    cv2.circle(frame, (x, y), 8, PATIENT_COLOR_BGR, -1, lineType=cv2.LINE_AA)
                    cv2.putText(frame, name, (x + 10, y - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, PATIENT_COLOR_BGR, 1, cv2.LINE_AA)
                if "P_L_ankle" in pj2d and "P_R_ankle" in pj2d:
                    la, ra = pj2d["P_L_ankle"], pj2d["P_R_ankle"]
                    cv2.line(frame, (int(round(la[0])), int(round(la[1]))),
                              (int(round(ra[0])), int(round(ra[1]))),
                              PATIENT_COLOR_BGR, 2, lineType=cv2.LINE_AA)

            n_overlaid += 1

        writer.write(frame)

    if writer is not None:
        writer.release()

    print(f"Overlaid {n_overlaid}/{len(frame_paths)} frames")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
