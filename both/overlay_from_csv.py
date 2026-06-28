"""
Create overlay video from arm_trajectory.csv — no JSON or GPU needed.

Reads the CSV produced by extract_arm_trajectory.py and projects joints
onto the source frames using weak-perspective:
    u = fx * (joint_x + cam_t_x) / (joint_z + cam_t_z) + cx

Drawn joints:
  Therapist left arm  (blue)   T_L_elbow → T_L_wrist
  Therapist right arm (red)    T_R_elbow → T_R_wrist
  Patient ankles      (green)  P_L_ankle, P_R_ankle (connected by line)

Usage:
  python overlay_from_csv.py \
      --csv ./output/arm_trajectory.csv \
      --frames ./frames \
      --output ./output/overlay_video.mp4 \
      [--fps 30] [--fx 1000] [--fy 1000]
"""

import argparse
import glob
import os

import cv2
import numpy as np
import pandas as pd

THERAPIST_LEFT  = ["T_L_elbow", "T_L_wrist"]
THERAPIST_RIGHT = ["T_R_elbow", "T_R_wrist"]
PATIENT_ANKLES  = ["P_L_ankle", "P_R_ankle"]

LEFT_COLOR    = (255, 80,   0)   # blue
RIGHT_COLOR   = (0,   80, 255)   # red
PATIENT_COLOR = (0,  220,  80)   # green


def project(joint_xyz_root: np.ndarray, cam_t: np.ndarray,
            fx: float, fy: float, cx: float, cy: float):
    """Root-relative joint + cam_t → (u, v) pixel, or None if behind camera."""
    xyz = joint_xyz_root + cam_t
    if xyz[2] <= 0:
        return None
    u = fx * xyz[0] / xyz[2] + cx
    v = fy * xyz[1] / xyz[2] + cy
    return (int(round(u)), int(round(v)))


def get_uv(row: pd.Series, joint: str, cam_t: np.ndarray,
            fx: float, fy: float, cx: float, cy: float):
    """Return pixel coords for one joint from a CSV row, or None if NaN."""
    x, y, z = row.get(f"{joint}_x"), row.get(f"{joint}_y"), row.get(f"{joint}_z")
    if any(v is None or (isinstance(v, float) and np.isnan(v)) for v in (x, y, z)):
        return None
    return project(np.array([x, y, z], dtype=np.float64), cam_t, fx, fy, cx, cy)


def draw_chain(img, uvs: list, color: tuple, radius: int = 7) -> None:
    valid = [uv for uv in uvs if uv is not None]
    for uv in valid:
        cv2.circle(img, uv, radius, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(img, uv, radius + 1, (255, 255, 255), 1, lineType=cv2.LINE_AA)
    for a, b in zip(valid[:-1], valid[1:]):
        cv2.line(img, a, b, color, 3, lineType=cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",     default="./output/arm_trajectory.csv")
    parser.add_argument("--frames",  default="./frames")
    parser.add_argument("--output",  default="./output/overlay_video.mp4")
    parser.add_argument("--fps",     type=float, default=30.0)
    parser.add_argument("--fx",      type=float, default=1000.0)
    parser.add_argument("--fy",      type=float, default=1000.0)
    args = parser.parse_args()

    if not os.path.isfile(args.csv):
        print(f"[ERROR] CSV not found: {args.csv}")
        print("  Run: python extract_arm_trajectory.py")
        raise SystemExit(1)

    df = pd.read_csv(args.csv)
    # Index by frame number for O(1) lookup
    df = df.set_index("frame")

    required_cam_t = ["T_cam_t_x", "T_cam_t_y", "T_cam_t_z"]
    if not all(c in df.columns for c in required_cam_t):
        print("[ERROR] CSV is missing cam_t columns (T_cam_t_x/y/z).")
        print("  Re-run: python extract_arm_trajectory.py  (no GPU needed)")
        raise SystemExit(1)

    frame_paths = sorted(glob.glob(os.path.join(args.frames, "frame_*.jpg")))
    if not frame_paths:
        print(f"[ERROR] No frame_*.jpg files found in {args.frames}")
        raise SystemExit(1)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    writer = None
    n_overlaid = 0

    for frame_path in frame_paths:
        # Extract frame index from filename (e.g. frame_000042.jpg → 42)
        basename = os.path.splitext(os.path.basename(frame_path))[0]  # "frame_000042"
        frame_idx = int(basename.split("_", 1)[1])

        img = cv2.imread(frame_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        cx, cy = w / 2.0, h / 2.0

        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(args.output, fourcc, args.fps, (w, h))

        if frame_idx in df.index:
            row = df.loc[frame_idx]

            # --- Therapist ---
            t_cam_t = np.array([row["T_cam_t_x"], row["T_cam_t_y"], row["T_cam_t_z"]],
                                dtype=np.float64)
            if not np.any(np.isnan(t_cam_t)):
                kw = dict(cam_t=t_cam_t, fx=args.fx, fy=args.fy, cx=cx, cy=cy)
                left_uvs  = [get_uv(row, j, **kw) for j in THERAPIST_LEFT]
                right_uvs = [get_uv(row, j, **kw) for j in THERAPIST_RIGHT]
                draw_chain(img, left_uvs,  LEFT_COLOR)
                draw_chain(img, right_uvs, RIGHT_COLOR)

                # Label wrists
                for joint, uvs in [("T_L_wrist", left_uvs), ("T_R_wrist", right_uvs)]:
                    uv = uvs[-1] if uvs else None
                    if uv:
                        cv2.putText(img, joint, (uv[0] + 9, uv[1] - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                    LEFT_COLOR if "L" in joint else RIGHT_COLOR,
                                    1, cv2.LINE_AA)

            # --- Patient ankles ---
            p_cam_t_cols = ["P_cam_t_x", "P_cam_t_y", "P_cam_t_z"]
            if all(c in df.columns for c in p_cam_t_cols):
                p_cam_t = np.array([row[c] for c in p_cam_t_cols], dtype=np.float64)
                if not np.any(np.isnan(p_cam_t)):
                    kw_p = dict(cam_t=p_cam_t, fx=args.fx, fy=args.fy, cx=cx, cy=cy)
                    ankle_uvs = [get_uv(row, j, **kw_p) for j in PATIENT_ANKLES]
                    draw_chain(img, ankle_uvs, PATIENT_COLOR, radius=9)
                    for joint, uv in zip(PATIENT_ANKLES, ankle_uvs):
                        if uv:
                            cv2.putText(img, joint, (uv[0] + 11, uv[1] - 5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                        PATIENT_COLOR, 1, cv2.LINE_AA)

            # Frame counter
            cv2.putText(img, f"frame {frame_idx}", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2, cv2.LINE_AA)
            n_overlaid += 1

        writer.write(img)

    if writer is not None:
        writer.release()

    print(f"Overlaid {n_overlaid}/{len(frame_paths)} frames")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
