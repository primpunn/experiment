"""
Step 5: Extract contact-point trajectories from SAM-3D-Body output.

Focuses on the positions that matter for calf-stretching motion capture:
  - Therapist arm endpoints (wrists) — where hands contact the patient
  - Therapist elbows — for arm configuration context
  - Patient ankles — the body part being lifted/pressed

mhr70 joint indices (see ~/sam-3d-body/sam_3d_body/metadata/mhr70.py):
  7  = left_elbow       8  = right_elbow
  62 = left_wrist       41 = right_wrist     (therapist arm endpoints)
  13 = left_ankle       14 = right_ankle      (patient contact points)

JSON structure (body_params_per_frame.json):
  {
    "frame_0": {
      "therapist": {"pred_keypoints_3d": [...], "pred_cam_t": [...], "bbox": [...]},
      "patient":   {"pred_keypoints_3d": [...], "pred_cam_t": [...], "bbox": [...]}
    }, ...
  }

Usage:
  python extract_arm_trajectory.py \
      --input ./output/body_params_per_frame.json \
      --output ./output/arm_trajectory.csv \
      --fps 30
"""

import argparse
import json
import os
import sys
import numpy as np
import pandas as pd

# Therapist: arm endpoints (wrists) + elbows for configuration context
THERAPIST_JOINTS = {
    "T_L_elbow": 7,
    "T_R_elbow": 8,
    "T_L_wrist": 62,   # left arm endpoint — contacts patient
    "T_R_wrist": 41,   # right arm endpoint — contacts patient
}

# Patient: ankles — the body part being lifted/pressed during calf-stretching
PATIENT_JOINTS = {
    "P_L_ankle": 13,
    "P_R_ankle": 14,
}


def load_body_params(json_path: str) -> tuple[dict, int]:
    with open(json_path) as f:
        data = json.load(f)
    return data, len(data)


def get_keypoints_per_frame(body_params: dict) -> tuple[dict, dict, dict, dict]:
    """Return (therapist_kp, patient_kp, therapist_cam_t, patient_cam_t).

    Accepts two JSON formats:
      New: {frame_N: {therapist: {pred_keypoints_3d, pred_cam_t}, patient: {...}}}
      Old: {frame_N: {pred_keypoints_3d, pred_cam_t}}  ← treated as therapist-only
    """
    therapist_kp:    dict = {}
    patient_kp:      dict = {}
    therapist_cam_t: dict = {}
    patient_cam_t:   dict = {}
    for frame_key, params in body_params.items():
        if "therapist" in params or "patient" in params:
            # New nested format
            t_data = params.get("therapist", {})
            p_data = params.get("patient", {})
        elif "pred_keypoints_3d" in params:
            # Old flat format — treat as therapist only
            t_data = params
            p_data = {}
        else:
            continue

        kp = t_data.get("pred_keypoints_3d")
        ct = t_data.get("pred_cam_t")
        if kp is not None:
            therapist_kp[frame_key] = np.array(kp, dtype=np.float32)
        if ct is not None:
            therapist_cam_t[frame_key] = np.array(ct, dtype=np.float32)

        pkp = p_data.get("pred_keypoints_3d")
        pct = p_data.get("pred_cam_t")
        if pkp is not None:
            patient_kp[frame_key] = np.array(pkp, dtype=np.float32)
        if pct is not None:
            patient_cam_t[frame_key] = np.array(pct, dtype=np.float32)

    return therapist_kp, patient_kp, therapist_cam_t, patient_cam_t


def build_trajectory_df(therapist_kp: dict, patient_kp: dict,
                         therapist_cam_t: dict, patient_cam_t: dict,
                         fps: float, sorted_keys: list) -> pd.DataFrame:
    all_joints = list(THERAPIST_JOINTS.items()) + list(PATIENT_JOINTS.items())
    cam_t_cols = ["T_cam_t_x", "T_cam_t_y", "T_cam_t_z",
                  "P_cam_t_x", "P_cam_t_y", "P_cam_t_z"]
    cols = (["frame", "time_sec"]
            + cam_t_cols
            + [f"{j}_{c}" for j, _ in all_joints for c in ("x", "y", "z")])
    rows = []
    for frame_idx, frame_key in enumerate(sorted_keys):
        row: dict = {"frame": frame_idx, "time_sec": frame_idx / fps}

        t_ct = therapist_cam_t.get(frame_key, np.full(3, np.nan, dtype=np.float32))
        row["T_cam_t_x"], row["T_cam_t_y"], row["T_cam_t_z"] = float(t_ct[0]), float(t_ct[1]), float(t_ct[2])

        p_ct = patient_cam_t.get(frame_key, np.full(3, np.nan, dtype=np.float32))
        row["P_cam_t_x"], row["P_cam_t_y"], row["P_cam_t_z"] = float(p_ct[0]), float(p_ct[1]), float(p_ct[2])

        t_joints = therapist_kp.get(frame_key)
        for joint_name, joint_idx in THERAPIST_JOINTS.items():
            pos = t_joints[joint_idx] if (t_joints is not None and joint_idx < len(t_joints)) else [float("nan")] * 3
            row[f"{joint_name}_x"] = float(pos[0])
            row[f"{joint_name}_y"] = float(pos[1])
            row[f"{joint_name}_z"] = float(pos[2])

        p_joints = patient_kp.get(frame_key)
        for joint_name, joint_idx in PATIENT_JOINTS.items():
            pos = p_joints[joint_idx] if (p_joints is not None and joint_idx < len(p_joints)) else [float("nan")] * 3
            row[f"{joint_name}_x"] = float(pos[0])
            row[f"{joint_name}_y"] = float(pos[1])
            row[f"{joint_name}_z"] = float(pos[2])

        rows.append(row)

    return pd.DataFrame(rows, columns=cols)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="./output/body_params_per_frame.json")
    parser.add_argument("--output", default="./output/arm_trajectory.csv")
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"[ERROR] Input not found: {args.input}")
        print("  Run: python run_sam_body4d.py --mock   (for testing)")
        sys.exit(1)

    print(f"Loading: {args.input}")
    body_params, n_frames = load_body_params(args.input)
    print(f"  {n_frames} frames loaded.")

    sorted_keys = sorted(body_params.keys(), key=lambda k: int(k.split("_")[1]))

    therapist_kp, patient_kp, therapist_cam_t, patient_cam_t = get_keypoints_per_frame(body_params)
    print(f"  Therapist detected in {len(therapist_kp)} frames.")
    print(f"  Patient detected in {len(patient_kp)} frames.")

    if not therapist_kp:
        print("[ERROR] No therapist joint data found. Check pred_keypoints_3d in JSON.")
        sys.exit(1)

    df = build_trajectory_df(therapist_kp, patient_kp, therapist_cam_t, patient_cam_t, args.fps, sorted_keys)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df.to_csv(args.output, index=False, float_format="%.6f")

    print(f"\nSaved {len(df)} frames → {args.output}")
    print(f"Columns: {list(df.columns)}")
    print(df.describe().to_string())


if __name__ == "__main__":
    main()
