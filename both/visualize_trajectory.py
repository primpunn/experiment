"""
Steps 6 & 7: Visualize arm trajectories + apply Savitzky-Golay smoothing.

Reads arm_trajectory.csv (raw) and optionally arm_trajectory_smoothed.csv,
then produces:
  ./output/wrist_trajectory_3d.png   — 3D wrist paths coloured by time
  ./output/wrist_trajectory_xyz.png  — X/Y/Z components vs time (raw + smoothed)
  ./output/arm_trajectory_smoothed.csv — SG-filtered version of the raw CSV

Usage:
  python visualize_trajectory.py \
      --input ./output/arm_trajectory.csv \
      --output_dir ./output \
      [--window 11] [--polyorder 3] [--no_smooth]
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless — no display needed
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d projection)


# Joint groups for plotting
# T_ = therapist arm joints, P_ = patient contact joints
FOCUS_JOINTS = ["T_L_elbow", "T_R_elbow", "T_L_wrist", "T_R_wrist", "P_L_ankle", "P_R_ankle"]
CONTACT_JOINTS = ["T_L_wrist", "T_R_wrist", "P_L_ankle", "P_R_ankle"]
COLORS = {
    "T_L_wrist":        "#1f77b4",  # blue  — therapist left hand
    "T_R_wrist":        "#d62728",  # red   — therapist right hand
    "P_L_ankle":        "#2ca02c",  # green — patient left ankle
    "P_R_ankle":        "#ff7f0e",  # orange — patient right ankle
    "T_L_wrist_smooth": "#aec7e8",
    "T_R_wrist_smooth": "#ffb1b1",
    "P_L_ankle_smooth": "#98df8a",
    "P_R_ankle_smooth": "#ffbb78",
}


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------
def apply_savgol(df: pd.DataFrame, window: int, polyorder: int) -> pd.DataFrame:
    from scipy.signal import savgol_filter

    # window must be odd and > polyorder
    if window % 2 == 0:
        window += 1
    window = max(window, polyorder + 1)
    # Cap window at number of valid samples
    n = len(df)
    if window > n:
        window = n if n % 2 == 1 else n - 1
        print(f"[WARN] Window capped to {window} (only {n} frames available).")
    if window < polyorder + 1:
        print(f"[WARN] Not enough frames to smooth — returning raw data.")
        return df.copy()

    df_smooth = df.copy()
    coord_cols = [c for c in df.columns if c not in ("frame", "time_sec")]
    for col in coord_cols:
        series = df[col].values.astype(float)
        nan_mask = np.isnan(series)
        if nan_mask.all():
            continue
        # Interpolate NaNs before filtering
        x = np.arange(len(series))
        series[nan_mask] = np.interp(x[nan_mask], x[~nan_mask], series[~nan_mask])
        df_smooth[col] = savgol_filter(series, window, polyorder)
        df_smooth.loc[nan_mask, col] = np.nan  # restore NaNs at edges
    return df_smooth


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_3d_contact_points(df_raw: pd.DataFrame, df_smooth: pd.DataFrame | None,
                            output_path: str) -> None:
    """3D trajectory for each contact-point joint (therapist wrists + patient ankles)."""
    fig = plt.figure(figsize=(14, 10))
    titles = {
        "T_L_wrist": "Therapist Left Wrist",
        "T_R_wrist": "Therapist Right Wrist",
        "P_L_ankle": "Patient Left Ankle",
        "P_R_ankle": "Patient Right Ankle",
    }

    for i, joint in enumerate(CONTACT_JOINTS):
        ax = fig.add_subplot(2, 2, i + 1, projection="3d")
        col = f"{joint}_"
        x = df_raw[f"{col}x"].values if f"{col}x" in df_raw.columns else np.full(len(df_raw), np.nan)
        y = df_raw[f"{col}y"].values if f"{col}y" in df_raw.columns else np.full(len(df_raw), np.nan)
        z = df_raw[f"{col}z"].values if f"{col}z" in df_raw.columns else np.full(len(df_raw), np.nan)
        t = df_raw["time_sec"].values
        valid = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))

        if valid.any():
            sc = ax.scatter(x[valid], y[valid], z[valid],
                            c=t[valid], cmap="viridis", s=12, label="raw", alpha=0.7)
            ax.plot(x[valid], y[valid], z[valid],
                    color=COLORS[joint], alpha=0.3, linewidth=0.8)
            plt.colorbar(sc, ax=ax, pad=0.1, shrink=0.6, label="time (s)")

        if df_smooth is not None and valid.any():
            xs = df_smooth[f"{col}x"].values
            ys = df_smooth[f"{col}y"].values
            zs = df_smooth[f"{col}z"].values
            ax.plot(xs[valid], ys[valid], zs[valid],
                    color=COLORS[f"{joint}_smooth"], linewidth=1.5,
                    label="smoothed", zorder=5)

        ax.set_title(titles.get(joint, joint))
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.legend(fontsize=7)

    plt.suptitle("Contact-Point 3D Trajectories (colour = time)", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_xyz_vs_time(df_raw: pd.DataFrame, df_smooth: pd.DataFrame | None,
                     output_path: str) -> None:
    """X/Y/Z vs time for each contact-point joint."""
    axes_labels = ["X (m)", "Y (m)", "Z (m)"]
    coords = ["x", "y", "z"]
    fig, axs = plt.subplots(3, len(CONTACT_JOINTS), figsize=(5 * len(CONTACT_JOINTS), 10), sharex=True)

    col_titles = {
        "T_L_wrist": "Therapist L Wrist",
        "T_R_wrist": "Therapist R Wrist",
        "P_L_ankle": "Patient L Ankle",
        "P_R_ankle": "Patient R Ankle",
    }

    for col_i, joint in enumerate(CONTACT_JOINTS):
        for row_i, (coord, ylabel) in enumerate(zip(coords, axes_labels)):
            ax = axs[row_i, col_i]
            t = df_raw["time_sec"].values
            col_name = f"{joint}_{coord}"
            raw = df_raw[col_name].values if col_name in df_raw.columns else np.full(len(t), np.nan)

            ax.plot(t, raw, color=COLORS[joint], linewidth=1, alpha=0.6, label="raw")
            if df_smooth is not None and col_name in df_smooth.columns:
                sm = df_smooth[col_name].values
                ax.plot(t, sm, color=COLORS[f"{joint}_smooth"], linewidth=2,
                        alpha=0.9, label="smoothed")

            ax.set_ylabel(ylabel)
            ax.set_title(f"{col_titles.get(joint, joint)} — {coord.upper()}", fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7)

    for ax in axs[-1]:
        ax.set_xlabel("Time (s)")

    plt.suptitle("Contact-Point Trajectories vs Time", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_all_joints_xyz(df_raw: pd.DataFrame, df_smooth: pd.DataFrame | None,
                        output_path: str) -> None:
    """All tracked joints (therapist elbows+wrists, patient ankles) X/Y/Z vs time."""
    joint_pairs = [
        ("T_L_elbow",  "T_R_elbow",  "Elbow"),
        ("T_L_wrist",  "T_R_wrist",  "Therapist Wrist"),
        ("P_L_ankle",  "P_R_ankle",  "Patient Ankle"),
    ]
    coords = ["x", "y", "z"]

    fig, axs = plt.subplots(3, 3, figsize=(16, 10), sharex=True)
    t = df_raw["time_sec"].values

    for row_i, (left, right, row_label) in enumerate(joint_pairs):
        for col_i, coord in enumerate(coords):
            ax = axs[row_i, col_i]
            for joint, color in [(left, COLORS.get(left, "#1f77b4")),
                                  (right, COLORS.get(right, "#d62728"))]:
                col_name = f"{joint}_{coord}"
                if col_name not in df_raw.columns:
                    continue
                ax.plot(t, df_raw[col_name].values, color=color, alpha=0.5,
                        linewidth=1, label=joint)
                if df_smooth is not None and col_name in df_smooth.columns:
                    ax.plot(t, df_smooth[col_name].values, color=color, alpha=0.9,
                            linewidth=1.8, linestyle="--")
            ax.set_title(f"{row_label} — {coord.upper()}", fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=6)
            if col_i == 0:
                ax.set_ylabel("Position (m)")

    for ax in axs[-1]:
        ax.set_xlabel("Time (s)")

    plt.suptitle("All Tracked Joints — X/Y/Z vs Time  (dashed = smoothed)", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="./output/arm_trajectory.csv")
    parser.add_argument("--output_dir", default="./output")
    parser.add_argument("--window", type=int, default=11,
                        help="Savitzky-Golay window size (must be odd, > polyorder)")
    parser.add_argument("--polyorder", type=int, default=3,
                        help="Savitzky-Golay polynomial order")
    parser.add_argument("--no_smooth", action="store_true",
                        help="Skip smoothing; only plot raw data")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"[ERROR] Input CSV not found: {args.input}")
        print("  Run extract_arm_trajectory.py first.")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading: {args.input}")
    df_raw = pd.read_csv(args.input)
    print(f"  {len(df_raw)} frames, {len(df_raw.columns)} columns.")

    # Smoothing
    df_smooth = None
    if not args.no_smooth:
        try:
            from scipy.signal import savgol_filter  # noqa: F401
            df_smooth = apply_savgol(df_raw, args.window, args.polyorder)
            smooth_path = os.path.join(args.output_dir, "arm_trajectory_smoothed.csv")
            df_smooth.to_csv(smooth_path, index=False, float_format="%.6f")
            print(f"Smoothed CSV saved: {smooth_path}")
        except ImportError:
            print("[WARN] scipy not installed — skipping smoothing. pip install scipy")

    # Plots
    plot_3d_contact_points(
        df_raw, df_smooth,
        os.path.join(args.output_dir, "contact_points_3d.png")
    )
    plot_xyz_vs_time(
        df_raw, df_smooth,
        os.path.join(args.output_dir, "contact_points_xyz.png")
    )
    plot_all_joints_xyz(
        df_raw, df_smooth,
        os.path.join(args.output_dir, "all_joints_xyz.png")
    )

    print("\nDone. Outputs in:", args.output_dir)


if __name__ == "__main__":
    main()
