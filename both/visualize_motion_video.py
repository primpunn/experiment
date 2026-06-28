#!/usr/bin/env python3
"""
Render an arm-skeleton animation video from a saved session.

Shows right arm (wrist–elbow) and left arm (wrist–elbow) moving in 3D world
frame. Two panels side-by-side: Raw data (with NaN gaps) vs Processed data.
Motion trails show the last TRAIL_LEN frames. Each joint dot is larger and
brighter when the value is interpolated (so you can see where the pipeline
filled gaps).

Output: <session_dir>/motion_video.mp4  (or .avi if MP4 codec unavailable)

Usage:
    conda activate massage
    python visualize_motion_video.py saved_data/2026-05-01_16-30-06
    python visualize_motion_video.py saved_data/2026-05-01_16-30-06 --fps 15 --step 2
"""

import argparse
import os
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401

# ── Constants ─────────────────────────────────────────────────────────────────
ARM_JOINTS = ['right_wrist', 'right_elbow', 'left_wrist', 'left_elbow']

JOINT_COLOR = {
    'right_wrist':  '#e74c3c',
    'right_elbow':  '#e67e22',
    'left_wrist':   '#2980b9',
    'left_elbow':   '#27ae60',
}
ARM_LINKS = [
    ('right_elbow', 'right_wrist'),
    ('left_elbow',  'left_wrist'),
]
ARM_LINK_COLOR = {
    ('right_elbow', 'right_wrist'): '#e74c3c',
    ('left_elbow',  'left_wrist'):  '#2980b9',
}

TRAIL_LEN  = 20     # number of past frames to show as fading trail
FIG_W_IN   = 12     # figure width inches
FIG_H_IN   = 5      # figure height inches
FIG_DPI    = 100    # rendering DPI → frame = 1200 × 500 px


# ── Data loading ──────────────────────────────────────────────────────────────

def load_poses_and_flag(session_dir: str, joint: str, suffix: str):
    """
    Returns:
        xyz      — (N, 3) float, NaN where missing
        is_interp — (N,) bool, True where the value came from interpolation
    """
    raw_suffix = ''
    frames = sorted(
        [d for d in os.listdir(session_dir) if d.startswith('frame_')],
        key=lambda x: int(x.split('_')[1])
    )
    n = len(frames)
    xyz       = np.full((n, 3), np.nan)
    is_interp = np.zeros(n, dtype=bool)

    for i, frame in enumerate(frames):
        # Check raw first to know if this was originally NaN
        raw_path = os.path.join(session_dir, frame, f'pose_{joint}.txt')
        tgt_path = os.path.join(session_dir, frame, f'pose_{joint}{suffix}.txt')

        raw_valid = False
        if os.path.exists(raw_path):
            T_raw = np.loadtxt(raw_path)
            raw_valid = not np.any(np.isnan(T_raw))

        if os.path.exists(tgt_path):
            T = np.loadtxt(tgt_path)
            if not np.any(np.isnan(T)):
                xyz[i] = T[:3, 3]
                is_interp[i] = not raw_valid   # interpolated if raw was missing
        # If both missing, xyz[i] stays NaN

    return xyz, is_interp


# ── Axis limits helper ────────────────────────────────────────────────────────

def compute_limits(data_dict: dict, margin: float = 0.1):
    """Compute equal-aspect 3D limits from all joint data."""
    all_pts = np.vstack([v for v in data_dict.values()
                         if not np.all(np.isnan(v))])
    valid = all_pts[~np.any(np.isnan(all_pts), axis=1)]
    if len(valid) == 0:
        return (-1, 1), (-1, 1), (-1, 1)
    lo, hi = valid.min(axis=0), valid.max(axis=0)
    span = max((hi - lo).max(), 0.1)
    mid  = (lo + hi) / 2
    half = span / 2 * (1 + margin)
    return (mid[0]-half, mid[0]+half), \
           (mid[1]-half, mid[1]+half), \
           (mid[2]-half, mid[2]+half)


# ── Single frame renderer ─────────────────────────────────────────────────────

def render_frame(fig, axes, frame_idx: int,
                 xyz_raw: dict, xyz_proc: dict, is_interp: dict,
                 lims_raw, lims_proc):
    for ax in axes:
        ax.cla()

    titles = [f'RAW  (frame {frame_idx})', f'PROCESSED  (frame {frame_idx})']
    data_sets = [(xyz_raw,  lims_raw,  False),
                 (xyz_proc, lims_proc, True)]

    for ax, (xyz_d, lims, show_interp), title in zip(axes, data_sets, titles):
        ax.set_title(title, fontsize=9, pad=3)
        xl, yl, zl = lims
        ax.set_xlim(xl); ax.set_ylim(yl); ax.set_zlim(zl)
        ax.set_xlabel('X', fontsize=7, labelpad=1)
        ax.set_ylabel('Y', fontsize=7, labelpad=1)
        ax.set_zlabel('Z', fontsize=7, labelpad=1)
        ax.tick_params(labelsize=5, pad=0)

        # Light ground plane
        gx = np.array([xl[0], xl[1], xl[1], xl[0]])
        gy = np.array([yl[0], yl[0], yl[1], yl[1]])
        gz = np.full(4, zl[0])
        ax.plot_surface(
            np.array([[xl[0], xl[1]], [xl[0], xl[1]]]),
            np.array([[yl[0], yl[0]], [yl[1], yl[1]]]),
            np.full((2, 2), zl[0]),
            alpha=0.08, color='grey', zorder=0
        )

        # Trails
        trail_start = max(0, frame_idx - TRAIL_LEN)
        for joint in ARM_JOINTS:
            traj = xyz_d[joint][trail_start: frame_idx + 1]
            valid = ~np.any(np.isnan(traj), axis=1)
            if valid.sum() < 2:
                continue
            t = traj[valid]
            alphas = np.linspace(0.1, 0.55, len(t))
            color  = JOINT_COLOR[joint]
            for k in range(len(t) - 1):
                ax.plot(t[k:k+2, 0], t[k:k+2, 1], t[k:k+2, 2],
                        color=color, lw=1.0, alpha=alphas[k], zorder=1)

        # Skeleton links at current frame
        for (j_a, j_b), link_color in ARM_LINK_COLOR.items():
            pa = xyz_d[j_a][frame_idx]
            pb = xyz_d[j_b][frame_idx]
            if not (np.any(np.isnan(pa)) or np.any(np.isnan(pb))):
                ax.plot([pa[0], pb[0]], [pa[1], pb[1]], [pa[2], pb[2]],
                        color=link_color, lw=2.5, alpha=0.9, zorder=3)

        # Joint dots
        for joint in ARM_JOINTS:
            pt = xyz_d[joint][frame_idx]
            if np.any(np.isnan(pt)):
                continue
            interp = show_interp and is_interp[joint][frame_idx]
            color  = JOINT_COLOR[joint]
            size   = 80 if interp else 50
            marker = '*' if interp else 'o'
            ax.scatter(*pt, s=size, color=color, marker=marker,
                       edgecolors='white', linewidths=0.5,
                       zorder=4, depthshade=False)

    # Legend (drawn once, bottom of right axis)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    legend_elems = [
        Line2D([0],[0], color=JOINT_COLOR['right_wrist'],  lw=2, label='Right wrist'),
        Line2D([0],[0], color=JOINT_COLOR['right_elbow'],  lw=2, label='Right elbow'),
        Line2D([0],[0], color=JOINT_COLOR['left_wrist'],   lw=2, label='Left wrist'),
        Line2D([0],[0], color=JOINT_COLOR['left_elbow'],   lw=2, label='Left elbow'),
        Line2D([0],[0], color='w', marker='*', ms=8, markerfacecolor='grey',
               label='★ = interpolated'),
    ]
    axes[-1].legend(handles=legend_elems, loc='upper right',
                    fontsize=6, framealpha=0.6)

    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    frame_img = np.asarray(buf)[..., :3]   # RGBA → RGB
    return frame_img


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Render arm skeleton motion video')
    parser.add_argument('session_dir')
    parser.add_argument('--fps',  type=int, default=15,
                        help='Output video FPS (default 15)')
    parser.add_argument('--step', type=int, default=2,
                        help='Render every Nth frame (default 2 → 15fps from 30fps)')
    args = parser.parse_args()

    session_dir = args.session_dir.rstrip('/')
    if not os.path.isdir(session_dir):
        print(f"Error: {session_dir} is not a directory"); return

    session_name = os.path.basename(session_dir)
    print(f"Session: {session_name}")

    # Load data
    print("Loading poses...")
    xyz_raw, _      = {}, {}
    xyz_proc        = {}
    is_interp_proc  = {}
    for j in ARM_JOINTS:
        xyz_raw[j],  _                = load_poses_and_flag(session_dir, j, '')
        xyz_proc[j], is_interp_proc[j] = load_poses_and_flag(session_dir, j, '_processed')

    n_frames = len(xyz_raw[ARM_JOINTS[0]])
    render_indices = list(range(0, n_frames, args.step))
    print(f"Frames: {n_frames} total, rendering {len(render_indices)} "
          f"(every {args.step}) at {args.fps} fps")

    # Axis limits (fixed throughout video)
    lims_raw  = compute_limits(xyz_raw)
    lims_proc = compute_limits(xyz_proc)

    # Video writer
    out_mp4 = os.path.join(session_dir, 'motion_video.mp4')
    out_avi = os.path.join(session_dir, 'motion_video.avi')

    frame_w = int(FIG_W_IN * FIG_DPI)
    frame_h = int(FIG_H_IN * FIG_DPI)

    writer = cv2.VideoWriter(
        out_mp4,
        cv2.VideoWriter_fourcc(*'mp4v'),
        args.fps,
        (frame_w, frame_h)
    )
    if not writer.isOpened():
        print("mp4v codec failed, falling back to MJPG .avi")
        writer = cv2.VideoWriter(
            out_avi,
            cv2.VideoWriter_fourcc(*'MJPG'),
            args.fps,
            (frame_w, frame_h)
        )
        out_path = out_avi
    else:
        out_path = out_mp4

    # Set up figure
    fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN), dpi=FIG_DPI)
    fig.patch.set_facecolor('#1a1a2e')
    ax1 = fig.add_subplot(121, projection='3d', facecolor='#16213e')
    ax2 = fig.add_subplot(122, projection='3d', facecolor='#16213e')
    for ax in [ax1, ax2]:
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('#333355')
        ax.yaxis.pane.set_edgecolor('#333355')
        ax.zaxis.pane.set_edgecolor('#333355')
        ax.tick_params(colors='#aaaaaa')
        ax.xaxis.label.set_color('#aaaaaa')
        ax.yaxis.label.set_color('#aaaaaa')
        ax.zaxis.label.set_color('#aaaaaa')
        ax.title.set_color('white')

    fig.suptitle(f'Arm Motion — {session_name}', color='white',
                 fontsize=11, fontweight='bold')
    plt.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.02,
                        wspace=0.05)

    print("Rendering frames...")
    for prog, fi in enumerate(render_indices):
        if prog % 30 == 0:
            print(f"  {prog}/{len(render_indices)} frames...", end='\r', flush=True)

        rgb = render_frame(fig, [ax1, ax2], fi,
                           xyz_raw, xyz_proc, is_interp_proc,
                           lims_raw, lims_proc)
        # matplotlib RGB → OpenCV BGR
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        bgr = cv2.resize(bgr, (frame_w, frame_h))
        writer.write(bgr)

    writer.release()
    plt.close(fig)
    print(f"\nVideo saved: {out_path}")
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"File size: {size_mb:.1f} MB")


if __name__ == '__main__':
    main()
