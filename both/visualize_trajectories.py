#!/usr/bin/env python3
"""
Visualize arm joint trajectories from a saved session.

Produces three figures:
  Fig 1 — Time-series comparison: Raw vs _filled vs _processed
           4 joints × 3 axes (x,y,z), showing where gaps and artifacts were fixed.
  Fig 2 — 3D spatial trajectories: Raw vs _processed side-by-side for each joint,
           colour-coded by frame number so you can see the motion flow over time.
  Fig 3 — Frame coverage: which version has valid data per frame.

Usage:
    conda activate massage
    python visualize_trajectories.py saved_data/2026-05-11 --save
    python visualize_trajectories.py saved_data/2026-05-11 --save --label depth
    python visualize_trajectories.py saved_data/2026-05-11 --save --label modify

With --label, output files are named  ts_<label>.png / 3d_<label>.png / cov_<label>.png
and saved inside the session directory (so both labels coexist without overwriting).
Without --label, files are saved as trajectory_timeseries/3d/coverage.png in the
parent directory (original behaviour).
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401 (registers 3D projection)

ARM_JOINTS   = ['right_wrist', 'right_elbow', 'left_wrist', 'left_elbow']
JOINT_COLORS = {
    'right_wrist':  '#e74c3c',   # red
    'right_elbow':  '#e67e22',   # orange
    'left_wrist':   '#2980b9',   # blue
    'left_elbow':   '#27ae60',   # green
}
AXES_LABELS = ['X (m)', 'Y (m)', 'Z (m)']


# ── Data loading ──────────────────────────────────────────────────────────────

def load_translations(session_dir: str, joint: str, suffix: str = '') -> np.ndarray:
    """
    Load translation time series for one joint.
    Returns (N, 3) float array with NaN where pose was missing.
    suffix: '' = raw, '_filled', '_processed'
    """
    frames = sorted(
        [d for d in os.listdir(session_dir) if d.startswith('frame_')],
        key=lambda x: int(x.split('_')[1])
    )
    xyz = np.full((len(frames), 3), np.nan)
    for i, frame in enumerate(frames):
        path = os.path.join(session_dir, frame, f'pose_{joint}{suffix}.txt')
        if os.path.exists(path):
            T = np.loadtxt(path)
            if not np.any(np.isnan(T)):
                xyz[i] = T[:3, 3]
    return xyz


def load_all(session_dir: str):
    """Return dicts: raw, filled, processed — each {joint: (N,3) array}."""
    raw, filled, processed = {}, {}, {}
    for j in ARM_JOINTS:
        raw[j]       = load_translations(session_dir, j, '')
        filled[j]    = load_translations(session_dir, j, '_filled')
        processed[j] = load_translations(session_dir, j, '_processed')
    return raw, filled, processed


# ── Figure 1: Time-series comparison ─────────────────────────────────────────

def plot_timeseries(raw: dict, filled: dict, processed: dict,
                    session_name: str, save: bool, label: str = ''):
    n_frames = len(next(iter(raw.values())))
    frames   = np.arange(n_frames)

    fig, axes = plt.subplots(4, 3, figsize=(18, 14), sharex=True)
    title_suffix = f'  [{label}]' if label else ''
    fig.suptitle(f'Arm Joint Trajectories — Time Series\n{session_name}{title_suffix}',
                 fontsize=13, fontweight='bold')

    for row, joint in enumerate(ARM_JOINTS):
        color = JOINT_COLORS[joint]
        for col, ax_label in enumerate(AXES_LABELS):
            ax  = axes[row, col]
            r   = raw[joint][:, col]
            f   = filled[joint][:, col]
            p   = processed[joint][:, col]

            # Raw gaps as light grey shading
            nan_mask = np.isnan(r)
            if nan_mask.any():
                ax.fill_between(frames, ax.get_ylim()[0] if ax.get_ylim()[0] != 0 else -2,
                                2, where=nan_mask,
                                color='#cccccc', alpha=0.35, label='_')

            ax.plot(frames, r, color='#aaaaaa', lw=1.2, alpha=0.7,
                    label='Raw', zorder=2)
            ax.plot(frames, f, color='#3498db', lw=1.0, alpha=0.8,
                    label='_filled', zorder=3, linestyle='--')
            ax.plot(frames, p, color=color, lw=1.5, alpha=0.9,
                    label='Processed', zorder=4)

            ax.set_ylabel(ax_label, fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=8)

            if row == 0:
                ax.set_title(ax_label, fontsize=10, fontweight='bold')
            if col == 0:
                ax.set_ylabel(f'{joint}\n{ax_label}', fontsize=8)
            if row == 3:
                ax.set_xlabel('Frame', fontsize=9)

    # Shared legend
    handles = [
        plt.Line2D([0], [0], color='#aaaaaa', lw=1.5, label='Raw (with NaN gaps)'),
        plt.Line2D([0], [0], color='#3498db', lw=1.5, linestyle='--',
                   label='_filled (interpolate_poses.py)'),
        plt.Line2D([0], [0], color='#e74c3c', lw=1.5, label='_processed (interpolate_modify.py)'),
        plt.Rectangle((0, 0), 1, 1, fc='#cccccc', alpha=0.5, label='Original NaN gap'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, 0.01))
    fig.tight_layout(rect=[0, 0.04, 1, 1])

    if save:
        if label:
            out = os.path.join(session_dir, f'ts_{label}.png')
        else:
            out = os.path.join(os.path.dirname(session_dir) or '.', 'trajectory_timeseries.png')
        fig.savefig(out, dpi=150, bbox_inches='tight')
        print(f"Saved: {out}")


# ── Figure 2: 3D spatial trajectories ────────────────────────────────────────

def plot_3d(raw: dict, processed: dict, session_name: str, save: bool, label: str = ''):
    fig = plt.figure(figsize=(18, 10))
    title_suffix = f'  [{label}]' if label else ''
    fig.suptitle(f'Arm Joint 3D Trajectories — Raw vs Processed\n{session_name}{title_suffix}',
                 fontsize=13, fontweight='bold')

    for idx, joint in enumerate(ARM_JOINTS):
        color = JOINT_COLORS[joint]
        n_frames = len(raw[joint])

        # Raw (left column)
        ax_raw  = fig.add_subplot(2, 4, idx + 1, projection='3d')
        r = raw[joint]
        valid_r = ~np.any(np.isnan(r), axis=1)
        if valid_r.any():
            sc = ax_raw.scatter(r[valid_r, 0], r[valid_r, 1], r[valid_r, 2],
                                c=np.where(valid_r)[0], cmap='viridis',
                                s=4, alpha=0.7)
        ax_raw.set_title(f'{joint}\n(Raw — {valid_r.sum()}/{n_frames} frames)',
                         fontsize=8)
        _style_3d(ax_raw)

        # Processed (right column)
        ax_pro  = fig.add_subplot(2, 4, idx + 5, projection='3d')
        p = processed[joint]
        valid_p = ~np.any(np.isnan(p), axis=1)
        if valid_p.any():
            cvals = np.arange(n_frames)[valid_p]
            ax_pro.scatter(p[valid_p, 0], p[valid_p, 1], p[valid_p, 2],
                           c=cvals, cmap='viridis', s=4, alpha=0.7)
            # Connect with a thin line to show motion flow
            ax_pro.plot(p[valid_p, 0], p[valid_p, 1], p[valid_p, 2],
                        color=color, lw=0.6, alpha=0.4)
        ax_pro.set_title(f'{joint}\n(Processed — {valid_p.sum()}/{n_frames} frames)',
                         fontsize=8, color=color)
        _style_3d(ax_pro)

    # Colour bar for time
    sm = plt.cm.ScalarMappable(cmap='viridis',
                                norm=plt.Normalize(0, n_frames - 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=fig.axes, shrink=0.4, pad=0.02)
    cbar.set_label('Frame number (early → late)', fontsize=9)

    fig.tight_layout()

    if save:
        if label:
            out = os.path.join(session_dir, f'3d_{label}.png')
        else:
            out = os.path.join(os.path.dirname(session_dir) or '.', 'trajectory_3d.png')
        fig.savefig(out, dpi=150, bbox_inches='tight')
        print(f"Saved: {out}")


def _style_3d(ax):
    ax.set_xlabel('X', fontsize=7)
    ax.set_ylabel('Y', fontsize=7)
    ax.set_zlabel('Z', fontsize=7)
    ax.tick_params(labelsize=6)


# ── Figure 3: NaN / artifact coverage per joint ───────────────────────────────

def plot_coverage(raw: dict, filled: dict, processed: dict,
                  session_name: str, save: bool, label: str = ''):
    n_frames = len(next(iter(raw.values())))
    frames   = np.arange(n_frames)

    fig, axes = plt.subplots(4, 1, figsize=(16, 7), sharex=True)
    title_suffix = f'  [{label}]' if label else ''
    fig.suptitle(f'Frame Coverage: where each version has valid data\n{session_name}{title_suffix}',
                 fontsize=12, fontweight='bold')

    for row, joint in enumerate(ARM_JOINTS):
        ax    = axes[row]
        color = JOINT_COLORS[joint]

        valid_raw  = ~np.any(np.isnan(raw[joint]), axis=1)
        valid_fill = ~np.any(np.isnan(filled[joint]), axis=1)
        valid_proc = ~np.any(np.isnan(processed[joint]), axis=1)

        # Stack as horizontal bands
        ax.fill_between(frames, 2.0, 3.0, where=valid_proc,
                        color=color, alpha=0.85, label='Processed')
        ax.fill_between(frames, 1.0, 2.0, where=valid_fill,
                        color='#3498db', alpha=0.7, label='_filled')
        ax.fill_between(frames, 0.0, 1.0, where=valid_raw,
                        color='#555555', alpha=0.6, label='Raw')

        # NaN gaps in white
        ax.fill_between(frames, 0.0, 3.0, where=~valid_proc,
                        color='white', alpha=1.0)
        ax.fill_between(frames, 0.0, 3.0, where=~valid_proc,
                        color='#ffdddd', alpha=0.6, label='NaN (even after processing)')

        pct_raw  = 100 * valid_raw.sum()  / n_frames
        pct_fill = 100 * valid_fill.sum() / n_frames
        pct_proc = 100 * valid_proc.sum() / n_frames
        ax.set_ylabel(f'{joint}\n({pct_raw:.0f}% → {pct_fill:.0f}% → {pct_proc:.0f}%)',
                      fontsize=8)
        ax.set_yticks([0.5, 1.5, 2.5])
        ax.set_yticklabels(['Raw', 'Filled', 'Proc'], fontsize=7)
        ax.set_ylim(0, 3)
        ax.grid(axis='x', alpha=0.3)

        if row == 3:
            ax.set_xlabel('Frame', fontsize=9)

    handles = [
        plt.Rectangle((0, 0), 1, 1, fc='#555555', alpha=0.7, label='Raw valid'),
        plt.Rectangle((0, 0), 1, 1, fc='#3498db', alpha=0.8, label='_filled valid'),
        plt.Rectangle((0, 0), 1, 1, fc='#e74c3c', alpha=0.9, label='_processed valid'),
        plt.Rectangle((0, 0), 1, 1, fc='#ffdddd', alpha=0.8, label='Still NaN'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=[0, 0.04, 1, 1])

    if save:
        if label:
            out = os.path.join(session_dir, f'cov_{label}.png')
        else:
            out = os.path.join(os.path.dirname(session_dir) or '.', 'trajectory_coverage.png')
        fig.savefig(out, dpi=150, bbox_inches='tight')
        print(f"Saved: {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Visualize arm joint trajectories from a session')
    parser.add_argument('session_dir', help='Path to a saved session directory')
    parser.add_argument('--save', action='store_true',
                        help='Save figures as PNG instead of displaying interactively')
    parser.add_argument('--label', default='',
                        help='Output filename label, e.g. "depth" or "modify". '
                             'Saves as ts_<label>.png / 3d_<label>.png / cov_<label>.png '
                             'inside the session directory.')
    args = parser.parse_args()

    global session_dir
    session_dir = args.session_dir.rstrip('/')

    if not os.path.isdir(session_dir):
        print(f"Error: {session_dir} is not a directory")
        return

    session_name = os.path.basename(session_dir)
    print(f"Loading trajectories from: {session_dir}")

    raw, filled, processed = load_all(session_dir)

    # Check which output files exist
    has_filled    = any(os.path.exists(
        os.path.join(session_dir, 'frame_0', f'pose_{j}_filled.txt'))
        for j in ARM_JOINTS)
    has_processed = any(os.path.exists(
        os.path.join(session_dir, 'frame_0', f'pose_{j}_processed.txt'))
        for j in ARM_JOINTS)

    print(f"  _filled.txt present:    {has_filled}")
    print(f"  _processed.txt present: {has_processed}")
    print(f"  Generating figures...\n")

    plot_timeseries(raw, filled, processed, session_name, args.save, args.label)
    plot_3d(raw, processed, session_name, args.save, args.label)
    plot_coverage(raw, filled, processed, session_name, args.save, args.label)

    if not args.save:
        plt.show()
    else:
        print("All figures saved.")


if __name__ == '__main__':
    main()
