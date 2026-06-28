#!/usr/bin/env python3
"""
Post-process a saved session to fill short NaN gaps in arm pose files.

For gaps of <= MAX_GAP frames that have valid poses on both sides, this script
interpolates translation linearly and rotation via SLERP, then writes
pose_<joint>_filled.txt alongside the originals.

Usage:
    conda activate massage
    python interpolate_poses.py <session_dir> [--max_gap 5]
"""

import argparse
import os
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


MAX_GAP_DEFAULT = 5   # frames; gaps longer than this are left as NaN

ARM_JOINTS = ['right_wrist', 'right_elbow', 'left_wrist', 'left_elbow']


def mat_to_rt(T: np.ndarray):
    """Split 4x4 into (Rotation, translation)."""
    return Rotation.from_matrix(T[:3, :3]), T[:3, 3]


def rt_to_mat(R: Rotation, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R.as_matrix()
    T[:3, 3] = t
    return T


def interpolate_gap(T_before: np.ndarray, T_after: np.ndarray,
                    n_fill: int) -> list:
    """
    Return n_fill interpolated 4x4 matrices between T_before and T_after (exclusive).
    Uses SLERP for rotation, linear for translation.
    """
    R_b, t_b = mat_to_rt(T_before)
    R_a, t_a = mat_to_rt(T_after)
    slerp = Slerp([0.0, 1.0], Rotation.concatenate([R_b, R_a]))
    filled = []
    for k in range(1, n_fill + 1):
        alpha = k / (n_fill + 1)
        R_k = slerp(alpha)
        t_k = (1 - alpha) * t_b + alpha * t_a
        filled.append(rt_to_mat(R_k, t_k))
    return filled


def fill_joint(session_dir: str, joint: str, max_gap: int, dry_run: bool) -> dict:
    frames = sorted(
        [d for d in os.listdir(session_dir) if d.startswith('frame_')],
        key=lambda x: int(x.split('_')[1])
    )
    n = len(frames)

    # Load all poses
    poses = []
    for frame in frames:
        path = os.path.join(session_dir, frame, f'pose_{joint}.txt')
        if os.path.exists(path):
            T = np.loadtxt(path)
            poses.append(None if np.any(np.isnan(T)) else T)
        else:
            poses.append(None)

    nan_before = sum(1 for p in poses if p is None)

    # Find NaN runs and interpolate
    filled = list(poses)  # copy
    i = 0
    gaps_filled = 0
    gaps_skipped = 0
    while i < n:
        if filled[i] is None:
            # Find end of NaN run
            j = i
            while j < n and filled[j] is None:
                j += 1
            gap_len = j - i
            # Look for valid pose before (at i-1) and after (at j)
            if i > 0 and j < n and filled[i - 1] is not None and filled[j] is not None:
                if gap_len <= max_gap:
                    interp = interpolate_gap(filled[i - 1], filled[j], gap_len)
                    for k, T in enumerate(interp):
                        filled[i + k] = T
                    gaps_filled += 1
                else:
                    gaps_skipped += 1
            i = j
        else:
            i += 1

    nan_after = sum(1 for p in filled if p is None)

    if not dry_run:
        for idx, (frame, T) in enumerate(zip(frames, filled)):
            out_path = os.path.join(session_dir, frame, f'pose_{joint}_filled.txt')
            if T is None:
                np.savetxt(out_path, np.full((4, 4), np.nan))
            else:
                np.savetxt(out_path, T)

    return {
        'nan_before': nan_before,
        'nan_after': nan_after,
        'recovered': nan_before - nan_after,
        'gaps_filled': gaps_filled,
        'gaps_skipped': gaps_skipped,
    }


def main():
    parser = argparse.ArgumentParser(description='Interpolate short NaN gaps in arm pose files')
    parser.add_argument('session_dir', help='Path to a saved session directory')
    parser.add_argument('--max_gap', type=int, default=MAX_GAP_DEFAULT,
                        help=f'Maximum gap length to fill (default: {MAX_GAP_DEFAULT})')
    parser.add_argument('--dry_run', action='store_true',
                        help='Report stats without writing files')
    args = parser.parse_args()

    session_dir = args.session_dir.rstrip('/')
    if not os.path.isdir(session_dir):
        print(f"Error: {session_dir} is not a directory")
        return

    frames = [d for d in os.listdir(session_dir) if d.startswith('frame_')]
    n = len(frames)
    print(f"Session: {session_dir}")
    print(f"Frames:  {n}   max_gap: {args.max_gap}{'  [DRY RUN]' if args.dry_run else ''}\n")

    print(f"{'Joint':<20} {'NaN before':>11} {'NaN after':>10} {'Recovered':>10} "
          f"{'Gaps filled':>12} {'Gaps skipped (too long)':>23}")
    print('-' * 90)
    for joint in ARM_JOINTS:
        stats = fill_joint(session_dir, joint, args.max_gap, args.dry_run)
        pct_b = 100 * stats['nan_before'] / n if n else 0
        pct_a = 100 * stats['nan_after'] / n if n else 0
        print(f"{joint:<20} {stats['nan_before']:>6} ({pct_b:4.1f}%) "
              f"{stats['nan_after']:>5} ({pct_a:4.1f}%) "
              f"{stats['recovered']:>10} "
              f"{stats['gaps_filled']:>12} "
              f"{stats['gaps_skipped']:>23}")

    if not args.dry_run:
        print(f"\nWrote pose_<joint>_filled.txt in each frame directory.")
    else:
        print("\n(Dry run — no files written)")


if __name__ == '__main__':
    main()
