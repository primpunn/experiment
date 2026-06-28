#!/usr/bin/env python3
"""
Level 2 verification — physical plausibility checks on interpolated arm poses.

Checks:
  1. Arm segment length consistency (wrist-to-elbow distance should be ~20-45 cm)
  2. Joint velocity — raw vs processed (flag frames still above physical limit)
  3. Coverage summary (raw / _filled / _processed valid-frame counts)

Usage:
    conda activate massage
    python verify_l2.py <session_dir>
    python verify_l2.py <session_dir> --suffix _processed   # default
    python verify_l2.py <session_dir> --suffix _filled
"""

import argparse
import os
import numpy as np

ARM_JOINTS = ['right_wrist', 'right_elbow', 'left_wrist', 'left_elbow']
SEGMENT_PAIRS = [
    ('right_wrist', 'right_elbow', 'Right forearm'),
    ('left_wrist',  'left_elbow',  'Left forearm'),
]
SEGMENT_MIN_CM = 20.0   # expected forearm min (cm)
SEGMENT_MAX_CM = 45.0   # expected forearm max (cm)
VEL_WARN_CM    = 3.0    # max plausible displacement per frame at 30 fps (~0.9 m/s)


def load_translations(session_dir: str, joint: str, suffix: str = '') -> np.ndarray:
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


def check_segment_lengths(session_dir: str, suffix: str) -> bool:
    print("\n── 1. Arm Segment Length Consistency ────────────────────────────────────")
    print(f"   Expected forearm: {SEGMENT_MIN_CM:.0f}–{SEGMENT_MAX_CM:.0f} cm   suffix: '{suffix}'\n")

    all_ok = True
    for ja, jb, label in SEGMENT_PAIRS:
        pa = load_translations(session_dir, ja, suffix)
        pb = load_translations(session_dir, jb, suffix)
        both = ~np.any(np.isnan(pa), axis=1) & ~np.any(np.isnan(pb), axis=1)
        if both.sum() == 0:
            print(f"   {label}: NO frames where both joints are valid")
            continue
        lengths = np.linalg.norm(pa[both] - pb[both], axis=1) * 100   # cm
        bad = (lengths < SEGMENT_MIN_CM) | (lengths > SEGMENT_MAX_CM)
        status = "PASS" if not bad.any() else f"WARN — {bad.sum()} outlier frames"
        print(f"   {label}:")
        print(f"     Valid frame pairs : {both.sum()}")
        print(f"     Length  mean={lengths.mean():.1f} cm  std={lengths.std():.2f} cm  "
              f"min={lengths.min():.1f}  max={lengths.max():.1f} cm")
        print(f"     Status : {status}")
        if bad.any():
            all_ok = False
            sample = sorted(lengths[bad].tolist())[:8]
            print(f"     Outlier lengths (cm): {[f'{v:.1f}' for v in sample]}")
    return all_ok


def check_velocity(session_dir: str, proc_suffix: str) -> bool:
    print("\n── 2. Joint Velocity — Raw vs Processed ─────────────────────────────────")
    print(f"   Spike threshold: >{VEL_WARN_CM:.0f} cm/frame  (consecutive frames only)\n")

    def velocities(suffix):
        per_joint = {}
        for j in ARM_JOINTS:
            xyz = load_translations(session_dir, j, suffix)
            valid = ~np.any(np.isnan(xyz), axis=1)
            idxs = np.where(valid)[0]
            dists = [
                np.linalg.norm(xyz[idxs[k + 1]] - xyz[idxs[k]]) * 100
                for k in range(len(idxs) - 1)
                if idxs[k + 1] - idxs[k] == 1
            ]
            per_joint[j] = np.array(dists) if dists else np.array([0.0])
        return per_joint

    raw_v  = velocities('')
    proc_v = velocities(proc_suffix)

    print(f"   {'Joint':<18} {'Raw mean':>9} {'Raw max':>9} {'Proc mean':>10} "
          f"{'Proc max':>9} {'Spikes raw':>11} {'Spikes proc':>12}")
    print("   " + "─" * 85)

    all_ok = True
    for j in ARM_JOINTS:
        r, p = raw_v[j], proc_v[j]
        r_spk = int((r > VEL_WARN_CM).sum())
        p_spk = int((p > VEL_WARN_CM).sum())
        if p_spk > 0:
            all_ok = False
        print(f"   {j:<18} {r.mean():>8.2f}cm {r.max():>8.2f}cm "
              f"{p.mean():>9.2f}cm {p.max():>8.2f}cm "
              f"{r_spk:>11}  {p_spk:>11}")

    verdict = "PASS" if all_ok else f"WARN — processed frames still exceed {VEL_WARN_CM:.0f} cm/frame"
    print(f"\n   Velocity check: {verdict}")
    return all_ok


def coverage_summary(session_dir: str):
    print("\n── 3. Coverage Summary ───────────────────────────────────────────────────")
    frames = sorted(
        [d for d in os.listdir(session_dir) if d.startswith('frame_')],
        key=lambda x: int(x.split('_')[1])
    )
    n = len(frames)
    suffixes = [('', 'Raw'), ('_filled', 'Filled'), ('_processed', 'Processed')]

    print(f"   {'Joint':<18}", end='')
    for _, label in suffixes:
        print(f"  {label:>14}", end='')
    print()
    print("   " + "─" * 62)

    for j in ARM_JOINTS:
        print(f"   {j:<18}", end='')
        for suffix, _ in suffixes:
            xyz = load_translations(session_dir, j, suffix)
            valid = (~np.any(np.isnan(xyz), axis=1)).sum()
            print(f"  {valid:4d}/{n} ({100*valid/n:3.0f}%)", end='')
        print()


def main():
    parser = argparse.ArgumentParser(
        description='Level 2 physical plausibility checks on interpolated poses')
    parser.add_argument('session_dir', help='Path to saved session directory')
    parser.add_argument('--suffix', default='_processed',
                        help='File suffix to evaluate (default: _processed)')
    args = parser.parse_args()

    session_dir  = args.session_dir.rstrip('/')
    session_name = os.path.basename(session_dir)

    print(f"{'='*60}")
    print(f"Level 2 Verification — Physical Plausibility")
    print(f"Session : {session_name}")
    print(f"{'='*60}")

    ok1 = check_segment_lengths(session_dir, args.suffix)
    ok2 = check_velocity(session_dir, args.suffix)
    coverage_summary(session_dir)

    print(f"\n{'='*60}")
    print(f"Summary: segment_length={'PASS' if ok1 else 'WARN'}  "
          f"velocity={'PASS' if ok2 else 'WARN'}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
