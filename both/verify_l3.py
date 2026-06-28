#!/usr/bin/env python3
"""
Level 3 verification — mask-and-compare ground truth accuracy test.

Artificially masks every --step-th valid frame for the chosen joint(s),
runs the interpolation script, then measures translation and rotation error
at the masked frames against the original ground truth.
Original pose_{joint}.txt files are always restored (try/finally).

Usage:
    conda activate massage
    python verify_l3.py <session_dir>
    python verify_l3.py <session_dir> --joint left_wrist --step 4
    python verify_l3.py <session_dir> --all-joints
    python verify_l3.py <session_dir> --all-joints --script interpolate_modify.py
"""

import argparse
import os
import sys
import subprocess
import numpy as np

ARM_JOINTS    = ['right_wrist', 'right_elbow', 'left_wrist', 'left_elbow']
TRANS_WARN_CM = 5.0    # translation error above this is concerning
ROT_WARN_DEG  = 15.0   # rotation error above this is concerning


def get_frames(session_dir: str) -> list:
    return sorted(
        [d for d in os.listdir(session_dir) if d.startswith('frame_')],
        key=lambda x: int(x.split('_')[1])
    )


def load_pose(path: str):
    if not os.path.exists(path):
        return None
    T = np.loadtxt(path)
    return None if np.any(np.isnan(T)) else T


def rotation_error_deg(T_gt: np.ndarray, T_out: np.ndarray) -> float:
    dR = T_gt[:3, :3] @ T_out[:3, :3].T
    cos_a = float(np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_a)))


def run_test(session_dir: str, joint: str, step: int, script_path: str) -> dict:
    frames = get_frames(session_dir)

    # Collect valid frame indices for this joint
    valid_indices = [
        i for i, fr in enumerate(frames)
        if load_pose(os.path.join(session_dir, fr, f'pose_{joint}.txt')) is not None
    ]

    if len(valid_indices) < step + 1:
        return {'joint': joint,
                'error': f'only {len(valid_indices)} valid frames — need >{step}'}

    # Every step-th valid frame becomes a masked (held-out) test frame
    mask_indices = valid_indices[::step]

    # Backup originals and overwrite with NaN
    backups = {}
    for idx in mask_indices:
        path = os.path.join(session_dir, frames[idx], f'pose_{joint}.txt')
        T_orig = np.loadtxt(path)
        backups[idx] = (path, T_orig)
        np.savetxt(path, np.full((4, 4), np.nan))

    try:
        result = subprocess.run(
            ['conda', 'run', '--no-capture-output', '-n', 'massage',
             'python', script_path, session_dir],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            return {'joint': joint,
                    'error': f'script exited with code {result.returncode}:\n'
                             + result.stderr[-600:]}

        errors_t, errors_R = [], []
        still_nan = 0
        for idx, (path, T_gt) in backups.items():
            proc = os.path.join(session_dir, frames[idx], f'pose_{joint}_processed.txt')
            T_out = load_pose(proc)
            if T_out is None:
                still_nan += 1
                continue
            errors_t.append(np.linalg.norm(T_gt[:3, 3] - T_out[:3, 3]) * 100)  # cm
            errors_R.append(rotation_error_deg(T_gt, T_out))

    finally:
        # Always restore originals regardless of success/failure
        for idx, (path, T_orig) in backups.items():
            np.savetxt(path, T_orig)

    if not errors_t:
        return {'joint': joint,
                'error': f'all {len(mask_indices)} masked frames remained NaN after interpolation'}

    return {
        'joint':       joint,
        'n_valid':     len(valid_indices),
        'n_masked':    len(mask_indices),
        'n_recovered': len(errors_t),
        'still_nan':   still_nan,
        'mean_t':      float(np.mean(errors_t)),
        'max_t':       float(np.max(errors_t)),
        'rmse_t':      float(np.sqrt(np.mean(np.array(errors_t) ** 2))),
        'mean_R':      float(np.mean(errors_R)),
        'max_R':       float(np.max(errors_R)),
    }


def print_results(results: list, script_name: str):
    print(f"\n{'='*75}")
    print(f"Level 3 Results — script: {script_name}")
    print(f"{'='*75}")
    print(f"\n  {'Joint':<18} {'Valid':>6} {'Masked':>7} {'Recov':>6} {'NaN':>5} "
          f"{'t mean':>8} {'t RMSE':>8} {'t max':>8} {'R mean':>7} {'R max':>7}  Pass?")
    print("  " + "─" * 92)

    any_warn = False
    for r in results:
        if 'error' in r:
            print(f"  {r['joint']:<18}  ERROR: {r['error']}")
            continue
        t_ok = r['max_t'] < TRANS_WARN_CM
        R_ok = r['max_R'] < ROT_WARN_DEG
        ok   = t_ok and R_ok
        if not ok:
            any_warn = True
        flag = "PASS" if ok else "WARN"
        print(f"  {r['joint']:<18} {r['n_valid']:>6} {r['n_masked']:>7} "
              f"{r['n_recovered']:>6} {r['still_nan']:>5} "
              f"{r['mean_t']:>7.2f}cm {r['rmse_t']:>7.2f}cm {r['max_t']:>7.2f}cm "
              f"{r['mean_R']:>6.1f}°  {r['max_R']:>6.1f}°  {flag}")

    print(f"\n  Thresholds: translation max < {TRANS_WARN_CM:.0f} cm,  "
          f"rotation max < {ROT_WARN_DEG:.0f}°")
    overall = "PASS — interpolation error within acceptable bounds" if not any_warn \
              else "WARN — some joints exceed thresholds"
    print(f"  Overall: {overall}")
    print(f"{'='*75}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Level 3 mask-and-compare interpolation accuracy test')
    parser.add_argument('session_dir',
                        help='Path to saved session directory')
    parser.add_argument('--joint', default='left_wrist',
                        choices=ARM_JOINTS,
                        help='Single joint to test (default: left_wrist)')
    parser.add_argument('--all-joints', action='store_true',
                        help='Test all 4 joints sequentially')
    parser.add_argument('--step', type=int, default=5,
                        help='Mask every N-th valid frame (default: 5, ~20%% held out)')
    parser.add_argument('--script', default='interpolate_depth.py',
                        help='Interpolation script to evaluate '
                             '(default: interpolate_depth.py)')
    args = parser.parse_args()

    session_dir = args.session_dir.rstrip('/')
    if not os.path.isdir(session_dir):
        print(f"Error: not a directory: {session_dir}")
        sys.exit(1)

    # Resolve script path relative to this file
    script_dir  = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(script_dir, args.script)
    if not os.path.exists(script_path):
        script_path = args.script

    joints = ARM_JOINTS if args.all_joints else [args.joint]

    print(f"{'='*60}")
    print(f"Level 3 Verification — Mask-and-Compare")
    print(f"Session : {os.path.basename(session_dir)}")
    print(f"Script  : {args.script}")
    print(f"Joints  : {joints}")
    print(f"Step    : every {args.step}th valid frame masked (~{100//args.step}% held out)")
    print(f"{'='*60}")

    results = []
    for joint in joints:
        print(f"\nTesting {joint}...", flush=True)
        r = run_test(session_dir, joint, args.step, script_path)
        results.append(r)
        if 'error' in r:
            print(f"  ERROR: {r['error']}")
        else:
            print(f"  Masked {r['n_masked']} / {r['n_valid']} valid frames  →  "
                  f"recovered {r['n_recovered']}, still NaN: {r['still_nan']}")
            print(f"  Translation: mean={r['mean_t']:.2f} cm  "
                  f"RMSE={r['rmse_t']:.2f} cm  max={r['max_t']:.2f} cm")
            print(f"  Rotation:    mean={r['mean_R']:.2f}°   "
                  f"max={r['max_R']:.2f}°")

    print_results(results, args.script)


if __name__ == '__main__':
    main()
