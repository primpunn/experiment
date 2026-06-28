#!/usr/bin/env python3
"""
MuJoCo playback of interpolated massage therapy arm motion.

Loads pose_{joint}<suffix>.txt (world-frame 4x4 transforms) for every frame
and animates the therapist's joints in the MuJoCo scene defined in
mujoco_scene.xml.

Controls inside the viewer window:
  Mouse drag   — rotate camera
  Scroll       — zoom
  Right drag   — pan
  Space        — pause / resume
  Esc          — quit

Usage:
    conda activate massage
    python mujoco_playback.py saved_data/2026-05-11
    python mujoco_playback.py saved_data/2026-05-11 --fps 15 --loop
    python mujoco_playback.py saved_data/2026-05-11 --suffix _filled
    python mujoco_playback.py saved_data/2026-05-11 --suffix ""   # raw only
"""

import argparse
import os
import sys
import time
import numpy as np
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation

# ── Constants ─────────────────────────────────────────────────────────────────

JOINTS    = ['right_wrist', 'right_elbow', 'left_wrist', 'left_elbow']
SCENE_XML = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mujoco_scene.xml')
HIDDEN    = np.array([0.0, 0.0, 10.0])   # off-screen park position for missing joints


# ── Data loading ──────────────────────────────────────────────────────────────

def load_session(session_dir: str, suffix: str):
    """
    Load all per-frame poses from the session directory.

    Returns
    -------
    joint_data  : {joint_name: list of (pos(3,), quat(4,)) or None}
    head_data   : list of (pos(3,), quat(4,)) or None   — D435i camera = head
    n_frames    : int
    l515_pos    : (3,) world position of L515 camera, or None
    """
    frames = sorted(
        [d for d in os.listdir(session_dir) if d.startswith('frame_')],
        key=lambda x: int(x.split('_')[1])
    )
    n = len(frames)

    joint_data = {j: [] for j in JOINTS}
    head_data  = []

    for fr in frames:
        frame_dir = os.path.join(session_dir, fr)

        for j in JOINTS:
            path = os.path.join(frame_dir, f'pose_{j}{suffix}.txt')
            entry = _load_pose(path)
            joint_data[j].append(entry)

        head_path = os.path.join(frame_dir, 'pose.txt')
        head_data.append(_load_pose(head_path))

    # L515 static pose
    l515_path = os.path.join(session_dir, 'T_world_L515.txt')
    l515_pos  = None
    if os.path.exists(l515_path):
        T = np.loadtxt(l515_path)
        l515_pos = T[:3, 3]

    return joint_data, head_data, n, l515_pos


def _load_pose(path: str):
    """Return (pos(3,), quat(4,)[w,x,y,z]) or None if file missing / NaN."""
    if not os.path.exists(path):
        return None
    T = np.loadtxt(path)
    if np.any(np.isnan(T)):
        return None
    pos  = T[:3, 3].copy()
    quat = _mat_to_quat(T[:3, :3])
    return pos, quat


def _mat_to_quat(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix → [w, x, y, z] quaternion (MuJoCo convention)."""
    q_xyzw = Rotation.from_matrix(R).as_quat()   # scipy: [x,y,z,w]
    return q_xyzw[[3, 0, 1, 2]]                   # reorder to [w,x,y,z]


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _align_z_to(v: np.ndarray) -> np.ndarray:
    """
    Quaternion [w,x,y,z] that rotates the local +Z axis to point along v.
    Used to orient forearm capsules (MuJoCo capsules extend along local Z).
    """
    v = v / (np.linalg.norm(v) + 1e-10)
    z = np.array([0.0, 0.0, 1.0])
    cross = np.cross(z, v)
    dot   = float(np.dot(z, v))

    if np.linalg.norm(cross) < 1e-6:
        # Parallel or anti-parallel
        if dot > 0.0:
            return np.array([1.0, 0.0, 0.0, 0.0])   # no rotation needed
        return np.array([0.0, 1.0, 0.0, 0.0])        # 180° around X

    axis  = cross / np.linalg.norm(cross)
    angle = np.arccos(np.clip(dot, -1.0, 1.0))
    s     = np.sin(angle / 2.0)
    return np.array([np.cos(angle / 2.0), s * axis[0], s * axis[1], s * axis[2]])


# ── MuJoCo ID cache ───────────────────────────────────────────────────────────

def _mocap_id(model, body_name: str) -> int:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    return model.body_mocapid[body_id]


def _geom_id(model, geom_name: str) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)


def build_id_cache(model):
    mocap = {
        'right_wrist':    _mocap_id(model, 'right_wrist'),
        'right_elbow':    _mocap_id(model, 'right_elbow'),
        'right_forearm':  _mocap_id(model, 'right_forearm'),
        'left_wrist':     _mocap_id(model, 'left_wrist'),
        'left_elbow':     _mocap_id(model, 'left_elbow'),
        'left_forearm':   _mocap_id(model, 'left_forearm'),
        'therapist_head': _mocap_id(model, 'therapist_head'),
        'l515_camera':    _mocap_id(model, 'l515_camera'),
    }
    geom = {
        'right_forearm_geom': _geom_id(model, 'right_forearm_geom'),
        'left_forearm_geom':  _geom_id(model, 'left_forearm_geom'),
    }
    return mocap, geom


# ── Per-frame update ──────────────────────────────────────────────────────────

FOREARM_PAIRS = [
    ('right_wrist', 'right_elbow', 'right_forearm', 'right_forearm_geom'),
    ('left_wrist',  'left_elbow',  'left_forearm',  'left_forearm_geom'),
]
IDENTITY_QUAT = np.array([1.0, 0.0, 0.0, 0.0])


def apply_frame(model, data, joint_data, head_data, frame_idx,
                mocap_ids, geom_ids):
    """Push one frame's pose data into the MuJoCo data struct."""

    joint_pos = {}

    # ── Joint spheres ─────────────────────────────────────────────────────
    for j in JOINTS:
        mid  = mocap_ids[j]
        pose = joint_data[j][frame_idx]
        if pose is not None:
            data.mocap_pos[mid]  = pose[0]
            data.mocap_quat[mid] = pose[1]
            joint_pos[j]         = pose[0]
        else:
            data.mocap_pos[mid]  = HIDDEN
            data.mocap_quat[mid] = IDENTITY_QUAT

    # ── Forearm capsules (midpoint + orientation + dynamic half-length) ───
    for wk, ek, fk, fg in FOREARM_PAIRS:
        mid = mocap_ids[fk]
        if wk in joint_pos and ek in joint_pos:
            w   = joint_pos[wk]
            e   = joint_pos[ek]
            vec = e - w
            dist = np.linalg.norm(vec)

            data.mocap_pos[mid]  = (w + e) / 2.0
            data.mocap_quat[mid] = _align_z_to(vec)

            # Capsule half-length = dist/2 − radius  (so tips touch the joint spheres)
            radius   = model.geom_size[geom_ids[fg], 0]
            half_len = max(0.001, dist / 2.0 - radius)
            model.geom_size[geom_ids[fg], 1] = half_len
        else:
            data.mocap_pos[mid]  = HIDDEN
            data.mocap_quat[mid] = IDENTITY_QUAT

    # ── Head ──────────────────────────────────────────────────────────────
    mid  = mocap_ids['therapist_head']
    pose = head_data[frame_idx]
    if pose is not None:
        data.mocap_pos[mid]  = pose[0]
        data.mocap_quat[mid] = pose[1]
    else:
        data.mocap_pos[mid]  = HIDDEN
        data.mocap_quat[mid] = IDENTITY_QUAT


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='MuJoCo playback of interpolated arm motion from a session')
    parser.add_argument('session_dir',
                        help='Path to saved session directory')
    parser.add_argument('--fps', type=float, default=15.0,
                        help='Playback speed in frames per second (default: 15)')
    parser.add_argument('--loop', action='store_true',
                        help='Loop playback continuously until window is closed')
    parser.add_argument('--suffix', default='_processed',
                        help='Pose file suffix: _processed (default), _filled, or "" for raw')
    parser.add_argument('--xml', default=SCENE_XML,
                        help='Path to MuJoCo scene XML (default: mujoco_scene.xml)')
    args = parser.parse_args()

    session_dir = args.session_dir.rstrip('/')
    if not os.path.isdir(session_dir):
        print(f'Error: directory not found: {session_dir}')
        sys.exit(1)
    if not os.path.exists(args.xml):
        print(f'Error: scene XML not found: {args.xml}')
        sys.exit(1)

    print(f'Session : {session_dir}')
    print(f'Suffix  : "{args.suffix}"')
    print(f'FPS     : {args.fps}')
    print(f'Loop    : {args.loop}')
    print(f'XML     : {args.xml}')
    print('Loading poses...', flush=True)

    joint_data, head_data, n_frames, l515_pos = load_session(session_dir, args.suffix)

    for j in JOINTS:
        valid = sum(1 for p in joint_data[j] if p is not None)
        print(f'  {j:<18}: {valid:>4}/{n_frames} valid frames')

    print(f'\nLoading MuJoCo model...', flush=True)
    model = mujoco.MjModel.from_xml_path(args.xml)
    data  = mujoco.MjData(model)

    mocap_ids, geom_ids = build_id_cache(model)

    # Place L515 camera marker at its world position (static for whole session)
    if l515_pos is not None:
        mid = mocap_ids['l515_camera']
        data.mocap_pos[mid]  = l515_pos
        data.mocap_quat[mid] = IDENTITY_QUAT
        print(f'  L515 camera placed at world pos: {l515_pos}')

    frame_dt = 1.0 / args.fps
    paused   = False

    print(f'\nStarting playback — {n_frames} frames at {args.fps} fps')
    print('  Space = pause/resume   |   Esc = quit\n', flush=True)

    with mujoco.viewer.launch_passive(model, data) as viewer:

        # Initial camera: looking down at the therapy area from the side
        viewer.cam.distance  = 3.2
        viewer.cam.azimuth   = 40.0
        viewer.cam.elevation = -28.0
        viewer.cam.lookat[:] = [0.0, -0.35, 0.25]

        frame_idx = 0
        while viewer.is_running():

            t0 = time.perf_counter()

            if not paused:
                apply_frame(model, data, joint_data, head_data,
                            frame_idx, mocap_ids, geom_ids)
                mujoco.mj_forward(model, data)
                viewer.sync()

                frame_idx += 1
                if frame_idx >= n_frames:
                    if args.loop:
                        frame_idx = 0
                    else:
                        # Hold last frame until window is closed
                        print('Playback complete — close window to exit.')
                        while viewer.is_running():
                            viewer.sync()
                            time.sleep(0.05)
                        break

            elapsed = time.perf_counter() - t0
            sleep_t = frame_dt - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)


if __name__ == '__main__':
    main()
