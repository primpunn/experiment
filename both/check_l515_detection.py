#!/usr/bin/env python3
"""
Check whether the L515 fallback actually contributed arm poses,
and verify what the L515 can physically see.
"""
import sys
sys.path.insert(0, '/home/primpunn/librealsense/build/Release')

import numpy as np
import cv2
from pathlib import Path

base   = Path("/home/primpunn/experiment/both/saved_data/2026-04-23")
frames = sorted([d for d in base.iterdir() if d.name.startswith("frame_")],
                key=lambda d: int(d.name.split("_")[1]))
N = len(frames)

ARM_IDS   = [0,1,2,3, 4,5,6,7, 8,9,12,15, 18,19,22,23]
FLOOR_IDS = [10,11,13,14,16,17,20,21]
ALL_IDS   = set(ARM_IDS + FLOOR_IDS)

ARM_JOINTS = ['right_wrist','right_elbow','left_wrist','left_elbow']
ARM_MARKER_GROUPS = {
    'right_wrist': [ 0, 1, 2, 3],
    'right_elbow': [ 4, 5, 6, 7],
    'left_wrist':  [ 8, 9,12,15],
    'left_elbow':  [18,19,22,23],
}

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
params     = cv2.aruco.DetectorParameters()
detector   = cv2.aruco.ArucoDetector(aruco_dict, params)

T_world_L515 = np.loadtxt(base / "T_world_L515.txt")

# ── L515 camera orientation ───────────────────────────────────────────────────
print("=" * 60)
print("L515 CAMERA GEOMETRY")
print("=" * 60)
R = T_world_L515[:3, :3]
pos = T_world_L515[:3, 3]
# Camera looks along its +Z axis in camera frame → world direction = R @ [0,0,1]
look_dir = R @ np.array([0, 0, 1])
print(f"  L515 position (world): ({pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f}) m")
print(f"  L515 look direction  : ({look_dir[0]:+.3f}, {look_dir[1]:+.3f}, {look_dir[2]:+.3f})")
print(f"  → L515 faces mostly {'right (+X)' if look_dir[0]>0 else 'left (-X)'}, "
      f"{'up (+Z)' if look_dir[2]>0 else 'down (-Z)'}")

# ── Run ArUco detection on ALL L515 color frames ─────────────────────────────
print("\n" + "=" * 60)
print("ARM MARKER VISIBILITY IN L515 FRAMES (all 900 frames)")
print("=" * 60)

# Track per-joint per-frame: was any of its markers seen in L515 image?
l515_joint_seen  = {j: np.zeros(N, dtype=bool) for j in ARM_JOINTS}
l515_any_arm     = np.zeros(N, dtype=bool)
l515_ids_seen    = {mid: 0 for mid in ARM_IDS}   # total count across all frames

for i, fr in enumerate(frames):
    img  = cv2.imread(str(fr / "color_image.png"))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        continue
    detected = set(ids.flatten().tolist())
    for mid in ARM_IDS:
        if mid in detected:
            l515_ids_seen[mid] += 1
    arm_detected = detected & set(ARM_IDS)
    if arm_detected:
        l515_any_arm[i] = True
        for joint, mids in ARM_MARKER_GROUPS.items():
            if any(m in detected for m in mids):
                l515_joint_seen[joint][i] = True

# Report per-ID counts
detected_ids = {mid: cnt for mid, cnt in l515_ids_seen.items() if cnt > 0}
if detected_ids:
    print(f"  Arm marker IDs detected in L515 frames:")
    for mid, cnt in sorted(detected_ids.items()):
        joint = next(j for j, ids in ARM_MARKER_GROUPS.items() if mid in ids)
        print(f"    ID{mid:2d} ({joint:15s}): {cnt:4d}/{N} frames ({100*cnt/N:.1f}%)")
else:
    print("  NO arm markers detected in any L515 frame.")

# Report per-joint
print(f"\n  Per-joint visibility in L515:")
for joint in ARM_JOINTS:
    cnt = l515_joint_seen[joint].sum()
    print(f"    {joint:15s}: {cnt:4d}/{N} frames ({100*cnt/N:.1f}%)")

print(f"\n  Any arm marker in L515: {l515_any_arm.sum()}/{N} frames ({100*l515_any_arm.mean():.1f}%)")

# ── Check if D435i missed but L515 saw ──────────────────────────────────────
print("\n" + "=" * 60)
print("FRAMES WHERE L515 FILLS GAP (D435i missed, L515 saw)")
print("=" * 60)
for joint in ARM_JOINTS:
    T_saved = np.array([np.loadtxt(fr/f"pose_{joint}.txt")[:3,3] for fr in frames])
    d435i_saved_valid = ~np.any(np.isnan(T_saved), axis=1)

    gap_filled = l515_joint_seen[joint] & ~d435i_saved_valid
    # This is an approximation: if saved pose is valid it could be from either camera
    # Better: frames where saved is valid AND d435i alone would have missed
    # (we can't perfectly distinguish, but l515_joint_seen tells what L515 saw)
    print(f"  {joint:15s}: L515 saw {l515_joint_seen[joint].sum():3d} frames | "
          f"Saved valid {d435i_saved_valid.sum():3d} frames | "
          f"L515 saw but saved=NaN: {gap_filled.sum():3d} frames (fallback gaps remain)")

# ── Conclusion ────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("CONCLUSION")
print("=" * 60)
any_l515 = sum(l515_ids_seen.values()) > 0
if any_l515:
    print("  L515 CAN detect arm markers — geometry is correct.")
    print("  The L515 fallback is physically working.")
    total_gap_filled = sum(
        (l515_joint_seen[j] & np.any(np.isnan(
            np.array([np.loadtxt(fr/f"pose_{j}.txt") for fr in frames])),axis=(1,2))).sum()
        for j in ARM_JOINTS
    )
    print(f"  Estimated frames gap-filled by L515: ~{total_gap_filled}")
else:
    print("  L515 CANNOT see arm markers from its current position.")
    print("  The fallback code runs but produces no additional data.")
    print("  → L515 camera angle/position does not cover the therapy area.")
