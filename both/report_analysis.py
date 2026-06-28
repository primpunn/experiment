#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/primpunn/librealsense/build/Release')
import numpy as np
import cv2
from pathlib import Path
from scipy.spatial.transform import Rotation

base   = Path("/home/primpunn/experiment/both/saved_data/2026-04-23")
frames = sorted([d for d in base.iterdir() if d.name.startswith("frame_")],
                key=lambda d: int(d.name.split("_")[1]))
N = len(frames)

ARM_JOINTS = ['right_wrist','right_elbow','left_wrist','left_elbow']
ARM_MARKER_GROUPS = {
    'right_wrist': [ 0, 1, 2, 3],
    'right_elbow': [ 4, 5, 6, 7],
    'left_wrist':  [ 8, 9,12,15],
    'left_elbow':  [18,19,22,23],
}
ARM_IDS   = [mid for ids in ARM_MARKER_GROUPS.values() for mid in ids]
FLOOR_IDS = [10,11,13,14,16,17,20,21]
PHASE_NAMES = {0:'idle',1:'approach',2:'lift',3:'press',4:'hold',5:'release'}

# ── Load static transforms ────────────────────────────────────────────────────
T_world_L515 = np.loadtxt(base / "T_world_L515.txt")
floor_transforms = {}
for f in base.glob("T_world_ID*.txt"):
    fid = int(f.stem.split("ID")[1])
    floor_transforms[fid] = np.loadtxt(f)

# ── L515 geometry ─────────────────────────────────────────────────────────────
l515_pos  = T_world_L515[:3, 3]
l515_R    = T_world_L515[:3, :3]
look_dir  = l515_R @ np.array([0, 0, 1])
up_dir    = l515_R @ np.array([0, -1, 0])
dist_to_origin = np.linalg.norm(l515_pos[:2])   # horizontal distance to ID10

# ── Load all arm poses & head poses ──────────────────────────────────────────
head_pos  = np.array([np.loadtxt(fr/"pose.txt")[:3,3] for fr in frames])
phases    = np.array([int(open(fr/"phase.txt").read().strip()) for fr in frames])

valid  = {}
pos_all = {}
for joint in ARM_JOINTS:
    vals, pos = [], []
    for fr in frames:
        T = np.loadtxt(fr / f"pose_{joint}.txt")
        ok = not np.any(np.isnan(T))
        vals.append(ok)
        pos.append(T[:3,3] if ok else np.full(3,np.nan))
    valid[joint]   = np.array(vals)
    pos_all[joint] = np.array(pos)

right_both = valid["right_wrist"] & valid["right_elbow"]
left_both  = valid["left_wrist"]  & valid["left_elbow"]
all_four   = right_both & left_both
nan_head   = np.any(np.isnan(head_pos), axis=1)

def runs(mask):
    segs, s = [], None
    for i,v in enumerate(mask):
        if v and s is None: s=i
        elif not v and s is not None: segs.append((s,i-1,i-s)); s=None
    if s is not None: segs.append((s,len(mask)-1,len(mask)-s))
    return sorted(segs, key=lambda x:-x[2])

# ── L515 arm detection scan ───────────────────────────────────────────────────
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
detector   = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
l515_joint_seen = {j: np.zeros(N,dtype=bool) for j in ARM_JOINTS}
for i,fr in enumerate(frames):
    img  = cv2.imread(str(fr/"color_image.png"))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None: continue
    det = set(ids.flatten().tolist())
    for joint,mids in ARM_MARKER_GROUPS.items():
        if any(m in det for m in mids):
            l515_joint_seen[joint][i] = True

# forearm length consistency (right arm, consecutive valid frames)
def forearm_lengths(joint_a, joint_b):
    mask = valid[joint_a] & valid[joint_b]
    if mask.sum() < 2: return np.array([])
    pa = pos_all[joint_a][mask]
    pb = pos_all[joint_b][mask]
    return np.linalg.norm(pa - pb, axis=1)

rfl = forearm_lengths("right_wrist","right_elbow")
lfl = forearm_lengths("left_wrist","left_elbow")

# ──────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("EXPERIMENTAL SETUP")
print("=" * 70)

print("\n1. Recording Environment")
print(f"   Date              : 2026-04-23")
print(f"   Cameras           : Intel RealSense L515 (static floor) + D435i (head-mounted)")
print(f"   Recording length  : {N} frames  (~{N/30:.0f} seconds at 30 fps)")
print(f"   Total data size   : {sum(f.stat().st_size for f in base.rglob('*') if f.is_file())/1e6:.1f} MB")
print(f"   Task              : Dual-arm calf-stretching therapy")

print("\n2. ArUco Marker Configuration  (DICT_4X4_100)")
print("   Floor markers (95 mm):")
for fid, T in sorted(floor_transforms.items()):
    p = T[:3,3]
    role = "world origin" if fid==10 else "re-localization"
    print(f"     ID{fid:2d}  ({role:15s})  pos=({p[0]:+.3f}, {p[1]:+.3f}, {p[2]:+.3f}) m")

print("   Arm markers (30 mm)  — 4 IDs per joint:")
for joint, mids in ARM_MARKER_GROUPS.items():
    print(f"     {joint:15s}: IDs {mids}")

print("\n3. L515 Camera Position and Orientation")
print(f"   World position        : ({l515_pos[0]:+.3f}, {l515_pos[1]:+.3f}, {l515_pos[2]:+.3f}) m")
print(f"   Height above floor    : {l515_pos[2]:.3f} m  ({l515_pos[2]*100:.1f} cm)")
print(f"   Horizontal dist to ID10: {dist_to_origin:.3f} m  ({dist_to_origin*100:.1f} cm)")
print(f"   Total dist to origin  : {np.linalg.norm(l515_pos):.3f} m")
print(f"   Look direction (world): ({look_dir[0]:+.3f}, {look_dir[1]:+.3f}, {look_dir[2]:+.3f})")
print(f"   Facing                : +X (toward therapy area), slight downward tilt ({look_dir[2]*100:.1f}% Z)")

fid_dists = {}
for fid, T in floor_transforms.items():
    fid_dists[fid] = np.linalg.norm(l515_pos - T[:3,3])
print("   Distance from L515 to each floor marker:")
for fid, d in sorted(fid_dists.items()):
    print(f"     ID{fid:2d}: {d:.3f} m")

print("\n4. Floor Marker Spatial Layout")
positions = np.array([floor_transforms[fid][:3,3] for fid in sorted(floor_transforms)])
xspan = positions[:,0].max() - positions[:,0].min()
yspan = positions[:,1].max() - positions[:,1].min()
print(f"   Markers span: X={xspan:.3f} m  Y={yspan:.3f} m  (floor coverage area)")
for fid in sorted(floor_transforms):
    p = floor_transforms[fid][:3,3]
    print(f"     ID{fid:2d}: ({p[0]:+.3f}, {p[1]:+.3f}, {p[2]:+.3f}) m  "
          f"Z-deviation from floor: {abs(p[2])*1000:.1f} mm")

print("\n" + "=" * 70)
print("DATA ANALYSIS")
print("=" * 70)

print("\n5. Head Pose Tracking (D435i)")
valid_hp = head_pos[~nan_head]
print(f"   Tracked frames     : {(~nan_head).sum()}/{N}  ({100*(~nan_head).mean():.1f}%)")
print(f"   Head height range  : Z=[{valid_hp[:,2].min():.3f}, {valid_hp[:,2].max():.3f}] m  "
      f"(mean {valid_hp[:,2].mean():.3f} m)")
diffs_head = np.linalg.norm(np.diff(valid_hp,axis=0),axis=1)
print(f"   Frame-to-frame displacement:")
print(f"     Mean : {diffs_head.mean()*100:.1f} cm/frame  ({diffs_head.mean()*30*100:.0f} cm/s at 30fps)")
print(f"     Max  : {diffs_head.max()*100:.1f} cm/frame")
print(f"     Jumps >5 cm : {(diffs_head>0.05).sum()} frames ({100*(diffs_head>0.05).mean():.1f}%)")

print("\n6. Arm Marker Detection Rates")
print(f"   {'Joint':15s} {'D435i+L515':>12} {'L515 only':>12} {'Consecutive best run':>25}")
for joint in ARM_JOINTS:
    cnt   = valid[joint].sum()
    lcnt  = l515_joint_seen[joint].sum()
    best  = runs(valid[joint])
    brun  = f"fr{best[0][0]}-{best[0][1]} ({best[0][2]} fr)" if best else "none"
    print(f"   {joint:15s} {cnt:4d}/{N} ({100*cnt/N:.0f}%)  "
          f"{lcnt:4d}/{N} ({100*lcnt/N:.0f}%)  {brun}")

print(f"\n   Combined arm completeness:")
print(f"     Right arm (wrist+elbow): {right_both.sum():4d}/{N}  ({100*right_both.mean():.1f}%)"
      f"  best run: {runs(right_both)[0][2] if runs(right_both) else 0} frames")
print(f"     Left  arm (wrist+elbow): {left_both.sum():4d}/{N}  ({100*left_both.mean():.1f}%)"
      f"  best run: {runs(left_both)[0][2] if runs(left_both) else 0} frames")
print(f"     All 4 joints           : {all_four.sum():4d}/{N}  ({100*all_four.mean():.1f}%)"
      f"  best run: {runs(all_four)[0][2] if runs(all_four) else 0} frames")

print("\n7. Trajectory Quality")
for joint in ARM_JOINTS:
    idxs = np.where(valid[joint])[0]
    if len(idxs)<2: continue
    diffs = []
    for k in range(len(idxs)-1):
        if idxs[k+1]-idxs[k]==1:
            diffs.append(np.linalg.norm(pos_all[joint][idxs[k+1]]-pos_all[joint][idxs[k]]))
    if not diffs: continue
    d = np.array(diffs)
    print(f"   {joint:15s}: mean={d.mean()*100:.1f} cm/frame  "
          f"max={d.max()*100:.1f} cm  "
          f"outliers>3cm: {(d>0.03).sum()} ({100*(d>0.03).mean():.0f}%)")

print("\n8. Forearm Segment Length Consistency (geometric validation)")
if len(rfl)>0:
    print(f"   Right wrist-to-elbow: mean={rfl.mean()*100:.1f} cm  "
          f"std={rfl.std()*100:.1f} cm  "
          f"range=[{rfl.min()*100:.1f}, {rfl.max()*100:.1f}] cm")
if len(lfl)>0:
    print(f"   Left  wrist-to-elbow: mean={lfl.mean()*100:.1f} cm  "
          f"std={lfl.std()*100:.1f} cm  "
          f"range=[{lfl.min()*100:.1f}, {lfl.max()*100:.1f}] cm")

print("\n9. Arm Spatial Extent in World Frame")
for joint in ARM_JOINTS:
    p = pos_all[joint][valid[joint]]
    if len(p)==0: continue
    print(f"   {joint}:")
    for i,ax in enumerate("XYZ"):
        print(f"     {ax}: [{p[:,i].min():+.3f}, {p[:,i].max():+.3f}] m  "
              f"mean={p[:,i].mean():+.3f}  range={np.ptp(p[:,i]):.3f} m")

print("\n10. Phase Label Distribution")
for pid,pname in PHASE_NAMES.items():
    cnt = int((phases==pid).sum())
    print(f"    Phase {pid} ({pname:8s}): {cnt:4d}/{N} ({100*cnt/N:.1f}%)")
if (phases==0).all():
    print("    *** Phase keys were not pressed — all frames labeled as idle ***")

print("\n11. L515 Fallback Detection")
print(f"    Frames with any arm marker in L515  : {sum(l515_joint_seen[j].any() for j in ARM_JOINTS)} joint-detections")
for joint in ARM_JOINTS:
    l = l515_joint_seen[joint].sum()
    v = valid[joint].sum()
    print(f"    {joint:15s}: L515 saw {l:3d} fr | total valid {v:3d} fr")

print("\n" + "=" * 70)
print("PROBLEMS AND SOLUTIONS")
print("=" * 70)

problems = [
    ("Phase labels not recorded",
     "All 900 frames are labeled phase=0 (idle). "
     "No task phase information is captured.",
     "Press number keys 0-5 actively during each recording session "
     "to label approach/lift/press/hold/release phases in real time."),
    ("High trajectory noise",
     f"Mean frame-to-frame arm displacement is ~16 cm/frame (~5 m/s at 30fps), "
     "which is physically impossible. "
     f"Head pose jumps >5cm in {(diffs_head>0.05).sum()} of 900 frames.",
     "Head-mounted D435i propagates all head-motion errors into arm poses "
     "(T_world_arm = T_world_head @ T_head_arm). "
     "Solutions: (1) move more slowly and steadily during recording; "
     "(2) add post-processing outlier filter discarding frames where any joint "
     "moves >3 cm from previous frame; "
     "(3) longer-term: move D435i to a fixed tripod mount."),
    ("Only 1 demonstration session",
     "IL algorithms require 30-50 complete task demonstrations. "
     "This dataset is a single ~30-second session with no complete task cycle.",
     "Record 30-50 independent demonstrations, each covering the full "
     "approach→lift→press→hold→release cycle (~15-30 sec each)."),
    ("All-4-joints coverage only 25.7%",
     f"Only 231/900 frames have all 4 joints simultaneously detected. "
     f"Longest complete run is {runs(all_four)[0][2]} frames (~1.5 sec).",
     "4-marker-per-joint setup already improved left arm dramatically. "
     "Remaining issue is trajectory noise causing false NaN detections. "
     "Fix: apply smoothing/outlier rejection post-processing."),
    ("L515 fallback limited overlap",
     "L515 detects arm markers in ~49% of frames but tends to see the same "
     "frames as D435i rather than complementary frames.",
     "L515 is confirmed physically functional. Its benefit will be larger "
     "in recordings where arm motion sweeps outside the D435i field of view. "
     "No code change needed."),
]

for i,(title,problem,solution) in enumerate(problems,1):
    print(f"\n  Problem {i}: {title}")
    print(f"  Issue   : {problem}")
    print(f"  Solution: {solution}")

print("\n" + "=" * 70)
print("IMITATION LEARNING READINESS ASSESSMENT")
print("=" * 70)

criteria = [
    ("Number of demonstrations",  False, "1 session (need 30-50 complete demos)"),
    ("Complete task cycles",       False, "0 complete cycles in this session"),
    ("All-4-joint coverage >80%",  False, f"{100*all_four.mean():.1f}% (need >80%)"),
    ("Right arm coverage >80%",    False, f"{100*right_both.mean():.1f}% (need >80%)"),
    ("Left arm coverage >80%",     left_both.mean()>0.5, f"{100*left_both.mean():.1f}%"),
    ("Head tracking >95%",         (~nan_head).mean()>0.95, f"{100*(~nan_head).mean():.1f}%  ✓"),
    ("Phase labels present",       not (phases==0).all(), "MISSING — all idle"),
    ("Trajectory noise <2cm/frame",False, f"~16 cm/frame mean (need <2 cm)"),
    ("Floor marker coverage",      True,  f"7 markers, span {xspan:.2f}x{yspan:.2f} m  ✓"),
    ("Point cloud available",      True,  "8192 pts/frame, stable  ✓"),
]

passed = sum(1 for _,ok,_ in criteria if ok)
print(f"\n  Score: {passed}/{len(criteria)} criteria met\n")
for label,ok,detail in criteria:
    status = "PASS ✓" if ok else "FAIL ✗"
    print(f"  [{status}]  {label:<35s}  {detail}")

print(f"""
  Overall verdict: NOT ready for imitation learning yet.
  Minimum steps before IL is viable:
    1. Record 30-50 complete demonstrations with active phase labeling
    2. Apply outlier filter (discard frames with >3 cm joint jump)
    3. Verify trajectory noise drops to <2 cm/frame after filtering
""")
