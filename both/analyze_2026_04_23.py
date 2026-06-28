#!/usr/bin/env python3
import numpy as np
from pathlib import Path

base = Path("/home/primpunn/experiment/both/saved_data/2026-04-23")
frames = sorted([d for d in base.iterdir() if d.name.startswith("frame_")],
                key=lambda d: int(d.name.split("_")[1]))
N = len(frames)

PHASE_NAMES = {0:'idle',1:'approach',2:'lift',3:'press',4:'hold',5:'release'}
JOINTS = ["right_wrist","right_elbow","left_wrist","left_elbow"]

# ── 1. Overview ──────────────────────────────────────────────────────────────
print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print(f"  Total frames : {N}")
total_size = sum(f.stat().st_size for f in base.rglob("*") if f.is_file())
print(f"  Total size   : {total_size/1e6:.1f} MB")
print(f"  Files/frame  : {len(list(frames[0].iterdir()))}")

# ── 2. Top-level transforms ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TOP-LEVEL STATIC TRANSFORMS")
print("=" * 60)
for f in sorted(base.glob("*.txt")):
    T = np.loadtxt(f)
    pos = T[:3, 3]
    print(f"  {f.name:<22s}  pos=({pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f}) m")

# ── 3. File completeness ─────────────────────────────────────────────────────
EXPECTED = ["color_image.png","depth_image.png","pointcloud.npy",
            "pose.txt","pose_right_wrist.txt","pose_right_elbow.txt",
            "pose_left_wrist.txt","pose_left_elbow.txt","phase.txt"]
missing = []
for fr in frames:
    for ef in EXPECTED:
        if not (fr / ef).exists():
            missing.append(f"{fr.name}/{ef}")
print("\n" + "=" * 60)
print("FILE COMPLETENESS")
print("=" * 60)
if missing:
    print(f"  MISSING: {len(missing)} files")
    for m in missing[:10]:
        print(f"    {m}")
else:
    print(f"  All {N} x {len(EXPECTED)} = {N*len(EXPECTED)} expected files present  ✓")

# ── 4. Phase labels ──────────────────────────────────────────────────────────
phases = np.array([int(open(fr/"phase.txt").read().strip()) for fr in frames])
print("\n" + "=" * 60)
print("PHASE LABEL DISTRIBUTION")
print("=" * 60)
for pid, pname in PHASE_NAMES.items():
    cnt = int((phases == pid).sum())
    bar = "█" * int(40 * cnt / N)
    print(f"  {pid} {pname:8s}: {cnt:4d} ({100*cnt/N:5.1f}%)  {bar}")
all_zero = (phases == 0).all()
if all_zero:
    print("\n  ⚠  All frames are phase=0 (idle). Phase keys were not pressed during recording.")

# ── 5. Head pose ─────────────────────────────────────────────────────────────
head_pos = np.array([np.loadtxt(fr/"pose.txt")[:3,3] for fr in frames])
nan_head = np.any(np.isnan(head_pos), axis=1)
valid_hp = head_pos[~nan_head]
print("\n" + "=" * 60)
print("HEAD POSE (D435i camera in world frame)")
print("=" * 60)
print(f"  Tracked frames: {(~nan_head).sum()}/{N}  ({100*(~nan_head).sum()/N:.1f}%)")
print(f"  Lost frames   : {nan_head.sum()}")
if len(valid_hp) > 1:
    for i, ax in enumerate("XYZ"):
        print(f"  {ax}: [{valid_hp[:,i].min():+.3f}, {valid_hp[:,i].max():+.3f}] m  "
              f"std={valid_hp[:,i].std():.3f}")
    diffs = np.linalg.norm(np.diff(valid_hp, axis=0), axis=1)
    print(f"  Frame-to-frame: mean={diffs.mean()*100:.1f} cm  max={diffs.max()*100:.1f} cm")
    print(f"  Large jumps >5cm: {(diffs>0.05).sum()}")

# ── 6. Arm detection ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ARM MARKER DETECTION")
print("=" * 60)
valid = {}
positions = {}
for joint in JOINTS:
    v, p = [], []
    for fr in frames:
        T = np.loadtxt(fr / f"pose_{joint}.txt")
        ok = not np.any(np.isnan(T))
        v.append(ok)
        p.append(T[:3,3] if ok else np.full(3, np.nan))
    valid[joint] = np.array(v)
    positions[joint] = np.array(p)
    cnt = valid[joint].sum()
    print(f"  {joint:15s}: {cnt:4d}/{N}  ({100*cnt/N:.1f}%)")

right_both = valid["right_wrist"] & valid["right_elbow"]
left_both  = valid["left_wrist"]  & valid["left_elbow"]
all_four   = right_both & left_both

print(f"\n  Right arm (wrist+elbow): {right_both.sum():4d}/{N}  ({100*right_both.sum()/N:.1f}%)")
print(f"  Left  arm (wrist+elbow): {left_both.sum():4d}/{N}  ({100*left_both.sum()/N:.1f}%)")
print(f"  All 4 joints            : {all_four.sum():4d}/{N}  ({100*all_four.sum()/N:.1f}%)")

# ── 7. Consecutive runs ──────────────────────────────────────────────────────
def runs(mask):
    segs, s = [], None
    for i, v in enumerate(mask):
        if v and s is None: s = i
        elif not v and s is not None:
            segs.append((s, i-1, i-s)); s = None
    if s is not None: segs.append((s, len(mask)-1, len(mask)-s))
    return sorted(segs, key=lambda x: -x[2])

print("\n" + "=" * 60)
print("CONSECUTIVE COMPLETE-ARM RUNS  (top 3 each)")
print("=" * 60)
for label, mask in [("Right arm", right_both),("Left arm", left_both),("All 4 joints", all_four)]:
    segs = runs(mask)[:3]
    if segs:
        info = "  ".join(f"fr{r[0]}-{r[1]} ({r[2]} frames)" for r in segs)
        print(f"  {label:12s}: {info}")
    else:
        print(f"  {label:12s}: NO complete runs found")

# ── 8. Trajectory quality ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TRAJECTORY QUALITY (consecutive valid frames)")
print("=" * 60)
for joint in JOINTS:
    idxs = np.where(valid[joint])[0]
    if len(idxs) < 2:
        print(f"  {joint:15s}: too few detections"); continue
    diffs = []
    for k in range(len(idxs)-1):
        if idxs[k+1] - idxs[k] == 1:
            diffs.append(np.linalg.norm(positions[joint][idxs[k+1]] -
                                        positions[joint][idxs[k]]))
    if diffs:
        d = np.array(diffs)
        print(f"  {joint:15s}: mean={d.mean()*100:.1f} cm  "
              f"max={d.max()*100:.1f} cm  outliers>3cm: {(d>0.03).sum()}")
    else:
        print(f"  {joint:15s}: no consecutive valid frames")

# ── 9. Phase × detection cross-tab ──────────────────────────────────────────
print("\n" + "=" * 60)
print("DETECTION RATE PER PHASE")
print("=" * 60)
print(f"  {'Phase':10s}  {'N':>5}  {'R-wrist':>8} {'R-elbow':>8} {'L-wrist':>8} {'L-elbow':>8} {'RArm':>6} {'LArm':>6} {'All4':>6}")
for pid, pname in PHASE_NAMES.items():
    mask = phases == pid
    if not mask.any(): continue
    tot = mask.sum()
    def pct(j): return f"{valid[j][mask].sum()}({100*valid[j][mask].sum()/tot:.0f}%)"
    ra = right_both[mask].sum(); la = left_both[mask].sum(); af = all_four[mask].sum()
    print(f"  {pname:10s}  {tot:5d}  {pct('right_wrist'):>8} {pct('right_elbow'):>8} "
          f"{pct('left_wrist'):>8} {pct('left_elbow'):>8} "
          f"{ra}({100*ra/tot:.0f}%)  {la}({100*la/tot:.0f}%)  {af}({100*af/tot:.0f}%)")

# ── 10. Spatial extent of detected joints ───────────────────────────────────
print("\n" + "=" * 60)
print("SPATIAL EXTENT (valid frames only, world frame)")
print("=" * 60)
for joint in JOINTS:
    p = positions[joint][valid[joint]]
    if len(p) == 0: print(f"  {joint:15s}: no data"); continue
    print(f"  {joint}:")
    for i, ax in enumerate("XYZ"):
        print(f"    {ax}: [{p[:,i].min():+.3f}, {p[:,i].max():+.3f}] m  "
              f"mean={p[:,i].mean():+.3f}  range={p[:,i].max()-p[:,i].min():.3f}")

print("\n" + "=" * 60)
print("IMITATION LEARNING READINESS ASSESSMENT")
print("=" * 60)
criteria = [
    ("Frame count",        N >= 450,           f"{N} frames (~{N/30:.0f}s)"),
    ("Head tracked >90%",  (~nan_head).mean()>0.9, f"{100*(~nan_head).mean():.1f}%"),
    ("Right arm >50%",     right_both.mean()>0.5, f"{100*right_both.mean():.1f}%"),
    ("Left arm >50%",      left_both.mean()>0.5,  f"{100*left_both.mean():.1f}%"),
    ("All 4 joints >50%",  all_four.mean()>0.5,   f"{100*all_four.mean():.1f}%"),
    ("Phase labels used",  not all_zero,           "yes" if not all_zero else "ALL ZERO — not set"),
    ("Floor markers seen", len(list(base.glob("T_world_ID*.txt"))) >= 3,
                           f"{len(list(base.glob('T_world_ID*.txt')))} floor markers"),
]
for label, ok, detail in criteria:
    status = "PASS ✓" if ok else "FAIL ✗"
    print(f"  [{status}] {label:30s} {detail}")
