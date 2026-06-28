"""
Calculate arm marker motion velocity from all collected sessions.
Used to establish a data-driven MAX_ARM_SPEED threshold for gap interpolation.

Strategy: examine the velocity distribution to separate legitimate therapy
motion from spurious ArUco detection jumps (which dominate p90+).
"""
import numpy as np
from pathlib import Path

TARGET_FPS = 30
MARKERS = ["right_wrist", "right_elbow", "left_wrist", "left_elbow"]
DATA_ROOT = Path(__file__).parent / "saved_data"

sessions = sorted([
    d for d in DATA_ROOT.iterdir()
    if d.is_dir() and any(d.glob("frame_*"))
])

print(f"Sessions: {[s.name for s in sessions]}\n")

all_v = {m: [] for m in MARKERS}

for session in sessions:
    frame_dirs = sorted(session.glob("frame_*"), key=lambda d: int(d.name.split("_")[1]))
    if len(frame_dirs) < 2:
        continue
    poses = {m: [] for m in MARKERS}
    for fd in frame_dirs:
        for m in MARKERS:
            try:
                mat = np.loadtxt(fd / f"pose_{m}.txt")
            except Exception:
                mat = np.full((4, 4), np.nan)
            poses[m].append(mat)

    for m in MARKERS:
        ps = poses[m]
        valid = [not np.any(np.isnan(p)) for p in ps]
        for i in range(1, len(ps)):
            if valid[i - 1] and valid[i]:
                dist = np.linalg.norm(ps[i][:3, 3] - ps[i - 1][:3, 3])
                all_v[m].append(dist * TARGET_FPS)

combined = np.concatenate([all_v[m] for m in MARKERS])

# --- Percentile table ---
pcts = [10, 25, 50, 60, 70, 75, 80, 85, 90, 95, 99]
print("Cumulative velocity percentiles (ALL markers combined):")
print(f"  {'Pct':>5}  {'m/s':>8}  {'m/frame':>9}")
print(f"  {'-'*5}  {'-'*8}  {'-'*9}")
for p in pcts:
    v = np.percentile(combined, p)
    print(f"  {p:>4}%  {v:>8.3f}  {v/TARGET_FPS:>9.4f}")
print(f"  {'max':>5}  {combined.max():>8.3f}  {combined.max()/TARGET_FPS:>9.4f}")

# --- Histogram (ASCII) to spot bimodal distribution ---
print("\nVelocity histogram (m/s) — each * = ~1% of data:")
bins = [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 60.0]
counts, edges = np.histogram(combined, bins=bins)
total = len(combined)
for i in range(len(counts)):
    lo, hi = edges[i], edges[i + 1]
    frac = counts[i] / total
    bar = "*" * int(round(frac * 100))
    print(f"  [{lo:5.1f} – {hi:5.1f}) m/s  {frac*100:5.1f}%  {bar}")

# --- Per-marker table ---
print("\nPer-marker summary (m/s):")
hdr = f"  {'Marker':<15} {'N':>5} {'mean':>7} {'p25':>7} {'p50':>7} {'p75':>7} {'p80':>7} {'p85':>7} {'p90':>7}"
print(hdr)
print("  " + "-" * (len(hdr) - 2))
for m in MARKERS:
    v = np.array(all_v[m])
    p = np.percentile(v, [25, 50, 75, 80, 85, 90])
    print(f"  {m:<15} {len(v):>5} {v.mean():>7.3f} {p[0]:>7.3f} {p[1]:>7.3f} "
          f"{p[2]:>7.3f} {p[3]:>7.3f} {p[4]:>7.3f} {p[5]:>7.3f}")

# --- Fraction of data below candidate thresholds ---
print("\nFraction of observed velocities BELOW candidate thresholds:")
thresholds = [0.30, 0.50, 1.0, 2.0, 3.0, 5.0]
for t in thresholds:
    frac = (combined < t).mean() * 100
    print(f"  < {t:.2f} m/s  → {frac:.1f}% of pairs  "
          f"({'PDF value — rejects majority of real motion!' if t == 0.30 else ''})")

# --- Recommendation ---
print("\n" + "=" * 60)
print("INTERPRETATION")
print("-" * 60)
p75 = np.percentile(combined, 75)
p80 = np.percentile(combined, 80)
p85 = np.percentile(combined, 85)
frac_below_5 = (combined < 5.0).mean() * 100
print(f"  The histogram shows a bimodal distribution:")
print(f"    • Legitimate therapy motion : roughly 0 – 5 m/s ({frac_below_5:.0f}% of pairs)")
print(f"    • ArUco reacquisition jumps : 10 – 60 m/s  (remaining {100-frac_below_5:.0f}%)")
print(f"")
print(f"  p75 = {p75:.2f} m/s | p80 = {p80:.2f} m/s | p85 = {p85:.2f} m/s")
print(f"")
print(f"  PDF used 0.30 m/s — this rejects {(combined<0.30).mean()*100:.0f}% of pairs,")
print(f"  meaning it is far BELOW the typical observed motion speed.")
print(f"")
print(f"  For the velocity-capped interpolation, the cap should be ABOVE")
print(f"  real motion and BELOW the noise (10+ m/s). Suggested range: 2–5 m/s.")
print(f"")
for t, label in [(2.0, "conservative"), (3.0, "recommended"), (5.0, "permissive")]:
    mf = t / TARGET_FPS
    print(f"  {label:<14} {t:.1f} m/s  →  {mf:.4f} m/frame  "
          f"(covers {(combined<t).mean()*100:.0f}% of real motion)")
print(f"\n  Note: FPS assumed = {TARGET_FPS} (TARGET_FPS in data_recording.py)")
