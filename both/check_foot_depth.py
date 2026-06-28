import numpy as np
import matplotlib.pyplot as plt

SESSION = "/home/primpunn/experiment/both/calibration_session/2026-06-16_19-32-19"
FRAME   = "frame_0"

# pointcloud.npy is (N, 6): [X, Y, Z, B, G, R] in world frame (metres)
pc = np.load(f"{SESSION}/{FRAME}/pointcloud.npy")
xyz = pc[:, :3].astype(np.float64)

print(f"Total points : {len(xyz)}")
print(f"X range : {xyz[:,0].min():.3f} .. {xyz[:,0].max():.3f} m")
print(f"Y range : {xyz[:,1].min():.3f} .. {xyz[:,1].max():.3f} m")
print(f"Z range : {xyz[:,2].min():.3f} .. {xyz[:,2].max():.3f} m")

# World-frame Y is vertical (up). Feet are near the floor → low Y values.
# Adjust Y_FOOT_MAX if your floor level differs.
Y_FOOT_MAX = xyz[:, 1].min() + 0.3   # bottom 30 cm of the scene
foot_pts = xyz[xyz[:, 1] < Y_FOOT_MAX]

print(f"\nFoot region (Y < {Y_FOOT_MAX:.3f} m): {len(foot_pts)} points")
if len(foot_pts) > 0:
    print(f"  X : {foot_pts[:,0].min():.3f} .. {foot_pts[:,0].max():.3f} m")
    print(f"  Y : {foot_pts[:,1].min():.3f} .. {foot_pts[:,1].max():.3f} m")
    print(f"  Z : {foot_pts[:,2].min():.3f} .. {foot_pts[:,2].max():.3f} m")
else:
    print("  No points found in foot region!")

# ── Plot: side view (X vs Y) so you can see how low the cloud goes ───────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sub = xyz[np.random.choice(len(xyz), min(len(xyz), 10000), replace=False)]

axes[0].scatter(sub[:, 0], sub[:, 1], s=1, c='lightgray', alpha=0.4, label='all')
if len(foot_pts) > 0:
    axes[0].scatter(foot_pts[:, 0], foot_pts[:, 1], s=2, c='red', alpha=0.8, label='foot region')
axes[0].set_xlabel('X (m)'); axes[0].set_ylabel('Y (m)')
axes[0].set_title('Side view (X vs Y)')
axes[0].legend(markerscale=5)

axes[1].scatter(sub[:, 0], sub[:, 2], s=1, c='lightgray', alpha=0.4, label='all')
if len(foot_pts) > 0:
    axes[1].scatter(foot_pts[:, 0], foot_pts[:, 2], s=2, c='red', alpha=0.8, label='foot region')
axes[1].set_xlabel('X (m)'); axes[1].set_ylabel('Z (m)')
axes[1].set_title('Top view (X vs Z)')
axes[1].legend(markerscale=5)

plt.tight_layout()
plt.savefig("foot_depth_L515.png", dpi=150, bbox_inches="tight")
print("\nSaved foot_depth_L515.png")
