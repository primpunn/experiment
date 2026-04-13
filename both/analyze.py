import numpy as np
from pathlib import Path
base = Path('saved_data')
positions = []
for i in range(30):
    T = np.loadtxt(base / f'frame_{i}/pose.txt')
    positions.append(T[:3, 3])
pts = np.array(positions)
print('HEAD TRAJECTORY:')
for i,p in enumerate(pts):
    print(f'  f{i:02d}: X={p[0]:+.4f} Y={p[1]:+.4f} Z={p[2]:+.4f}')
print(f'Range X={pts[:,0].ptp():.4f} Y={pts[:,1].ptp():.4f} Z={pts[:,2].ptp():.4f} m')
print('\nLEFT WRIST detected frames:')
for i in range(30):
    T = np.loadtxt(base / f'frame_{i}/pose_left_wrist.txt')
    if not np.any(np.isnan(T)):
        print(f'  f{i:02d}: X={T[0,3]:+.4f} Y={T[1,3]:+.4f} Z={T[2,3]:+.4f}')
T_L515 = np.loadtxt(base / 'T_world_L515.txt')
R = T_L515[:3,:3]
t = T_L515[:3,3]
print(f'\nL515 det={np.linalg.det(R):.6f}  pos={t.round(4)} m  dist={np.linalg.norm(t):.4f}m')
