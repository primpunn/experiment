#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/primpunn/librealsense/build/Release')
import numpy as np, cv2, os

SESSION = 'saved_data/2026-05-01_16-30-06'
FX, FY, CX, CY = 914.0, 914.0, 653.0, 347.0

T_world_L515 = np.loadtxt(os.path.join(SESSION, 'T_world_L515.txt'))
T_L515_world = np.linalg.inv(T_world_L515)

ARM_ID_MAP = {0: 'right_wrist', 1: 'right_elbow', 2: 'left_wrist', 3: 'left_elbow'}

aruco_dict   = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
aruco_params = cv2.aruco.DetectorParameters()
detector     = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

frames = sorted(
    [d for d in os.listdir(SESSION) if d.startswith('frame_')],
    key=lambda x: int(x.split('_')[1])
)

print('Frame   ID  Joint           Detected          Projected         Error')
print('-' * 80)

count = 0
for frame in frames:
    frame_dir = os.path.join(SESSION, frame)
    img_path  = os.path.join(frame_dir, 'color_image.png')
    if not os.path.exists(img_path):
        continue
    img  = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        continue
    for i, mid in enumerate(ids.flatten()):
        if mid not in ARM_ID_MAP:
            continue
        joint = ARM_ID_MAP[mid]
        cx_det = corners[i][0][:, 0].mean()
        cy_det = corners[i][0][:, 1].mean()

        pose_path = os.path.join(frame_dir, 'pose_%s_processed.txt' % joint)
        if not os.path.exists(pose_path):
            continue
        T = np.loadtxt(pose_path)
        if np.any(np.isnan(T)):
            continue
        p_cam = T_L515_world @ np.array([T[0,3], T[1,3], T[2,3], 1.0])
        if p_cam[2] < 0.1:
            continue
        u_proj = FX * p_cam[0] / p_cam[2] + CX
        v_proj = FY * p_cam[1] / p_cam[2] + CY
        err = np.sqrt((cx_det - u_proj)**2 + (cy_det - v_proj)**2)
        fi = int(frame.split('_')[1])
        print('%5d  %3d  %-14s  (%6.1f,%6.1f)  (%6.1f,%6.1f)  %7.1f px' % (
            fi, mid, joint, cx_det, cy_det, u_proj, v_proj, err))
        count += 1
    if count >= 30:
        break
