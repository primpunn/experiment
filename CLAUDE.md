# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a robotics research project for a 1st paper on **dual-arm physical therapy assistance** — specifically, a robot performing calf-stretching on a human patient using two arms (one lifts the leg at the ankle, one presses the foot). The pipeline spans data collection, force optimization, simulation, and real-hardware control.

## Environment

- Python 3.13 (Miniconda), Ubuntu 22.04, kernel 6.8
- **Critical**: The pip `pyrealsense2` package does **not** support the L515 camera. All L515 scripts must use the locally built librealsense v2.54.2:
  ```python
  import sys
  sys.path.insert(0, '/home/primpunn/Desktop/claudetry/librealsense/build/release')
  ```
- The **D435i** works fine with standard pip `pyrealsense2`.
- udev rules: `/etc/udev/rules.d/99-realsense-libusb.rules`

## Running the scripts

```bash
# Test camera connection
python Depth/test_camera.py

# Live pose estimation (D435i only, YOLOv8 + Madgwick IMU)
python Depth/pose_estimation.py

# Full data recording (L515 + D435i + MediaPipe)
python Depth/data_recording.py -s -o ./Depth/saved_data
python Depth/data_recording.py -s -v -o ./Depth/saved_data --total_frame 5000
python Depth/data_recording.py -s -o ./Depth/saved_data --l515_serial ABC123 --d435i_serial XYZ456

# L515 point cloud capture (claudetry subfolder)
python claudetry/l515_pointcloud.py
```

## Code Architecture

### Data Collection (`Depth/`)

**Two-camera pipeline:**
- **L515** (LiDAR) → RGB-D scene capture: `color_image.jpg` + `depth_image.png`
- **D435i** (depth + IMU) → human pose via MediaPipe + IMU orientation: `pose.txt` + `*_arm_keypoints.txt`

**IMU fusion:** `data_recording.py` uses a **complementary filter** (accel + gyro SLERP blend, α=0.98); `pose_estimation.py` uses a **Madgwick filter** (β=0.1). Both output a 4×4 rotation matrix — translation is zero (IMU does not track position).

**Keypoints extracted (MediaPipe, 5 joints per arm):** elbow, wrist, index_tip, pinky_tip, thumb_tip — stored as (5,3) world coordinates in metres.

**Per-frame output structure:**
```
saved_data/frame_<N>/
├── color_image.jpg          # RGB from L515 (960×540)
├── depth_image.png          # uint16 depth from L515 (÷4 metric)
├── pose.txt                 # 4×4 rotation matrix from D435i IMU
├── right_arm_keypoints.txt  # (5, 3) right arm 3D world coords [m]
└── left_arm_keypoints.txt   # (5, 3) left arm 3D world coords [m]
```

Frames are buffered in RAM and flushed to disk in parallel via `ThreadPoolExecutor` after recording completes.

### Research Baseline (`Baseline Idea` document)

**Baseline 1 — Force Allocation Optimization (FAO):**
- Models the leg as a rigid shank–foot link with two contact points (ankle = Arm1, toe = Arm2)
- Solves a QP (via `cvxpy`) for optimal contact forces under friction cone + force limit constraints
- Two phases: *Lift* (Arm2=0, find F1_lift) and *Press* (Arm2 fixed, find F1_hold and ΔF1)
- Optional MuJoCo 3.x validation of "sag" before/after compensation

**Baseline 2 — Basic Control + Collision-Free Trajectory:**
- Uses FAO force references as feed-forward in simple force–position control
- Dual-arm trajectory planning via MoveIt 2 to avoid self-collision
- Tested in simulation (MuJoCo or Gazebo + ROS 2 Humble) then on real hardware

### Hardware in Use

| Camera | SDK | Use |
|--------|-----|-----|
| Intel RealSense L515 | librealsense v2.54.2 (local build, RSUSB) | RGB-D scene capture |
| Intel RealSense D435i | pip `pyrealsense2` | Human pose + IMU |

L515 support was dropped in SDK v2.55.1+. The local build at `/home/primpunn/Desktop/claudetry/librealsense/` was compiled with `FORCE_RSUSB_BACKEND=true` and `BUILD_PYTHON_BINDINGS=true`.

## GitHub Workflow

After **any** change to files in this folder (`/home/primpunn/Desktop/1st paper/`), always git push to:

**`https://github.com/primpunn/experiment.git`**

```bash
cd "/home/primpunn/Desktop/1st paper"
git add -A
git commit -m "your message"
git push origin main
```

If the repo is not yet initialized locally:
```bash
cd "/home/primpunn/Desktop/1st paper"
git init
git remote add origin https://github.com/primpunn/experiment.git
git add -A
git commit -m "initial commit"
git push -u origin main
```
