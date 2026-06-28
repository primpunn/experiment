# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Robotics research project for a **1st paper on dual-arm physical therapy assistance** — a robot performing calf-stretching on a human patient using two arms (one lifts the ankle, one presses the foot). This repo currently covers the **data collection pipeline** (`both/`). Downstream work (force optimization, simulation, control) is planned but not yet in this repo.

## Environment

- **Conda environment:** `massage` (Python 3.13, Ubuntu 22.04)
- **Run all scripts with:** `conda activate massage`
- **Critical:** The pip `pyrealsense2` does **not** support L515. The locally built librealsense at `/home/primpunn/librealsense/build/Release` must be on `sys.path` first. `both/data_recording.py` does this automatically via `sys.path.insert(0, '/home/primpunn/librealsense/build/Release')`.
- udev rules: `/etc/udev/rules.d/99-realsense-libusb.rules`

## Running the Scripts

```bash
conda activate massage
cd /home/primpunn/experiment/both

# Record data (default 200 frames, live preview)
python data_recording.py -o ./saved_data

# Record with custom frame count, no preview
python data_recording.py -o ./saved_data --total_frames 1000 --no-preview

# Quick post-hoc inspection of a saved session
python analyze.py   # edit base path in script to point at target session
```

### Startup sequence for `data_recording.py`

1. Both cameras warm up for 2 seconds (60-frame discard)
2. **Init phase**: live preview windows open — position D435i (head) so **ArUco ID10** is visible in **both** cameras simultaneously, then press `S` or ENTER. The script averages 60 valid frames to compute static transforms.
3. **Recording phase**: perform the task; press `Q` or `Ctrl+C` to stop. All frames are buffered in RAM then flushed to disk via `ThreadPoolExecutor`.

## Code Architecture (`both/`)

### `data_recording.py` — the main pipeline

**Two-camera setup:**
- **L515** (floor, static) → RGB scene + LiDAR point cloud
- **D435i** (head-mounted, physically rotated 90° CW) → arm pose tracking via ArUco

**ArUco marker assignment (`DICT_4X4_100`):**

| ID | Role | Size |
|----|------|------|
| 0 | right wrist | 3 cm |
| 1 | right elbow | 3 cm |
| 2 | left wrist | 3 cm |
| 3 | left elbow | 3 cm |
| 10 | world frame origin | 10 cm |
| 11,13,14,16,17,20,21 | extra floor markers (re-localization) | 10 cm |

**Transform convention:** `T_A_B` transforms points FROM frame B TO frame A.

**Per-frame pose computation:**
```
T_world_L515  — fixed, computed once at init from L515 seeing ID10
T_world_head  — updated every frame via any visible floor marker:
                T_world_head = T_world_IDk @ inv(T_head_IDk)
T_world_arm   = T_world_head @ T_head_arm   (for each arm marker in D435i view)
```

**D435i rotation correction:** physical 90° CW mount → all D435i images rotated 90° CCW before ArUco detection. Intrinsics adjusted accordingly (`fx↔fy`, `cx↔cy` with axis flip) via `rotate_intrinsics_90ccw()`.

**Point cloud:** L515 depth back-projected to camera frame, transformed to world frame via `T_world_L515`, randomly downsampled to `PC_NUM_POINTS = 8192`.

### Output structure per session

```
saved_data/<YYYY-MM-DD_HH-MM-SS>/
├── T_world_L515.txt        # L515 camera pose in world (4×4), fixed for session
├── T_world_ID<k>.txt       # each floor marker pose in world (4×4), fixed
└── frame_<N>/
    ├── color_image.png     # L515 BGR, 1280×720
    ├── depth_image.png     # D435i depth uint16, 720×1280 (after 90° CCW rotation)
    ├── pointcloud.npy      # (8192, 6) float32: [X,Y,Z,B,G,R] in world frame
    ├── pose.txt            # D435i camera (head) pose in world (4×4, full 6-DOF)
    ├── pose_right_wrist.txt  # ArUco ID0 in world (4×4), NaN matrix if not detected
    ├── pose_right_elbow.txt  # ArUco ID1 in world (4×4)
    ├── pose_left_wrist.txt   # ArUco ID2 in world (4×4)
    └── pose_left_elbow.txt   # ArUco ID3 in world (4×4)
```

**Important:** `pose.txt` stores a **full 6-DOF** camera pose (translation is the real head position in world frame, not zero). This is different from any IMU-only approach.

### Key classes and functions

| Symbol | File | Purpose |
|--------|------|---------|
| `DualCameraRecorder` | `data_recording.py:259` | Main class — camera init, per-frame recording, disk flush |
| `initialize_transforms()` | `data_recording.py:350` | Init phase: compute and save static `T_world_L515` and floor marker transforms |
| `_update_head_pose()` | `data_recording.py:519` | Re-localizes D435i head pose from any visible floor marker each frame |
| `record_frame()` | `data_recording.py:537` | Captures one frame from both cameras, runs ArUco detection, buffers result |
| `depth_to_pointcloud_cam()` | `data_recording.py:219` | Back-projects L515 aligned depth to (N,6) XYZbgr in camera frame |
| `detect_markers()` | `data_recording.py:148` | OpenCV ArUco detection → `{id: T_cam_marker (4×4)}` |
| `rotate_intrinsics_90ccw()` | `data_recording.py:108` | Adjusts `rs.intrinsics` after 90° CCW image rotation |

## GitHub Workflow

```bash
cd /home/primpunn/experiment
git add -A
git commit -m "your message"
git push origin main
```

Remote: `https://github.com/primpunn/experiment.git`

The `.gitignore` excludes all `saved_data/` directories, `.npy`, `.png`, `.jpg`, `.ply` files (data is not tracked in git).
