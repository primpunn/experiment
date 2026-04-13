# Lab Notebook — 1st Paper Progress

## Week of Mar 14–16, 2026 — L515 Point Cloud Pipeline (`claudetry/`)

**Goal:** Get the Intel RealSense L515 working and capture RGB-D point clouds.

**What was done:**
- Discovered that pip `pyrealsense2` does NOT support L515 → had to build librealsense v2.54.2 from source with `FORCE_RSUSB_BACKEND=true` (kernel 6.8 issue)
- Built Python bindings at `claudetry/librealsense/build/release/`
- **Mar 14:** First L515 point cloud captures saved as `.ply` files (`claudetry/pointclouds/`)
- **Mar 15:** `l515_pointcloud.py` — full capture script: depth (1024×768) + color (1920×1080), High Accuracy preset, statistical outlier removal, saves color PNG + raw depth NPY + colorized depth PNG + RGB point cloud PLY
- **Mar 15:** Added explicit RGB texture mapping onto point cloud vertices
- **Mar 16:** `sam_segment.py` — added SAM (Segment Anything Model, `sam_b.pt`) pipeline to isolate the human body from the point cloud background

**Output structure:**
```
claudetry/output/<date>/
├── color/            # RGB .png
├── depth/            # Raw uint16 .npy
├── depth_colormap/   # Colorized depth .png
└── rawpointclouds/   # RGB-colored .ply
```

---

## Week of Mar 23–25, 2026 — Pose Estimation + Full Data Recording (`Depth/`)

**Goal:** Build the full data collection pipeline for the paper — RGB-D scene + human arm keypoints + camera orientation per frame.

**What was done:**

**Mar 23:**
- `test_camera.py` — basic D435i test script (color + depth streams, center pixel depth display). Fixed black screen on startup by adding 60-frame warm-up. Fixed window sizing.

**Mar 24:**
- `pose_estimation.py` — live human pose estimation on D435i:
  - Started with MediaPipe Pose → switched to **YOLOv8n-pose** (Python 3.13 compatible)
  - Projects 2D keypoints to 3D using depth + camera intrinsics
  - Computes forearm midpoints (elbow + wrist average)
  - IMU fusion: tried callback-based approach → fixed by polling IMU from frameset instead
  - Added fallback if IMU streams unavailable
  - Final filter: **Madgwick filter** (β=0.1) for quaternion from accel + gyro
  - Displays: annotated video | joint 3D positions (m) | camera quaternion live

**Mar 25:**
- `data_recording.py` — full synchronized multi-camera recording:
  - **L515** → RGB-D scene (color_image.jpg + depth_image.png)
  - **D435i** → **MediaPipe Pose** arm keypoints + **IMU complementary filter** (α=0.98) orientation
  - Saves per frame: color, depth, pose (4×4 matrix), right_arm_keypoints (5×3), left_arm_keypoints (5×3)
  - IMU uses callback-based pipeline (separate from color pipeline) to avoid 5-second timeout bug
  - Parallel disk I/O via `ThreadPoolExecutor` for fast saving
  - Optional Open3D live point cloud visualization
  - **Data recorded:** `Depth/saved_data/` — 1000+ frames captured (Mar 25)

**Keypoints extracted (5 joints per arm):** elbow, wrist, index_tip, pinky_tip, thumb_tip

---

## Research Direction (from Baseline Idea document)

**Problem:** Dual-arm robot performing calf-stretching on a patient.
- Arm1 lifts leg at ankle
- Arm2 presses foot to stretch the calf
- When Arm2 pushes down, leg position drops → Arm1 must compensate

**Baseline 1 — Force Allocation Optimization (FAO):**
- Model: rigid shank–foot link, two contact points (ankle + toe), friction cone constraints
- Solve QP (`cvxpy`) for optimal forces in two phases: Lift (Arm2=0) and Press (Arm2 fixed)
- Validate in MuJoCo 3.x

**Baseline 2 — Control + Trajectory:**
- Use FAO force references as feed-forward in simple force–position control
- Dual-arm collision-free trajectory planning via MoveIt 2
- Test in simulation (MuJoCo / Gazebo + ROS 2 Humble) → then real hardware
