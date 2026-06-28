#!/usr/bin/env python3
"""
Phase A shape calibration: fit SMPL-X shape parameters β from a 360° rotation recording.

The therapist stands ~0.8 m from the L515 camera and rotates slowly for ~10 seconds.
data_recording.py captures this session as a normal session directory.  This script:

  Step 1  For each frame, run MediaPipe Selfie Segmentation on color_image.png to get
          a person mask, project that frame's pointcloud.npy (world-frame XYZ) back
          to L515 pixel coordinates (via T_world_L515.txt and L515_intrinsics.txt),
          keep only points whose pixel falls inside the person mask, voxel-downsample
          (0.01 m grid), and merge all frames into a single combined point cloud.

  Step 2  Statistical outlier removal (scipy KD-tree, nb_neighbors=20, std_ratio=2.0)
          to clean up residual segmentation-edge noise.

  Step 3  Align the body cloud to the SMPL-X canonical frame (centroid → origin,
          dominant XZ direction → +X via Y-axis rotation) then fit shape β (10
          components) by minimising a one-sided Chamfer distance using Adam.

  Step 4  Save smplx_beta.npy; optionally visualise the fit with open3d.

Requires numpy, scipy, torch, smplx, opencv-python, and mediapipe.
open3d is optional — only needed for --visualise.

The saved smplx_beta.npy should be copied into every future massage-session directory.
interpolate_mosh_pipeline.py will load it automatically (skipping its own Step 3a) as
long as --recalibrate is not passed.

Usage:
    conda activate massage
    python estimate_shape.py <calibration_session_dir>
    python estimate_shape.py <calibration_session_dir> --gender female --visualise
    python estimate_shape.py <calibration_session_dir> --dry-run
    python estimate_shape.py <calibration_session_dir> --smplx-model-dir ~/models/smplx
"""

import argparse
import math
import multiprocessing
import os
import sys
from typing import Optional

import numpy as np
import cv2
from scipy.spatial import cKDTree

# ── Required dependencies (core pipeline) ────────────────────────────────────
try:
    import torch
    import smplx
    import mediapipe as mp
    DEPS_AVAILABLE = True
except ImportError as e:
    DEPS_AVAILABLE = False
    missing = str(e)

# ── Optional dependency (--visualise only) ────────────────────────────────────
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False

# ── Optional dependency (debug raw-vs-segmented point cloud plot only) ────────
try:
    import matplotlib
    matplotlib.use('Agg')   # headless-safe backend
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# ── Person-segmentation constants ─────────────────────────────────────────────
SEGMENTATION_THRESHOLD = 0.7   # default MediaPipe mask value above which a pixel is "person"
SEGMENTATION_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'models', 'selfie_segmenter_landscape.tflite')

# ── Pose-bbox constants (extends selfie segmentation to cover legs/feet) ──────
# Selfie Segmentation often clips the lower legs/feet. We additionally run
# MediaPipe Pose and OR in a bounding box around all detected landmarks
# (expanded with margins, esp. downward for feet) so the combined mask keeps
# the full head-to-toe body even where segmentation confidence is low.
POSE_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'models', 'pose_landmarker_lite.task')
POSE_MIN_VISIBILITY     = 0.5   # ignore landmarks below this visibility when sizing the bbox
POSE_BBOX_MARGIN_SIDE   = 0.10  # extra fraction of bbox width added to left/right
POSE_BBOX_MARGIN_TOP    = 0.10  # extra fraction of bbox height added above
POSE_BBOX_MARGIN_BOTTOM = 0.20  # extra fraction of bbox height added below (covers feet)

# ── Point-cloud pre-processing ────────────────────────────────────────────────
VOXEL_SIZE           = 0.01   # m — per-frame voxel grid size before merging
DEPTH_RANGE_MIN      = 0.3    # m — discard points closer to the L515 than this
DEPTH_RANGE_MAX      = 2.5    # m — discard points farther from the L515 than this
FOOT_PIXEL_MARGIN    = 60     # px — allow points projecting this far outside the image border

# ── Shape optimisation constants ──────────────────────────────────────────────
CHAMFER_MAX_POINTS   = 5000   # max observed points sampled per optimisation step
SHAPE_LR             = 0.02   # Adam learning rate for β
SHAPE_ITERS          = 1000    # maximum optimisation iterations, default=500
SHAPE_CONV_THRESHOLD = 0.01   # m — warn if final RMS Chamfer distance exceeds this, default=0.05


# =============================================================================
# Step 1 — Per-frame person segmentation, projection, and merge
# =============================================================================

def voxel_downsample(pts: np.ndarray, voxel_size: float) -> np.ndarray:
    """
    Grid-based voxel downsampling in pure numpy.

    Assigns each point to a voxel cell and keeps one representative point
    per occupied cell (the first point encountered, after sorting by key).
    Equivalent to open3d.voxel_down_sample for the purpose of density reduction.
    """
    voxel_idx = np.floor(pts / voxel_size).astype(np.int64)
    min_idx   = voxel_idx.min(axis=0)
    voxel_idx -= min_idx
    max_idx   = voxel_idx.max(axis=0) + 1
    # Encode 3-D voxel coordinates as a single integer key
    keys      = (voxel_idx[:, 0] * int(max_idx[1]) * int(max_idx[2]) +
                 voxel_idx[:, 1] * int(max_idx[2]) +
                 voxel_idx[:, 2])
    _, first  = np.unique(keys, return_index=True)
    return pts[first]


def load_intrinsics(session_dir: str) -> dict:
    """
    Load L515 color intrinsics saved by data_recording.py.

    Returns dict with keys: fx, fy, ppx, ppy, width, height.
    Raises FileNotFoundError if the session was recorded before intrinsics
    saving was added (re-record the calibration session).
    """
    path = os.path.join(session_dir, 'L515_intrinsics.txt')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found — re-record this session with the current "
            "data_recording.py so L515 intrinsics are saved.")
    fx, fy, ppx, ppy, width, height = np.loadtxt(path)
    return dict(fx=fx, fy=fy, ppx=ppx, ppy=ppy,
                width=int(width), height=int(height))


def project_world_to_pixel(pts_world: np.ndarray,
                            T_world_L515: np.ndarray,
                            intr: dict) -> tuple:
    """
    Project world-frame points back to L515 color-image pixel coordinates.

    Inverse of the back-projection done in data_recording.depth_to_pointcloud_cam
    (pure pinhole, no distortion — consistent with how the cloud was generated).

    Also enforces a camera-frame depth range [DEPTH_RANGE_MIN, DEPTH_RANGE_MAX]
    (computed here, before any world transform was applied to get back to this
    camera-frame Z) to reject background points that survived segmentation.

    Returns
    -------
    u, v  : np.ndarray (N,) int   — pixel coordinates (may be out of bounds)
    valid : np.ndarray (N,) bool  — True where the point projects in front of
            the camera (Z > 0), within the image bounds, and within
            [DEPTH_RANGE_MIN, DEPTH_RANGE_MAX] m of the camera
    """
    T_L515_world = np.linalg.inv(T_world_L515)
    pts_h    = np.hstack([pts_world, np.ones((len(pts_world), 1))])
    pts_cam  = (T_L515_world @ pts_h.T).T[:, :3]

    z = pts_cam[:, 2]
    with np.errstate(divide='ignore', invalid='ignore'):
        u = pts_cam[:, 0] * intr['fx'] / z + intr['ppx']
        v = pts_cam[:, 1] * intr['fy'] / z + intr['ppy']

    u_i, v_i = u.round().astype(np.int64), v.round().astype(np.int64)
    valid = ((z >= DEPTH_RANGE_MIN) & (z <= DEPTH_RANGE_MAX) &
             (u_i >= 0) & (u_i < intr['width']) &
             (v_i >= -FOOT_PIXEL_MARGIN) & (v_i < intr['height'] + FOOT_PIXEL_MARGIN))
    return u_i, v_i, valid


def segment_person_mask(color_bgr: np.ndarray, segmenter,
                         seg_threshold: float = SEGMENTATION_THRESHOLD) -> np.ndarray:
    """
    Run MediaPipe Selfie Segmentation on a BGR image.

    Returns a boolean (H, W) mask, True where the pixel belongs to a person.
    """
    rgb      = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result   = segmenter.segment(mp_image)
    confidence_mask = result.confidence_masks[0].numpy_view()
    return confidence_mask.squeeze(-1) > seg_threshold


def _save_mask_debug(color_bgr: np.ndarray, mask: np.ndarray,
                     pose_mask: np.ndarray, bbox, frame_dir: str):
    """
    Save mask_debug.png into frame_dir showing how the two masks combine:
      - green   : pixels kept by Selfie Segmentation
      - blue    : pixels added by the Pose bbox (not in segmentation mask)
      - yellow  : pose bounding box border
    Useful for spotting where segmentation clips the body vs where Pose fills in.
    """
    overlay = color_bgr.copy().astype(np.float32)
    overlay[mask]            = overlay[mask]            * 0.5 + np.array([  0, 180,   0], np.float32) * 0.5
    overlay[pose_mask & ~mask] = overlay[pose_mask & ~mask] * 0.5 + np.array([180,   0,   0], np.float32) * 0.5
    overlay = overlay.clip(0, 255).astype(np.uint8)
    if bbox is not None:
        x_min, y_min, x_max, y_max = bbox
        cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
    cv2.imwrite(os.path.join(frame_dir, 'mask_debug.png'), overlay)


def compute_pose_bbox(color_bgr: np.ndarray, landmarker) -> Optional[tuple]:
    """
    Run MediaPipe Pose on a BGR image and return a pixel bounding box that
    covers the whole detected body, expanded with margins (extra downward
    margin for feet).

    This complements Selfie Segmentation, which sometimes cuts off the legs
    and feet: ORing this bbox into the person mask ensures the final body
    cloud covers head-to-toe even where segmentation confidence is low.

    Returns (x_min, y_min, x_max, y_max) in pixel coordinates, clamped to the
    image bounds, or None if pose detection fails or finds no landmarks
    (caller should then fall back to segmentation-only behaviour).
    """
    try:
        h, w = color_bgr.shape[:2]
        rgb      = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result   = landmarker.detect(mp_image)

        if not result.pose_landmarks:
            return None

        landmarks = result.pose_landmarks[0]
        visible = [lm for lm in landmarks if lm.visibility >= POSE_MIN_VISIBILITY]
        if not visible:
            visible = landmarks

        def _to_pixel(lm):
            return lm.x * w, lm.y * h

        xs, ys = zip(*(_to_pixel(lm) for lm in visible))
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        bbox_w = x_max - x_min
        bbox_h = y_max - y_min
        x_min -= bbox_w * POSE_BBOX_MARGIN_SIDE
        x_max += bbox_w * POSE_BBOX_MARGIN_SIDE
        y_min -= bbox_h * POSE_BBOX_MARGIN_TOP
        y_max += bbox_h * POSE_BBOX_MARGIN_BOTTOM

        x_min = int(max(0, math.floor(x_min)))
        y_min = int(max(0, math.floor(y_min)))
        x_max = int(min(w - 1, math.ceil(x_max)))
        y_max = int(min(h - 1, math.ceil(y_max)))

        return (x_min, y_min, x_max, y_max)
    except Exception as e:
        print(f"  WARNING: pose detection failed ({e}) — "
              "falling back to segmentation-only mask for this frame.")
        return None


def load_segment_and_merge(session_dir: str,
                            seg_threshold: float = SEGMENTATION_THRESHOLD,
                            mask_debug: bool = False) -> tuple:
    """
    For each frame: segment the person in color_image.png, project that frame's
    pointcloud.npy (world XYZ) back to pixel coordinates, keep only points whose
    pixel falls inside the person mask and within [DEPTH_RANGE_MIN, DEPTH_RANGE_MAX]
    m of the L515, voxel-downsample, and merge all frames.

    Returns
    -------
    merged_pts  : np.ndarray, shape (M, 3)  — world-frame XYZ, person-only
    n_frames    : int  — number of frames successfully processed
    n_raw       : int  — total points across all frames before segmentation
    raw_pts     : np.ndarray, shape (n_raw, 3) — world-frame XYZ, unsegmented
                  (concatenation of every frame's raw pointcloud.npy, for
                  debug comparison against merged_pts)
    """
    T_world_L515 = np.loadtxt(os.path.join(session_dir, 'T_world_L515.txt'))
    intr         = load_intrinsics(session_dir)

    frames = sorted(
        [d for d in os.listdir(session_dir) if d.startswith('frame_')],
        key=lambda x: int(x.split('_')[1])
    )

    chunks     = []
    raw_chunks = []
    n_loaded = 0
    n_raw    = 0

    seg_options = mp.tasks.vision.ImageSegmenterOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=SEGMENTATION_MODEL_PATH),
        output_confidence_masks=True)

    pose_options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=POSE_MODEL_PATH),
        num_poses=1,
        min_pose_detection_confidence=0.5)

    with mp.tasks.vision.ImageSegmenter.create_from_options(seg_options) as segmenter, \
         mp.tasks.vision.PoseLandmarker.create_from_options(pose_options) as landmarker:

        for frame in frames:
            frame_dir = os.path.join(session_dir, frame)
            pc_path    = os.path.join(frame_dir, 'pointcloud.npy')
            color_path = os.path.join(frame_dir, 'color_image.png')
            if not (os.path.exists(pc_path) and os.path.exists(color_path)):
                continue
            try:
                pc        = np.load(pc_path)
                color_bgr = cv2.imread(color_path)
            except Exception:
                continue
            if pc.shape[0] == 0 or color_bgr is None:
                continue

            n_raw += pc.shape[0]
            pts_world = pc[:, :3].astype(np.float64)
            raw_chunks.append(pts_world)

            mask = segment_person_mask(color_bgr, segmenter, seg_threshold)

            # Selfie Segmentation alone can clip legs/feet — OR in a Pose
            # bounding box (with margins) to extend coverage head-to-toe.
            bbox = compute_pose_bbox(color_bgr, landmarker)
            pose_mask = np.zeros_like(mask, dtype=bool)
            if bbox is not None:
                x_min, y_min, x_max, y_max = bbox
                pose_mask[y_min:y_max + 1, x_min:x_max + 1] = True
            combined_mask = mask | pose_mask

            if mask_debug:
                _save_mask_debug(color_bgr, mask, pose_mask, bbox, frame_dir)

            u, v, valid = project_world_to_pixel(pts_world, T_world_L515, intr)

            v_clamped = np.clip(v[valid], 0, intr['height'] - 1)
            u_clamped = np.clip(u[valid], 0, intr['width']  - 1)
            person = np.zeros(len(pts_world), dtype=bool)
            person[valid] = combined_mask[v_clamped, u_clamped]

            pts_person = pts_world[person]
            if len(pts_person) == 0:
                continue

            chunks.append(voxel_downsample(pts_person, VOXEL_SIZE))
            n_loaded += 1

    if not chunks:
        raise RuntimeError(
            f"No person points found under {session_dir} — "
            "check that the therapist is visible in color_image.png frames.")

    raw_pts = np.concatenate(raw_chunks, axis=0) if raw_chunks else np.empty((0, 3))
    return np.concatenate(chunks, axis=0), n_loaded, n_raw, raw_pts


# =============================================================================
# Step 2 — Outlier removal (residual segmentation-edge noise)
# =============================================================================

def filter_statistical_outliers(pts: np.ndarray,
                                  nb_neighbors: int = 20,
                                  std_ratio: float = 2.0) -> np.ndarray:
    """
    Remove floating noise via scipy KD-tree statistical outlier removal.

    For each point, computes the mean distance to its nb_neighbors nearest
    neighbours.  Points whose mean distance exceeds
        global_mean + std_ratio * global_std
    are discarded.  Matches open3d.remove_statistical_outlier behaviour.
    """
    tree      = cKDTree(pts)
    dists, _  = tree.query(pts, k=nb_neighbors + 1)   # k+1 because index 0 = self
    mean_dist = dists[:, 1:].mean(axis=1)              # exclude self-distance (0)
    threshold = mean_dist.mean() + std_ratio * mean_dist.std()
    return pts[mean_dist < threshold]


def isolate_body(pts: np.ndarray) -> np.ndarray:
    """
    Apply statistical outlier removal to the segmented body cloud.
    Returns the cleaned body point cloud.
    """
    if len(pts) < 100:
        print("  WARNING: very few points after segmentation — "
              "skipping statistical outlier removal.")
        return pts

    pts_clean = filter_statistical_outliers(pts)
    print(f"  After outlier removal        : {len(pts_clean):>10,} points")
    return pts_clean


# =============================================================================
# Step 3 — SMPL-X shape fitting via Chamfer distance
# =============================================================================

def align_to_canonical(pts: np.ndarray) -> np.ndarray:
    """
    Align the body point cloud to the SMPL-X canonical frame:

      1. Translate so the cloud centroid sits at the origin.
      2. Rotate around the Y axis so that the dominant horizontal spread direction
         (largest eigenvector of the XZ covariance matrix) aligns with +X.
         This places the shoulder axis along ±X and the facing direction along ±Z,
         matching the SMPL-X canonical pose (global_orient=0 → facing +Z).

    For a 360° rotation recording the XZ distribution is approximately circular,
    so the PCA direction is a best-effort estimate of the shoulder axis; the fit
    is not sensitive to small misalignments because β only captures shape, not pose.

    Returns the aligned point cloud as (N, 3) float64.
    """
    # ── 1. Centre ─────────────────────────────────────────────────────────────
    centroid = pts.mean(axis=0)
    pts_c    = pts - centroid

    # ── 2. PCA on XZ plane → dominant direction ───────────────────────────────
    xz       = pts_c[:, [0, 2]]                          # (N, 2)
    cov      = (xz.T @ xz) / max(len(xz) - 1, 1)
    _, evecs = np.linalg.eigh(cov)                       # ascending eigenvalues
    dominant = evecs[:, -1]                              # [dx, dz], unit vector

    # ── 3. Y-axis rotation: bring dominant to +X ─────────────────────────────
    # R_y(α) where α = atan2(dz, dx).
    # Proof: R_y(α) @ [cos α, 0, sin α]ᵀ = [1, 0, 0]ᵀ ✓
    alpha = np.arctan2(dominant[1], dominant[0])
    c, s  = np.cos(alpha), np.sin(alpha)
    R_y   = np.array([[c,  0.0, s  ],
                      [0.0, 1.0, 0.0],
                      [-s,  0.0, c  ]])

    return pts_c @ R_y.T                                 # (N, 3)


def chamfer_loss_one_sided(pts_obs: 'torch.Tensor',
                            verts: 'torch.Tensor') -> 'torch.Tensor':
    """
    One-sided Chamfer distance: observed points → mesh vertices.

    For each observed point, finds the nearest SMPL-X vertex (used as a surface
    proxy) and returns the mean squared distance across all observed points.

    pts_obs : (N, 3)  — subsampled observed body points
    verts   : (V, 3)  — SMPL-X mesh vertices for the current β
    Returns : scalar tensor in m² (differentiable w.r.t. verts)
    """
    dists2 = torch.cdist(pts_obs, verts) ** 2    # (N, V)
    return dists2.min(dim=1).values.mean()


def fit_shape(pts_body: np.ndarray, smplx_model_dir: str,
              gender: str, device: 'torch.device') -> tuple:
    """
    Fit SMPL-X shape β and global_orient to the aligned body point cloud via
    Adam optimisation.

    Minimises one-sided Chamfer distance (observed → mesh vertices) using
    CHAMFER_MAX_POINTS random points per iteration for computational efficiency.
    global_orient (3 values) is optimised jointly with β so the canonical
    A-pose mesh can rotate to match the person's actual standing orientation
    (e.g. arms at sides rather than A-pose). body_pose and transl are held
    fixed at zero throughout.

    Early stop: if loss does not improve by more than 1e-5 over 50 consecutive
    iterations, optimisation halts and prints "Converged early at iter N".

    Returns
    -------
    beta_numpy          : np.ndarray, shape (10,)
    global_orient_numpy : np.ndarray, shape (3,)
    final_loss_m        : float  — final RMS Chamfer distance in metres (√mean_sq_dist)
    """
    model = smplx.create(
        smplx_model_dir, model_type='smplx', gender=gender,
        use_pca=False, num_betas=10, batch_size=1
    ).to(device)

    pts_t = torch.tensor(pts_body, dtype=torch.float32, device=device)

    beta = torch.zeros(1, 10, dtype=torch.float32,
                       device=device, requires_grad=True)
    global_orient = torch.zeros(1, 3, dtype=torch.float32,
                                 device=device, requires_grad=True)
    optimizer = torch.optim.Adam([beta, global_orient], lr=SHAPE_LR)

    best_loss  = float('inf')
    no_improve = 0
    final_loss = float('inf')

    print(f"  Optimising β + global_orient over {SHAPE_ITERS} max iters "
          f"(lr={SHAPE_LR}, subsample={CHAMFER_MAX_POINTS} pts/iter)...")

    for it in range(SHAPE_ITERS):
        optimizer.zero_grad()

        out   = model(betas=beta,
                      global_orient=global_orient,
                      body_pose=torch.zeros(1, 63, device=device),
                      transl=torch.zeros(1, 3, device=device),
                      return_verts=True)
        verts = out.vertices[0]                          # (V, 3)

        # Random subsample per iteration — improves coverage, limits memory
        if pts_t.shape[0] > CHAMFER_MAX_POINTS:
            idx   = torch.randperm(pts_t.shape[0], device=device)[:CHAMFER_MAX_POINTS]
            pts_s = pts_t[idx]
        else:
            pts_s = pts_t

        loss = chamfer_loss_one_sided(pts_s, verts)
        loss.backward()
        optimizer.step()

        loss_val   = loss.item()
        final_loss = loss_val

        if (it + 1) % 50 == 0:
            print(f"    iter {it + 1:>4}/{SHAPE_ITERS}: "
                  f"Chamfer = {math.sqrt(max(loss_val, 0.0)):.6f} m")

        # ── Early-stop: no improvement > 1e-5 over 50 consecutive iters ──────
        if best_loss - loss_val < 1e-5:
            no_improve += 1
            if no_improve >= 50:
                print(f"  Converged early at iter {it + 1}")
                break
        else:
            no_improve = 0
            best_loss  = loss_val

    final_loss_m = math.sqrt(max(final_loss, 0.0))
    return (beta.detach().cpu().numpy().squeeze(0),
            global_orient.detach().cpu().numpy().squeeze(0),
            final_loss_m)


# =============================================================================
# Step 4 — Visualisation (open3d optional)
# =============================================================================

def visualise_fit(pts_body: np.ndarray, verts_fitted: np.ndarray,
                   output_path: str = None):
    """
    Display the observed body cloud (blue) and fitted SMPL-X mesh vertices (red)
    so the user can visually verify fit quality.

    Tries an interactive open3d window first. If that fails (e.g. no
    GLFW-compatible display, common under Wayland without XWayland), falls
    back to EGL-based offscreen rendering and writes a PNG to `output_path`.
    Requires open3d to be installed; skipped silently if unavailable.
    """
    if not OPEN3D_AVAILABLE:
        print("  open3d not available — skipping visualisation.")
        print("  Install with: pip install open3d")
        return

    pcd_obs        = o3d.geometry.PointCloud()
    pcd_obs.points = o3d.utility.Vector3dVector(pts_body)
    pcd_obs.paint_uniform_color([0.0, 0.0, 1.0])    # blue — observed body

    pcd_fit        = o3d.geometry.PointCloud()
    pcd_fit.points = o3d.utility.Vector3dVector(verts_fitted)
    pcd_fit.paint_uniform_color([1.0, 0.0, 0.0])    # red  — fitted SMPL-X vertices

    # Best-effort interactive window. open3d's GLFW backend may print a
    # C-level warning and silently fail to create a window (no exception
    # raised) — e.g. under native Wayland without XWayland. We can't reliably
    # detect that from Python, so always additionally save an offscreen
    # render below as a guaranteed-to-work fallback.
    try:
        o3d.visualization.draw_geometries(
            [pcd_obs, pcd_fit],
            window_name='Shape fitting — blue: observed, red: SMPL-X',
            width=1280, height=720,
        )
    except Exception as e:
        print(f"  Interactive window unavailable ({e}).")

    if output_path is None:
        return

    print("  Saving offscreen render...")
    width, height = 1280, 720
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])

    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader     = 'defaultUnlit'
    mat.point_size = 2.0
    renderer.scene.add_geometry('observed', pcd_obs, mat)
    renderer.scene.add_geometry('fitted', pcd_fit, mat)

    all_pts = np.concatenate([pts_body, verts_fitted], axis=0)
    center  = all_pts.mean(axis=0)
    extent  = np.linalg.norm(all_pts.max(axis=0) - all_pts.min(axis=0))
    eye     = center + np.array([0.0, 0.0, extent if extent > 0 else 1.0])
    renderer.setup_camera(60.0, center, eye, [0.0, 1.0, 0.0])

    img = renderer.render_to_image()
    o3d.io.write_image(output_path, img)
    print(f"  Saved render to {output_path}")


def debug_visualize_pointclouds(raw_points: np.ndarray, human_points: np.ndarray,
                                 output_path: str, max_points: int = 30000):
    """
    Save a 3-D scatter plot comparing the raw L515 point cloud (light gray,
    everything seen by the camera before any masking) against the
    segmented human-only point cloud (blue, what's actually fed to the
    SMPL-X fit) — useful for spotting points (e.g. legs) wrongly dropped by
    segmentation, or background points that leaked through.

    Both clouds are expected in the same world-frame XYZ (m) convention used
    throughout the rest of the script. Each cloud is randomly subsampled to
    at most `max_points` for plotting. Requires matplotlib; skipped silently
    if unavailable.

    Parameters
    ----------
    raw_points   : (N, 3) — pointcloud.npy XYZ, before masking/filtering
    human_points : (M, 3) — final body cloud used for SMPL-X fitting
    output_path  : path to write the PNG to
    """
    if not MATPLOTLIB_AVAILABLE:
        print("  matplotlib not available — skipping raw-vs-segmented debug plot.")
        print("  Install with: pip install matplotlib")
        return

    def _subsample(pts):
        if len(pts) > max_points:
            idx = np.random.choice(len(pts), max_points, replace=False)
            return pts[idx]
        return pts

    raw_sub   = _subsample(raw_points)
    human_sub = _subsample(human_points)

    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(111, projection='3d')

    ax.scatter(raw_sub[:, 0], raw_sub[:, 1], raw_sub[:, 2],
               s=1, c='lightgray', alpha=0.3,
               label=f'raw ({len(raw_points):,} pts)')
    ax.scatter(human_sub[:, 0], human_sub[:, 1], human_sub[:, 2],
               s=1, c='blue', alpha=0.6,
               label=f'segmented ({len(human_points):,} pts)')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Raw vs. segmented point cloud')
    ax.legend(loc='upper right', markerscale=10)

    # Equal aspect ratio across all three axes
    all_pts = np.concatenate([raw_sub, human_sub], axis=0)
    centers = (all_pts.max(axis=0) + all_pts.min(axis=0)) / 2
    radius  = max((all_pts.max(axis=0) - all_pts.min(axis=0)).max() / 2, 1e-6)
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)

    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved raw-vs-segmented debug plot to {output_path}")


# =============================================================================
# Main pipeline
# =============================================================================

def run_shape_estimation(session_dir: str, smplx_model_dir: str,
                          gender: str,
                          visualise: bool, dry_run: bool,
                          seg_threshold: float = SEGMENTATION_THRESHOLD,
                          mask_debug: bool = False):
    """
    Execute all four steps of SMPL-X shape calibration for a single therapist.

    Parameters
    ----------
    session_dir     : path to the 360° rotation calibration session directory
    smplx_model_dir : path to the folder containing SMPL-X model files
    gender          : 'neutral' | 'male' | 'female'
    visualise       : open an open3d window after fitting to inspect quality
    dry_run         : run all steps but skip writing smplx_beta.npy to disk
    seg_threshold   : MediaPipe confidence threshold above which a pixel is "person"
    """
    print(f"Session : {session_dir}")
    print(f"Gender  : {gender}   Visualise: {visualise}"
          f"{'   [DRY RUN]' if dry_run else ''}\n")

    # ── Step 1: segment, project, and merge ───────────────────────────────────
    # Run in a spawned subprocess: MediaPipe's TFLite/EGL runtime corrupts
    # process-wide thread state, causing segfaults in PyTorch's autograd
    # (Step 3) if both run in the same process.
    print("Step 1: Segmenting person and merging point clouds...")
    with multiprocessing.get_context('spawn').Pool(1) as pool:
        merged_pts, n_frames, n_raw, raw_points = pool.apply(
            load_segment_and_merge, (session_dir, seg_threshold, mask_debug))
    n_merged = len(merged_pts)
    print(f"  Frames loaded        : {n_frames}")
    print(f"  Points (raw, total)  : {n_raw:,}")
    print(f"  Points (person, merged): {n_merged:,}\n")

    # ── Step 2: outlier removal ────────────────────────────────────────────────
    print("Step 2: Cleaning up segmented body cloud...")
    body_pts = isolate_body(merged_pts)
    n_body   = len(body_pts)
    if n_body < 500:
        print(f"  WARNING: only {n_body} points remain after filtering. "
              "Check that the therapist is clearly visible in color_image.png.")
    print()

    # human_points: final body cloud actually used for SMPL-X fitting below,
    # kept alongside raw_points for the raw-vs-segmented debug plot.
    human_points = body_pts

    # ── Step 3: canonical alignment + shape fitting ───────────────────────────
    print("Step 3: Aligning body cloud to SMPL-X canonical frame...")
    body_aligned   = align_to_canonical(body_pts)
    centroid_check = body_aligned.mean(axis=0)
    print(f"  Aligned centroid (should be ≈0): "
          f"[{centroid_check[0]:+.4f}, {centroid_check[1]:+.4f}, {centroid_check[2]:+.4f}]")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}\n")

    print("  Fitting SMPL-X β via Chamfer distance...")
    beta_np, global_orient_np, final_loss_m = fit_shape(
        body_aligned, smplx_model_dir, gender, device)
    print(f"  Final Chamfer loss: {final_loss_m:.4f} m")
    print(f"  Fitted global_orient (rad): "
          f"[{global_orient_np[0]:+.4f}, {global_orient_np[1]:+.4f}, {global_orient_np[2]:+.4f}]")

    if final_loss_m > SHAPE_CONV_THRESHOLD:
        print(f"  WARNING: Shape fitting loss {final_loss_m:.4f} m is high. "
              "Check background removal or increase SHAPE_ITERS. Saving β anyway.")
    print()

    # ── Step 4: save ─────────────────────────────────────────────────────────
    beta_path = os.path.join(session_dir, 'smplx_beta.npy')
    if not dry_run:
        np.save(beta_path, beta_np)
        saved_str = beta_path
    else:
        saved_str = beta_path + '  (dry-run — not written)'

    print("Shape parameters β:")
    for i, v in enumerate(beta_np):
        print(f"  β[{i:2d}] = {v:+.6f}")
    print()

    # ── Visualisation (optional) ──────────────────────────────────────────────
    if visualise:
        print("Step 4: Launching open3d visualisation (close window to continue)...")
        model_vis = smplx.create(
            smplx_model_dir, model_type='smplx', gender=gender,
            use_pca=False, num_betas=10, batch_size=1)
        with torch.no_grad():
            beta_t          = torch.tensor(beta_np[np.newaxis], dtype=torch.float32)
            global_orient_t = torch.tensor(global_orient_np[np.newaxis], dtype=torch.float32)
            out    = model_vis(betas=beta_t,
                               global_orient=global_orient_t,
                               body_pose=torch.zeros(1, 63),
                               transl=torch.zeros(1, 3),
                               return_verts=True)
            verts_fitted = out.vertices[0].numpy()
        render_path = os.path.join(session_dir, 'fit_visualization.png')
        visualise_fit(body_aligned, verts_fitted, output_path=render_path)

        debug_render_path = os.path.join(session_dir, 'debug_raw_vs_segmented_pointcloud.png')
        debug_visualize_pointclouds(raw_points, human_points, debug_render_path)
        print()

    # ── Summary ───────────────────────────────────────────────────────────────
    sep = '═' * 38
    print(sep)
    print('Shape Estimation Summary')
    print(sep)
    print(f"  Frames loaded        : {n_frames}")
    print(f"  Points (person, merged): {n_merged:,}")
    print(f"  Points after filter  : {n_body:,}")
    print(f"  Final Chamfer loss   : {final_loss_m:.4f} m")
    print(f"  β saved to           : {saved_str}")
    print(sep)


def main():
    parser = argparse.ArgumentParser(
        description='Estimate SMPL-X shape β from a 360° rotation calibration recording')
    parser.add_argument('session_dir',
                        help='Path to calibration session directory')
    parser.add_argument('--smplx-model-dir',
                        default=os.path.expanduser('~/models/smplx'),
                        help='Path to SMPL-X model folder (default: ~/models/smplx)')
    parser.add_argument('--gender', default='neutral',
                        choices=['neutral', 'male', 'female'],
                        help='SMPL-X body gender (default: neutral)')
    parser.add_argument('--visualise', action='store_true',
                        help='Show open3d visualisation of fit quality after finishing '
                             '(requires open3d; install separately if needed)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Run all steps but do not write smplx_beta.npy')
    parser.add_argument('--seg-threshold', type=float, default=SEGMENTATION_THRESHOLD,
                        help='MediaPipe confidence threshold above which a pixel is '
                             f'"person" (default: {SEGMENTATION_THRESHOLD})')
    parser.add_argument('--mask-debug', action='store_true',
                        help='Save mask_debug.png into each frame directory showing '
                             'segmentation mask (green), pose-bbox extension (blue), '
                             'and pose bbox border (yellow)')
    args = parser.parse_args()

    if not DEPS_AVAILABLE:
        print(f"Missing dependency: {missing}")
        print("Install with: pip install smplx torch")
        sys.exit(1)

    session_dir = args.session_dir.rstrip('/')
    if not os.path.isdir(session_dir):
        print(f"Error: {session_dir} is not a directory")
        sys.exit(1)

    run_shape_estimation(
        session_dir     = session_dir,
        smplx_model_dir = args.smplx_model_dir,
        gender          = args.gender,
        visualise       = args.visualise,
        dry_run         = args.dry_run,
        seg_threshold   = args.seg_threshold,
        mask_debug      = args.mask_debug,
    )


if __name__ == '__main__':
    main()
