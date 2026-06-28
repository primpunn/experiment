"""
Step 4: Run SAM-3D-Body inference on extracted frames.

Uses the real SAM3DBodyEstimator API from ~/sam-3d-body (facebook/sam-3d-body-dinov3
checkpoint, MHR-based). Targets the therapist (standing person) and ignores the
patient (lying down).

VRAM budget notes (GTX 1650, 4GB):
  - We skip the repo's built-in ViTDet-Huge detector and MoGe2 FOV estimator
    (both heavy) and instead use a lightweight torchvision Faster R-CNN
    (MobileNetV3) just to get person bounding boxes per frame.
  - inference_type="body" skips the hand decoder (we only need shoulder/
    elbow/wrist), further reducing compute/VRAM.

Usage:
  # Standard run (auto-detect therapist by aspect ratio on frame 0, IoU-track after)
  python run_sam_body4d.py --frames ./frames --output ./output

  # Click on therapist in first frame to confirm selection
  python run_sam_body4d.py --frames ./frames --output ./output --interactive

  # Test mode: generate synthetic data to test downstream scripts without GPU
  python run_sam_body4d.py --frames ./frames --output ./output --mock

Prerequisites:
  - ~/sam-3d-body cloned
  - HuggingFace access to facebook/sam-3d-body-dinov3 + `hf auth login` done
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

SAM3D_REPO = os.environ.get("SAM3D_REPO", os.path.expanduser("~/sam-3d-body"))
SAM3D_CKPT = os.environ.get("SAM3D_CKPT", os.path.expanduser("~/checkpoints/sam3dbody"))


# ---------------------------------------------------------------------------
# Therapist selection: standing person has taller bounding box
# ---------------------------------------------------------------------------
def select_therapist_index(bboxes: np.ndarray) -> int:
    """Return index of bbox with highest height/width ratio (standing person)."""
    if len(bboxes) == 0:
        return 0
    ratios = (bboxes[:, 3] - bboxes[:, 1]) / (bboxes[:, 2] - bboxes[:, 0] + 1e-6)
    return int(np.argmax(ratios))


def select_patient_index(bboxes: np.ndarray) -> int:
    """Return index of bbox with lowest height/width ratio (lying person)."""
    if len(bboxes) == 0:
        return -1
    ratios = (bboxes[:, 3] - bboxes[:, 1]) / (bboxes[:, 2] - bboxes[:, 0] + 1e-6)
    return int(np.argmin(ratios))


def _track_by_iou(prev_bbox: np.ndarray, bboxes: np.ndarray) -> int:
    """Pick the detection with highest IoU vs previous frame's bbox."""
    x1 = np.maximum(prev_bbox[0], bboxes[:, 0])
    y1 = np.maximum(prev_bbox[1], bboxes[:, 1])
    x2 = np.minimum(prev_bbox[2], bboxes[:, 2])
    y2 = np.minimum(prev_bbox[3], bboxes[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_prev = (prev_bbox[2] - prev_bbox[0]) * (prev_bbox[3] - prev_bbox[1])
    area_cur = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
    iou = inter / (area_prev + area_cur - inter + 1e-6)
    return int(np.argmax(iou))


def _interactive_select(img_bgr: np.ndarray, bboxes: np.ndarray) -> int:
    import cv2
    vis = img_bgr.copy()
    for i, (x1, y1, x2, y2) in enumerate(bboxes.astype(int)):
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis, str(i), (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    print("Detected persons shown. Click on the THERAPIST's bounding box, then press any key.")
    click = []

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            click.append((x, y))

    cv2.namedWindow("Select Therapist")
    cv2.setMouseCallback("Select Therapist", on_mouse)
    cv2.imshow("Select Therapist", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if click:
        cx, cy = click[0]
        for i, (x1, y1, x2, y2) in enumerate(bboxes.astype(int)):
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                return i
    return select_therapist_index(bboxes)


# ---------------------------------------------------------------------------
# Mock mode: generate plausible random MHR70-keypoint data for testing
# ---------------------------------------------------------------------------
def generate_mock_data(frames_dir: str, output_dir: str, fps: float = 30.0) -> str:
    frame_paths = sorted(glob.glob(os.path.join(frames_dir, "frame_*.jpg")))
    n = len(frame_paths) if frame_paths else 200

    print(f"[MOCK] Generating synthetic MHR70 keypoints for {n} frames (therapist + patient)...")
    results = {}
    rng = np.random.default_rng(42)

    therapist_kp = rng.normal(0, 0.3, (70, 3)).astype(np.float32)
    patient_kp = rng.normal(0, 0.3, (70, 3)).astype(np.float32)
    for i in range(n):
        therapist_kp += rng.normal(0, 0.01, (70, 3)).astype(np.float32)
        patient_kp += rng.normal(0, 0.005, (70, 3)).astype(np.float32)

        results[f"frame_{i}"] = {
            "therapist": {
                "pred_keypoints_3d": therapist_kp.tolist(),
                "pred_cam_t": rng.normal(0, 0.1, 3).tolist(),
                "bbox": [100.0, 50.0, 400.0, 600.0],
            },
            "patient": {
                "pred_keypoints_3d": patient_kp.tolist(),
                "pred_cam_t": rng.normal(0, 0.05, 3).tolist(),
                "bbox": [50.0, 300.0, 700.0, 500.0],
            },
            "fps": fps,
        }

    os.makedirs(output_dir, exist_ok=True)
    out_json = os.path.join(output_dir, "body_params_per_frame.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[MOCK] Saved to {out_json}")
    return out_json


# ---------------------------------------------------------------------------
# Lightweight person detector (avoids loading the repo's heavy ViTDet-H)
# ---------------------------------------------------------------------------
class LightPersonDetector:
    def __init__(self, device: str, score_thresh: float = 0.6):
        import torch
        from torchvision.models.detection import (
            fasterrcnn_mobilenet_v3_large_320_fpn,
            FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
        )
        weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
        self.model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights, box_score_thresh=score_thresh)
        self.model.to(device).eval()
        self.device = device
        self.torch = torch

    def detect_persons(self, img_rgb: np.ndarray) -> np.ndarray:
        img_t = self.torch.from_numpy(img_rgb.transpose(2, 0, 1)).float().to(self.device) / 255.0
        with self.torch.no_grad():
            out = self.model([img_t])[0]
        labels = out["labels"].cpu().numpy()
        boxes = out["boxes"].cpu().numpy()
        person_mask = labels == 1  # COCO class 1 = person
        return boxes[person_mask]


# ---------------------------------------------------------------------------
# Real inference via SAM-3D-Body
# ---------------------------------------------------------------------------
def run_sam3d_body(frames_dir: str, output_dir: str, device: str, interactive: bool, max_frames: int = None) -> str:
    if not os.path.isdir(SAM3D_REPO):
        print(f"[ERROR] SAM-3D-Body repo not found at: {SAM3D_REPO}")
        print("  Clone it: git clone https://github.com/facebookresearch/sam-3d-body.git ~/sam-3d-body")
        sys.exit(1)

    sys.path.insert(0, SAM3D_REPO)

    import cv2
    import torch
    from sam_3d_body.build_models import load_sam_3d_body
    from sam_3d_body.sam_3d_body_estimator import SAM3DBodyEstimator

    ckpt_path = os.path.join(SAM3D_CKPT, "model.ckpt")
    mhr_path = os.path.join(SAM3D_CKPT, "assets", "mhr_model.pt")
    print(f"Loading SAM-3D-Body model from local checkpoint: {ckpt_path} ...")
    model, model_cfg = load_sam_3d_body(checkpoint_path=ckpt_path, mhr_path=mhr_path, device=device)

    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=None,   # we supply bboxes ourselves (lighter than ViTDet-H)
        human_segmentor=None,  # no mask-conditioned inference needed
        fov_estimator=None,    # skip MoGe2; use default FOV (relative trajectory shape is unaffected)
    )

    print("Loading lightweight person detector (torchvision MobileNetV3)...")
    detector = LightPersonDetector(device=device)

    frame_paths = sorted(glob.glob(os.path.join(frames_dir, "frame_*.jpg")))
    if not frame_paths:
        print(f"[ERROR] No frames found in {frames_dir}")
        sys.exit(1)
    if max_frames is not None:
        frame_paths = frame_paths[:max_frames]
    os.makedirs(output_dir, exist_ok=True)

    results = {}
    therapist_bbox = None
    patient_bbox = None

    for frame_idx, frame_path in enumerate(frame_paths):
        img_bgr = cv2.imread(frame_path)
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        bboxes = detector.detect_persons(img_rgb)
        if len(bboxes) == 0:
            print(f"  [WARN] Frame {frame_idx}: no person detected, skipping.")
            continue

        # --- Therapist ---
        if frame_idx == 0 and interactive:
            tidx = _interactive_select(img_bgr, bboxes)
        elif therapist_bbox is not None:
            tidx = _track_by_iou(therapist_bbox, bboxes)
        else:
            tidx = select_therapist_index(bboxes)

        therapist_bbox = bboxes[tidx]

        try:
            out_list = estimator.process_one_image(
                img_rgb, bboxes=bboxes[tidx:tidx + 1], inference_type="body"
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                print(f"  [WARN] Frame {frame_idx}: CUDA OOM (therapist), skipping frame.")
                continue
            raise

        if not out_list:
            print(f"  [WARN] Frame {frame_idx}: model returned no therapist output, skipping.")
            continue

        out = out_list[0]
        frame_data: dict = {
            "therapist": {
                "pred_keypoints_3d": np.asarray(out["pred_keypoints_3d"]).tolist(),
                "pred_cam_t": np.asarray(out["pred_cam_t"]).tolist(),
                "bbox": np.asarray(out["bbox"]).tolist(),
            }
        }

        # --- Patient (lying person among remaining detections) ---
        other_bboxes = np.array([bboxes[i] for i in range(len(bboxes)) if i != tidx])
        if len(other_bboxes) > 0:
            if patient_bbox is not None:
                pidx_local = _track_by_iou(patient_bbox, other_bboxes)
            else:
                pidx_local = select_patient_index(other_bboxes)
            patient_bbox_cur = other_bboxes[pidx_local]
            try:
                patient_out_list = estimator.process_one_image(
                    img_rgb, bboxes=patient_bbox_cur[np.newaxis], inference_type="body"
                )
                if patient_out_list:
                    p_out = patient_out_list[0]
                    patient_bbox = patient_bbox_cur
                    frame_data["patient"] = {
                        "pred_keypoints_3d": np.asarray(p_out["pred_keypoints_3d"]).tolist(),
                        "pred_cam_t": np.asarray(p_out["pred_cam_t"]).tolist(),
                        "bbox": np.asarray(p_out["bbox"]).tolist(),
                    }
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    print(f"  [WARN] Frame {frame_idx}: CUDA OOM (patient), skipping patient.")
                else:
                    raise

        results[f"frame_{frame_idx}"] = frame_data

        if frame_idx % 20 == 0:
            has_patient = "patient" in frame_data
            print(f"  Frame {frame_idx}/{len(frame_paths)} | patient={'yes' if has_patient else 'no'}")

    out_json = os.path.join(output_dir, "body_params_per_frame.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} frames → {out_json}")
    return out_json


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", default="./frames",
                        help="Directory with frame_XXXXXX.jpg files")
    parser.add_argument("--output", default="./output",
                        help="Output directory for body_params_per_frame.json")
    parser.add_argument("--device", default=None,
                        help="'cuda' or 'cpu' (auto-detected if omitted)")
    parser.add_argument("--interactive", action="store_true",
                        help="Click on therapist in first frame to confirm selection")
    parser.add_argument("--mock", action="store_true",
                        help="Generate synthetic data (no GPU/model needed — for testing)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Only process the first N frames (smoke testing)")
    args = parser.parse_args()

    if args.device is None:
        try:
            import torch
            args.device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            args.device = "cpu"

    print(f"Device: {args.device}")

    if args.mock:
        print("[MOCK MODE] Generating synthetic keypoints for testing...")
        generate_mock_data(args.frames, args.output)
    else:
        if args.device == "cpu":
            print("[WARN] Running on CPU — will be very slow. Use --mock for testing.")
        run_sam3d_body(args.frames, args.output, args.device, args.interactive, args.max_frames)


if __name__ == "__main__":
    main()
