"""
Step 3: Video Preprocessing
Extracts frames from rgb_video.mp4 into ./frames/ as JPGs.
Usage: python preprocess_video.py [--video rgb_video.mp4] [--output ./frames] [--fps 0]
  --fps 0 means extract all frames at native FPS
"""

import argparse
import os
import cv2
import sys


def extract_frames(video_path: str, output_dir: str, target_fps: float = 0) -> dict:
    if not os.path.exists(video_path):
        print(f"[ERROR] Video not found: {video_path}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        sys.exit(1)

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = total_frames / native_fps if native_fps > 0 else 0

    print(f"Video: {video_path}")
    print(f"  Resolution : {width}x{height}")
    print(f"  Native FPS : {native_fps:.2f}")
    print(f"  Total frames: {total_frames}")
    print(f"  Duration   : {duration_sec:.2f}s")

    # Decide frame sampling interval
    if target_fps <= 0 or target_fps >= native_fps:
        frame_interval = 1
        effective_fps = native_fps
    else:
        frame_interval = max(1, round(native_fps / target_fps))
        effective_fps = native_fps / frame_interval

    print(f"  Extracting every {frame_interval} frame(s) → effective {effective_fps:.2f} FPS")

    frame_idx = 0
    saved_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            out_path = os.path.join(output_dir, f"frame_{saved_idx:06d}.jpg")
            cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved_idx += 1
        frame_idx += 1

    cap.release()
    print(f"\nSaved {saved_idx} frames to: {output_dir}/")

    return {
        "video_path": video_path,
        "native_fps": native_fps,
        "effective_fps": effective_fps,
        "total_source_frames": total_frames,
        "saved_frames": saved_idx,
        "width": width,
        "height": height,
        "duration_sec": duration_sec,
        "output_dir": output_dir,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="./saved_data/2026-05-11/rgb_video.mp4")
    parser.add_argument("--output", default="./frames")
    parser.add_argument(
        "--fps",
        type=float,
        default=0,
        help="Target FPS for extraction (0 = all frames at native FPS)",
    )
    args = parser.parse_args()

    info = extract_frames(args.video, args.output, args.fps)

    # Save metadata alongside frames
    meta_path = os.path.join(args.output, "video_info.txt")
    with open(meta_path, "w") as f:
        for k, v in info.items():
            f.write(f"{k}: {v}\n")
    print(f"Metadata saved to: {meta_path}")


if __name__ == "__main__":
    main()
