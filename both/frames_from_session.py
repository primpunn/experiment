"""
Alternative to make_rgb_video.py + preprocess_video.py: copy a recording
session's frame_N/color_image.png directly into ./frames/frame_NNNNNN.jpg,
with a single JPEG encode (no PNG -> mp4v -> JPEG double-compression pass).

Use this instead of the video round-trip when you don't need rgb_video.mp4
itself and just want frames for run_sam_body4d.py. Writes the same
video_info.txt metadata format as preprocess_video.py so downstream scripts
(extract_arm_trajectory.py --fps, overlay_skeleton.py --fps) work the same
either way.

Usage:
    python frames_from_session.py <session_dir> [--output ./frames] [--fps 30]
"""

import argparse
import os
import sys
import cv2


def copy_frames(session_dir: str, output_dir: str, fps: float) -> dict:
    session_dir = session_dir.rstrip("/")
    if not os.path.isdir(session_dir):
        print(f"[ERROR] Session directory not found: {session_dir}")
        sys.exit(1)

    frames = sorted(
        [d for d in os.listdir(session_dir) if d.startswith("frame_")],
        key=lambda x: int(x.split("_")[1]),
    )
    if not frames:
        print(f"[ERROR] No frame_* directories found in {session_dir}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    first_img = cv2.imread(os.path.join(session_dir, frames[0], "color_image.png"))
    if first_img is None:
        print("[ERROR] Could not read first color_image.png — check the session directory.")
        sys.exit(1)
    height, width = first_img.shape[:2]
    print(f"Session   : {session_dir}")
    print(f"Resolution: {width}x{height}")
    print(f"Frames    : {len(frames)}")

    saved_idx = 0
    for i, frame in enumerate(frames):
        img_path = os.path.join(session_dir, frame, "color_image.png")
        img = cv2.imread(img_path)
        if img is None:
            print(f"  WARNING: skipping {frame} — could not read color_image.png")
            continue
        out_path = os.path.join(output_dir, f"frame_{saved_idx:06d}.jpg")
        cv2.imwrite(out_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved_idx += 1
        if saved_idx % 50 == 0:
            print(f"  {saved_idx}/{len(frames)} frames written...")

    print(f"\nSaved {saved_idx} frames to: {output_dir}/")

    return {
        "source_session_dir": session_dir,
        "method": "direct_png_copy",
        "native_fps": fps,
        "effective_fps": fps,
        "total_source_frames": len(frames),
        "saved_frames": saved_idx,
        "width": width,
        "height": height,
        "duration_sec": saved_idx / fps if fps > 0 else 0,
        "output_dir": output_dir,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("session_dir", help="Path to saved_data session directory")
    parser.add_argument("--output", default="./frames")
    parser.add_argument("--fps", type=float, default=30.0,
                         help="Capture FPS for metadata/timing (data_recording.py "
                              "configures the RealSense streams at TARGET_FPS=30)")
    args = parser.parse_args()

    info = copy_frames(args.session_dir, args.output, args.fps)

    meta_path = os.path.join(args.output, "video_info.txt")
    with open(meta_path, "w") as f:
        for k, v in info.items():
            f.write(f"{k}: {v}\n")
    print(f"Metadata saved to: {meta_path}")


if __name__ == "__main__":
    main()
