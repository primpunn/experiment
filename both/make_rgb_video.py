#!/usr/bin/env python3
"""
Assemble color_image.png frames from a saved_data session into an MP4 video.

Usage:
    python make_rgb_video.py <session_dir>
    python make_rgb_video.py <session_dir> --fps 30 --out video.mp4
"""

import argparse
import os
import cv2

def main():
    parser = argparse.ArgumentParser(
        description='Build RGB video from saved session color frames')
    parser.add_argument('session_dir', help='Path to session directory')
    parser.add_argument('--fps', type=float, default=30.0,
                        help='Output video frame rate (default: 30)')
    parser.add_argument('--out', default=None,
                        help='Output file path (default: <session_dir>/rgb_video.mp4)')
    args = parser.parse_args()

    session_dir = args.session_dir.rstrip('/')
    out_path = args.out or os.path.join(session_dir, 'rgb_video.mp4')

    frames = sorted(
        [d for d in os.listdir(session_dir) if d.startswith('frame_')],
        key=lambda x: int(x.split('_')[1])
    )
    if not frames:
        print(f"No frame_* directories found in {session_dir}")
        return

    # Read first frame to get resolution
    first_img = cv2.imread(os.path.join(session_dir, frames[0], 'color_image.png'))
    if first_img is None:
        print("Could not read first color_image.png — check the session directory.")
        return
    h, w = first_img.shape[:2]
    print(f"Resolution : {w}x{h},  frames : {len(frames)},  fps : {args.fps}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, args.fps, (w, h))

    for i, frame in enumerate(frames):
        img_path = os.path.join(session_dir, frame, 'color_image.png')
        img = cv2.imread(img_path)
        if img is None:
            print(f"  WARNING: skipping {frame} — could not read image")
            continue
        writer.write(img)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(frames)} frames written...")

    writer.release()
    print(f"\nSaved video to {out_path}")

if __name__ == '__main__':
    main()
