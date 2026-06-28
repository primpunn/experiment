#!/usr/bin/env python3
"""Generate a PDF sheet of ArUco DICT_4X4_100 markers ID22-27, 40mm each, 10mm gaps."""

import sys
sys.path.insert(0, '/home/primpunn/librealsense/build/Release')

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

MARKER_IDS     = [22, 23, 24, 25, 26, 27]
MARKER_MM      = 30
GAP_MM         = 10
MARGIN_MM      = 20
LABEL_MM       = 5
COLS, ROWS     = 3, 2
DPI            = 300

def mm2px(mm):
    return int(round(mm / 25.4 * DPI))

MARKER_PX  = mm2px(MARKER_MM)
GAP_PX     = mm2px(GAP_MM)
MARGIN_PX  = mm2px(MARGIN_MM)
LABEL_PX   = mm2px(LABEL_MM)
CELL_W     = MARKER_PX
CELL_H     = MARKER_PX + LABEL_PX

PAGE_W = MARGIN_PX * 2 + COLS * CELL_W + (COLS - 1) * GAP_PX
PAGE_H = MARGIN_PX * 2 + ROWS * CELL_H + (ROWS - 1) * GAP_PX

page = Image.new('L', (PAGE_W, PAGE_H), color=255)
draw = ImageDraw.Draw(page)

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)

# Title
try:
    font_title = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', mm2px(4))
    font_label = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', mm2px(2.2))
except OSError:
    font_title = ImageFont.load_default()
    font_label = font_title

title = f'ArUco DICT_4X4_100  —  Left arm markers  —  {MARKER_MM} x {MARKER_MM} mm each'
bbox = draw.textbbox((0, 0), title, font=font_title)
tw = bbox[2] - bbox[0]
draw.text(((PAGE_W - tw) // 2, mm2px(6)), title, fill=0, font=font_title)

for i, mid in enumerate(MARKER_IDS):
    row = i // COLS
    col = i % COLS

    # Top-left of marker image on page
    x = MARGIN_PX + col * (CELL_W + GAP_PX)
    y = MARGIN_PX + row * (CELL_H + GAP_PX)

    # Generate marker (grayscale, 0=black, 255=white from OpenCV)
    marker_cv = cv2.aruco.generateImageMarker(aruco_dict, mid, MARKER_PX)
    # Add thin white padding so adjacent markers stay distinct when printed
    pad = mm2px(1)
    marker_cv = cv2.copyMakeBorder(marker_cv, pad, pad, pad, pad,
                                   cv2.BORDER_CONSTANT, value=255)
    marker_cv = cv2.resize(marker_cv, (MARKER_PX, MARKER_PX),
                           interpolation=cv2.INTER_NEAREST)
    marker_pil = Image.fromarray(marker_cv, mode='L')
    page.paste(marker_pil, (x, y))

    # Label centred below marker
    label = f'ID {mid}  ({MARKER_MM} x {MARKER_MM} mm)'
    lbbox = draw.textbbox((0, 0), label, font=font_label)
    lw = lbbox[2] - lbbox[0]
    lh = lbbox[3] - lbbox[1]
    draw.text((x + (MARKER_PX - lw) // 2,
               y + MARKER_PX + (LABEL_PX - lh) // 2),
              label, fill=0, font=font_label)

out = 'aruco_markers_ID22-27.pdf'
page.save(out, 'PDF', resolution=DPI)
print(f'Saved: {out}  ({PAGE_W}x{PAGE_H} px at {DPI} DPI)')
