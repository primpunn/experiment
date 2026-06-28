#!/usr/bin/env python3
"""
Generate ArUco DICT_4X4_100 markers for Prim's experiment.
IDs : 0-9, 12, 15, 18-19, 22-23  (16 markers total)
Size: 45 mm each
Paper: A4 (210 × 297 mm), 300 DPI, multi-page PDF

Output: ~/Desktop/aruco_markers_prim_45mm.pdf
"""

import sys
sys.path.insert(0, '/home/primpunn/librealsense/build/Release')

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

# ── marker IDs ──────────────────────────────────────────────────────────────
MARKER_IDS = (
    list(range(0, 10)) +   # 0-9
    [12, 15] +             # 12, 15
    [18, 19] +             # 18-19
    [22, 23]               # 22-23
)  # 16 total

# ── layout constants (all in mm unless marked _PX) ─────────────────────────
MARKER_MM  = 45
GAP_MM     = 9          # gap between markers
MARGIN_MM  = 15         # page margin (left / right / bottom)
TOP_MM     = 22         # top margin (includes title area)
LABEL_MM   = 7          # height of ID label below each marker
COLS       = 3
DPI        = 300

def mm2px(mm: float) -> int:
    return int(round(mm / 25.4 * DPI))

MARKER_PX = mm2px(MARKER_MM)
GAP_PX    = mm2px(GAP_MM)
MARGIN_PX = mm2px(MARGIN_MM)
TOP_PX    = mm2px(TOP_MM)
LABEL_PX  = mm2px(LABEL_MM)
CELL_W    = MARKER_PX
CELL_H    = MARKER_PX + LABEL_PX   # marker + label, no inter-row gap here

# A4 page size
A4_W_PX = mm2px(210)
A4_H_PX = mm2px(297)

# Usable area for markers
usable_w_px = A4_W_PX - 2 * MARGIN_PX
usable_h_px = A4_H_PX - TOP_PX - MARGIN_PX

# How many rows fit per page?
# n rows need: n*CELL_H + (n-1)*GAP_PX ≤ usable_h_px
def max_rows(usable_h):
    for n in range(20, 0, -1):
        if n * CELL_H + (n - 1) * GAP_PX <= usable_h:
            return n
    return 1

ROWS_PER_PAGE = max_rows(usable_h_px)
PER_PAGE      = COLS * ROWS_PER_PAGE

# Horizontal start: centre the 3-column block
block_w = COLS * CELL_W + (COLS - 1) * GAP_PX
x_start = (A4_W_PX - block_w) // 2

# ── fonts ───────────────────────────────────────────────────────────────────
try:
    font_title = ImageFont.truetype(
        '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', mm2px(4))
    font_label = ImageFont.truetype(
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',      mm2px(2.5))
except OSError:
    font_title = ImageFont.load_default()
    font_label = font_title

# ── ArUco dict ───────────────────────────────────────────────────────────────
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)

# ── helper: draw one page ───────────────────────────────────────────────────
def make_page(ids_on_page: list, page_num: int, total_pages: int) -> Image.Image:
    page = Image.new('L', (A4_W_PX, A4_H_PX), color=255)
    draw = ImageDraw.Draw(page)

    # Title
    title = (f'ArUco DICT_4X4_100  —  {MARKER_MM} × {MARKER_MM} mm each  '
             f'(page {page_num}/{total_pages})')
    try:
        tbbox = draw.textbbox((0, 0), title, font=font_title)
    except AttributeError:          # older Pillow fallback
        tbbox = (0, 0, *draw.textsize(title, font=font_title))
    tw = tbbox[2] - tbbox[0]
    draw.text(((A4_W_PX - tw) // 2, mm2px(6)), title, fill=0, font=font_title)

    # Markers
    for i, mid in enumerate(ids_on_page):
        row = i // COLS
        col = i % COLS
        x = x_start + col * (CELL_W + GAP_PX)
        y = TOP_PX  + row * (CELL_H + GAP_PX)

        # Generate marker image at target size
        raw = cv2.aruco.generateImageMarker(aruco_dict, mid, MARKER_PX)

        # Add 1 mm white border so adjacent markers are visually distinct
        pad = mm2px(1)
        raw = cv2.copyMakeBorder(raw, pad, pad, pad, pad,
                                 cv2.BORDER_CONSTANT, value=255)
        raw = cv2.resize(raw, (MARKER_PX, MARKER_PX),
                         interpolation=cv2.INTER_NEAREST)

        marker_pil = Image.fromarray(raw, mode='L')
        page.paste(marker_pil, (x, y))

        # Outer black border (0.5 mm)
        border_w = max(1, mm2px(0.5))
        for b in range(border_w):
            draw.rectangle(
                [x - b - 1, y - b - 1, x + MARKER_PX + b, y + MARKER_PX + b],
                outline=0
            )

        # Label centred below marker
        label = f'ID {mid}  ({MARKER_MM} × {MARKER_MM} mm)'
        try:
            lbbox = draw.textbbox((0, 0), label, font=font_label)
        except AttributeError:
            lbbox = (0, 0, *draw.textsize(label, font=font_label))
        lw = lbbox[2] - lbbox[0]
        lh = lbbox[3] - lbbox[1]
        draw.text(
            (x + (MARKER_PX - lw) // 2,
             y + MARKER_PX + mm2px(1.5)),
            label, fill=0, font=font_label
        )

    return page

# ── build pages & save PDF ───────────────────────────────────────────────────
pages_ids = [MARKER_IDS[i:i + PER_PAGE]
             for i in range(0, len(MARKER_IDS), PER_PAGE)]
total_pages = len(pages_ids)

rendered = [make_page(ids, p + 1, total_pages)
            for p, ids in enumerate(pages_ids)]

out_path = os.path.expanduser('~/Desktop/aruco_markers_prim_45mm.pdf')
rendered[0].save(
    out_path, 'PDF', resolution=DPI,
    save_all=True, append_images=rendered[1:]
)

print(f'✓  Saved: {out_path}')
print(f'   Pages  : {total_pages}')
print(f'   Markers: {len(MARKER_IDS)}  (IDs: {MARKER_IDS})')
print(f'   Size   : {MARKER_MM} × {MARKER_MM} mm  @ {DPI} DPI')
print(f'   Layout : {COLS} cols × {ROWS_PER_PAGE} rows per page')
