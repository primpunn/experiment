#!/usr/bin/env python3
"""
Generate academic-style pipeline flowchart PNGs for all pipeline scripts.
Output: /home/primpunn/experiment/both/flowchart_<name>.png
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon
import os

OUT = '/home/primpunn/experiment/both'
DPI = 200
plt.rcParams['font.family'] = 'DejaVu Sans'

# ── colour map: key → (facecolour, textcolour) ────────────────────────────────
C = dict(
    term =('#1a237e', 'white'),   # deep indigo       start / end
    io   =('#01579b', 'white'),   # ocean blue        input / output
    proc =('#37474f', 'white'),   # blue-grey         generic process
    det1 =('#b71c1c', 'white'),   # deep red          Stage-1 detect
    det2 =('#6a1b9a', 'white'),   # deep violet       Stage-2 detect
    det3 =('#004d40', 'white'),   # dark teal         Stage-3 detect
    det4 =('#e65100', 'white'),   # burnt orange      Stage-4 detect
    fil  =('#1b5e20', 'white'),   # deep green        fill / repair
    dep  =('#006064', 'white'),   # dark cyan         depth step
    kin  =('#880e4f', 'white'),   # deep pink         kinematic step
    cal  =('#4a148c', 'white'),   # deep purple       calibration
    dec  =('#f9a825', '#1a1a1a'), # amber             decision diamond
    note =('#eceff1', '#37474f'), # light bg          annotation note
    r2   =('#3e2723', 'white'),   # dark brown        R2-specific note
    cam  =('#263238', 'white'),   # dark blue-grey    camera step
    init =('#004d40', 'white'),   # dark teal         init phase
    rec  =('#1b5e20', 'white'),   # deep green        recording phase
)

# ── default geometry (inches, matching coordinate space) ─────────────────────
W  = 3.8   # standard box width
H  = 0.52  # box height
S  = 0.88  # vertical step (centre-to-centre)
WD = 2.8   # decision diamond width
HD = 0.76  # decision diamond height

# ── primitives ────────────────────────────────────────────────────────────────

def _new(fw, fh, title):
    fig = plt.figure(figsize=(fw, fh), dpi=DPI, facecolor='white')
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(-fw/2, fw/2)
    ax.set_ylim(-fh, 0)
    ax.axis('off')
    ax.patch.set_visible(False)   # transparent bg → bbox_inches='tight' crops to content
    fig.text(0.5, 0.993, title, ha='center', va='top',
             fontsize=10.5, fontweight='bold', color='#1a237e')
    return fig, ax


def rb(ax, cx, cy, w, h, text, ck='proc', fs=8.5, r=0.07):
    fc, tc = C[ck]
    ax.add_patch(FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle=f'round,pad={r}', fc=fc, ec='white', lw=1.5, zorder=3))
    ax.text(cx, cy, text, ha='center', va='center', fontsize=fs,
            color=tc, multialignment='center', linespacing=1.3, zorder=4)


def db(ax, cx, cy, w, h, text, ck='dec', fs=8):
    fc, tc = C[ck]
    pts = [(cx, cy+h/2), (cx+w/2, cy), (cx, cy-h/2), (cx-w/2, cy)]
    ax.add_patch(Polygon(pts, closed=True, fc=fc, ec='white', lw=1.5, zorder=3))
    ax.text(cx, cy, text, ha='center', va='center', fontsize=fs,
            color=tc, fontweight='bold', multialignment='center', zorder=4)


def ar(ax, x1, y1, x2, y2, lbl='', ldx=0.18, ldy=0, fs=7.5):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='#444', lw=1.2,
                                mutation_scale=10), zorder=2)
    if lbl:
        ax.text((x1+x2)/2+ldx, (y1+y2)/2+ldy, lbl,
                ha='center', va='center', fontsize=fs,
                color='#666', fontstyle='italic', zorder=5)


def pa(ax, pts, lbl='', ldx=0.2, ldy=0.1, fs=7.5):
    """Polyline with arrowhead at last segment; optional label near first point."""
    for i in range(len(pts)-1):
        x1, y1 = pts[i]; x2, y2 = pts[i+1]
        if i < len(pts)-2:
            ax.plot([x1,x2],[y1,y2], c='#444', lw=1.2, zorder=2,
                    solid_capstyle='round')
        else:
            ax.annotate('', xy=(x2,y2), xytext=(x1,y1),
                        arrowprops=dict(arrowstyle='->', color='#444', lw=1.2,
                                        mutation_scale=10), zorder=2)
    if lbl:
        ax.text(pts[0][0]+ldx, pts[0][1]+ldy, lbl,
                ha='center', va='center', fontsize=fs,
                color='#666', fontstyle='italic', zorder=5)


def sbg(ax, cx, yt, yb, w, lbl, fc='#e8eaf6', ec='#9fa8da'):
    """Light background band with vertical side label."""
    hh = yt - yb
    ax.add_patch(FancyBboxPatch(
        (cx-w/2-0.1, yb-0.08), w+0.2, hh+0.16,
        boxstyle='round,pad=0.04', fc=fc, ec=ec, lw=0.8,
        alpha=0.38, zorder=1))
    ax.text(cx-w/2-0.14, (yt+yb)/2, lbl,
            ha='right', va='center', fontsize=7.5, color='#5c6bc0',
            fontstyle='italic', fontweight='bold', rotation=90, zorder=5)


# ─────────────────────────────────────────────────────────────────────────────
# Chart 1 — interpolate_poses.py
# ─────────────────────────────────────────────────────────────────────────────
def chart_poses():
    fw, fh = 5.8, 8.5
    fig, ax = _new(fw, fh, 'Pipeline: interpolate_poses.py')

    # y positions (top → bottom)
    y0  = -0.45   # START
    y1  = y0  - S         # Load poses
    y2  = y1  - S         # Identify gaps
    y3  = y2  - S*1.05    # Decision
    y4l = y3  - S*1.1     # Yes: fill box   (cx = -0.9)
    y4r = y4l             # No: skip        (cx = +1.1)
    y5  = y4l - H*0.6     # merge horizontal line
    y6  = y5  - 0.52      # Write filled
    y7  = y6  - S         # END

    rb(ax,  0, y0,  2.4,  H, 'START', 'term', fs=9.5, r=0.26)
    rb(ax,  0, y1,  W,    H, 'Load raw pose files\n(4 arm joints × N frames)', 'io')
    rb(ax,  0, y2,  W,    H, 'Identify NaN gap runs per joint', 'proc')
    db(ax,  0, y3,  WD,   HD,'gap_len ≤ MAX_GAP\n(default: 5 frames)?', 'dec')

    # branches
    rb(ax, -0.95, y4l, 2.10, H*1.1,
       'Linear interp. (translation)\n+ SLERP (rotation)\n→ fill gap', 'fil', fs=8)
    rb(ax,  1.10, y4r, 1.70, H*0.85,
       'Skip —\nleave as NaN', 'note', fs=8)

    rb(ax,  0, y6,  W,    H, 'Write pose_<joint>_filled.txt\n(all 4 joints)', 'io')
    rb(ax,  0, y7,  2.4,  H, 'END',   'term', fs=9.5, r=0.26)

    # arrows
    ar(ax, 0, y0-H/2,      0, y1+H/2)
    ar(ax, 0, y1-H/2,      0, y2+H/2)
    ar(ax, 0, y2-H/2,      0, y3+HD/2)

    # diamond → YES left
    pa(ax, [(0, y3-HD/2), (0, y3-HD/2-0.18),
            (-0.95, y3-HD/2-0.18), (-0.95, y4l+H*1.1/2)],
       'Yes', ldx=-0.35, ldy=0.12)

    # diamond → NO right
    pa(ax, [(WD/2, y3), (1.95, y3), (1.95, y4r+H*0.85/2)],
       'No', ldx=0.25, ldy=0.12)

    # both branches → merge line → write
    merge_y = y4l - H*1.1/2 - 0.22
    ax.plot([-0.95, -0.95], [y4l-H*1.1/2, merge_y], c='#444', lw=1.2, zorder=2)
    ax.plot([1.10,  1.10],  [y4r-H*0.85/2, merge_y], c='#444', lw=1.2, zorder=2)
    ax.plot([-0.95, 1.10],  [merge_y, merge_y], c='#444', lw=1.2, zorder=2)
    ar(ax, 0, merge_y, 0, y6+H/2)
    ar(ax, 0, y6-H/2,  0, y7+H/2)

    # loop-back note
    ax.text(-fw/2+0.12, (y2+y3)/2, 'Repeat for each\ngap run and joint',
            ha='left', va='center', fontsize=7, color='#999',
            fontstyle='italic', linespacing=1.3)

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# shared builder for interpolate_modify-family charts
# ─────────────────────────────────────────────────────────────────────────────
def _modify_core(ax, y0, has_r2=False):
    """Draw the 4-stage modify pipeline starting at y0. Returns final y."""
    y = y0

    rb(ax, 0, y, W, H, 'Load raw poses + initialise interp_mask\n(NaN frames flagged)', 'proc')
    y -= S

    # ── STAGE 1 ──────────────────────────────────────────────────────────────
    sbg(ax, 0, y+H/2+0.08, y-H-0.08, W+0.15, 'Stage 1')
    rb(ax, 0, y, W, H,
       'Detect single peaks\n(median HP filter + amplitude ratio + top-hat)',
       'det1')
    y -= S
    rb(ax, 0, y, W, H,
       'B&L Low-dim. Kalman Smooth → repair peaks\n'
       '[Translation: Kalman fill   |   Rotation: SLERP]',
       'fil')
    y -= S*0.72
    ax.text(0, y+0.05, '↓  update interp_mask', ha='center', va='center',
            fontsize=7.5, color='#555', fontstyle='italic')
    y -= S*0.45

    # ── STAGE 2 ──────────────────────────────────────────────────────────────
    sbg(ax, 0, y+H/2+0.08, y-H-0.08, W+0.15, 'Stage 2')
    rb(ax, 0, y, W, H,
       'Detect heavy noise — exclude interp_mask\n'
       '(SG high-pass + morphological closing)',
       'det2')
    y -= S
    rb(ax, 0, y, W, H,
       'B&L Low-dim. Kalman Smooth → repair noise\n'
       '[Translation: Kalman fill   |   Rotation: SQUAD]',
       'fil')
    y -= S*0.72
    ax.text(0, y+0.05, '↓  update interp_mask', ha='center', va='center',
            fontsize=7.5, color='#555', fontstyle='italic')
    y -= S*0.45

    # ── STAGE 3 ──────────────────────────────────────────────────────────────
    sbg(ax, 0, y+H/2+0.08, y-H-0.08, W+0.15, 'Stage 3')
    rb(ax, 0, y, W, H,
       'Detect step changes — exclude interp_mask\n'
       '(SG derivative + complementary spike pairs)',
       'det3')
    y -= S
    rb(ax, 0, y, W, H,
       'Linear bridging → fill step region\n'
       '[Translation: linear   |   Rotation: CuBsp]',
       'fil')
    y -= S*0.72
    ax.text(0, y+0.05, '↓  update interp_mask', ha='center', va='center',
            fontsize=7.5, color='#555', fontstyle='italic')
    y -= S*0.45

    # ── STAGE 4 ──────────────────────────────────────────────────────────────
    sbg(ax, 0, y+H/2+0.08, y-H-0.08, W+0.15, 'Stage 4')
    rb(ax, 0, y, W, H,
       'Train FFNN on clean frames\n'
       '(3 siblings × k lags + polynomial + MA self)',
       'det4')
    y -= S
    rb(ax, 0, y, W, H,
       'Detect slow drift (hysteresis on SG residual)\n'
       '→ replace with FFNN prediction + CuBsp rotation',
       'fil')
    y -= S

    # R2 note
    if has_r2:
        rb(ax, 0, y, W, H*0.78,
           'R2 rule — right_elbow: contribute to SVD subspace\n'
           'but do NOT fill originally-NaN frames',
           'r2', fs=7.8)
        y -= S*0.85

    return y


def _draw_arrows_modify(ax, y_start, y_end, fw):
    """Draw a continuous downward arrow chain between y_start and y_end."""
    # We draw one long continuous arrow from y_start to y_end on the right edge
    # Actually, just let each box connect via the ar() calls in the main function
    pass


def chart_modify():
    fw, fh = 5.8, 17.0
    fig, ax = _new(fw, fh, 'Pipeline: interpolate_modify.py\n'
                             '(Skurowski & Pawlyta 2022  ·  Burke & Lasenby 2016)')

    y = -1.0
    rb(ax, 0, y, 2.4, H, 'START', 'term', fs=9.5, r=0.26);  y -= S
    rb(ax, 0, y, W,   H, 'Input: session directory\n(N frames, 4 arm joints)', 'io'); y -= S

    y_after = _modify_core(ax, y, has_r2=False)
    y = y_after

    rb(ax, 0, y, W, H, 'Write pose_<joint>_processed.txt\n(4 joints, all N frames)', 'io'); y -= S
    rb(ax, 0, y, 2.4, H, 'END', 'term', fs=9.5, r=0.26)

    # single vertical arrow spanning the whole chart
    ax.annotate('', xy=(2.55, y), xytext=(2.55, -0.45-H/2),
                arrowprops=dict(arrowstyle='->', color='#aaa', lw=0.8,
                                mutation_scale=8,
                                connectionstyle='arc3,rad=0'),
                zorder=1)
    ax.text(2.62, (-0.45 + y)/2, 'flow', ha='left', va='center',
            fontsize=7, color='#bbb', fontstyle='italic', rotation=-90)

    return fig


def chart_modify_R2():
    fw, fh = 5.8, 18.0
    fig, ax = _new(fw, fh, 'Pipeline: interpolate_modify_R2.py\n'
                             '(Gløersen & Federolf 2016  +  Burke & Lasenby 2016)')

    y = -1.0
    rb(ax, 0, y, 2.4, H, 'START', 'term', fs=9.5, r=0.26);  y -= S
    rb(ax, 0, y, W,   H, 'Input: session directory\n(N frames, 4 arm joints)', 'io'); y -= S

    y_after = _modify_core(ax, y, has_r2=True)
    y = y_after

    rb(ax, 0, y, W, H, 'Write pose_<joint>_processed.txt\n'
                        '(right_elbow: NaN retained where originally NaN)', 'io'); y -= S
    rb(ax, 0, y, 2.4, H, 'END', 'term', fs=9.5, r=0.26)

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# shared depth-pipeline core
# ─────────────────────────────────────────────────────────────────────────────
def _depth_core(ax, y0, has_r2=False, has_kin=False):
    """Draw depth-assisted pipeline starting at y0. Returns final y."""
    y = y0

    rb(ax, 0, y, W, H,
       'Load raw poses + initialise interp_mask\n(NaN frames flagged)', 'proc')
    y -= S

    # ── STEP 0 ────────────────────────────────────────────────────────────────
    sbg(ax, 0, y+H/2+0.08, y-H*2-S-0.08, W+0.15, 'Step 0',
        fc='#e0f2f1', ec='#80cbc4')
    rb(ax, 0, y, W, H,
       'Preload all L515 point clouds into RAM\n'
       '(200 pointcloud.npy files, 8192 pts each)',
       'dep')
    y -= S
    rb(ax, 0, y, W, H,
       'Compute depth fallback positions\n'
       '(linear interp → point cloud search, for B&L-fallback path)',
       'dep')
    y -= S

    if has_kin:
        rb(ax, 0, y, W, H,
           'Calibrate forearm bone lengths from raw data\n'
           '(median wrist↔elbow distance; used in Stages 1 & 2)',
           'cal')
        y -= S

    # ── STAGE 1 ───────────────────────────────────────────────────────────────
    sbg(ax, 0, y+H/2+0.08, y-H-0.08, W+0.15, 'Stage 1')
    rb(ax, 0, y, W, H,
       'Detect single peaks — on raw data\n'
       '(median HP + amplitude ratio + morphological top-hat)',
       'det1')
    y -= S

    kin_note = '\n+ Kinematic constraint (bone length → elbow estimate)' if has_kin else ''
    rb(ax, 0, y, W, H,
       'Depth-assisted B&L Kalman Smooth\n'
       '  R_RGB < R_depth < R_kinematic   ·   Adaptive Q'
       + kin_note + '\n[Rotation: SLERP]',
       'fil', fs=8 if has_kin else 8.5)
    y -= S*0.72
    ax.text(0, y+0.05, '↓  update interp_mask', ha='center', va='center',
            fontsize=7.5, color='#555', fontstyle='italic')
    y -= S*0.45

    # ── STAGE 2 ───────────────────────────────────────────────────────────────
    sbg(ax, 0, y+H/2+0.08, y-H-0.08, W+0.15, 'Stage 2')
    rb(ax, 0, y, W, H,
       'Detect heavy noise — exclude interp_mask\n'
       '(SG high-pass + morphological closing)',
       'det2')
    y -= S
    rb(ax, 0, y, W, H,
       'Depth-assisted B&L Kalman Smooth\n'
       '  R_RGB < R_depth < R_kinematic   ·   Adaptive Q'
       + kin_note + '\n[Rotation: SQUAD]',
       'fil', fs=8 if has_kin else 8.5)
    y -= S*0.72
    ax.text(0, y+0.05, '↓  update interp_mask', ha='center', va='center',
            fontsize=7.5, color='#555', fontstyle='italic')
    y -= S*0.45

    # ── STAGE 3 ───────────────────────────────────────────────────────────────
    sbg(ax, 0, y+H/2+0.08, y-H-0.08, W+0.15, 'Stage 3')
    rb(ax, 0, y, W, H,
       'Detect step changes — exclude interp_mask\n'
       '(SG derivative + complementary spike pairs)',
       'det3')
    y -= S
    rb(ax, 0, y, W, H,
       'Linear bridging → fill step region\n'
       '[Translation: linear   |   Rotation: CuBsp]',
       'fil')
    y -= S*0.72
    ax.text(0, y+0.05, '↓  update interp_mask', ha='center', va='center',
            fontsize=7.5, color='#555', fontstyle='italic')
    y -= S*0.45

    # ── STAGE 4 ───────────────────────────────────────────────────────────────
    sbg(ax, 0, y+H/2+0.08, y-H-0.08, W+0.15, 'Stage 4')
    rb(ax, 0, y, W, H,
       'Train FFNN on clean frames\n'
       '(3 siblings × k lags + polynomial + MA self)',
       'det4')
    y -= S
    rb(ax, 0, y, W, H,
       'Detect slow drift (hysteresis on SG residual)\n'
       '→ replace with FFNN prediction + CuBsp rotation',
       'fil')
    y -= S

    if has_r2:
        rb(ax, 0, y, W, H*0.78,
           'R2 rule — right_elbow: contribute to SVD subspace\n'
           'but do NOT fill originally-NaN frames',
           'r2', fs=7.8)
        y -= S*0.85

    return y


def chart_depth():
    fw, fh = 5.8, 19.0
    fig, ax = _new(fw, fh, 'Pipeline: interpolate_depth.py\n'
                             '(+ L515 depth assistance  ·  Zhao et al. 2012)')

    y = -1.0
    rb(ax, 0, y, 2.4, H, 'START', 'term', fs=9.5, r=0.26);  y -= S
    rb(ax, 0, y, W,   H, 'Input: session directory\n(N frames, 4 arm joints)', 'io'); y -= S

    y = _depth_core(ax, y, has_r2=False, has_kin=False)

    rb(ax, 0, y, W, H, 'Write pose_<joint>_processed.txt\n(4 joints, all N frames)', 'io'); y -= S
    rb(ax, 0, y, 2.4, H, 'END', 'term', fs=9.5, r=0.26)

    return fig


def chart_depth_R2():
    fw, fh = 5.8, 20.0
    fig, ax = _new(fw, fh, 'Pipeline: interpolate_depth_R2.py\n'
                             '(Depth assistance  +  Gløersen & Federolf 2016)')

    y = -1.0
    rb(ax, 0, y, 2.4, H, 'START', 'term', fs=9.5, r=0.26);  y -= S
    rb(ax, 0, y, W,   H, 'Input: session directory\n(N frames, 4 arm joints)', 'io'); y -= S

    y = _depth_core(ax, y, has_r2=True, has_kin=False)

    rb(ax, 0, y, W, H,
       'Write pose_<joint>_processed.txt\n'
       '(right_elbow: NaN retained where originally NaN)', 'io'); y -= S
    rb(ax, 0, y, 2.4, H, 'END', 'term', fs=9.5, r=0.26)

    return fig


def chart_depth_3dhuman():
    fw, fh = 5.8, 21.0
    fig, ax = _new(fw, fh, 'Pipeline: interpolate_depth_3dhuman.py\n'
                             '(Depth assistance + 3-D Human Kinematic Constraint)')

    y = -1.0
    rb(ax, 0, y, 2.4, H, 'START', 'term', fs=9.5, r=0.26);  y -= S
    rb(ax, 0, y, W,   H, 'Input: session directory\n(N frames, 4 arm joints)', 'io'); y -= S

    y = _depth_core(ax, y, has_r2=False, has_kin=True)

    rb(ax, 0, y, W, H, 'Write pose_<joint>_processed.txt\n(4 joints, all N frames)', 'io'); y -= S
    rb(ax, 0, y, 2.4, H, 'END', 'term', fs=9.5, r=0.26)

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Chart 7 — data_recording_45mm.py
# ─────────────────────────────────────────────────────────────────────────────
def chart_data_recording():
    fw, fh = 5.8, 22.0
    fig, ax = _new(fw, fh, 'Pipeline: data_recording_45mm.py\n'
                             '(L515 floor  +  D435i head-mounted, 45 mm ArUco)')

    y = -1.0
    rb(ax, 0, y, 2.4, H, 'START', 'term', fs=9.5, r=0.26);  y -= S

    # ── Setup ────────────────────────────────────────────────────────────────
    sbg(ax, 0, y+H/2+0.08, y-H*2-S-0.08, W+0.15, 'Setup',
        fc='#e3f2fd', ec='#90caf9')
    rb(ax, 0, y, W, H,
       'Initialise L515 + D435i pipelines\n'
       '(30 fps, aligned depth, DICT_4X4_100)',
       'cam'); y -= S
    rb(ax, 0, y, W, H,
       'Discard 60 warm-up frames\n'
       '(auto-exposure + gain stabilisation)',
       'cam'); y -= S

    # ── Init Phase ────────────────────────────────────────────────────────────
    sbg(ax, 0, y+H/2+0.08, y-H*3-S*2-0.08, W+0.15, 'Init Phase',
        fc='#f3e5f5', ec='#ce93d8')
    rb(ax, 0, y, W, H,
       'Live preview — position cameras until\n'
       'ArUco ID 10 (world origin) is visible in both',
       'init'); y -= S
    rb(ax, 0, y, W, H,
       'User presses S / Enter\n'
       '→ begin static-transform averaging',
       'proc'); y -= S
    rb(ax, 0, y, W, H,
       'Average 60 valid frames per camera\n'
       '→ compute T_world_L515 and T_world_ID<k>',
       'init'); y -= S
    rb(ax, 0, y, W, H,
       'Save T_world_L515.txt  +  T_world_ID<k>.txt\n'
       '(static transforms, fixed for this session)',
       'io'); y -= S

    # ── Recording Phase ───────────────────────────────────────────────────────
    sbg(ax, 0, y+H/2+0.08, y-H*5-S*4-0.08, W+0.15, 'Recording Phase',
        fc='#e8f5e9', ec='#a5d6a7')
    rb(ax, 0, y, W, H,
       'Capture synchronised frame from L515 + D435i', 'rec'); y -= S

    rb(ax, 0, y, W, H*1.15,
       'L515 processing:\n'
       '  • Align depth → colour\n'
       '  • Back-project → 3-D point cloud (world frame)\n'
       '  • Brightness-weighted downsample → 8 192 pts',
       'dep', fs=8.2); y -= S*1.05

    rb(ax, 0, y, W, H*1.15,
       'D435i processing (90° CCW correction):\n'
       '  • Detect floor markers → update T_world_head\n'
       '  • Detect arm markers (IDs 0–23, 45 mm)\n'
       '  • Compute T_world_arm per joint (world frame)',
       'cam', fs=8.2); y -= S*1.05

    rb(ax, 0, y, W, H,
       'Buffer frame data in RAM\n'
       '(colour, depth, point cloud, poses, phase label)',
       'proc'); y -= S

    # Decision: stop?
    db(ax, 0, y, WD, HD,
       'Q pressed  or\ntotal_frames reached?', 'dec'); y_dec = y
    y -= S*1.1

    # YES branch continues down
    rb(ax, 0, y, W, H,
       'Flush buffer to disk\n'
       '(ThreadPoolExecutor — parallel frame writes)',
       'rec'); y -= S*0.9

    # NO loop-back (right side, back to "Capture frame")
    loop_x = fw/2 - 0.22
    loop_y_top = y_dec - HD/2 - 0.15   # just below diamond
    loop_y_bot = y_dec + HD/2 + 0.12 + S*4.3  # above "Capture frame"

    pa(ax, [(WD/2, y_dec),
            (loop_x, y_dec),
            (loop_x, loop_y_bot),
            (W/2+0.05, loop_y_bot)],
       'No', ldx=0.22, ldy=0.12)

    # YES arrow (down from diamond)
    ar(ax, 0, y_dec-HD/2, 0, y_dec-HD/2-S*1.1+H/2, 'Yes', ldx=0.25, ldy=0)

    rb(ax, 0, y, W, H,
       'Output: saved_data/<timestamp>/frame_N/\n'
       '  colour.png · depth.png · pointcloud.npy\n'
       '  pose_<joint>.txt · pose.txt · phase.txt',
       'io', fs=8); y -= S

    rb(ax, 0, y, 2.4, H, 'END', 'term', fs=9.5, r=0.26)

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
CHARTS = [
    ('flowchart_interpolate_poses',          chart_poses),
    ('flowchart_interpolate_modify',         chart_modify),
    ('flowchart_interpolate_modify_R2',      chart_modify_R2),
    ('flowchart_interpolate_depth',          chart_depth),
    ('flowchart_interpolate_depth_R2',       chart_depth_R2),
    ('flowchart_interpolate_depth_3dhuman',  chart_depth_3dhuman),
    ('flowchart_data_recording_45mm',        chart_data_recording),
]

if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    for name, fn in CHARTS:
        print(f'Generating {name}.png ...')
        fig = fn()
        path = os.path.join(OUT, name + '.png')
        fig.savefig(path, dpi=DPI, bbox_inches='tight', pad_inches=0.25,
                    facecolor='white', edgecolor='none')
        plt.close(fig)
        print(f'  Saved → {path}')
    print('Done.')
