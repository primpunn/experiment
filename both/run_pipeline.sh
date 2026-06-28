#!/usr/bin/env bash
# ============================================================
# End-to-end arm tracking pipeline runner
#
# Usage:
#   # Full run with SAM-3D-Body (needs sam_3d_body env + GPU)
#   ./run_pipeline.sh
#
#   # Dry-run with synthetic data (no GPU needed, for testing)
#   ./run_pipeline.sh --mock
#
#   # Use a specific video file
#   ./run_pipeline.sh --video /path/to/my_video.mp4
# ============================================================
set -e

VIDEO="./saved_data/2026-05-11/rgb_video.mp4"
FRAMES_DIR="./frames"
OUTPUT_DIR="./output"
MOCK=false
ENV="massage"

for arg in "$@"; do
    case $arg in
        --mock) MOCK=true ;;
        --video=*) VIDEO="${arg#*=}" ;;
        --video) shift; VIDEO="$1" ;;
    esac
done

CONDA_RUN="conda run -n $ENV --no-capture-output"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=============================="
echo " Arm Tracking Pipeline"
echo " Mode: $([ "$MOCK" = true ] && echo MOCK || echo REAL)"
echo " Video: $VIDEO"
echo " Output: $OUTPUT_DIR"
echo "=============================="

# Step 3: Extract frames
echo ""
echo "[Step 3] Extracting frames..."
$CONDA_RUN python "$SCRIPT_DIR/preprocess_video.py" \
    --video "$VIDEO" \
    --output "$FRAMES_DIR"

# Step 4: Run inference
echo ""
echo "[Step 4] Running body pose inference..."
if [ "$MOCK" = true ]; then
    $CONDA_RUN python "$SCRIPT_DIR/run_sam_body4d.py" \
        --frames "$FRAMES_DIR" \
        --output "$OUTPUT_DIR" \
        --mock
else
    $CONDA_RUN python "$SCRIPT_DIR/run_sam_body4d.py" \
        --frames "$FRAMES_DIR" \
        --output "$OUTPUT_DIR"
fi

# Step 5: Extract arm trajectories
echo ""
echo "[Step 5] Extracting arm joint trajectories..."
$CONDA_RUN python "$SCRIPT_DIR/extract_arm_trajectory.py" \
    --input "$OUTPUT_DIR/body_params_per_frame.json" \
    --output "$OUTPUT_DIR/arm_trajectory.csv" \
    --fps 30

# Steps 6 & 7: Visualize + smooth
echo ""
echo "[Steps 6-7] Visualizing trajectories + applying smoothing..."
$CONDA_RUN python "$SCRIPT_DIR/visualize_trajectory.py" \
    --input "$OUTPUT_DIR/arm_trajectory.csv" \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "=============================="
echo " Pipeline complete."
echo " Outputs:"
echo "   $OUTPUT_DIR/body_params_per_frame.json"
echo "   $OUTPUT_DIR/arm_trajectory.csv"
echo "   $OUTPUT_DIR/arm_trajectory_smoothed.csv"
echo "   $OUTPUT_DIR/wrist_trajectory_3d.png"
echo "   $OUTPUT_DIR/wrist_trajectory_xyz.png"
echo "   $OUTPUT_DIR/arm_joints_xyz.png"
echo "=============================="
