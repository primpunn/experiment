#!/usr/bin/env bash
# ============================================================
# Lab computer setup — clones all repos, builds librealsense,
# creates the 'massage' conda environment, and downloads all
# model files. Run once on a fresh Ubuntu machine with a GPU.
#
# BEFORE running this script, you (the user) must do 2 things:
#
#   1. Install NVIDIA driver and reboot:
#        sudo ubuntu-drivers autoinstall && sudo reboot
#
#   2. Log in to HuggingFace (needs facebook/sam-3d-body-dinov3 access):
#        pip install huggingface_hub
#        huggingface-cli login
#        (enter your HuggingFace token when prompted)
#
#   3. Download SMPLX body models MANUALLY (requires website registration):
#        - Go to: https://smpl-x.is.tue.mpg.de/
#        - Register and download "SMPL-X Models" (models_smplx_v1_1.zip)
#        - Extract and place files at: ~/models/smplx/
#        - Required file: ~/models/smplx/SMPLX_NEUTRAL.npz
#        Claude Code CANNOT do this step — it requires your personal login.
#
# Usage:
#   chmod +x setup_lab.sh
#   ./setup_lab.sh
# ============================================================
set -e

EXPERIMENT_REPO="https://github.com/primpunn/experiment.git"
SAM3D_BODY_REPO="https://github.com/facebookresearch/sam-3d-body.git"
SAM_BODY4D_REPO="https://github.com/gaomingqi/sam-body4d.git"
LIBREALSENSE_COMMIT="2dbaaf596"

EXPERIMENT_DIR="$HOME/experiment"
SAM3D_DIR="$HOME/sam-3d-body"
SAM_BODY4D_DIR="$HOME/sam-body4d"
LIBREALSENSE_DIR="$HOME/librealsense"
CKPT_DIR="$HOME/checkpoints/sam3dbody"
SMPLX_DIR="$HOME/models/smplx"
MEDIAPIPE_MODELS_DIR="$EXPERIMENT_DIR/both/models"

echo "=============================================="
echo " Lab Environment Setup"
echo "=============================================="

# ---- 0. Check NVIDIA driver -------------------------------------------
echo ""
echo "[Step 0] Checking NVIDIA driver..."
if ! nvidia-smi &>/dev/null; then
    echo "  [ERROR] NVIDIA driver not found."
    echo "  Fix: sudo ubuntu-drivers autoinstall && sudo reboot"
    echo "  Then re-run this script."
    exit 1
fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo "  Driver OK."

# ---- 1. System packages -----------------------------------------------
echo ""
echo "[Step 1] Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y \
    git cmake build-essential \
    libssl-dev libusb-1.0-0-dev libudev-dev \
    pkg-config libgtk-3-dev \
    libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev \
    python3-dev curl wget

# ---- 2. Install Miniconda (if not present) ----------------------------
echo ""
echo "[Step 2] Checking Miniconda..."
if ! command -v conda &>/dev/null; then
    echo "  conda not found — installing Miniconda..."
    curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
    rm /tmp/miniconda.sh
    export PATH="$HOME/miniconda3/bin:$PATH"
    conda init bash
    echo "  Miniconda installed. You may need to restart your shell after setup."
else
    echo "  conda found: $(conda --version)"
fi

# ---- 3. Clone repositories --------------------------------------------
echo ""
echo "[Step 3] Cloning repositories..."

if [ -d "$EXPERIMENT_DIR" ]; then
    echo "  experiment: already exists, pulling latest..."
    git -C "$EXPERIMENT_DIR" pull
else
    git clone "$EXPERIMENT_REPO" "$EXPERIMENT_DIR"
    echo "  experiment: cloned."
fi

if [ -d "$SAM3D_DIR" ]; then
    echo "  sam-3d-body: already exists, skipping."
else
    git clone "$SAM3D_BODY_REPO" "$SAM3D_DIR"
    echo "  sam-3d-body: cloned."
fi

if [ -d "$SAM_BODY4D_DIR" ]; then
    echo "  sam-body4d: already exists, skipping."
else
    git clone "$SAM_BODY4D_REPO" "$SAM_BODY4D_DIR"
    echo "  sam-body4d: cloned."
fi

# ---- 4. Build librealsense from source --------------------------------
# pip pyrealsense2 does NOT support the L515 camera.
# Must build from source with FORCE_RSUSB_BACKEND=ON.
echo ""
echo "[Step 4] Building librealsense from source (L515 + D435i support)..."

if [ -f "$LIBREALSENSE_DIR/build/Release/pyrealsense2.cpython-310-x86_64-linux-gnu.so" ]; then
    echo "  librealsense already built, skipping."
else
    if [ ! -d "$LIBREALSENSE_DIR" ]; then
        git clone https://github.com/IntelRealSense/librealsense.git "$LIBREALSENSE_DIR"
    fi
    cd "$LIBREALSENSE_DIR"
    git checkout "$LIBREALSENSE_COMMIT"

    mkdir -p build && cd build
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_EXAMPLES=ON \
        -DBUILD_PYTHON_BINDINGS=ON \
        -DBUILD_WITH_CUDA=OFF \
        -DFORCE_RSUSB_BACKEND=ON \
        -DPYTHON_INSTALL_DIR=OFF

    make -j"$(nproc)"
    echo "  librealsense built successfully."
    cd "$HOME"
fi

# ---- 5. udev rules for RealSense cameras ------------------------------
echo ""
echo "[Step 5] Setting up udev rules for RealSense cameras..."
UDEV_SRC="$LIBREALSENSE_DIR/config/99-realsense-libusb.rules"
UDEV_DST="/etc/udev/rules.d/99-realsense-libusb.rules"
if [ -f "$UDEV_DST" ]; then
    echo "  udev rules already installed."
else
    sudo cp "$UDEV_SRC" "$UDEV_DST"
    sudo udevadm control --reload-rules
    sudo udevadm trigger
    echo "  udev rules installed."
fi

# ---- 6. Create conda environment --------------------------------------
echo ""
echo "[Step 6] Creating 'massage' conda environment from environment.yml..."
ENV_YML="$EXPERIMENT_DIR/environment.yml"

if conda env list | grep -q "^massage"; then
    echo "  'massage' env already exists, skipping."
    echo "  To recreate: conda env remove -n massage -y  then re-run."
else
    conda env create -f "$ENV_YML"
    echo "  'massage' environment created."
fi

# ---- 7. Install sam-3d-body package into env --------------------------
echo ""
echo "[Step 7] Installing sam-3d-body into massage env..."
if [ -f "$SAM3D_DIR/pyproject.toml" ] || [ -f "$SAM3D_DIR/setup.py" ]; then
    conda run -n massage --no-capture-output pip install -e "$SAM3D_DIR" --no-deps
    echo "  sam-3d-body installed."
else
    echo "  No pyproject.toml/setup.py in sam-3d-body — skipping."
fi

# ---- 8. Download SAM-3D-Body checkpoints from HuggingFace ------------
echo ""
echo "[Step 8] Downloading SAM-3D-Body checkpoints (~2GB)..."
mkdir -p "$CKPT_DIR"

if [ -f "$CKPT_DIR/model.ckpt" ]; then
    echo "  Checkpoint already at $CKPT_DIR, skipping."
else
    conda run -n massage --no-capture-output python - <<'PYEOF'
import os
from huggingface_hub import snapshot_download
ckpt_dir = os.path.expanduser("~/checkpoints/sam3dbody")
print(f"  Downloading to: {ckpt_dir}")
snapshot_download(
    repo_id="facebook/sam-3d-body-dinov3",
    local_dir=ckpt_dir,
    ignore_patterns=["*.md", "*.txt"],
)
print("  Download complete.")
PYEOF
fi

# ---- 9. Download MediaPipe model files --------------------------------
# Used by estimate_shape.py and interpolate_mosh_pipeline.py.
# These are public files — no login needed.
echo ""
echo "[Step 9] Downloading MediaPipe model files..."
mkdir -p "$MEDIAPIPE_MODELS_DIR"

POSE_MODEL="$MEDIAPIPE_MODELS_DIR/pose_landmarker_lite.task"
SEG_MODEL="$MEDIAPIPE_MODELS_DIR/selfie_segmenter_landscape.tflite"

if [ -f "$POSE_MODEL" ]; then
    echo "  pose_landmarker_lite.task already exists, skipping."
else
    wget -q --show-progress \
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task" \
        -O "$POSE_MODEL"
    echo "  pose_landmarker_lite.task downloaded."
fi

if [ -f "$SEG_MODEL" ]; then
    echo "  selfie_segmenter_landscape.tflite already exists, skipping."
else
    wget -q --show-progress \
        "https://storage.googleapis.com/mediapipe-assets/selfie_segmenter_landscape.tflite" \
        -O "$SEG_MODEL"
    echo "  selfie_segmenter_landscape.tflite downloaded."
fi

# ---- 10. Check SMPLX models (must be done manually) ------------------
echo ""
echo "[Step 10] Checking SMPLX body models..."
if [ -f "$SMPLX_DIR/SMPLX_NEUTRAL.npz" ]; then
    echo "  SMPLX models found at $SMPLX_DIR  (OK)"
else
    echo "  [WARNING] SMPLX models NOT found at $SMPLX_DIR"
    echo ""
    echo "  *** YOU MUST DO THIS MANUALLY ***"
    echo "  1. Go to: https://smpl-x.is.tue.mpg.de/"
    echo "  2. Register and download: models_smplx_v1_1.zip"
    echo "  3. Extract and place files at: $SMPLX_DIR/"
    echo "  Required file: $SMPLX_DIR/SMPLX_NEUTRAL.npz"
    echo "  Used by: estimate_shape.py, check_requirements.py"
    echo ""
fi

# ---- Final summary ----------------------------------------------------
echo ""
echo "=============================================="
echo " Setup complete. Final checklist:"
echo ""
echo "  [AUTO] experiment repo     → $EXPERIMENT_DIR"
echo "  [AUTO] sam-3d-body         → $SAM3D_DIR"
echo "  [AUTO] sam-body4d          → $SAM_BODY4D_DIR"
echo "  [AUTO] librealsense build  → $LIBREALSENSE_DIR/build/Release"
echo "  [AUTO] udev rules          → /etc/udev/rules.d/"
echo "  [AUTO] conda env massage   → Python 3.10, all packages"
echo "  [AUTO] SAM-3D-Body ckpt    → $CKPT_DIR"
echo "  [AUTO] MediaPipe models    → $MEDIAPIPE_MODELS_DIR"
echo ""
if [ ! -f "$SMPLX_DIR/SMPLX_NEUTRAL.npz" ]; then
echo "  [MANUAL - NOT DONE] SMPLX models → $SMPLX_DIR"
echo "    Download from https://smpl-x.is.tue.mpg.de/"
else
echo "  [MANUAL] SMPLX models      → $SMPLX_DIR  (OK)"
fi
echo ""
echo " To start working:"
echo "  conda activate massage"
echo "  cd $EXPERIMENT_DIR/both"
echo "  python data_recording.py -o ./saved_data"
echo "=============================================="
