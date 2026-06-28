#!/usr/bin/env bash
# ============================================================
# Lab computer setup — clones all repos, builds librealsense,
# creates the 'massage' conda environment, and downloads model
# checkpoints. Run once on a fresh Ubuntu machine with a GPU.
#
# Prerequisites (do these manually before running):
#   1. Install NVIDIA driver:  sudo ubuntu-drivers autoinstall && sudo reboot
#   2. Log in to HuggingFace:  huggingface-cli login   (needs facebook/sam-3d-body-dinov3 access)
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
HF_REPO="facebook/sam-3d-body-dinov3"

EXPERIMENT_DIR="$HOME/experiment"
SAM3D_DIR="$HOME/sam-3d-body"
SAM_BODY4D_DIR="$HOME/sam-body4d"
LIBREALSENSE_DIR="$HOME/librealsense"
CKPT_DIR="$HOME/checkpoints/sam3dbody"

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
    python3-dev curl

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
# The pip pyrealsense2 does NOT support the L515 camera.
# We must build from source with FORCE_RSUSB_BACKEND=ON.
echo ""
echo "[Step 4] Building librealsense from source (L515 support)..."

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
    echo "  udev rules installed. Plug in cameras AFTER this step."
fi

# ---- 6. Create conda environment --------------------------------------
echo ""
echo "[Step 6] Creating 'massage' conda environment from environment.yml..."
ENV_YML="$EXPERIMENT_DIR/environment.yml"

if conda env list | grep -q "^massage"; then
    echo "  'massage' env already exists, skipping create."
    echo "  To recreate: conda env remove -n massage -y && re-run this script."
else
    conda env create -f "$ENV_YML"
    echo "  'massage' environment created."
fi

# ---- 7. Install sam-3d-body package into env --------------------------
echo ""
echo "[Step 7] Installing sam-3d-body package into massage env..."
if [ -f "$SAM3D_DIR/pyproject.toml" ] || [ -f "$SAM3D_DIR/setup.py" ]; then
    conda run -n massage --no-capture-output pip install -e "$SAM3D_DIR" --no-deps
    echo "  sam-3d-body installed."
else
    echo "  No pyproject.toml/setup.py found in sam-3d-body — skipping pip install."
fi

# ---- 8. Download model checkpoints ------------------------------------
echo ""
echo "[Step 8] Downloading SAM-3D-Body checkpoints from HuggingFace (~2GB)..."
mkdir -p "$CKPT_DIR"

if [ -f "$CKPT_DIR/model.ckpt" ]; then
    echo "  Checkpoint already downloaded at $CKPT_DIR"
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

# ---- 9. Summary -------------------------------------------------------
echo ""
echo "=============================================="
echo " Setup complete. Summary:"
echo "  experiment    → $EXPERIMENT_DIR"
echo "  sam-3d-body   → $SAM3D_DIR"
echo "  sam-body4d    → $SAM_BODY4D_DIR"
echo "  librealsense  → $LIBREALSENSE_DIR/build/Release"
echo "  checkpoints   → $CKPT_DIR"
echo "  conda env     → massage (Python 3.10)"
echo ""
echo " Next steps:"
echo "  1. Plug in L515 and D435i cameras"
echo "  2. conda activate massage"
echo "  3. cd $EXPERIMENT_DIR/both"
echo "  4. python data_recording.py -o ./saved_data"
echo ""
echo " NOTE: Scripts that use RealSense already prepend"
echo "  $LIBREALSENSE_DIR/build/Release to sys.path."
echo "  No extra steps needed for L515 support."
echo "=============================================="
