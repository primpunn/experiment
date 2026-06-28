#!/usr/bin/env bash
# ============================================================
# Complete setup script for SAM-3D-Body arm tracking pipeline
# Run this AFTER installing the NVIDIA driver and rebooting.
#
# Usage:
#   chmod +x setup_sam_pipeline.sh
#   ./setup_sam_pipeline.sh
# ============================================================
set -e

echo "=============================="
echo " SAM-3D-Body Pipeline Setup"
echo "=============================="

# ---- 0. Free disk space -----------------------------------------------
echo ""
echo "[Step 0] Freeing disk space..."
BEFORE=$(df -BG / | awk 'NR==2{print $4}')
echo "  Disk free before: $BEFORE"

pip cache purge 2>/dev/null || true
conda clean --all -y 2>/dev/null || true
# Uncomment the following lines if 'focap' or 'hil' environments are unused:
# conda env remove -n hil -y 2>/dev/null || true
# conda env remove -n focap -y 2>/dev/null || true

AFTER=$(df -BG / | awk 'NR==2{print $4}')
echo "  Disk free after:  $AFTER"

FREE_GB=$(df -BG / | awk 'NR==2{gsub("G",""); print $4}')
if [ "$FREE_GB" -lt 12 ]; then
    echo ""
    echo "[ERROR] Only ${FREE_GB}G free. Need ≥12 GB."
    echo "  To free more space, uncomment the 'conda env remove' lines above"
    echo "  (remove hil env: ~5 GB, focap env: ~2 GB)."
    exit 1
fi

# ---- 1. Verify NVIDIA driver -------------------------------------------
echo ""
echo "[Step 1] Checking NVIDIA driver..."
if ! nvidia-smi &>/dev/null; then
    echo "[ERROR] NVIDIA driver not loaded."
    echo "  Fix: sudo ubuntu-drivers autoinstall && sudo reboot"
    echo "  (GTX 1650 Mobile is detected at hardware level — driver missing)"
    exit 1
fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo "Driver OK."

# ---- 2. Create conda environment ---------------------------------------
echo ""
echo "[Step 2] Creating conda environment sam_3d_body (Python 3.11)..."
if conda env list | grep -q "^sam_3d_body"; then
    echo "  Environment already exists, skipping create."
else
    conda create -n sam_3d_body python=3.11 -y
fi

# Activate (for scripting use conda run)
CONDA_RUN="conda run -n sam_3d_body --no-capture-output"

# ---- 3. Install PyTorch ------------------------------------------------
echo ""
echo "[Step 3] Installing PyTorch with CUDA 12.1..."
$CONDA_RUN pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# Quick CUDA check
$CONDA_RUN python -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
"

# ---- 4. Install Python dependencies ------------------------------------
echo ""
echo "[Step 4] Installing Python dependencies..."
$CONDA_RUN pip install \
    pytorch-lightning \
    pyrender \
    "opencv-python>=4.8" \
    yacs \
    scikit-image \
    einops \
    timm \
    dill \
    pandas \
    rich \
    hydra-core \
    hydra-submitit-launcher \
    hydra-colorlog \
    pyrootutils \
    webdataset \
    "networkx==3.2.1" \
    roma \
    joblib \
    seaborn \
    wandb \
    appdirs \
    cython \
    jsonlines \
    pytest \
    loguru \
    optree \
    fvcore \
    pycocotools \
    tensorboard \
    huggingface_hub \
    smplx \
    matplotlib \
    numpy \
    scipy \
    xtcocotools

# ---- 5. Install Detectron2 ---------------------------------------------
echo ""
echo "[Step 5] Installing Detectron2..."
$CONDA_RUN pip install \
    'git+https://github.com/facebookresearch/detectron2.git@a1ce2f9' \
    --no-build-isolation --no-deps

# ---- 6. Clone SAM-3D-Body ----------------------------------------------
echo ""
echo "[Step 6] Cloning SAM-3D-Body..."
SAM3D_DIR="$HOME/sam-3d-body"
if [ -d "$SAM3D_DIR" ]; then
    echo "  Already cloned at $SAM3D_DIR"
else
    git clone https://github.com/facebookresearch/sam-3d-body.git "$SAM3D_DIR"
fi

# Install the repo itself if it has a setup.py / pyproject.toml
if [ -f "$SAM3D_DIR/setup.py" ] || [ -f "$SAM3D_DIR/pyproject.toml" ]; then
    $CONDA_RUN pip install -e "$SAM3D_DIR" --no-deps
fi

# ---- 7. Download checkpoints from HuggingFace --------------------------
echo ""
echo "[Step 7] Downloading SAM-3D-Body checkpoints..."
CKPT_DIR="$HOME/checkpoints/sam3dbody"
mkdir -p "$CKPT_DIR"

$CONDA_RUN python - <<'PYEOF'
import os
from huggingface_hub import snapshot_download

ckpt_dir = os.path.expanduser("~/checkpoints/sam3dbody")
print(f"Downloading to: {ckpt_dir}")
snapshot_download(
    repo_id="facebook/sam-3d-body-dinov3",
    local_dir=ckpt_dir,
    ignore_patterns=["*.md", "*.txt"],
)
print("Download complete.")
PYEOF

echo ""
echo "[Step 8] SMPLX body model: checking..."
SMPLX_FILE="$HOME/models/smplx/SMPLX_NEUTRAL.npz"
if [ -f "$SMPLX_FILE" ]; then
    echo "  Found: $SMPLX_FILE  (OK)"
else
    echo "  NOT FOUND at $SMPLX_FILE"
    echo "  Download from https://smpl-x.is.tue.mpg.de/ and place at $SMPLX_FILE"
fi

# ---- Final check -------------------------------------------------------
echo ""
echo "[Done] Running requirements check..."
conda run -n sam_3d_body --no-capture-output \
    python "$(dirname "$0")/check_requirements.py"
