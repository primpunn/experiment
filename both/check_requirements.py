"""
Environment requirements check for SAM-3D-Body pipeline.
Run this before setup to get a clear go/no-go on each requirement.

Usage: python check_requirements.py
"""

import os
import shutil
import subprocess
import sys


def check(label: str, ok: bool, detail: str = "", fix: str = ""):
    status = "OK" if ok else "FAIL"
    print(f"  [{status}] {label}")
    if detail:
        print(f"         {detail}")
    if not ok and fix:
        print(f"         FIX: {fix}")
    return ok


def bytes_to_gb(b: int) -> float:
    return b / 1024 ** 3


def main():
    print("=" * 60)
    print("SAM-3D-Body Pipeline — Requirements Check")
    print("=" * 60)
    all_ok = True

    # ---- Python version ----
    print("\n[Python]")
    maj, minor = sys.version_info.major, sys.version_info.minor
    py_ok = (maj == 3 and minor == 11)
    all_ok &= check(
        f"Python 3.11 (found {maj}.{minor})", py_ok,
        fix="conda create -n sam_3d_body python=3.11 -y  &&  conda activate sam_3d_body"
    )

    # ---- Disk space ----
    print("\n[Disk space]")
    root_usage = shutil.disk_usage("/")
    free_gb = bytes_to_gb(root_usage.free)
    disk_ok = free_gb >= 12
    all_ok &= check(
        f"≥12 GB free on / (found {free_gb:.1f} GB)", disk_ok,
        fix=(
            "Run the following to free space:\n"
            "         pip cache purge               # frees ~6 GB\n"
            "         conda clean --all -y          # frees ~1 GB\n"
            "         conda env remove -n hil       # frees ~5 GB if unused"
        )
    )

    pip_cache = os.path.expanduser("~/.cache/pip")
    if os.path.isdir(pip_cache):
        pip_gb = bytes_to_gb(sum(
            os.path.getsize(os.path.join(dp, f))
            for dp, _, fns in os.walk(pip_cache)
            for f in fns
        ))
        check(
            f"Pip cache size (reclaimable): {pip_gb:.1f} GB",
            True,
            detail="Run: pip cache purge   to free this"
        )

    # ---- GPU / CUDA ----
    print("\n[GPU & CUDA]")
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        gpu_ok = result.returncode == 0 and bool(result.stdout.strip())
        gpu_detail = result.stdout.strip() if gpu_ok else result.stderr.strip()
    except FileNotFoundError:
        gpu_ok = False
        gpu_detail = "nvidia-smi not found"

    all_ok &= check(
        "NVIDIA driver loaded", gpu_ok, detail=gpu_detail,
        fix=(
            "sudo ubuntu-drivers autoinstall  &&  sudo reboot\n"
            "         (GTX 1650 Mobile detected at PCI level — driver just not loaded)"
        )
    )

    cuda_path = shutil.which("nvcc") or ""
    cuda_dir = "/usr/local/cuda"
    cuda_installed = bool(cuda_path) or os.path.isdir(cuda_dir)
    check(
        f"CUDA toolkit ({'found' if cuda_installed else 'not found'})",
        cuda_installed,
        detail=cuda_path or cuda_dir if cuda_installed else "",
        fix="sudo apt install nvidia-cuda-toolkit  (or install via conda-forge)"
    )

    try:
        import torch
        torch_ok = True
        cuda_torch = torch.cuda.is_available()
        check(f"PyTorch installed ({torch.__version__})", torch_ok)
        all_ok &= check(
            f"CUDA available in PyTorch ({cuda_torch})", cuda_torch,
            fix=(
                "pip install torch torchvision torchaudio "
                "--index-url https://download.pytorch.org/whl/cu121"
            )
        )
        if cuda_torch:
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            check(f"VRAM: {vram_gb:.1f} GB (≥3 GB needed)", vram_gb >= 3,
                  detail=torch.cuda.get_device_name(0))
    except ImportError:
        all_ok &= check("PyTorch installed", False,
                        fix="pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")

    # ---- RAM ----
    print("\n[RAM]")
    try:
        with open("/proc/meminfo") as f:
            lines = f.readlines()
        mem_total = int([l for l in lines if l.startswith("MemTotal")][0].split()[1]) * 1024
        mem_avail = int([l for l in lines if l.startswith("MemAvailable")][0].split()[1]) * 1024
        ram_gb = bytes_to_gb(mem_total)
        avail_gb = bytes_to_gb(mem_avail)
        check(f"Total RAM: {ram_gb:.1f} GB (8 GB recommended)", ram_gb >= 6,
              detail=f"Available now: {avail_gb:.1f} GB")
    except Exception:
        check("RAM check", False, detail="Could not read /proc/meminfo")

    # ---- Key packages ----
    print("\n[Key packages]")
    packages = {
        "cv2": "opencv-python",
        "smplx": "smplx",
        "numpy": "numpy",
        "pandas": "pandas",
        "scipy": "scipy",
        "matplotlib": "matplotlib",
        "huggingface_hub": "huggingface_hub",
    }
    for mod, pkg in packages.items():
        try:
            m = __import__(mod)
            ver = getattr(m, "__version__", "?")
            check(f"{pkg} ({ver})", True)
        except ImportError:
            check(f"{pkg}", False, fix=f"pip install {pkg}")

    # ---- SMPL model files ----
    print("\n[SMPL body model]")
    smpl_paths = [
        os.path.expanduser("~/models/smplx/SMPLX_NEUTRAL.npz"),
        os.path.expanduser("~/body_models/smplx/SMPLX_NEUTRAL.npz"),
        "./models/smplx/SMPLX_NEUTRAL.npz",
    ]
    smpl_found = any(os.path.isfile(p) for p in smpl_paths)
    found_path = next((p for p in smpl_paths if os.path.isfile(p)), None)
    check(
        "SMPL/SMPLX body model files", smpl_found,
        detail=found_path or "not found at: " + ", ".join(smpl_paths),
        fix=(
            "Download from https://smpl-x.is.tue.mpg.de/  (free registration)\n"
            "         Place at: ~/models/smplx/SMPLX_NEUTRAL.npz"
        )
    )

    # ---- SAM-3D-Body repo ----
    print("\n[SAM-3D-Body repo]")
    sam3d_paths = [
        os.path.expanduser("~/sam-3d-body"),
        "./sam-3d-body",
    ]
    sam3d_path = next((p for p in sam3d_paths if os.path.isdir(p)), None)
    check(
        "sam-3d-body repo cloned",
        sam3d_path is not None,
        detail=sam3d_path or "",
        fix="git clone https://github.com/facebookresearch/sam-3d-body.git ~/sam-3d-body"
    )

    # ---- Summary ----
    print("\n" + "=" * 60)
    if all_ok:
        print("All checks passed — ready to run the pipeline.")
    else:
        print("Some checks FAILED. Fix the items above before running.")
        print("Critical path: disk space → driver → CUDA → packages → model files.")
    print("=" * 60)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
