#!/bin/bash
# Install both MEMO conda envs.
#
#   memo          - data preparation, training, rendering, extraction, evaluation.
#   memo-blender  - Blender (bpy 4.5.9, numpy<2) for the video script's default renderer.
#
# Detects the local CUDA toolkit version via nvcc and installs the newest
# torch and torchvision built for it into memo, from the PyTorch index for
# that CUDA version only. The torch-dependent packages and the CUDA
# extensions then build against that torch.
#
# Usage:  bash install.sh
#
# Prereqs:
#  - conda on PATH
#  - CUDA toolkit available (nvcc on PATH)
#  - gcc available for the CUDA extension builds

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEMO_YAML="${SCRIPT_DIR}/environment.yml"
BLENDER_YAML="${SCRIPT_DIR}/environment_blender.yml"

# gDel3D (GPU Delaunay) through its Python binding. MEMO's meshes are built
# with the single-precision triangulation, whereas the binding defaults to
# double precision.
PYGDEL3D_URL="git+https://github.com/half-potato/pyGDel3D.git@245c717"
NVDIFFRAST_URL="git+https://github.com/NVlabs/nvdiffrast.git@v0.3.5"
# pytorch3d's GPU k-nearest-neighbor kernel serves the 4D initialization's
# track matching. It has no wheels for recent torch builds, so it is built
# from source against the env's torch.
PYTORCH3D_URL="git+https://github.com/facebookresearch/pytorch3d.git@stable"

echo "==[install.sh]== Step 1/6: Prerequisites"

if ! command -v conda &>/dev/null; then
    echo "ERROR: conda not on PATH. Install miniconda first." >&2
    exit 1
fi

if ! command -v nvcc &>/dev/null; then
    echo "ERROR: nvcc not on PATH. Make the CUDA toolkit available" >&2
    echo "  (for example by loading your cluster's CUDA module) before running this script." >&2
    exit 1
fi

CUDA_VERSION=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+' | head -1)
if [ -z "${CUDA_VERSION}" ]; then
    echo "ERROR: failed to parse CUDA version from nvcc --version" >&2
    exit 1
fi
CUDA_TAG="cu$(echo "${CUDA_VERSION}" | tr -d .)"
TORCH_INDEX="https://download.pytorch.org/whl/${CUDA_TAG}"

echo "  conda:   $(command -v conda)"
echo "  CUDA:    ${CUDA_VERSION} → torch index: ${TORCH_INDEX}"

echo
echo "==[install.sh]== Step 2/6: Create 'memo' env"
conda env create -f "${MEMO_YAML}"

echo
echo "==[install.sh]== Step 3/6: torch for CUDA ${CUDA_VERSION}, then the torch-dependent packages"
if ! conda run -n memo pip install torch torchvision --index-url "${TORCH_INDEX}"; then
    echo "ERROR: no torch build for CUDA ${CUDA_VERSION} at ${TORCH_INDEX}." >&2
    echo "  Use a CUDA toolkit that PyTorch publishes wheels for (see pytorch.org)." >&2
    exit 1
fi
conda run -n memo pip install torchmetrics

echo
echo "==[install.sh]== Step 4/6: Build the CUDA extensions against the env's torch"
echo "  (takes several minutes, pytorch3d the longest)"
# FORCE_CUDA builds the GPU kernels even where the build cannot see a GPU.
FORCE_CUDA=1 conda run -n memo pip install "${PYTORCH3D_URL}" --no-build-isolation
for ext in diff-gaussian-rasterization_memo diff-gaussian-rasterization_sof fused-ssim simple-knn; do
    conda run -n memo pip install "${SCRIPT_DIR}/submodules/${ext}" --no-build-isolation
done
USE_SINGLE_PRECISION=1 conda run -n memo pip install "${PYGDEL3D_URL}" --no-deps --no-build-isolation
conda run -n memo pip install "${NVDIFFRAST_URL}" --no-build-isolation

echo
echo "==[install.sh]== Step 5/6: Build nvdiffrast's CUDA plugin"
# nvdiffrast otherwise compiles its plugin on first use, for the local GPU.
if conda run -n memo python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)"; then
    conda run -n memo python -c "import nvdiffrast.torch as dr; dr.RasterizeCudaContext()"
else
    echo "  No GPU visible here, so the plugin is built on first use instead."
fi

echo
echo "==[install.sh]== Step 6/6: Create 'memo-blender' env"
conda env create -f "${BLENDER_YAML}"

conda run -n memo python -c \
    "import torch; print(f'  torch {torch.__version__}, CUDA {torch.version.cuda}')"
echo
echo "==[install.sh]== Done."
echo "  conda activate memo"
