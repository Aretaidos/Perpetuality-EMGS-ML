#!/bin/bash

##############################################################################
# PyTorch CUDA Compatibility Fix Script
#
# Problem: The conda environment has PyTorch built for CUDA 12.4, but UVA
#          GPU servers have CUDA 11.8 or 12.8.1 available. This mismatch
#          prevents torch.cuda.is_available() from returning True.
#
# Solution: Reinstall PyTorch with CUDA 11.8 support using pip.
#
# Usage:
#   bash fix_pytorch_cuda.sh [cuda_version]
#
# Examples:
#   bash fix_pytorch_cuda.sh         # Uses CUDA 11.8 (default)
#   bash fix_pytorch_cuda.sh cu118   # Explicit CUDA 11.8
#   bash fix_pytorch_cuda.sh cu121   # CUDA 12.1
##############################################################################

set -e  # Exit on error

echo "==========================================="
echo "  PyTorch CUDA Compatibility Fix"
echo "==========================================="
echo ""

# Determine CUDA version to use
CUDA_VERSION="${1:-cu118}"  # Default to CUDA 11.8

case ${CUDA_VERSION} in
    cu118)
        CUDA_NAME="CUDA 11.8"
        MODULE_VERSION="cuda/11.8.0"
        ;;
    cu121)
        CUDA_NAME="CUDA 12.1"
        MODULE_VERSION="cuda/12.8.1"  # UVA has 12.8.1, compatible with 12.1
        ;;
    *)
        echo "ERROR: Unknown CUDA version '${CUDA_VERSION}'"
        echo "Supported versions: cu118, cu121"
        exit 1
        ;;
esac

echo "[1/6] Configuration:"
echo "  CUDA version: ${CUDA_NAME} (${CUDA_VERSION})"
echo "  Module: ${MODULE_VERSION}"
echo ""

# Load modules
echo "[2/6] Loading modules..."
module purge
module load miniforge
module load ${MODULE_VERSION}

# Check if CUDA module loaded successfully
if ! module list 2>&1 | grep -q cuda; then
    echo "ERROR: Failed to load CUDA module"
    exit 1
fi

echo "  ✓ Modules loaded successfully"
echo ""

# Activate conda environment
echo "[3/6] Activating conda environment..."
eval "$(conda shell.bash hook)"

if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" == "base" ]; then
    conda activate neuromotor
fi

if [ "$CONDA_DEFAULT_ENV" != "neuromotor" ]; then
    echo "ERROR: Failed to activate neuromotor environment"
    echo "Current environment: $CONDA_DEFAULT_ENV"
    exit 1
fi

echo "  ✓ Environment activated: $CONDA_DEFAULT_ENV"
echo ""

# Check current PyTorch version and CUDA support
echo "[4/6] Current PyTorch status:"
python3 -c "import torch; print(f'  PyTorch version: {torch.__version__}'); print(f'  CUDA available: {torch.cuda.is_available()}'); print(f'  PyTorch built for CUDA: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')" || echo "  (PyTorch not properly configured)"
echo ""

# Uninstall current PyTorch
echo "[5/6] Uninstalling current PyTorch packages..."
pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
echo "  ✓ PyTorch uninstalled"
echo ""

# Reinstall PyTorch with correct CUDA version
echo "[6/6] Installing PyTorch with ${CUDA_NAME} support..."
echo "  This may take 2-3 minutes..."
echo ""

pip install torch==2.4.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/${CUDA_VERSION}

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Failed to install PyTorch"
    exit 1
fi

echo ""
echo "  ✓ PyTorch installed successfully"
echo ""

# Verify installation
echo "==========================================="
echo "  Verification"
echo "==========================================="
echo ""

# Check PyTorch version
echo "New PyTorch status:"
python3 -c "import torch; print(f'  PyTorch version: {torch.__version__}'); print(f'  PyTorch built for CUDA: {torch.version.cuda}')"
echo ""

# Check CUDA availability
echo "CUDA Detection Test:"
python3 -c "
import torch
import sys

cuda_available = torch.cuda.is_available()
print(f'  CUDA available: {cuda_available}')

if cuda_available:
    device_count = torch.cuda.device_count()
    print(f'  GPU count: {device_count}')
    print('')
    print('  Available GPUs:')
    for i in range(device_count):
        name = torch.cuda.get_device_name(i)
        memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f'    GPU {i}: {name} ({memory:.1f} GB)')
    print('')
    print('  ✓ GPU access verified!')
    sys.exit(0)
else:
    print('')
    print('  ✗ CUDA not available!')
    print('')
    print('  Troubleshooting:')
    print('    1. Check CUDA module is loaded: module list')
    print('    2. Check LD_LIBRARY_PATH includes CUDA: echo \$LD_LIBRARY_PATH')
    print('    3. Try loading CUDA module again: module load ${MODULE_VERSION}')
    print('    4. Try different CUDA version: bash fix_pytorch_cuda.sh cu121')
    sys.exit(1)
"

VERIFICATION_EXIT=$?

echo ""
echo "==========================================="

if [ ${VERIFICATION_EXIT} -eq 0 ]; then
    echo "  ✓ PyTorch CUDA fix successful!"
    echo ""
    echo "Next steps:"
    echo "  1. Verify GPU access: bash verify_gpu.sh"
    echo "  2. Submit training job: sbatch slurm_train_all_gpus.sbatch"
else
    echo "  ✗ PyTorch CUDA fix incomplete"
    echo ""
    echo "Please check the troubleshooting steps above."
    exit 1
fi

echo "==========================================="
