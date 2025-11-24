#!/bin/bash

##############################################################################
# GPU Verification Script for UVA CS GPU Servers
#
# This script performs comprehensive checks to verify GPU access and
# PyTorch CUDA compatibility for sEMG model training.
#
# Usage:
#   bash verify_gpu.sh
##############################################################################

echo "==========================================="
echo "  GPU Verification Script"
echo "==========================================="
echo "Start time: $(date)"
echo ""

# Check if on GPU server
HOSTNAME=$(hostname)
echo "[1/8] System Information:"
echo "  Hostname: ${HOSTNAME}"
echo "  User: $(whoami)"
echo "  Working directory: $(pwd)"
echo ""

if [[ ! ${HOSTNAME} =~ ^gpusrv[0-9]+ ]]; then
    echo "  ⚠ Warning: Not on a gpusrv server!"
    echo "  Current host: ${HOSTNAME}"
    echo "  You may need to SSH to a GPU server (e.g., gpusrv17, gpusrv19)"
    echo ""
fi

# Check SLURM environment
echo "[2/8] SLURM Environment:"
if [ -n "$SLURM_JOB_ID" ]; then
    echo "  ✓ Running in SLURM job"
    echo "    Job ID: $SLURM_JOB_ID"
    echo "    Job name: $SLURM_JOB_NAME"
    echo "    Node: $SLURM_NODELIST"
    echo "    Partition: $SLURM_JOB_PARTITION"
    echo "    GPUs requested: $SLURM_GPUS"
else
    echo "  ⚠ Not running in SLURM job (direct SSH access)"
    echo "    Note: For production training, use SLURM"
fi
echo ""

# Check loaded modules
echo "[3/8] Module Status:"
module list 2>&1 | head -20
echo ""

# Check if CUDA module is loaded
if module list 2>&1 | grep -q cuda; then
    CUDA_MODULE=$(module list 2>&1 | grep cuda | awk '{print $1}')
    echo "  ✓ CUDA module loaded: ${CUDA_MODULE}"
else
    echo "  ✗ CUDA module NOT loaded"
    echo "    Fix: module load cuda/11.8.0"
fi
echo ""

# Check nvidia-smi
echo "[4/8] Hardware Detection (nvidia-smi):"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,driver_version,memory.total,compute_cap --format=csv
    echo ""
    echo "  GPU Utilization:"
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv
    echo ""
    NVIDIA_SMI_EXIT=0
else
    echo "  ✗ nvidia-smi not found!"
    echo "    This indicates no GPU driver or no GPUs available."
    NVIDIA_SMI_EXIT=1
fi
echo ""

# Check CUDA environment variables
echo "[5/8] CUDA Environment Variables:"
echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-'(not set - all GPUs visible)'}"
echo "  LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:0:200}..."
echo ""

# Check conda environment
echo "[6/8] Conda Environment:"
eval "$(conda shell.bash hook)" 2>/dev/null || true

if [ "$CONDA_DEFAULT_ENV" == "neuromotor" ]; then
    echo "  ✓ neuromotor environment activated"
elif [ -n "$CONDA_DEFAULT_ENV" ]; then
    echo "  ⚠ Different environment active: $CONDA_DEFAULT_ENV"
    echo "    Activating neuromotor..."
    conda activate neuromotor 2>/dev/null || echo "    ✗ Failed to activate neuromotor"
else
    echo "  ⚠ No conda environment active"
    echo "    Activating neuromotor..."
    conda activate neuromotor 2>/dev/null || echo "    ✗ Failed to activate neuromotor"
fi

if [ "$CONDA_DEFAULT_ENV" == "neuromotor" ]; then
    echo "  Current environment: $CONDA_DEFAULT_ENV"
    echo "  Python: $(which python3)"
    echo "  Python version: $(python3 --version)"
else
    echo "  ✗ Failed to activate neuromotor environment"
    CONDA_EXIT=1
fi
echo ""

# Check PyTorch installation
echo "[7/8] PyTorch Status:"
python3 << 'PYEOF'
import sys
try:
    import torch
    print(f"  ✓ PyTorch installed")
    print(f"    Version: {torch.__version__}")
    print(f"    Location: {torch.__file__}")
    print(f"    Built with CUDA: {torch.version.cuda}")

    # Check CUDA availability
    print("")
    print("  CUDA Detection:")
    cuda_available = torch.cuda.is_available()
    print(f"    torch.cuda.is_available(): {cuda_available}")

    if cuda_available:
        device_count = torch.cuda.device_count()
        print(f"    torch.cuda.device_count(): {device_count}")
        print("")
        print("  ✓ GPU Access Verified:")
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            print(f"    GPU {i}:")
            print(f"      Name: {torch.cuda.get_device_name(i)}")
            print(f"      Memory: {props.total_memory / 1024**3:.1f} GB")
            print(f"      Compute Capability: {props.major}.{props.minor}")
            print(f"      Multi-processors: {props.multi_processor_count}")

        # Test actual tensor allocation
        print("")
        print("  GPU Memory Allocation Test:")
        try:
            for i in range(device_count):
                test_tensor = torch.randn(1000, 1000, device=f'cuda:{i}')
                print(f"    ✓ GPU {i}: Successfully allocated 1000x1000 tensor")
                del test_tensor
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"    ✗ Allocation failed: {e}")
            sys.exit(1)
    else:
        print("")
        print("  ✗ CUDA NOT AVAILABLE in PyTorch!")
        print("")
        print("  Common causes:")
        print("    1. PyTorch CUDA version mismatch with system CUDA")
        print("       Solution: bash fix_pytorch_cuda.sh")
        print("    2. CUDA module not loaded")
        print("       Solution: module load cuda/11.8.0")
        print("    3. No GPUs available on this node")
        print("       Solution: Request GPU with SLURM or SSH to GPU server")
        sys.exit(1)

except ImportError:
    print("  ✗ PyTorch not installed!")
    print("    Solution: conda install pytorch")
    sys.exit(1)
except Exception as e:
    print(f"  ✗ Error checking PyTorch: {e}")
    sys.exit(1)

sys.exit(0)
PYEOF

PYTORCH_EXIT=$?
echo ""

# Check training script exists
echo "[8/8] Training Script Verification:"
TRAIN_SCRIPT="generic_neuromotor_interface/scripts/train_all_models_fixed.py"
if [ -f "${TRAIN_SCRIPT}" ]; then
    echo "  ✓ Training script found: ${TRAIN_SCRIPT}"

    # Check if script is executable
    if python3 -c "import sys; sys.exit(0)"; then
        echo "  ✓ Python can execute scripts"
    fi
else
    echo "  ✗ Training script not found: ${TRAIN_SCRIPT}"
    echo "    Are you in the project root directory?"
fi
echo ""

# Summary
echo "==========================================="
echo "  Verification Summary"
echo "==========================================="
echo ""

OVERALL_STATUS=0

if [ ${NVIDIA_SMI_EXIT:-0} -eq 0 ]; then
    echo "✓ Hardware: GPUs detected"
else
    echo "✗ Hardware: No GPUs detected"
    OVERALL_STATUS=1
fi

if module list 2>&1 | grep -q cuda; then
    echo "✓ CUDA Module: Loaded"
else
    echo "✗ CUDA Module: Not loaded"
    OVERALL_STATUS=1
fi

if [ "$CONDA_DEFAULT_ENV" == "neuromotor" ]; then
    echo "✓ Conda: neuromotor environment active"
else
    echo "✗ Conda: neuromotor environment not active"
    OVERALL_STATUS=1
fi

if [ ${PYTORCH_EXIT} -eq 0 ]; then
    echo "✓ PyTorch: CUDA available and working"
else
    echo "✗ PyTorch: CUDA not available"
    OVERALL_STATUS=1
fi

echo ""
echo "==========================================="

if [ ${OVERALL_STATUS} -eq 0 ]; then
    echo "  ✓ ALL CHECKS PASSED!"
    echo ""
    echo "  Your system is ready for GPU training."
    echo ""
    echo "  Next steps:"
    echo "    1. Download data (if not already done):"
    echo "       python -m generic_neuromotor_interface.scripts.download_data \\"
    echo "         --task discrete_gestures --output-dir ~/emg_data"
    echo ""
    echo "    2. Submit multi-GPU training job:"
    echo "       sbatch slurm_train_all_gpus.sbatch"
    echo ""
    echo "    3. Monitor training:"
    echo "       squeue -u $(whoami)"
    echo "       tail -f logs/slurm_<jobid>.out"
else
    echo "  ✗ SOME CHECKS FAILED"
    echo ""
    echo "  Please fix the issues above before training."
    echo ""
    echo "  Quick fixes:"
    echo "    - Load CUDA module: module load cuda/11.8.0"
    echo "    - Activate environment: conda activate neuromotor"
    echo "    - Fix PyTorch CUDA: bash fix_pytorch_cuda.sh"
    echo "    - Request GPU node: salloc -p gnolim --gres=gpu:1"
fi

echo "==========================================="
echo "End time: $(date)"
echo ""

exit ${OVERALL_STATUS}
