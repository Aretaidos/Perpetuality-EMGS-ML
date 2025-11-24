#!/bin/bash
set -e

echo "════════════════════════════════════════════════════════════════"
echo "Step 1: Fixing PyTorch for GPU Support"
echo "════════════════════════════════════════════════════════════════"

module purge
module load miniforge cuda/11.8.0
eval "$(conda shell.bash hook)"
conda activate neuromotor

echo "Uninstalling current PyTorch..."
pip uninstall -y torch torchvision torchaudio

echo "Installing PyTorch with CUDA 11.8 support..."
pip install torch==2.4.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo ""
echo "Verifying GPU access..."
python3 << 'PYEOF'
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA version: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    echo "ERROR: GPU still not detected!"
    exit 1
fi
PYEOF

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "Step 2: Downloading Full 100-User Dataset"
echo "════════════════════════════════════════════════════════════════"

cd /u/usz7pc/Perpetuality/generic-neuromotor-interface

# Remove small subset
echo "Removing small subset (3 users)..."
rm -f /u/usz7pc/emg_data/discrete_gestures_user_*.hdf5

# Download full dataset
echo "Downloading full 100-user dataset..."
echo "WARNING: This is ~50-100GB and will take 1-3 hours!"
echo ""

python -m generic_neuromotor_interface.scripts.download_data \
    --task discrete_gestures \
    --output-dir ~/emg_data

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "✅ Setup Complete!"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "GPU: Ready ✅"
echo "Dataset: Full 100 users ✅"
echo ""
echo "Next: Run training with GPU"
echo "  ./scripts/run_comparison_uva_gpu.sh ~/emg_data ./full_gpu_comparison 0"
echo ""
