#!/bin/bash
set -e

##############################################################################
# Complete GPU Training Script for All 3 Models
# Channels: {5, 6, 7, 8, 9, 13, 15} (7 isolated EMG locations)
# Models: M1 (CNN+LSTM), M2 (CNN-Only), M3 (Random Forest)
##############################################################################

echo "════════════════════════════════════════════════════════════════"
echo "🚀 Complete GPU Training - All 3 Models"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Get arguments
DATA_LOCATION="${1:-/u/usz7pc/emg_data}"
OUTPUT_DIR="${2:-./full_gpu_results}"
GPU_DEVICE="${3:-0}"

echo "Configuration:"
echo "  Data: $DATA_LOCATION"
echo "  Output: $OUTPUT_DIR"
echo "  GPU: $GPU_DEVICE"
echo "  Channels: {5, 6, 7, 8, 9, 13, 15}"
echo ""

# Step 1: Load modules and environment
echo "════════════════════════════════════════════════════════════════"
echo "Step 1: Setting up environment"
echo "════════════════════════════════════════════════════════════════"

module purge
module load miniforge cuda/11.8.0
eval "$(conda shell.bash hook)"
conda activate neuromotor

# Step 2: Fix PyTorch for GPU
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "Step 2: Fixing PyTorch for CUDA 11.8"
echo "════════════════════════════════════════════════════════════════"

echo "Current PyTorch status:"
python3 << 'EOF'
import torch
print(f"  PyTorch: {torch.__version__}")
print(f"  CUDA version: {torch.version.cuda}")
print(f"  CUDA available: {torch.cuda.is_available()}")
EOF

if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "✅ GPU already working!"
else
    echo "⚠️  GPU not detected - reinstalling PyTorch..."
    pip uninstall -y torch torchvision torchaudio
    pip install torch==2.4.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
fi

echo ""
echo "Verifying GPU after fix:"
python3 << 'EOF'
import torch
print(f"  PyTorch: {torch.__version__}")
print(f"  CUDA version: {torch.version.cuda}")  
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    echo "❌ ERROR: GPU still not available!"
    exit 1
fi
EOF

# Step 3: Download full dataset (if not already downloaded)
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "Step 3: Checking dataset"
echo "════════════════════════════════════════════════════════════════"

USER_COUNT=$(ls $DATA_LOCATION/discrete_gestures_user_*.hdf5 2>/dev/null | wc -l)
echo "Current user count: $USER_COUNT"

if [ "$USER_COUNT" -lt 100 ]; then
    echo "⚠️  Need to download full dataset (100 users)..."
    echo "This will take 1-3 hours for ~50-100GB"
    echo ""
    
    # Remove small subset if present
    rm -f $DATA_LOCATION/discrete_gestures_user_00[0-2]_dataset_000.hdf5
    
    python -m generic_neuromotor_interface.scripts.download_data \
        --task discrete_gestures \
        --output-dir $DATA_LOCATION \
        --cleanup
else
    echo "✅ Full dataset already downloaded ($USER_COUNT users)"
fi

# Step 4: Run all 3 models on GPU
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "Step 4: Training All 3 Models on GPU"
echo "════════════════════════════════════════════════════════════════"
echo ""

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

# Set CUDA device
export CUDA_VISIBLE_DEVICES=$GPU_DEVICE
echo "Using GPU device: $GPU_DEVICE"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run comparison script
python3 scripts/compare_models.py \
    --data-location "$DATA_LOCATION" \
    --output-dir "$OUTPUT_DIR" \
    --channel-indices 9 8 7 6 5 13 15 \
    --gpu-device $GPU_DEVICE

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "✅ TRAINING COMPLETE!"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "View results:"
echo "  cat $OUTPUT_DIR/model_comparison_report.md"
echo ""
echo "Expected accuracies:"
echo "  M1 (CNN+LSTM): 93-96%"
echo "  M2 (CNN-Only): 88-92%"
echo "  M3 (Random Forest): 82-88%"
echo ""

