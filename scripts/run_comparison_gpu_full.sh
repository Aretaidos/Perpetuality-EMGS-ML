#!/bin/bash
set -e

##############################################
# Run All 3 Models on GPU with Full Dataset
##############################################
# This runs M1, M2, M3 on GPU with 100 users
# Expected time: 12-24 hours
# Expected accuracy: M1: 93-96%
##############################################

# Parse arguments
DATA_LOCATION="${1:-/u/usz7pc/emg_data}"
OUTPUT_DIR="${2:-./full_comparison_gpu}"
GPU_DEVICE="${3:-0}"

echo "=========================================="
echo "🚀 GPU Training - Full 100-User Dataset"
echo "=========================================="
echo ""
echo "Data location: $DATA_LOCATION"
echo "Output directory: $OUTPUT_DIR"
echo "GPU device: $GPU_DEVICE"
echo ""

# Step 1: Verify dataset
USER_COUNT=$(ls $DATA_LOCATION/discrete_gestures_user_*.hdf5 2>/dev/null | wc -l)
echo "Users in dataset: $USER_COUNT"
if [ "$USER_COUNT" -lt 90 ]; then
    echo "❌ ERROR: Not enough users! Expected ~100, found $USER_COUNT"
    echo "Run ./scripts/fix_gpu_and_download.sh first to download full dataset"
    exit 1
fi
echo "✅ Full dataset detected"
echo ""

# Step 2: Load modules
echo "Loading modules..."
module purge
module load miniforge
module load cuda/11.8.0

# Step 3: Activate conda
echo "Activating neuromotor environment..."
eval "$(conda shell.bash hook)"
conda activate neuromotor

# Step 4: Verify GPU
echo ""
echo "Verifying GPU access..."
python3 << 'EOF'
import torch
cuda_available = torch.cuda.is_available()
print(f"CUDA available: {cuda_available}")
if cuda_available:
    print(f"Device count: {torch.cuda.device_count()}")
    print(f"Device 0: {torch.cuda.get_device_name(0)}")
    print("✅ GPU is ready!")
else:
    print("❌ GPU not detected!")
    print("Run ./scripts/fix_gpu_and_download.sh to fix PyTorch")
    exit(1)
EOF

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ GPU verification failed!"
    echo "Run: ./scripts/fix_gpu_and_download.sh"
    exit 1
fi

# Step 5: Navigate to project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

# Step 6: Set GPU device
export CUDA_VISIBLE_DEVICES="$GPU_DEVICE"
echo ""
echo "Using GPU: $GPU_DEVICE"
echo ""

# Step 7: Run comparison
echo "=========================================="
echo "Starting Model Comparison (GPU)"
echo "=========================================="
echo ""
echo "Training all 3 models on GPU with 100 users:"
echo "  • M1: CNN+LSTM (Expected: 93-96% accuracy)"
echo "  • M2: CNN-Only (Expected: 88-92% accuracy)"
echo "  • M3: Random Forest (Expected: 82-88% accuracy)"
echo ""
echo "Estimated time: 12-24 hours"
echo ""
echo "You can monitor progress with:"
echo "  tail -f $OUTPUT_DIR/training.log"
echo ""
echo "Close this terminal - training will continue in background"
echo ""

# Run with GPU enabled
python3 scripts/compare_models.py \
    --data-location "$DATA_LOCATION" \
    --output-dir "$OUTPUT_DIR" \
    --channel-indices 9 8 7 6 5 13 15 \
    --gpu-device "$GPU_DEVICE"

echo ""
echo "=========================================="
echo "✅ TRAINING COMPLETE!"
echo "=========================================="
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "View results:"
echo "  cat $OUTPUT_DIR/model_comparison_report.md"
echo ""
echo "Expected results:"
echo "  M1: 93-96% validation accuracy ✅"
echo "  M2: 88-92% validation accuracy ✅"
echo "  M3: 82-88% validation accuracy ✅"
echo ""

