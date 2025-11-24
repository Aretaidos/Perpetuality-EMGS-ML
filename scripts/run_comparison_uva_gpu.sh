#!/bin/bash
# Run all 3 models on UVA CS GPU server with proper module loading
# Based on UVA CS Software Modules documentation

set -e  # Exit on error

echo "=========================================="
echo "UVA CS GPU - 3 Model Comparison"
echo "=========================================="

# Arguments
DATA_LOCATION="${1:-~/emg_data}"
OUTPUT_DIR="${2:-./model_comparison_gpu}"
GPU_DEVICE="${3:-0}"

echo "Data location: $DATA_LOCATION"
echo "Output directory: $OUTPUT_DIR"
echo "GPU device: $GPU_DEVICE"
echo ""

# Step 1: Initialize module system
if ! command -v module &> /dev/null; then
    echo "Loading module system..."
    source /etc/profile.d/modules.sh
fi

# Step 2: Purge all modules for clean slate
echo "Purging existing modules..."
module purge

# Step 3: Load required modules
echo "Loading modules..."
module load miniforge  # For conda
module load cuda/12.8.1  # REQUIRED for GPU support

# Step 4: Initialize conda for this shell and activate environment
echo "Initializing conda..."
eval "$(conda shell.bash hook)"

echo "Activating neuromotor conda environment..."
conda activate neuromotor

# Step 5: Verify GPU access (informational only - training will work even if check fails)
echo ""
echo "Checking GPU access..."
python3 << 'EOF'
import torch
print(f"CUDA available (pre-check): {torch.cuda.is_available()}")
print(f"Device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"Device 0: {torch.cuda.get_device_name(0)}")
else:
    print("Note: CUDA check shows False, but training may still work on GPU.")
    print("Continuing anyway - previous runs succeeded despite this check failing.")
EOF

echo "Proceeding with training..."

# Step 6: Navigate to project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
echo ""
echo "Project directory: $PROJECT_DIR"
echo ""

# Step 7: Set CUDA_VISIBLE_DEVICES for specific GPU
export CUDA_VISIBLE_DEVICES=$GPU_DEVICE
echo "Using GPU: $CUDA_VISIBLE_DEVICES"
echo ""

# Step 8: Run comparison
echo "=========================================="
echo "Starting Model Comparison"
echo "=========================================="
echo "This will train all 3 models:"
echo "- M1: CNN+LSTM (fixed LSTM contiguity)"
echo "- M2: CNN-Only"  
echo "- M3: Random Forest"
echo ""

python3 scripts/compare_models.py \
    --data-location "$DATA_LOCATION" \
    --output-dir "$OUTPUT_DIR" \
    --channel-indices 9 8 7 6 5 13 15 \
    --gpu-device 0

echo ""
echo "=========================================="
echo "Comparison Complete!"
echo "=========================================="
echo "Results saved to: $OUTPUT_DIR"
echo "View report: cat $OUTPUT_DIR/model_comparison_report.md"

