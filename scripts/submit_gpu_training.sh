#!/bin/bash
# Submit training job to GPU server
# This script can be used with job schedulers or run directly on gpusrv17

# Set job parameters
JOB_NAME="emg_model_comparison"
DATA_LOCATION="${1:-~/emg_data}"
OUTPUT_DIR="${2:-./model_comparison}"
GPU_DEVICE="${3:-0}"  # Default to GPU 0

echo "=========================================="
echo "EMG Model Comparison Training"
echo "=========================================="
echo "Job Name: $JOB_NAME"
echo "Data Location: $DATA_LOCATION"
echo "Output Directory: $OUTPUT_DIR"
echo "GPU Device: $GPU_DEVICE"
echo "=========================================="

# Change to project directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Verify GPU access
echo ""
echo "=== Verifying GPU Access ==="
if ! command -v nvidia-smi &> /dev/null; then
    echo "WARNING: nvidia-smi not found. Training will use CPU."
    USE_GPU="--no-gpu"
else
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
    USE_GPU=""
fi

# Check PyTorch CUDA
echo ""
echo "=== Checking PyTorch CUDA ==="
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device count: {torch.cuda.device_count()}')"

# Set GPU visibility if specified
if [ -n "$GPU_DEVICE" ] && [ "$GPU_DEVICE" != "all" ]; then
    export CUDA_VISIBLE_DEVICES=$GPU_DEVICE
    echo "Using GPU(s): $GPU_DEVICE"
fi

# Run training
echo ""
echo "=== Starting Training ==="
echo "This may take several hours..."
echo ""

python3 scripts/compare_models.py \
    --data-location "$DATA_LOCATION" \
    --output-dir "$OUTPUT_DIR" \
    --channel-indices 9 8 7 6 5 13 15 \
    $USE_GPU

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Training Completed Successfully!"
    echo "=========================================="
    echo "Results saved to: $OUTPUT_DIR"
    echo ""
    echo "View report:"
    echo "  cat $OUTPUT_DIR/model_comparison_report.md"
    echo ""
else
    echo ""
    echo "=========================================="
    echo "Training Failed!"
    echo "=========================================="
    echo "Check logs in: $OUTPUT_DIR"
    exit 1
fi


