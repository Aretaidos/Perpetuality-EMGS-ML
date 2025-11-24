#!/bin/bash
# Script to run model comparison training on GPU server (gpusrv17)
# Usage: ./run_training_on_gpu.sh

set -e  # Exit on error

echo "=========================================="
echo "GPU Model Training Script"
echo "=========================================="

# Check if we're on a GPU server
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. Are you on a GPU server?"
    exit 1
fi

# Display GPU status
echo ""
echo "=== GPU Status ==="
nvidia-smi

echo ""
echo "=== PyTorch CUDA Check ==="
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device count: {torch.cuda.device_count()}'); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else print('No CUDA devices')"

# Navigate to project directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo ""
echo "=== Starting Model Comparison Training ==="
echo "Project directory: $PROJECT_DIR"
echo "Data location: ~/emg_data"
echo ""

# Set data location (adjust if needed)
DATA_LOCATION="${1:-~/emg_data}"
OUTPUT_DIR="${2:-./model_comparison}"

# Run comparison script with GPU
echo "Running model comparison with GPU support..."
python3 scripts/compare_models.py \
    --data-location "$DATA_LOCATION" \
    --output-dir "$OUTPUT_DIR" \
    --channel-indices 9 8 7 6 5 13 15

echo ""
echo "=========================================="
echo "Training Complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="


