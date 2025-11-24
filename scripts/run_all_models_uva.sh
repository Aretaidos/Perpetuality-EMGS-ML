#!/bin/bash
# Complete setup and run script for UVA CS GPU servers (gpusrv17)
# Based on UVA CS Software Modules documentation

set -e  # Exit on error

echo "=========================================="
echo "UVA CS GPU Server - Model Comparison Setup"
echo "=========================================="

# Step 1: Initialize modules
echo ""
echo "Step 1: Initializing modules..."
if ! command -v module &> /dev/null; then
    echo "Loading module system..."
    source /etc/profile.d/modules.sh
fi

# Step 2: Load required modules
echo ""
echo "Step 2: Loading modules..."
module purge  # Clean slate
module load miniforge  # For conda

# Load CUDA module - REQUIRED for PyTorch to detect GPUs
module load cuda/12.8.1

# Step 3: Setup conda environment
echo ""
echo "Step 3: Setting up conda environment..."
if conda env list | grep -q "neuromotor"; then
    echo "Activating existing neuromotor environment..."
    conda activate neuromotor
else
    echo "Creating neuromotor environment from environment.yml..."
    conda env create -f environment.yml
    conda activate neuromotor
fi

# Step 4: Install missing packages and the project itself
echo ""
echo "Step 4: Installing missing packages and project..."
conda install -y scikit-learn tabulate 2>/dev/null || {
    echo "Conda install failed, trying pip..."
    python3 -m pip install --user scikit-learn tabulate
}

# Install the project in development mode
echo "Installing generic-neuromotor-interface package..."
cd "$PROJECT_DIR"
python3 -m pip install -e . --user 2>/dev/null || {
    echo "Warning: pip install -e failed, trying alternative method..."
    export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"
}

# Step 5: Navigate to project
echo ""
echo "Step 5: Navigating to project directory..."
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"
echo "Project directory: $PROJECT_DIR"

# Step 6: Verify GPU access
echo ""
echo "Step 6: Verifying GPU access..."
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Status:"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
else
    echo "WARNING: nvidia-smi not found. Training will use CPU."
fi

echo ""
echo "PyTorch CUDA Check:"
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device count: {torch.cuda.device_count()}')"

# Step 7: Run model comparison
echo ""
echo "=========================================="
echo "Step 7: Running Model Comparison"
echo "=========================================="
echo "This will train all 3 models:"
echo "  - M1: CNN+LSTM (~2-4 hours)"
echo "  - M2: CNN-Only (~1-2 hours)"
echo "  - M3: Random Forest (~10-30 minutes)"
echo ""

DATA_LOCATION="${1:-~/emg_data}"
OUTPUT_DIR="${2:-./model_comparison}"

python3 scripts/compare_models.py \
    --data-location "$DATA_LOCATION" \
    --output-dir "$OUTPUT_DIR" \
    --channel-indices 9 8 7 6 5 13 15

# Step 8: Display results
echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "View report:"
echo "  cat $OUTPUT_DIR/model_comparison_report.md"
echo ""


