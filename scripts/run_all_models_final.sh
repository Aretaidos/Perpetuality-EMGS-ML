#!/bin/bash
set -e

##############################################
# Run all 3 models on CPU (RELIABLE & FAST)
##############################################
# This avoids GPU version mismatch issues
# CPU is actually fast enough for this dataset
##############################################

echo "=========================================="
echo "🚀 Running All 3 Models on CPU"
echo "=========================================="
echo ""

# Step 1: Load modules
echo "Loading modules..."
module purge
module load miniforge

# Step 2: Activate conda environment
echo "Activating neuromotor environment..."
eval "$(conda shell.bash hook)"
conda activate neuromotor

# Step 3: Navigate to project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

echo "Project directory: $PROJECT_DIR"
echo ""

# Step 4: Set data location
DATA_LOCATION="${1:-/u/usz7pc/emg_data}"
OUTPUT_DIR="${2:-./model_comparison_final}"

echo "Data location: $DATA_LOCATION"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Step 5: Run comparison on CPU
echo "=========================================="
echo "Training Models (M1, M2, M3)"
echo "=========================================="
echo "Using: CPU (avoids CUDA version mismatch)"
echo "Expected time: ~2-3 hours total"
echo ""

python3 scripts/compare_models.py \
    --data-location "$DATA_LOCATION" \
    --output-dir "$OUTPUT_DIR" \
    --channel-indices 9 8 7 6 5 13 15 \
    --no-gpu

echo ""
echo "=========================================="
echo "✅ COMPLETE!"
echo "=========================================="
echo "Results in: $OUTPUT_DIR"
echo "View report:"
echo "  cat $OUTPUT_DIR/model_comparison_report.md"
echo ""

