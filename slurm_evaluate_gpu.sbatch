#!/bin/bash
#SBATCH --job-name=semg_eval
#SBATCH --partition=gnolim
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000
#SBATCH --time=2:00:00
#SBATCH --output=/u/usz7pc/Perpetuality/generic-neuromotor-interface/logs/slurm_eval_%j.out
#SBATCH --error=/u/usz7pc/Perpetuality/generic-neuromotor-interface/logs/slurm_eval_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=usz7pc@virginia.edu

##############################################################################
# Comprehensive Model Evaluation Script for sEMG Gesture Recognition
#
# This script performs detailed evaluation of all trained models (M1, M2, M3)
# and generates extensive metrics beyond just accuracy:
#
# Visual Analysis (27+ figures):
#   - Confusion matrices (9×9 heatmaps) for each model
#   - ROC curves with AUC scores (9 classes per model)
#   - Precision-Recall curves with AP scores (9 classes per model)
#   - Feature importance visualization (M3 only)
#
# Quantitative Metrics (100+ values):
#   - Overall: accuracy, macro precision, recall, F1
#   - Per-class (9 gestures): precision, recall, F1, AUC, AP
#   - Performance: inference latency, throughput
#   - Complexity: model size, parameter count
#
# Interpretability (M3):
#   - Feature importance by channel and type
#   - Top contributing features
#   - Anatomical muscle group analysis
#
# Usage:
#   sbatch slurm_evaluate_gpu.sbatch [models_directory]
#
# Example:
#   sbatch slurm_evaluate_gpu.sbatch ./models_gpu_multi_12345
##############################################################################

echo "==========================================="
echo "  sEMG Comprehensive Evaluation Job"
echo "==========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "==========================================="

# Load required modules
echo ""
echo "[1/5] Loading modules..."
module purge
module load miniforge
module load cuda/11.8.0

# Initialize conda
echo ""
echo "[2/5] Activating conda environment..."
eval "$(conda shell.bash hook)"
conda activate neuromotor

# Verify GPU access
echo ""
echo "[3/5] Verifying GPU access..."
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo ""
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
echo ""

# Navigate to project directory
cd /u/usz7pc/Perpetuality/generic-neuromotor-interface

# Parse arguments
MODELS_DIR="${1:-./models_gpu_multi_latest}"

# If argument looks like a job ID (all digits), construct path
if [[ "${1}" =~ ^[0-9]+$ ]]; then
    MODELS_DIR="./models_gpu_multi_${1}"
fi

echo "[4/5] Configuration:"
echo "  Models directory: ${MODELS_DIR}"

# Check if models directory exists
if [ ! -d "${MODELS_DIR}" ]; then
    echo ""
    echo "ERROR: Models directory not found: ${MODELS_DIR}"
    echo ""
    echo "Usage: sbatch slurm_evaluate_gpu.sbatch <models_directory>"
    echo "Example: sbatch slurm_evaluate_gpu.sbatch ./models_gpu_multi_12345"
    echo ""
    echo "Available model directories:"
    ls -d ./models_gpu_multi_* 2>/dev/null || echo "  (none found)"
    exit 1
fi

# Check if model files exist
echo ""
echo "Checking for trained models:"
MODELS_FOUND=0

if [ -f "${MODELS_DIR}/m1_best.pt" ]; then
    echo "  ✓ M1 model found"
    MODELS_FOUND=$((MODELS_FOUND + 1))
else
    echo "  ✗ M1 model not found: ${MODELS_DIR}/m1_best.pt"
fi

if [ -f "${MODELS_DIR}/m2_best.pt" ]; then
    echo "  ✓ M2 model found"
    MODELS_FOUND=$((MODELS_FOUND + 1))
else
    echo "  ✗ M2 model not found: ${MODELS_DIR}/m2_best.pt"
fi

if [ -f "${MODELS_DIR}/m3_best.pkl" ]; then
    echo "  ✓ M3 model found"
    MODELS_FOUND=$((MODELS_FOUND + 1))
else
    echo "  ✗ M3 model not found: ${MODELS_DIR}/m3_best.pkl"
fi

if [ ${MODELS_FOUND} -eq 0 ]; then
    echo ""
    echo "ERROR: No trained models found in ${MODELS_DIR}"
    echo "Please train models first using: sbatch slurm_train_all_gpus.sbatch"
    exit 1
fi

echo ""
echo "Found ${MODELS_FOUND} trained model(s)"
echo ""

# Setup evaluation output directory
OUTPUT_DIR="${MODELS_DIR}_evaluation"
mkdir -p ${OUTPUT_DIR}

echo "  Output directory: ${OUTPUT_DIR}"
echo ""

# Define data directory
DATA_DIR="${HOME}/emg_data"

# Check if data exists
if [ ! -d "${DATA_DIR}" ]; then
    echo "ERROR: Data directory not found: ${DATA_DIR}"
    echo "Please download the dataset first."
    exit 1
fi

echo "[5/5] Running comprehensive evaluation..."
echo "==========================================="
echo ""
echo "This will generate:"
echo "  • Confusion matrices (9×9 heatmaps)"
echo "  • ROC curves with AUC scores"
echo "  • Precision-Recall curves with AP scores"
echo "  • Per-class precision, recall, F1-score"
echo "  • Inference latency and throughput"
echo "  • Model size and complexity metrics"
echo "  • Feature importance analysis (M3)"
echo "  • Model comparison summary table"
echo ""
echo "Expected time: 10-15 minutes"
echo "==========================================="
echo ""

# Run evaluation script
python3 generic_neuromotor_interface/scripts/evaluate_models_comprehensive.py \
    --data-dir ${DATA_DIR} \
    --models-dir ${MODELS_DIR} \
    --output-dir ${OUTPUT_DIR} \
    --gpu

EVAL_EXIT=$?

echo ""
echo "==========================================="
echo "  Evaluation Complete"
echo "==========================================="
echo "End time: $(date)"
echo ""

if [ ${EVAL_EXIT} -eq 0 ]; then
    echo "✓ Evaluation completed successfully!"
    echo ""
    echo "Results saved to: ${OUTPUT_DIR}"
    echo ""
    echo "Generated outputs:"
    echo ""
    echo "1. Visual Analysis (figures/):"
    ls -1 ${OUTPUT_DIR}/figures/*.png 2>/dev/null | head -20 | sed 's/^/     • /'
    echo ""
    echo "2. Summary Files:"
    echo "     • ${OUTPUT_DIR}/model_comparison.csv"
    echo "     • ${OUTPUT_DIR}/all_results.json"
    echo ""
    echo "3. Detailed Metrics (metrics/):"
    ls -1 ${OUTPUT_DIR}/metrics/*.json 2>/dev/null | sed 's/^/     • /'
    echo ""
    echo "Quick view of results:"
    echo "-------------------------------------------"
    if [ -f "${OUTPUT_DIR}/model_comparison.csv" ]; then
        cat ${OUTPUT_DIR}/model_comparison.csv
    fi
    echo "-------------------------------------------"
    echo ""
    echo "To view detailed results:"
    echo "  cat ${OUTPUT_DIR}/all_results.json | python3 -m json.tool | less"
    echo ""
    echo "To download figures for analysis:"
    echo "  scp -r usz7pc@gpusrv17.cs.virginia.edu:${OUTPUT_DIR}/figures/ ."
    echo ""
    echo "Next steps:"
    echo "  1. Review confusion matrices to identify gesture confusions"
    echo "  2. Analyze ROC/PR curves for threshold optimization"
    echo "  3. Compare model_comparison.csv for performance tradeoffs"
    echo "  4. Check feature importance (M3) for interpretability"
else
    echo "✗ Evaluation failed with exit code ${EVAL_EXIT}"
    echo ""
    echo "Check the logs above for error details."
    echo ""
    echo "Common issues:"
    echo "  - Missing model files"
    echo "  - Incompatible model format"
    echo "  - Insufficient GPU memory (reduce batch size)"
    echo "  - Missing dependencies (matplotlib, seaborn, sklearn)"
fi

echo "==========================================="

exit ${EVAL_EXIT}
