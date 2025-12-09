#!/bin/bash
#SBATCH --job-name=m1_4ch_optimized
#SBATCH --partition=gnolim
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000
#SBATCH --time=04:00:00
#SBATCH --output=/u/usz7pc/Perpetuality/generic-neuromotor-interface/logs/m1_4ch_optimized_%j.out
#SBATCH --error=/u/usz7pc/Perpetuality/generic-neuromotor-interface/logs/m1_4ch_optimized_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=usz7pc@virginia.edu

##############################################################################
# M1 4-Channel Optimized Training (Variant 2 - Attention)
#
# Optimized 4-channel architecture with attention mechanism:
#   Multi-scale Conv (4→64) -> Attention -> Downsample Conv (64→96)
#   -> Residual -> BiLSTM(96, 48, 1 layer) -> FC(9)
#   Parameters: ~135-155K | Expected Accuracy: 99.55-99.65%
#
# Key Innovation:
#   - Multi-scale convolutions capture different EMG speeds
#   - Channel attention dynamically weights channel importance
#   - Bidirectional LSTM for desktop (can look ahead)
#
# Training uses Multi-GPU DDP:
#   - 2x GPUs (DDP strategy)
#   - Expected time: 1.5-2.0 hours
#   - Inference: 15-25ms CPU (BiLSTM slower but accurate)
##############################################################################

echo "==========================================="
echo "  M1 4-Channel Optimized Training (Variant 2)"
echo "  Architecture Optimized for 4-Channel Input"
echo "==========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Partition: $SLURM_JOB_PARTITION"
echo "GPUs: 2x (DDP training)"
echo "Start time: $(date)"
echo "==========================================="
echo ""
echo "Model Architecture (Variant 2 - Optimized):"
echo "  Multi-scale Conv (kernels 7/15/25) -> Channel Attention"
echo "  -> Downsample Conv -> Residual -> BiLSTM(48×2) -> FC(9)"
echo "  Parameters: ~135-155K (3× TinyML)"
echo "  Target: Balance accuracy and efficiency"
echo ""
echo "Key Innovations:"
echo "  1. Multi-scale feature extraction (3 parallel conv paths)"
echo "  2. Channel attention (Squeeze-Excitation module)"
echo "  3. Bidirectional LSTM (desktop can look ahead)"
echo "  4. Residual connections for gradient flow"
echo "==========================================="
echo ""

# Load required modules
echo "[1/6] Loading modules..."
module purge
module load miniforge
module load cuda/11.8.0

# Initialize conda
echo "[2/6] Activating conda environment..."
eval "$(conda shell.bash hook)"
conda activate neuromotor

# Verify GPU access
echo "[3/6] Verifying GPU access..."
nvidia-smi --query-gpu=index,name,memory.total --format=csv
echo ""
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"

if ! python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    echo "ERROR: CUDA not available!"
    exit 1
fi

# Navigate to project directory
cd /u/usz7pc/Perpetuality/generic-neuromotor-interface

# Create output directories
OUTPUT_DIR="./checkpoints_4channel_optimized"
LOG_DIR="./logs"
mkdir -p ${OUTPUT_DIR}
mkdir -p ${LOG_DIR}

echo "[4/6] Output directory: ${OUTPUT_DIR}"

# Training configuration
DATA_DIR="${HOME}/emg_data"
CONFIG_FILE="config/discrete_gestures_m1_4channel_optimized.yaml"

echo "[5/6] Training configuration:"
echo "  Config: ${CONFIG_FILE}"
echo "  Data: ${DATA_DIR}"
echo "  Output: ${OUTPUT_DIR}"
echo ""

# Check prerequisites
if [ ! -f "${CONFIG_FILE}" ]; then
    echo "ERROR: Config file not found: ${CONFIG_FILE}"
    exit 1
fi

if [ ! -d "${DATA_DIR}" ]; then
    echo "ERROR: Data directory not found: ${DATA_DIR}"
    exit 1
fi

echo "[6/6] Launching DDP training on 2 GPUs..."
echo "==========================================="
echo ""

# Set distributed training environment (different port than Variant 1)
export MASTER_ADDR=localhost
export MASTER_PORT=29503
export WORLD_SIZE=2

# Run training with PyTorch Lightning DDP
torchrun \
    --nproc_per_node=2 \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    -m generic_neuromotor_interface.train \
    --config-name discrete_gestures_m1_4channel_optimized \
    data_location=${DATA_DIR}

TRAIN_EXIT=$?

echo ""
echo "==========================================="
echo "  Training Summary"
echo "==========================================="
echo "End time: $(date)"
echo "Exit code: ${TRAIN_EXIT}"
echo ""

if [ ${TRAIN_EXIT} -eq 0 ]; then
    echo "SUCCESS: M1 4-Channel Optimized training completed!"
    echo ""
    echo "Model Variant 2 (Optimized) Performance:"
    echo "  - Architecture: Multi-scale Conv + Attention + BiLSTM"
    echo "  - Parameters: ~135-155K"
    echo "  - Expected Accuracy: 99.55-99.65%"
    echo "  - Inference: 15-25ms CPU"
    echo ""
    echo "Architecture Highlights:"
    echo "  ✓ Multi-scale convolutions (3 kernels: 7/15/25)"
    echo "  ✓ Channel attention (dynamic weighting)"
    echo "  ✓ Bidirectional LSTM (looks ahead + behind)"
    echo "  ✓ Optimized for 4-channel characteristics"
    echo ""
    echo "Next steps:"
    echo ""
    echo "1. Find best checkpoint:"
    echo "   ls -la ${OUTPUT_DIR}/*/checkpoints/*.ckpt"
    echo ""
    echo "2. Compare with other variants:"
    echo "   python benchmark_inference.py --compare-all"
    echo ""
else
    echo "FAILED: Training failed!"
    echo "Check logs: ${LOG_DIR}/m1_4ch_optimized_${SLURM_JOB_ID}.err"
fi

echo "==========================================="

exit ${TRAIN_EXIT}
