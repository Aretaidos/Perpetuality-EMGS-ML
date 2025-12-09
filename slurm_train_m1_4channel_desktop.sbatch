#!/bin/bash
#SBATCH --job-name=m1_4ch_desktop
#SBATCH --partition=gnolim
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000
#SBATCH --time=04:00:00
#SBATCH --output=/u/usz7pc/Perpetuality/generic-neuromotor-interface/logs/m1_4ch_desktop_%j.out
#SBATCH --error=/u/usz7pc/Perpetuality/generic-neuromotor-interface/logs/m1_4ch_desktop_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=usz7pc@virginia.edu

##############################################################################
# M1 4-Channel Desktop Training (Variant 1 - Scaled)
#
# Full-capacity 4-channel architecture for desktop inference:
#   Conv1d(4, 144, k=15, s=10) -> BatchNorm -> LSTM(144, 128, 3 layers) -> FC(9)
#   Parameters: ~405K | Expected Accuracy: 99.3-99.6%
#
# 4-Channel Selection (flexor-extensor balance, 66.2% importance):
#   - Ch7 (idx 6): Index flexor (24.5%) - press detection
#   - Ch8 (idx 7): Ring flexor (13.2%) - press detection
#   - Ch13 (idx 12): Index/middle extensor (19.8%) - release detection
#   - Ch15 (idx 14): Ring/pinky extensor (8.7%) - release detection
#
# Training uses Multi-GPU DDP:
#   - 2x GPUs (DDP strategy)
#   - Expected time: 1.8-2.2 hours
#   - Inference: 10-12ms CPU, 2-4ms GPU
##############################################################################

echo "==========================================="
echo "  M1 4-Channel Desktop Training (Variant 1)"
echo "  Full-Capacity Model for Desktop Inference"
echo "==========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Partition: $SLURM_JOB_PARTITION"
echo "GPUs: 2x (DDP training)"
echo "Start time: $(date)"
echo "==========================================="
echo ""
echo "Model Architecture (Variant 1 - Scaled):"
echo "  Conv(4->144, k=15, s=10) -> BatchNorm -> LSTM(144, 128, 3 layers) -> FC(9)"
echo "  Parameters: ~405K (matches 7-channel M1)"
echo "  Target: Desktop CPU/GPU inference"
echo ""
echo "4-Channel Selection:"
echo "  Ch7 (idx 6): Index flexor - 24.5% importance"
echo "  Ch8 (idx 7): Ring flexor - 13.2% importance"
echo "  Ch13 (idx 12): Index/middle extensor - 19.8% importance"
echo "  Ch15 (idx 14): Ring/pinky extensor - 8.7% importance"
echo "  Total: 66.2% importance (flexor-extensor balance)"
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
OUTPUT_DIR="./checkpoints_4channel_desktop"
LOG_DIR="./logs"
mkdir -p ${OUTPUT_DIR}
mkdir -p ${LOG_DIR}

echo "[4/6] Output directory: ${OUTPUT_DIR}"

# Training configuration
DATA_DIR="${HOME}/emg_data"
CONFIG_FILE="config/discrete_gestures_m1_4channel_desktop.yaml"

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

# Set distributed training environment
export MASTER_ADDR=localhost
export MASTER_PORT=29502
export WORLD_SIZE=2

# Run training with PyTorch Lightning DDP
torchrun \
    --nproc_per_node=2 \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    -m generic_neuromotor_interface.train \
    --config-name discrete_gestures_m1_4channel_desktop \
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
    echo "SUCCESS: M1 4-Channel Desktop training completed!"
    echo ""
    echo "Model Variant 1 (Scaled) Performance:"
    echo "  - Architecture: Conv(4→144) + LSTM(3×128)"
    echo "  - Parameters: ~405K"
    echo "  - Expected Accuracy: 99.3-99.6%"
    echo "  - Inference: 10-12ms CPU, 2-4ms GPU"
    echo ""
    echo "Next steps:"
    echo ""
    echo "1. Find best checkpoint:"
    echo "   ls -la ${OUTPUT_DIR}/*/checkpoints/*.ckpt"
    echo ""
    echo "2. Compare with other variants:"
    echo "   - Variant 2 (Optimized): BiLSTM + Attention"
    echo "   - Variant 3 (Efficient): Pure CNN (2-3ms inference)"
    echo ""
    echo "3. Run inference benchmark:"
    echo "   python benchmark_inference.py --variant 1 --device cpu"
    echo "   python benchmark_inference.py --variant 1 --device cuda"
    echo ""
else
    echo "FAILED: Training failed!"
    echo "Check logs: ${LOG_DIR}/m1_4ch_desktop_${SLURM_JOB_ID}.err"
fi

echo "==========================================="

exit ${TRAIN_EXIT}
