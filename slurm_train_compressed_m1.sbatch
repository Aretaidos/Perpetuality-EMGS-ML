#!/bin/bash
#SBATCH --job-name=m1_compressed
#SBATCH --partition=gnolim
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000
#SBATCH --time=08:00:00
#SBATCH --output=/u/usz7pc/Perpetuality/generic-neuromotor-interface/logs/compressed_m1_%j.out
#SBATCH --error=/u/usz7pc/Perpetuality/generic-neuromotor-interface/logs/compressed_m1_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=usz7pc@virginia.edu

##############################################################################
# Compressed M1 CNN+LSTM Training for XIAO nRF52840 TinyML Deployment
#
# This trains a memory-optimized version of M1 that fits in 256KB RAM:
#   Original M1:     Conv(7→128) → LSTM(128, 3 layers) → FC(9) [~398KB RAM - NO FIT]
#   Compressed M1:   Conv(7→64)  → LSTM(32, 1 layer)   → FC(9) [~80KB RAM - FITS!]
#
# Training uses Multi-GPU DDP for faster convergence:
#   - 4x RTX A4000 GPUs (20GB each)
#   - Expected time: 2-4 hours
#   - Expected accuracy: 94-96% (down from 99.55% original)
#
# After training, convert to TFLite:
#   cd tinyml_deployment/conversion
#   python convert_m1_compressed.py --checkpoint ../../checkpoints_compressed/best.ckpt
##############################################################################

echo "==========================================="
echo "  Compressed M1 Training for XIAO nRF52840"
echo "==========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Partition: $SLURM_JOB_PARTITION"
echo "GPUs: 4x (DDP training)"
echo "Start time: $(date)"
echo "==========================================="
echo ""
echo "Model Architecture:"
echo "  Conv(7→64, k=15, s=10) → LSTM(32, 1 layer) → FC(9)"
echo "  Parameters: ~45,000 (down from 411,000)"
echo "  Target: XIAO nRF52840 (256KB RAM)"
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
OUTPUT_DIR="./checkpoints_compressed"
LOG_DIR="./logs"
mkdir -p ${OUTPUT_DIR}
mkdir -p ${LOG_DIR}

echo "[4/6] Output directory: ${OUTPUT_DIR}"

# Training configuration
DATA_DIR="${HOME}/emg_data"
CONFIG_FILE="config/discrete_gestures_m1_compressed.yaml"

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

echo "[6/6] Launching DDP training on 4 GPUs..."
echo "==========================================="
echo ""

# Set distributed training environment
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export WORLD_SIZE=4

# Run training with PyTorch Lightning DDP
# Using torchrun for robust distributed training
torchrun \
    --nproc_per_node=4 \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    -m generic_neuromotor_interface.train \
    --config ${CONFIG_FILE} \
    --trainer.devices=4 \
    --trainer.accelerator=gpu \
    --trainer.strategy=ddp \
    --trainer.default_root_dir=${OUTPUT_DIR} \
    --data_module.data_location=${DATA_DIR}

TRAIN_EXIT=$?

echo ""
echo "==========================================="
echo "  Training Summary"
echo "==========================================="
echo "End time: $(date)"
echo "Exit code: ${TRAIN_EXIT}"
echo ""

if [ ${TRAIN_EXIT} -eq 0 ]; then
    echo "✓ Compressed M1 training successful!"
    echo ""
    echo "Next steps for XIAO nRF52840 deployment:"
    echo ""
    echo "1. Find best checkpoint:"
    echo "   ls -la ${OUTPUT_DIR}/*.ckpt"
    echo ""
    echo "2. Convert to TFLite:"
    echo "   cd tinyml_deployment/conversion"
    echo "   python convert_m1_compressed.py \\"
    echo "       --checkpoint ../../${OUTPUT_DIR}/best.ckpt \\"
    echo "       --output-dir ../converted_models"
    echo ""
    echo "3. Copy header to firmware:"
    echo "   cp ../converted_models/m1_compressed_model.h ../platformio_project/include/"
    echo ""
    echo "4. Build and flash:"
    echo "   cd ../platformio_project"
    echo "   pio run -e xiao_nrf52840 --target upload"
    echo ""
    echo "5. Test on device:"
    echo "   pio device monitor -b 115200"
    echo "   # Press 't' for test inference"
else
    echo "✗ Training failed!"
    echo "Check logs: ${LOG_DIR}/compressed_m1_${SLURM_JOB_ID}.err"
fi

echo "==========================================="

exit ${TRAIN_EXIT}
