#!/bin/bash

##############################################################################
# Multi-GPU Parallel Training - Direct Execution Script
# Runs all 3 models in parallel on separate GPUs for maximum speed
##############################################################################

set -e

echo "==========================================="
echo "  sEMG Multi-GPU Training Started"
echo "==========================================="
echo "Start time: $(date)"
echo "Host: $(hostname)"
echo ""

# Load modules and activate environment
echo "[1/5] Loading environment..."
module purge 2>/dev/null || true
module load miniforge 2>/dev/null || true
module load cuda/11.8.0 2>/dev/null || true
eval "$(conda shell.bash hook)" 2>/dev/null || true
conda activate neuromotor 2>/dev/null || true

# Verify GPU access
echo ""
echo "[2/5] Verifying GPUs..."
nvidia-smi --query-gpu=index,name,memory.total --format=csv

# Create output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="./models_gpu_multi_${TIMESTAMP}"
mkdir -p ${OUTPUT_DIR}

echo ""
echo "[3/5] Output directory: ${OUTPUT_DIR}"

# Configuration
DATA_DIR="${HOME}/emg_data"
EPOCHS=100
BATCH_SIZE=64
NUM_WORKERS=8
LR=0.001

echo ""
echo "[4/5] Configuration:"
echo "  Data: ${DATA_DIR}"
echo "  Epochs: ${EPOCHS}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Workers: ${NUM_WORKERS}"

echo ""
echo "[5/5] Launching parallel training..."
echo "==========================================="
echo ""

# Launch M1 on GPU 0
echo "[GPU 0] Starting M1 (CNN+LSTM) at $(date)..."
CUDA_VISIBLE_DEVICES=0 nohup python3 generic_neuromotor_interface/scripts/train_all_models_fixed.py \
    --model m1 \
    --data-dir ${DATA_DIR} \
    --output-dir ${OUTPUT_DIR} \
    --gpu \
    --epochs ${EPOCHS} \
    --batch-size ${BATCH_SIZE} \
    --lr ${LR} \
    --num-workers ${NUM_WORKERS} \
    > ${OUTPUT_DIR}/m1_training.log 2>&1 &
M1_PID=$!
echo "  M1 PID: ${M1_PID}"

sleep 2

# Launch M2 on GPU 1
echo "[GPU 1] Starting M2 (CNN-only) at $(date)..."
CUDA_VISIBLE_DEVICES=1 nohup python3 generic_neuromotor_interface/scripts/train_all_models_fixed.py \
    --model m2 \
    --data-dir ${DATA_DIR} \
    --output-dir ${OUTPUT_DIR} \
    --gpu \
    --epochs ${EPOCHS} \
    --batch-size ${BATCH_SIZE} \
    --lr ${LR} \
    --num-workers ${NUM_WORKERS} \
    > ${OUTPUT_DIR}/m2_training.log 2>&1 &
M2_PID=$!
echo "  M2 PID: ${M2_PID}"

sleep 2

# Launch M3 on CPU
echo "[CPU] Starting M3 (Random Forest) at $(date)..."
nohup python3 generic_neuromotor_interface/scripts/train_all_models_fixed.py \
    --model m3 \
    --data-dir ${DATA_DIR} \
    --output-dir ${OUTPUT_DIR} \
    --epochs ${EPOCHS} \
    > ${OUTPUT_DIR}/m3_training.log 2>&1 &
M3_PID=$!
echo "  M3 PID: ${M3_PID}"

echo ""
echo "==========================================="
echo "✓ All training processes launched!"
echo ""
echo "Process IDs:"
echo "  M1 (GPU 0): ${M1_PID}"
echo "  M2 (GPU 1): ${M2_PID}"
echo "  M3 (CPU):   ${M3_PID}"
echo ""
echo "Output directory: ${OUTPUT_DIR}"
echo ""
echo "Monitor training with:"
echo "  tail -f ${OUTPUT_DIR}/m1_training.log"
echo "  tail -f ${OUTPUT_DIR}/m2_training.log"
echo "  tail -f ${OUTPUT_DIR}/m3_training.log"
echo ""
echo "Check GPU usage:"
echo "  watch -n 2 nvidia-smi"
echo ""
echo "Check processes:"
echo "  ps aux | grep train_all_models_fixed"
echo ""
echo "Training will continue in background even if you disconnect SSH."
echo "Expected completion time: ~12 hours"
echo "==========================================="

# Save PIDs
echo "M1_PID=${M1_PID}" > ${OUTPUT_DIR}/training_pids.txt
echo "M2_PID=${M2_PID}" >> ${OUTPUT_DIR}/training_pids.txt
echo "M3_PID=${M3_PID}" >> ${OUTPUT_DIR}/training_pids.txt
echo "OUTPUT_DIR=${OUTPUT_DIR}" >> ${OUTPUT_DIR}/training_pids.txt
echo "TIMESTAMP=${TIMESTAMP}" >> ${OUTPUT_DIR}/training_pids.txt

