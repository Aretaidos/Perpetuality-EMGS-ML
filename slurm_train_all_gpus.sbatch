#!/bin/bash
#SBATCH --job-name=semg_multi_gpu
#SBATCH --partition=gnolim
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000
#SBATCH --time=24:00:00
#SBATCH --output=/u/usz7pc/Perpetuality/generic-neuromotor-interface/logs/slurm_train_%j.out
#SBATCH --error=/u/usz7pc/Perpetuality/generic-neuromotor-interface/logs/slurm_train_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=usz7pc@virginia.edu

##############################################################################
# Multi-GPU Parallel Training Script for sEMG Gesture Recognition
#
# This script trains all 3 models (M1, M2, M3) in parallel across 4 GPUs:
#   - GPU 0: M1 (CNN+LSTM) - Highest complexity, ~6-12 hours
#   - GPU 1: M2 (CNN-only) - Medium complexity, ~4-8 hours
#   - CPU:   M3 (Random Forest) - Fast, ~5-10 minutes
#
# Expected speedup: 3x faster than sequential training
# Total time: ~12 hours (vs ~20 hours sequential)
##############################################################################

echo "==========================================="
echo "  sEMG Multi-GPU Training Job Started"
echo "==========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Partition: $SLURM_JOB_PARTITION"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Start time: $(date)"
echo "==========================================="

# Load required modules
echo ""
echo "[1/6] Loading modules..."
module purge
module load miniforge
module load cuda/11.8.0

# Initialize conda
echo ""
echo "[2/6] Activating conda environment..."
eval "$(conda shell.bash hook)"
conda activate neuromotor

# Verify GPU access
echo ""
echo "[3/6] Verifying GPU access..."
nvidia-smi
echo ""
echo "PyTorch CUDA status:"
python3 -c "import torch; print(f'  CUDA available: {torch.cuda.is_available()}'); print(f'  GPU count: {torch.cuda.device_count()}'); [print(f'  GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"

if ! python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    echo ""
    echo "ERROR: CUDA not available in PyTorch!"
    echo "Please run the fix_pytorch_cuda.sh script first."
    exit 1
fi

# Navigate to project directory
cd /u/usz7pc/Perpetuality/generic-neuromotor-interface

# Create output directory
OUTPUT_DIR="./models_gpu_multi_${SLURM_JOB_ID}"
mkdir -p ${OUTPUT_DIR}

echo ""
echo "[4/6] Output directory: ${OUTPUT_DIR}"

# Define training parameters (optimized for multi-GPU)
DATA_DIR="${HOME}/emg_data"
EPOCHS=100
BATCH_SIZE=64    # Increased for RTX A4000 (20GB VRAM)
NUM_WORKERS=8    # More workers for faster data loading
LR=0.001

echo ""
echo "[5/6] Training configuration:"
echo "  Data directory: ${DATA_DIR}"
echo "  Epochs: ${EPOCHS}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Learning rate: ${LR}"
echo "  Num workers: ${NUM_WORKERS}"

# Check if data exists
if [ ! -d "${DATA_DIR}" ]; then
    echo ""
    echo "ERROR: Data directory not found: ${DATA_DIR}"
    echo "Please download the dataset first using:"
    echo "  python -m generic_neuromotor_interface.scripts.download_data --task discrete_gestures --output-dir ~/emg_data"
    exit 1
fi

echo ""
echo "[6/6] Launching parallel training on 4 GPUs..."
echo "==========================================="
echo ""

# Launch M1 training on GPU 0 in background
echo "[GPU 0] Starting M1 (CNN+LSTM) training at $(date)..."
CUDA_VISIBLE_DEVICES=0 python3 generic_neuromotor_interface/scripts/train_all_models_fixed.py \
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

# Small delay to avoid race conditions
sleep 2

# Launch M2 training on GPU 1 in background
echo "[GPU 1] Starting M2 (CNN-only) training at $(date)..."
CUDA_VISIBLE_DEVICES=1 python3 generic_neuromotor_interface/scripts/train_all_models_fixed.py \
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

# Small delay
sleep 2

# Launch M3 training on CPU in background (Random Forest doesn't use GPU)
echo "[CPU] Starting M3 (Random Forest) training at $(date)..."
python3 generic_neuromotor_interface/scripts/train_all_models_fixed.py \
    --model m3 \
    --data-dir ${DATA_DIR} \
    --output-dir ${OUTPUT_DIR} \
    --epochs ${EPOCHS} \
    > ${OUTPUT_DIR}/m3_training.log 2>&1 &
M3_PID=$!
echo "  M3 PID: ${M3_PID}"

echo ""
echo "All training processes launched successfully!"
echo "  M1 (GPU 0): PID ${M1_PID}"
echo "  M2 (GPU 1): PID ${M2_PID}"
echo "  M3 (CPU):   PID ${M3_PID}"
echo ""
echo "Monitoring progress (use Ctrl+C to stop monitoring, training will continue):"
echo "  tail -f ${OUTPUT_DIR}/m1_training.log"
echo "  tail -f ${OUTPUT_DIR}/m2_training.log"
echo "  tail -f ${OUTPUT_DIR}/m3_training.log"
echo ""

# Monitor GPU utilization every 60 seconds in background
(
    while true; do
        echo "=== GPU Utilization at $(date) ===" >> ${OUTPUT_DIR}/gpu_monitor.log
        nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv >> ${OUTPUT_DIR}/gpu_monitor.log
        echo "" >> ${OUTPUT_DIR}/gpu_monitor.log
        sleep 60
    done
) &
MONITOR_PID=$!

# Wait for M3 to complete (should be fast)
echo "Waiting for M3 (Random Forest) to complete..."
wait ${M3_PID}
M3_EXIT=$?
echo "[CPU] M3 training completed at $(date) with exit code ${M3_EXIT}"

# Check M3 success
if [ ${M3_EXIT} -eq 0 ]; then
    echo "  ✓ M3 training successful!"
else
    echo "  ✗ M3 training failed! Check ${OUTPUT_DIR}/m3_training.log"
fi

# Wait for M2 to complete
echo ""
echo "Waiting for M2 (CNN-only) to complete..."
wait ${M2_PID}
M2_EXIT=$?
echo "[GPU 1] M2 training completed at $(date) with exit code ${M2_EXIT}"

# Check M2 success
if [ ${M2_EXIT} -eq 0 ]; then
    echo "  ✓ M2 training successful!"
else
    echo "  ✗ M2 training failed! Check ${OUTPUT_DIR}/m2_training.log"
fi

# Wait for M1 to complete (typically the longest)
echo ""
echo "Waiting for M1 (CNN+LSTM) to complete..."
wait ${M1_PID}
M1_EXIT=$?
echo "[GPU 0] M1 training completed at $(date) with exit code ${M1_EXIT}"

# Check M1 success
if [ ${M1_EXIT} -eq 0 ]; then
    echo "  ✓ M1 training successful!"
else
    echo "  ✗ M1 training failed! Check ${OUTPUT_DIR}/m1_training.log"
fi

# Stop GPU monitoring
kill ${MONITOR_PID} 2>/dev/null

echo ""
echo "==========================================="
echo "  Training Summary"
echo "==========================================="
echo "End time: $(date)"
echo ""
echo "Exit codes:"
echo "  M1: ${M1_EXIT} (0 = success)"
echo "  M2: ${M2_EXIT} (0 = success)"
echo "  M3: ${M3_EXIT} (0 = success)"
echo ""

# Overall exit code (fail if any model failed)
OVERALL_EXIT=$((M1_EXIT + M2_EXIT + M3_EXIT))

if [ ${OVERALL_EXIT} -eq 0 ]; then
    echo "✓ All models trained successfully!"
    echo ""
    echo "Trained models saved to: ${OUTPUT_DIR}"
    echo "  - ${OUTPUT_DIR}/m1_best.pt"
    echo "  - ${OUTPUT_DIR}/m2_best.pt"
    echo "  - ${OUTPUT_DIR}/m3_best.pkl"
    echo ""
    echo "Training logs:"
    echo "  - ${OUTPUT_DIR}/m1_training.log"
    echo "  - ${OUTPUT_DIR}/m2_training.log"
    echo "  - ${OUTPUT_DIR}/m3_training.log"
    echo ""
    echo "GPU utilization log:"
    echo "  - ${OUTPUT_DIR}/gpu_monitor.log"
    echo ""
    echo "Next step: Run comprehensive evaluation"
    echo "  sbatch slurm_evaluate_gpu.sbatch ${OUTPUT_DIR}"
else
    echo "✗ Some models failed to train!"
    echo "Check individual log files for details."
fi

echo "==========================================="

exit ${OVERALL_EXIT}
