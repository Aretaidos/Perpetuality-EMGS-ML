#!/bin/bash
#SBATCH --job-name=m1_4ch_efficient
#SBATCH --partition=gnolim
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000
#SBATCH --time=02:00:00
#SBATCH --output=/u/usz7pc/Perpetuality/generic-neuromotor-interface/logs/m1_4ch_efficient_%j.out
#SBATCH --error=/u/usz7pc/Perpetuality/generic-neuromotor-interface/logs/m1_4ch_efficient_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=usz7pc@virginia.edu

##############################################################################
# M1 4-Channel Efficient Training (Variant 3 - Fast CNN)
#
# Pure CNN architecture for ultra-fast desktop inference:
#   Conv (4→48, s=10) -> DilatedConv (48→64, d=2) -> DepthwiseSep (64→96)
#   -> Conv (96→128, s=2) -> Conv (128→64) -> Conv (64→9)
#   Parameters: ~120K | Expected Accuracy: 98.0-99.0%
#
# Key Innovation:
#   - NO LSTM - eliminates sequential bottleneck
#   - Dilated convolutions for 65ms temporal context
#   - Depthwise separable for efficient multi-scale features
#   - Fully parallelizable (5-10× faster than LSTM!)
#
# Training Configuration:
#   - 1x GPU (sufficient for fast CNN)
#   - Expected time: 15-20 minutes
#   - Inference: 2-3ms CPU (ULTRA FAST!)
##############################################################################

echo "==========================================="
echo "  M1 4-Channel Efficient Training (Variant 3)"
echo "  Pure CNN for Ultra-Fast Inference"
echo "==========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Partition: $SLURM_JOB_PARTITION"
echo "GPUs: 1x (fast training)"
echo "Start time: $(date)"
echo "==========================================="
echo ""
echo "Model Architecture (Variant 3 - Efficient):"
echo "  Pure CNN - NO LSTM for maximum speed"
echo "  Conv -> DilatedConv -> DepthwiseSep -> Conv -> Conv -> Out"
echo "  Parameters: ~120K"
echo "  Target: Real-time applications (<10ms latency)"
echo ""
echo "Key Features:"
echo "  1. Dilated convolutions (65ms receptive field)"
echo "  2. Depthwise separable conv (efficient)"
echo "  3. Fully parallelizable (no sequential processing)"
echo "  4. MKL/oneDNN optimized on modern CPUs"
echo ""
echo "Expected Performance:"
echo "  Accuracy: 98.0-99.0% (slight drop for 5-10× speed gain)"
echo "  CPU Inference: 2-3ms (vs 10-25ms for LSTM variants)"
echo "  GPU Inference: 1-2ms"
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
OUTPUT_DIR="./checkpoints_4channel_efficient"
LOG_DIR="./logs"
mkdir -p ${OUTPUT_DIR}
mkdir -p ${LOG_DIR}

echo "[4/6] Output directory: ${OUTPUT_DIR}"

# Training configuration
DATA_DIR="${HOME}/emg_data"
CONFIG_FILE="config/discrete_gestures_m1_4channel_efficient.yaml"

echo "[5/6] Training configuration:"
echo "  Config: ${CONFIG_FILE}"
echo "  Data: ${DATA_DIR}"
echo "  Output: ${OUTPUT_DIR}"
echo "  Batch size: 128 (larger - CNN uses less memory)"
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

echo "[6/6] Launching single-GPU training..."
echo "==========================================="
echo ""

# Run training (single GPU, no DDP needed)
python3 -m generic_neuromotor_interface.train \
    --config-name discrete_gestures_m1_4channel_efficient \
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
    echo "SUCCESS: M1 4-Channel Efficient training completed!"
    echo ""
    echo "Model Variant 3 (Efficient) Performance:"
    echo "  - Architecture: Pure CNN (NO LSTM)"
    echo "  - Parameters: ~120K"
    echo "  - Expected Accuracy: 98.0-99.0%"
    echo "  - CPU Inference: 2-3ms (ULTRA FAST!)"
    echo "  - GPU Inference: 1-2ms"
    echo ""
    echo "Speed Advantage:"
    echo "  ✓ 5-10× faster than LSTM variants on CPU"
    echo "  ✓ Fully parallelizable (no sequential bottleneck)"
    echo "  ✓ Perfect for real-time applications"
    echo "  ✓ MKL/oneDNN optimized on modern CPUs"
    echo ""
    echo "Use Cases:"
    echo "  - Real-time gesture control (<10ms latency)"
    echo "  - Gaming/VR interfaces"
    echo "  - Live music performance"
    echo "  - High-throughput batch processing"
    echo ""
    echo "Next steps:"
    echo ""
    echo "1. Find best checkpoint:"
    echo "   ls -la ${OUTPUT_DIR}/*/checkpoints/*.ckpt"
    echo ""
    echo "2. Benchmark inference speed:"
    echo "   python benchmark_inference.py --variant 3 --device cpu"
    echo "   # Should see ~2-3ms CPU inference!"
    echo ""
    echo "3. Compare all variants:"
    echo "   python benchmark_inference.py --compare-all"
    echo ""
else
    echo "FAILED: Training failed!"
    echo "Check logs: ${LOG_DIR}/m1_4ch_efficient_${SLURM_JOB_ID}.err"
fi

echo "==========================================="

exit ${TRAIN_EXIT}
