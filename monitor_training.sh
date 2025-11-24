#!/bin/bash

##############################################################################
# Training Monitoring Helper Script
#
# Quick commands to monitor your running training jobs
##############################################################################

OUTPUT_DIR="./models_gpu_multi_20251124_121916"

echo "==========================================="
echo "  sEMG Training Monitor"
echo "==========================================="
echo ""

# Check if processes are running
echo "[1] Process Status:"
if ps aux | grep -E "(3342003|3342014|3342048)" | grep -v grep > /dev/null; then
    echo "  ✓ Training processes are running"
    ps aux | grep train_all_models_fixed | grep -v grep | awk '{printf "    PID %s: %s%%CPU, %sMB RAM\n", $2, $3, int($6/1024)}'
else
    echo "  ⚠ No training processes found"
fi

echo ""
echo "[2] GPU Utilization:"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader | \
    awk -F', ' '{printf "  GPU %s: %s util, %s/%s memory, %s°C\n", $1, $3, $4, $5, $6}'

echo ""
echo "[3] Training Progress:"
echo ""
echo "  M1 (CNN+LSTM) - GPU 0:"
tail -3 ${OUTPUT_DIR}/m1_training.log 2>/dev/null | sed 's/^/    /' || echo "    (loading...)"

echo ""
echo "  M2 (CNN-only) - GPU 1:"
tail -3 ${OUTPUT_DIR}/m2_training.log 2>/dev/null | sed 's/^/    /' || echo "    (loading...)"

echo ""
echo "  M3 (Random Forest) - CPU:"
tail -3 ${OUTPUT_DIR}/m3_training.log 2>/dev/null | sed 's/^/    /' || echo "    (loading...)"

echo ""
echo "==========================================="
echo ""
echo "Commands:"
echo "  Watch logs:  tail -f ${OUTPUT_DIR}/m1_training.log"
echo "  GPU monitor: watch -n 2 nvidia-smi"
echo "  This script: bash monitor_training.sh"
echo ""
echo "Output directory: ${OUTPUT_DIR}"
echo "==========================================="
