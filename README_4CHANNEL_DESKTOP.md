# M1 4-Channel Desktop Models - Quick Start Guide

This guide covers the three M1 4-channel model variants optimized for desktop inference.

## Overview

Three architectural variants have been implemented to explore different trade-offs between accuracy, speed, and model complexity:

| Variant | Architecture | Parameters | Expected Accuracy | CPU Inference | Use Case |
|---------|-------------|------------|-------------------|---------------|----------|
| **Variant 1 (Desktop)** | Conv(4→144) + LSTM(3×128) | ~405K | 99.3-99.6% | 10-12ms | Maximum accuracy |
| **Variant 2 (Optimized)** | Multi-scale Conv + Attention + BiLSTM | ~135-155K | 99.55-99.65% | 15-25ms | Balanced |
| **Variant 3 (Efficient)** | Pure CNN (no LSTM) | ~120K | 98.0-99.0% | **2-3ms** | Ultra-fast inference |

## Quick Start

### Training All Variants in Parallel

The three variants can be trained simultaneously on your 4-GPU cluster:

```bash
# Create logs directory
mkdir -p /u/usz7pc/Perpetuality/generic-neuromotor-interface/logs

# Launch all three training jobs
cd /u/usz7pc/Perpetuality/generic-neuromotor-interface

# Variant 1 (Desktop) - uses GPU 0-1
sbatch slurm_train_m1_4channel_desktop.sbatch

# Variant 2 (Optimized) - uses GPU 2-3
sbatch slurm_train_m1_4channel_optimized.sbatch

# Variant 3 (Efficient) - uses GPU 1 (or submit after others)
sbatch slurm_train_m1_4channel_efficient.sbatch

# Monitor training
squeue -u $USER
watch nvidia-smi
```

### Expected Training Times

- **Variant 1 (Desktop)**: 1.8-2.2 hours on 2 GPUs
- **Variant 2 (Optimized)**: 1.5-2.0 hours on 2 GPUs
- **Variant 3 (Efficient)**: 15-20 minutes on 1 GPU

**Total wall time**: ~2-2.5 hours (variants run concurrently)

## Model Details

### Variant 1: Desktop (Scaled)

**Goal**: Maximum accuracy by scaling up to match 7-channel capacity

**Architecture**:
```
Conv1d(4→144, k=15, s=10) → BatchNorm → ReLU → Dropout(0.2)
→ LSTM(144→128, 3 layers) → LayerNorm → FC(9)
```

**Configuration**: `config/discrete_gestures_m1_4channel_desktop.yaml`
**Training Script**: `slurm_train_m1_4channel_desktop.sbatch`
**Checkpoints**: `./checkpoints_4channel_desktop/`

**When to use**:
- You need maximum accuracy (≥99.5%)
- Inference speed <50ms is acceptable
- Have GPU available for inference

---

### Variant 2: Optimized (Attention)

**Goal**: Custom architecture exploiting 4-channel characteristics

**Architecture**:
```
Multi-scale Conv (4→64, kernels 7/15/25) → Channel Attention
→ Downsample Conv (64→96, s=10) → Residual Conv
→ BiLSTM(48, 1 layer) → LayerNorm → FC(9)
```

**Key Innovations**:
- Multi-scale convolutions capture different EMG speeds
- Channel attention dynamically weights importance
- Bidirectional LSTM (desktop can look ahead)

**Configuration**: `config/discrete_gestures_m1_4channel_optimized.yaml`
**Training Script**: `slurm_train_m1_4channel_optimized.sbatch`
**Checkpoints**: `./checkpoints_4channel_optimized/`

**When to use**:
- You want balance between accuracy and efficiency
- Interested in attention visualization
- Acceptable 15-25ms inference time

---

### Variant 3: Efficient (Fast CNN)

**Goal**: Ultra-fast inference via pure CNN

**Architecture**:
```
Conv(4→48, s=10) → DilatedConv(48→64, d=2) → DepthwiseSep(64→96)
→ Conv(96→128, s=2) → Conv(128→64) → Conv(64→9)
```

**Key Features**:
- **NO LSTM** - eliminates sequential bottleneck
- Dilated convolutions for 65ms temporal context
- Fully parallelizable (MKL/oneDNN optimized)

**Configuration**: `config/discrete_gestures_m1_4channel_efficient.yaml`
**Training Script**: `slurm_train_m1_4channel_efficient.sbatch`
**Checkpoints**: `./checkpoints_4channel_efficient/`

**When to use**:
- Real-time applications requiring <10ms latency
- Gaming/VR control systems
- High-throughput batch processing
- Acceptable 98%+ accuracy (vs 99.5%+)

## After Training

### 1. Check Results

```bash
# Find best checkpoints for each variant
ls -lh checkpoints_4channel_desktop/*/checkpoints/best*.ckpt
ls -lh checkpoints_4channel_optimized/*/checkpoints/best*.ckpt
ls -lh checkpoints_4channel_efficient/*/checkpoints/best*.ckpt

# View training logs
tail -100 logs/m1_4ch_desktop_*.out
tail -100 logs/m1_4ch_optimized_*.out
tail -100 logs/m1_4ch_efficient_*.out
```

### 2. Benchmark Inference Speed

```bash
# Activate conda environment
conda activate neuromotor

# Compare all variants on CPU
python benchmark_inference.py --compare-all --device cpu

# Compare on GPU
python benchmark_inference.py --compare-all --device cuda

# Benchmark specific variant with checkpoint
python benchmark_inference.py \
    --variant 3 \
    --checkpoint checkpoints_4channel_efficient/lightning_logs/version_0/checkpoints/best.ckpt \
    --device cpu
```

### 3. TensorBoard Visualization

```bash
# Launch TensorBoard for all variants
tensorboard --logdir_spec \
  Desktop:checkpoints_4channel_desktop,\
  Optimized:checkpoints_4channel_optimized,\
  Efficient:checkpoints_4channel_efficient

# Open browser to http://localhost:6006
```

## Channel Selection

All three variants use the same 4-channel subset:

| Channel | 1-Based | 0-Based Index | Muscle | Type | Importance |
|---------|---------|---------------|--------|------|------------|
| Ch7 | 7 | 6 | Index flexor | Press | 24.5% |
| Ch8 | 8 | 7 | Ring flexor | Press | 13.2% |
| Ch13 | 13 | 12 | Index/middle extensor | Release | 19.8% |
| Ch15 | 15 | 14 | Ring/pinky extensor | Release | 8.7% |

**Total Importance**: 66.2% (balanced flexor-extensor design)

## Comparison with Existing Models

| Model | Channels | Params | Val Acc | Inference | Deployment |
|-------|----------|--------|---------|-----------|------------|
| **M1 Full (7ch)** | 7 | 411K | 99.55% | 15-20ms CPU | Reference |
| **M1 TinyML (4ch)** | 4 | 47K | 99.59% | 40ms ARM | XIAO nRF52840 |
| **M1 Desktop (4ch)** | 4 | 405K | 99.3-99.6% | 10-12ms CPU | **New - Max accuracy** |
| **M1 Optimized (4ch)** | 4 | 135-155K | 99.55-99.65% | 15-25ms CPU | **New - Balanced** |
| **M1 Efficient (4ch)** | 4 | 120K | 98.0-99.0% | **2-3ms CPU** | **New - Ultra fast** |

## Troubleshooting

### Training Issues

**GPU Out of Memory**:
```bash
# Reduce batch size in config
# Edit config file: batch_size: 64 → 32
```

**Training Too Slow**:
```bash
# Check GPU utilization
nvidia-smi dmon -s u

# Ensure using DDP
# Should see multiple processes in nvidia-smi
```

**Poor Convergence**:
```bash
# Check learning rate schedule in logs
# May need to adjust warmup epochs or LR milestones
```

### Inference Issues

**Slow CPU Inference**:
```bash
# Ensure using optimized PyTorch build with MKL
python -c "import torch; print(torch.__config__.parallel_info())"

# Should see: "ATen/Parallel: OpenMP"
```

## Next Steps

1. **Compare Variants**: Use `benchmark_inference.py --compare-all` to determine which variant best fits your use case

2. **Deploy Best Model**:
   - Variant 1/2: For applications requiring ≥99% accuracy
   - Variant 3: For real-time applications (<10ms latency)

3. **Export for Production**:
   ```bash
   # ONNX export (for cross-platform deployment)
   python -m generic_neuromotor_interface.export_onnx \
       --checkpoint best_model.ckpt \
       --output model.onnx
   ```

4. **Test on Real Data**: Integrate best variant into your desktop application using the signal_drift demo as template

## File Structure

```
generic-neuromotor-interface/
├── config/
│   ├── discrete_gestures_m1_4channel_desktop.yaml     # Variant 1 config
│   ├── discrete_gestures_m1_4channel_optimized.yaml   # Variant 2 config
│   └── discrete_gestures_m1_4channel_efficient.yaml   # Variant 3 config
├── generic_neuromotor_interface/
│   └── networks_isolated.py                            # Network classes
├── slurm_train_m1_4channel_desktop.sbatch             # Variant 1 training
├── slurm_train_m1_4channel_optimized.sbatch           # Variant 2 training
├── slurm_train_m1_4channel_efficient.sbatch           # Variant 3 training
├── benchmark_inference.py                              # Performance analysis
├── checkpoints_4channel_desktop/                       # Variant 1 outputs
├── checkpoints_4channel_optimized/                     # Variant 2 outputs
└── checkpoints_4channel_efficient/                     # Variant 3 outputs
```

## Questions?

- Training issues: Check SLURM logs in `logs/`
- Model architecture: See network classes in `generic_neuromotor_interface/networks_isolated.py`
- Performance: Run `benchmark_inference.py --compare-all`

Happy training! 🚀
