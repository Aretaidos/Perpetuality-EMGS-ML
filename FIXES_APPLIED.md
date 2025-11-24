# Fixes Applied for 3-Model Comparison

## Date: November 16, 2025

## Issues Fixed

### 1. M1 (CNN+LSTM) - LSTM cuDNN Error ✅
**Problem**: `cuDNN error: CUDNN_STATUS_NOT_SUPPORTED: non-contiguous input`  
**Fix**: Added `.contiguous()` call before LSTM layer in `networks.py` line 171
**File**: `generic_neuromotor_interface/networks.py`
**Change**:
```python
# Before LSTM
x = x.contiguous()  # NEW LINE
x, _ = self.lstm(x)
```

### 2. .bashrc Syntax Error ✅  
**Problem**: Missing `;;` in case statement caused bash errors
**Fix**: Added `;;` to line 10 of `/u/usz7pc/.bashrc`
**File**: `/u/usz7pc/.bashrc`

### 3. GPU Detection - CUDA Module Loading ✅
**Problem**: PyTorch couldn't detect GPUs (CUDA available: False)
**Fix**: Updated scripts to load `cuda/12.8.1` module before training
**Files**:
- `scripts/run_all_models_uva.sh` 
- `scripts/run_all_models_uva_background.sh`
- `scripts/run_comparison_uva_gpu.sh` (NEW)

### 4. Environment Variables for Subprocesses ✅
**Problem**: LD_LIBRARY_PATH not preserved in subprocesses
**Fix**: `compare_models.py` now copies parent environment to subprocesses
**File**: `scripts/compare_models.py`

### 5. Created UVA GPU-Optimized Script ✅
**New File**: `scripts/run_comparison_uva_gpu.sh`
**Features**:
- Follows UVA CS module guidelines
- Purges modules before loading
- Loads miniforge + cuda/12.8.1
- Verifies GPU access before training
- Sets CUDA_VISIBLE_DEVICES for specific GPU selection

## Remaining Issues to Fix

### M2 & M3: Metrics Not Extracted
**Problem**: Training completed but no metrics in final report
**Root Cause**: 
- M2: Subprocess stdout not parsed for metrics
- M3: Training script doesn't include evaluation step

**Solution Needed**: Update comparison script to parse subprocess output or save metrics to files

## How to Run

### Quick Run (Default GPU 0):
```bash
cd /u/usz7pc/Perpetuality/generic-neuromotor-interface
./scripts/run_comparison_uva_gpu.sh
```

### Specify GPU:
```bash
./scripts/run_comparison_uva_gpu.sh /u/usz7pc/emg_data ./model_comparison_gpu 2
```

### Background (with nohup):
```bash
nohup ./scripts/run_comparison_uva_gpu.sh > training_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

## Expected Behavior

With these fixes:
- ✅ M1 should complete training AND testing without cuDNN errors
- ✅ M2 should complete training (metrics extraction still pending)
- ✅ M3 should complete training (evaluation still pending)
- ✅ All models use correct isolated channels [9, 8, 7, 6, 5, 13, 15]
- ✅ All models run on GPU with CUDA support

## Verification

After running, check:
1. M1 trained 88+ epochs and completed test phase
2. M2 trained for ~28 minutes
3. M3 trained in ~31 seconds
4. All used GPU (check with `nvidia-smi` during training)
5. Report generated at `./model_comparison_gpu/model_comparison_report.md`

