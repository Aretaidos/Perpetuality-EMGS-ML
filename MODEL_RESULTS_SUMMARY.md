# Model Comparison Results Summary

## Overview
Comparison run completed on November 16, 2025 at 01:40 (started 00:36)

## Results

### M1: CNN+LSTM (Kaifosh et al. 2025 Architecture)
- **Status**: ❌ Failed during test phase
- **Training**: ✅ Completed successfully 
- **Training Time**: ~35 minutes (88 epochs)
- **Validation Accuracy**: 22.36%
- **Validation Loss**: 0.0187
- **Best Checkpoint**: `epoch=88-step=890.ckpt`
- **Error**: cuDNN LSTM error during test (`CUDNN_STATUS_NOT_SUPPORTED: non-contiguous input`)
- **Note**: Training was successful, only testing failed

### M2: CNN-Only (Inception Block Architecture)
- **Status**: ✅ Completed
- **Training Time**: 28 minutes (0.46 hours)
- **Metrics**: Not extracted by comparison script
- **Issue**: Subprocess output not parsed for metrics
- **Logs**: Captured by subprocess but not saved to Hydra logs directory
- **Next Step**: Need to re-run or modify script to extract metrics from stdout

### M3: Random Forest (Classical ML Baseline)
- **Status**: ✅ Completed  
- **Training Time**: 31 seconds (0.0086 hours)
- **Metrics**: Not extracted by comparison script
- **Issue**: Model trained but no evaluation metrics captured
- **Next Step**: Need to add evaluation step to Random Forest training

## Channel Configuration
✅ All models correctly used isolated channels: [9, 8, 7, 6, 5, 13, 15]

## Issues Identified

1. **M1 LSTM Error**: Needs `.contiguous()` call before LSTM layer
2. **Metrics Extraction**: Comparison script doesn't parse subprocess output for M2
3. **M3 Evaluation**: Random Forest training doesn't include test/validation evaluation
4. **Logging**: Hydra logs not created for M2 (subprocess issue)

## Recommendations

1. Fix M1 LSTM contiguity issue in `networks.py`
2. Update `compare_models.py` to parse metrics from subprocess stdout
3. Add evaluation metrics to Random Forest training script
4. Consider saving checkpoint paths and loading them for evaluation

