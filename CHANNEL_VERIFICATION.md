# EMG Channel Verification for All 3 Models

## Required Isolated Channels
[9, 8, 7, 6, 5, 13, 15]

## Verification Results

### ✅ M1: CNN+LSTM
- **Config**: `discrete_gestures_isolated.yaml`
- **Data Module**: `discrete_gestures_data_module_isolated.yaml`
- **Channel Indices**: `[9, 8, 7, 6, 5, 13, 15]` ✅
- **Location**: Line 13 of `config/data_module/discrete_gestures_data_module_isolated.yaml`
- **Status**: CORRECT

### ✅ M2: CNN-Only  
- **Config**: `discrete_gestures_cnn_isolated.yaml`
- **Data Module**: `discrete_gestures_data_module_isolated.yaml` (same as M1)
- **Channel Indices**: `[9, 8, 7, 6, 5, 13, 15]` ✅
- **Location**: Line 13 of `config/data_module/discrete_gestures_data_module_isolated.yaml`
- **Status**: CORRECT

### ✅ M3: Random Forest
- **Command**: `train_random_forest.py --channel-indices 9 8 7 6 5 13 15`
- **Channel Indices**: `[9, 8, 7, 6, 5, 13, 15]` ✅
- **Source**: Command-line arguments from `compare_models.py` line 451
- **Status**: CORRECT

## Summary
✅ **All 3 models are correctly using the isolated EMG channel indices**

All models are training on the exact same 7 channels identified from the importance analysis:
- Channels 5, 6, 7, 8, 9 (wrist extensors region)
- Channel 13 (forearm flexors)  
- Channel 15 (forearm extensors)

This ensures a fair comparison between models as they all use the same input features.

