# Isolated EMG Channel Models

This document describes the isolated channel models created based on channel importance analysis.

## Selected Channels

Based on handwriting channel importance analysis, the following channels were identified as most critical:
- **Channels: 9, 8, 7, 6, 5, 13, 15** (7 channels total, out of original 16)

These channels showed the highest impact when occluded during validation testing.

## Architecture Changes

### Handwriting Model
- **Input channels**: Reduced from 16 to 7
- **Network**: `HandwritingArchitecture` with `num_channels=7`
- **Transform**: `IsolatedHandwritingTransform` that selects only the specified channels

### Discrete Gestures Models  
- **Input channels**: Reduced from 16 to 7
- **Network options**:
  - `DiscreteGesturesArchitecture` (CNN+LSTM) with `input_channels=7` - Original architecture
  - `DiscreteGesturesCNNArchitecture` (1D CNN-only) with `input_channels=7` - New streamlined architecture
- **Transform**: `IsolatedDiscreteGesturesTransform` that selects only the specified channels

## New Components

### 1. Channel Selector (`channel_selector.py`)
- Utility class to select specific channels from EMG data
- Handles both numpy arrays and torch tensors
- Supports various tensor shapes (2D, 3D)

### 2. Isolated Transforms (`transforms_isolated.py`)
- `IsolatedHandwritingTransform`: Handwriting transform with channel selection
- `IsolatedDiscreteGesturesTransform`: Discrete gestures transform with channel selection
- `IsolatedWristTransform`: Wrist transform with channel selection (for future use)

### 3. Configuration Files

#### Handwriting Isolated
- `config/handwriting_isolated.yaml`: Main config
- `config/data_module/handwriting_data_module_isolated.yaml`: Data module with isolated transform
- `config/lightning_module/handwriting_module_isolated.yaml`: Network with 7 channels

#### Discrete Gestures Isolated
- `config/discrete_gestures_isolated.yaml`: Main config
- `config/data_module/discrete_gestures_data_module_isolated.yaml`: Data module with isolated transform
- `config/lightning_module/discrete_gestures_module_isolated.yaml`: Network with 7 input channels

## Usage

### Training Handwriting Isolated Model

```bash
python -m generic_neuromotor_interface.train --config-name=handwriting_isolated
```

### Training Discrete Gestures Isolated Models

#### CNN+LSTM Model (Original Architecture)
```bash
python -m generic_neuromotor_interface.train --config-name=discrete_gestures_isolated
```

#### 1D CNN-Only Model (Streamlined Architecture)
```bash
python -m generic_neuromotor_interface.train --config-name=discrete_gestures_cnn_isolated
```

#### Random Forest Model (Classical ML Baseline)
```bash
python -m generic_neuromotor_interface.scripts.train_random_forest \
    --data-location ~/emg_data \
    --channel-indices 9 8 7 6 5 13 15 \
    --n-estimators 200 \
    --output-dir ./models
```

### Training on GPU Cluster

To run on the GPU cluster (gpusrv17), you may need to:
1. Set up SSH access or use a job scheduler
2. Ensure CUDA is available: `trainer.accelerator=gpu`
3. Adjust batch sizes if needed for GPU memory

Example with GPU:
```bash
python -m generic_neuromotor_interface.train \
    --config-name=handwriting_isolated \
    trainer.accelerator=gpu \
    trainer.devices=1
```

## Channel Importance Results

### Handwriting (Occlusion Analysis)
- **Baseline CER**: 100.0
- **Top channels by impact** (when removed, CER increases):
  - Channel 9: +550.0 CER increase
  - Channel 6: +487.5 CER increase
  - Channel 5: +81.25 CER increase
  - Channel 1: +75.0 CER increase
  - Channels 7, 8, 13: +25.0 CER increase each
  - Channel 15: +12.5 CER increase

### Discrete Gestures (Weight Analysis)
- All channels showed relatively uniform importance
- Channels 13, 5, 1, 12, 0 showed slightly higher weight magnitudes

## Additional Models

### 1D CNN Architecture (Model M2)

A pure CNN architecture with Inception blocks, optimized for embedded deployment:
- **Location**: `generic_neuromotor_interface.networks.DiscreteGesturesCNNArchitecture`
- **Features**:
  - Multi-scale temporal receptive fields via Inception blocks
  - ~5.2 MFLOPs per inference (vs ~16.3 MFLOPs for CNN+LSTM)
  - Suitable for on-device inference on ARM Cortex-M4
  - Uses same 7 isolated channels: {9, 8, 7, 6, 5, 13, 15}

**Architecture**:
- Initial conv block (7→64 channels, kernel 11, MaxPool)
- Inception block (3 parallel paths: kernel 5, 15, 25)
- Second conv block (192→128 channels, kernel 7, MaxPool)
- Additional conv block (128→128 channels, kernel 5)
- Per-time-step classification head

**Configuration**: `config/discrete_gestures_cnn_isolated.yaml`

### Random Forest Model (Model M3)

A classical ML approach with engineered features:
- **Location**: `generic_neuromotor_interface.random_forest_model.RandomForestGestureModel`
- **Features**:
  - 10 features per channel (6 time-domain + 4 frequency-domain)
  - Total: 70 features (10 × 7 channels)
  - ~5 ms inference time on Cortex-M4
  - Requires per-user calibration
  - Highly interpretable (feature importance analysis)

**Feature Extraction**:
- Time-domain: MAV, RMS, Zero Crossings, Slope Sign Changes, Waveform Length, iEMG
- Frequency-domain: Mean Frequency, Median Frequency, Peak Frequency, Total Power

**Training Script**: `generic_neuromotor_interface/scripts/train_random_forest.py`

**Configuration**: `config/discrete_gestures_rf_isolated.yaml`

## Model Comparison

| Model | Architecture | FLOPs/Inference | Inference Time (M4) | Expected Accuracy | Use Case |
|-------|-------------|-----------------|---------------------|------------------|----------|
| CNN+LSTM | Deep learning | ~16.3M | ~30-40ms | 93-96% | Maximum accuracy, cross-user |
| 1D CNN | Streamlined CNN | ~5.2M | ~8-12ms | 88-92% | On-device, moderate accuracy |
| Random Forest | Classical ML | ~0.002M | ~5ms | 82-88% | Ultra-low power, per-user calibration |

## Notes

- The isolated models use only 7 channels instead of 16, reducing input dimensionality by ~56%
- This may improve training speed and reduce model complexity
- Performance may be slightly reduced compared to full 16-channel models
- The selected channels (9, 8, 7, 6, 5, 13, 15) represent the most informative locations for handwriting recognition
- All three discrete gesture models use the same isolated channel configuration for fair comparison

