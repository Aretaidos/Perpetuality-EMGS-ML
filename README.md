# Hand Gesture Recognition Model (HGAG Dataset)

A real-time hand gesture classifier trained on the HGAG (Hand Gesture Accelerometer Gyroscope) dataset using Edge Impulse, optimized for deployment on **Seeed XIAO nRF52840** with **ICM20948** IMU sensor. There are several issues with how this prototype classifier will be used: 
1) there is no "neutral" class, meaning resting is also improperly classified such that proper inferencing will have to occur "manually" 
2) low accuracy for class differentiation between wrist extension and flexion (inherent to the uncertainty of the IMU sensor to differentiate muscle group ) 

---

## Model Performance

- **Test Accuracy:** 92.11%
- **Training Accuracy:** 96-99%
- **Inference Time:** ~1ms
- **Memory Usage:** ~27.9K flash, ~1.9K Peak RAM
- **ROC-AUC:** 0.98-0.99

---

## Supported Gestures (6 Classes)

| Gesture | Description | Test Accuracy |
|---------|-------------|---------------|
| **Clapping** | Hands clapping together | 98.2% |
| **Index Thumb Tap** | Tapping index finger to thumb | 100% |
| **Horizontal Wrist Extension** | Extending wrist horizontally | 98.2% |
| **Fist Making** | Closing hand into fist | 79.8% |
| **Wrist Extension** | Extending wrist upward | 76% |
| **Wrist Flexion** | Flexing wrist downward | 73.1% |

---

## Hardware Requirements

- **Microcontroller:** Seeed XIAO nRF52840 (256KB RAM, 1MB Flash)
- **IMU Sensor:** ICM20948 (9-DOF: 3-axis accel, gyro, mag)
- **Connection:** I2C communication
- **Power:** USB-C or battery (3.3V-5V)

---

## Dataset Information

**Source:** HGAG-DATA (Hand Gesture Accelerometer Gyroscope)  
**Origin:** Mendeley Data  
**Total Samples:** 23,650 gesture recordings  
**Participants:** 43 individuals  
**Sampling Rate:** 200 Hz  
**Sensor Data:** 6-axis (3-axis accelerometer + 3-axis gyroscope)

### Data Characteristics
- **Window Size:** 2000ms (2 seconds per gesture)
- **Window Overlap:** 100ms sliding window
- **Total Features:** 1200 (200 samples × 6 axes)
- **Original Gestures:** 11 (reduced to 6 for optimal accuracy)

---

## Model Architecture

### Processing Pipeline
1. **Wavelet Transform:** rbio3.1, level 1, scale 0.0098, cutoff 94Hz
2. **Neural Network:** 4-layer dense architecture with BatchNorm + Dropout

### Network Structure
```
Input (1200 features)
    ↓
Dense(56, ReLU) → BatchNorm → Dropout(0.35)
    ↓
Dense(40, ReLU) → BatchNorm → Dropout(0.30)
    ↓
Dense(28, ReLU) → Dropout(0.25)
    ↓
Dense(16, ReLU) → Dropout(0.20)
    ↓
Output(6, Softmax)
```

### Training Configuration
- **Epochs:** 200
- **Learning Rate:** 0.0006
- **Batch Size:** 20
- **Optimizer:** Adam with gradient clipping (clipnorm=1.0)
- **Callbacks:** EarlyStopping (patience=25), ReduceLROnPlateau (patience=12)

---

## Deployment

### Edge Impulse Configuration
- **Optimization:** EON Compiler enabled
- **Quantization:** int8 (reduces model size by ~4x)
- **Target Device:** Seeed XIAO nRF52840

### Sensor Configuration
- **Accelerometer Range:** ±16G
- **Gyroscope Range:** ±2000 DPS
- **Sampling Rate:** 200 Hz (5ms intervals)
- **Data Units:** 
  - Accel: m/s²
  - Gyro: rad/s (not deg/s)

### Axis Order (CRITICAL)
```
accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z
```

---

## Performance by Gesture

```
Gesture                      | Precision | Recall | F1-Score
-----------------------------|-----------|--------|----------
Clapping                     |   0.95    |  0.96  |   0.95
Index Thumb Tap              |   1.00    |  1.00  |   1.00
Horizontal Wrist Extension   |   0.97    |  0.98  |   0.97
Fist Making                  |   0.75    |  0.80  |   0.75
Wrist Extension              |   0.80    |  0.76  |   0.80
Wrist Flexion                |   0.79    |  0.73  |   0.79
```

---

## Optimization History

### Initial Training (11 Gestures)
- **Accuracy:** 76.89% test, 85-90% train
- **Issues:** Confused gestures (Coin Flipping, Shooting, Index Finger Flicking, Thumb Up)
- **High "Uncertain" predictions:** 15-18% on problematic gestures

### After Gesture Reduction (6 Gestures)
- **Accuracy Gain:** +15.2% (76.89% → 92.11%)
- **Memory Reduction:** -% flash, -% RAM
- **Inference Speed:** +% faster
- **Removed Gestures:** Coin Flipping, Shooting, Index Finger Flicking, Thumb Up, Finger Snapping

---

## Technical Details

### Feature Extraction
- **Method:** Wavelet transform (rbio3.1 mother wavelet)
- **Decomposition Level:** 1
- **Scale Factor:** 0.0098
- **Frequency Cutoff:** 94 Hz
- **Purpose:** Captures both time and frequency domain characteristics for gesture discrimination

### Regularization Techniques
- **L2 Regularization:** 0.0008-0.001 on dense layers (prevents overfitting)
- **Batch Normalization:** Stabilizes training, enables higher learning rates
- **Dropout:** 0.20-0.35 (progressively higher in early layers)
- **Gradient Clipping:** clipnorm=1.0 (prevents exploding gradients)

### Data Preprocessing
- **Normalization:** Standard scaling applied to sensor data
- **Window Strategy:** Sliding window with 100% overlap during training

---

## 📖 References

- **HGAG Dataset:** [*Hand Gesture Accelerometer and Gyroscope Dataset*, Mendeley Data (2024)](https://data.mendeley.com/datasets/mkhn7kxjvy/1)
- **Edge Impulse:** [https://edgeimpulse.com](https://edgeimpulse.com)
- **Wavelet Analysis:** Mallat, S. (1999). *A Wavelet Tour of Signal Processing*
- **Seeed XIAO nRF52840:** [Product Documentation](https://wiki.seeedstudio.com/XIAO_BLE/)