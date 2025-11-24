# ML Model Comparison Report

## Overview

This report compares three ML models for discrete gesture recognition using isolated EMG channels {5, 6, 7, 8, 9, 13, 15}.

## Models Compared

1. **M1: CNN+LSTM** - Hybrid architecture with convolutional front-end and LSTM layers
2. **M2: CNN-Only** - Pure CNN architecture with Inception blocks
3. **M3: Random Forest** - Classical ML with engineered features

## Results Summary

| Model             |   Training Time (hours) | Status    |   Val Accuracy |   Test CLER |
|:------------------|------------------------:|:----------|---------------:|------------:|
| M1: CNN+LSTM      |              15.2225    | completed |            nan |         nan |
| M2: CNN-Only      |               8.04037   | completed |            nan |         nan |
| M3: Random Forest |               0.0113602 | completed |            nan |         nan |

## Detailed Metrics

### M1: CNN+LSTM

- **Training Time**: 15.22 hours
- **Status**: completed
- **Metrics**:
  - No metrics available

### M2: CNN-Only

- **Training Time**: 8.04 hours
- **Status**: completed
- **Metrics**:
  - No metrics available

### M3: Random Forest

- **Training Time**: 0.01 hours
- **Status**: completed
- **Metrics**:
  - No metrics available

## Recommendations

- **Best Accuracy**: M1: CNN+LSTM (0.0000)
