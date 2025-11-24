#!/usr/bin/env python3
"""
Quick test script to verify all fixes are working correctly.

Run this to ensure:
1. Imports work correctly
2. Models can be instantiated
3. Forward passes work with correct shapes
4. Channel indices are correct
"""

import torch
import numpy as np
from generic_neuromotor_interface.transforms_isolated import (
    IsolatedDiscreteGesturesTransform,
    ISOLATED_CHANNELS,
)
from generic_neuromotor_interface.networks_isolated import (
    FixedDiscreteGesturesLSTM,
    FixedDiscreteGesturesCNN,
    count_parameters,
)
from generic_neuromotor_interface.random_forest_model import (
    FixedRandomForestGestureModel,
    RobustEMGFeatureExtractor,
)

def test_channel_indices():
    """Test channel indices are correct (0-based)."""
    print("Testing channel indices...")
    assert ISOLATED_CHANNELS == [4, 5, 6, 7, 8, 12, 14], "Channel indices incorrect!"
    print("✓ Channel indices correct: [4, 5, 6, 7, 8, 12, 14]")

def test_m1_model():
    """Test M1 (CNN+LSTM) model."""
    print("\nTesting M1 (CNN+LSTM) model...")

    model = FixedDiscreteGesturesLSTM(input_channels=7)

    # Test forward pass
    batch_size = 4
    seq_len = 2000  # 1 second at 2kHz
    x = torch.randn(batch_size, 7, seq_len)

    output = model(x)

    # Check output shape
    expected_time_out = (seq_len - model.left_context) // model.stride
    assert output.shape[0] == batch_size, f"Batch size mismatch: {output.shape[0]} vs {batch_size}"
    assert output.shape[1] == 9, f"Output channels mismatch: {output.shape[1]} vs 9"

    print(f"✓ M1 forward pass successful")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Parameters: {count_parameters(model):,}")
    print(f"  left_context: {model.left_context}, stride: {model.stride}")

def test_m2_model():
    """Test M2 (CNN-only) model."""
    print("\nTesting M2 (CNN-only) model...")

    model = FixedDiscreteGesturesCNN(input_channels=7)

    # Test forward pass
    batch_size = 4
    seq_len = 2000
    x = torch.randn(batch_size, 7, seq_len)

    output = model(x)

    assert output.shape[0] == batch_size
    assert output.shape[1] == 9

    print(f"✓ M2 forward pass successful")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Parameters: {count_parameters(model):,}")
    print(f"  left_context: {model.left_context}, stride: {model.stride}")

def test_m3_model():
    """Test M3 (Random Forest) model."""
    print("\nTesting M3 (Random Forest) model...")

    # Create model
    model = FixedRandomForestGestureModel(
        n_estimators=10,  # Small for testing
        max_depth=5,
        n_channels=7,
    )

    # Create dummy training data
    n_samples = 100
    window_len = 400  # 200ms at 2kHz

    X_train = np.random.randn(n_samples, 7, window_len).astype(np.float32)
    y_train = np.random.randint(0, 9, n_samples)

    # Test feature extraction
    extractor = RobustEMGFeatureExtractor()
    features = extractor.extract_features(X_train[:10])

    assert features.shape == (10, 7 * 18), f"Feature shape mismatch: {features.shape}"

    print(f"✓ M3 feature extraction successful")
    print(f"  Input shape: {X_train[:10].shape}")
    print(f"  Feature shape: {features.shape}")
    print(f"  Features per channel: 18")
    print(f"  Total features: {7 * 18}")

    # Test training
    print("  Training RF model...")
    results = model.fit(X_train, y_train)

    print(f"✓ M3 training successful")
    print(f"  CV Accuracy: {results['cv_accuracy_mean']:.3f}")

    # Test prediction
    X_test = np.random.randn(10, 7, window_len).astype(np.float32)
    y_pred = model.predict(X_test)

    assert y_pred.shape == (10,), f"Prediction shape mismatch: {y_pred.shape}"

    print(f"✓ M3 prediction successful")
    print(f"  Test shape: {X_test.shape}")
    print(f"  Prediction shape: {y_pred.shape}")

def test_gpu_availability():
    """Test GPU availability."""
    print("\nTesting GPU availability...")

    if torch.cuda.is_available():
        print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

        # Test model on GPU
        model = FixedDiscreteGesturesLSTM(input_channels=7).cuda()
        x = torch.randn(2, 7, 1000).cuda()
        output = model(x)
        print(f"✓ GPU inference successful")
    else:
        print("⚠ No GPU available - will use CPU")

def main():
    print("="*60)
    print("Testing Fixed sEMG Gesture Recognition Models")
    print("="*60)

    try:
        test_channel_indices()
        test_m1_model()
        test_m2_model()
        test_m3_model()
        test_gpu_availability()

        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        print("\nYour models are ready to train!")
        print("\nNext steps:")
        print("1. Download data: python -m generic_neuromotor_interface.download_utils --data-dir ~/emg_data")
        print("2. Train models: python generic_neuromotor_interface/scripts/train_all_models_fixed.py --model all --data-dir ~/emg_data --gpu")
        print("3. Evaluate: python generic_neuromotor_interface/scripts/evaluate_models_comprehensive.py --data-dir ~/emg_data --models-dir ./models --gpu")
        print("\nSee TRAINING_GUIDE.md for detailed instructions.")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == '__main__':
    exit(main())
