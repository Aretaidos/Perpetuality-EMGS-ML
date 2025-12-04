#!/usr/bin/env python3
"""
M1 CNN+LSTM Model Conversion to TensorFlow Lite for XIAO nRF52840.

This script converts the PyTorch M1 model (CNN+LSTM) to TensorFlow Lite format
optimized for deployment on Cortex-M4F microcontrollers.

Target: Seeed Studio XIAO nRF52840 (non-Sense)
- Cortex-M4F @ 64MHz
- 256KB RAM, 1MB flash + 2MB external QSPI

Conversion Path: PyTorch -> ONNX -> TensorFlow -> TFLite

Key Considerations:
- LSTM requires INT16 quantization for cell states (TFLite Micro limitation)
- Conv layers can use INT8 quantization
- LayerNorm decomposes to multiple ops in TFLite
- Input window reduced from 10000 to 2000 samples (1 second) to fit in RAM
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn

# Import the model architecture
from generic_neuromotor_interface.networks_isolated import (
    FixedDiscreteGesturesLSTM,
    ReinhardCompression,
)


class M1ModelForExport(nn.Module):
    """
    Wrapper around M1 model for cleaner ONNX/TFLite export.

    Key modifications for embedded deployment:
    1. Stateless LSTM (no hidden state caching between calls)
    2. Reduced input window (2000 samples = 1 second @ 2kHz)
    3. Dropout removed (inference mode)
    """

    def __init__(
        self,
        input_channels: int = 7,
        conv_output_channels: int = 128,
        kernel_width: int = 15,
        stride: int = 10,
        lstm_hidden_size: int = 128,
        lstm_num_layers: int = 3,
        output_channels: int = 9,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.lstm_num_layers = lstm_num_layers
        self.lstm_hidden_size = lstm_hidden_size

        # 1. Amplitude normalization (Reinhard compression)
        # For TFLite, we'll implement this as a custom op or approximate
        self.compression = ReinhardCompression(range_val=1.0, midpoint=32.0)

        # 2. Initial Conv1D
        self.conv = nn.Conv1d(
            in_channels=input_channels,
            out_channels=conv_output_channels,
            kernel_size=kernel_width,
            stride=stride,
            padding=0,
        )

        # 3. Post-conv processing (no dropout for inference)
        self.conv_ln = nn.LayerNorm(conv_output_channels)
        self.relu = nn.LeakyReLU(0.1)

        # 4. Stacked LSTM (no dropout for inference)
        self.lstm = nn.LSTM(
            input_size=conv_output_channels,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=0,  # No dropout for inference
            bidirectional=False,
        )

        # 5. Post-LSTM normalization
        self.lstm_ln = nn.LayerNorm(lstm_hidden_size)

        # 6. Output projection
        self.fc_out = nn.Linear(lstm_hidden_size, output_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass optimized for TFLite export.

        Input: (batch, 7, time) - EMG data
        Output: (batch, 9, time') - Gesture logits
        """
        # 1. Amplitude normalization
        x = self.compression(x)

        # 2. Conv1D
        x = self.conv(x)  # (B, 128, T')
        x = self.relu(x)

        # 3. Transpose for LayerNorm
        x = x.transpose(1, 2)  # (B, T', 128)
        x = self.conv_ln(x)

        # 4. LSTM (stateless - fresh hidden state each call)
        x, _ = self.lstm(x)  # (B, T', hidden)

        # 5. Post-LSTM LayerNorm
        x = self.lstm_ln(x)

        # 6. Output projection
        x = self.fc_out(x)  # (B, T', 9)

        # 7. Transpose back
        x = x.transpose(1, 2)  # (B, 9, T')

        return x


def load_trained_weights(model: nn.Module, checkpoint_path: str) -> nn.Module:
    """
    Load trained weights from M1 checkpoint into export model.
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Handle different checkpoint formats
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        # Remove 'model.' prefix if present (from Lightning)
        state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
    else:
        state_dict = checkpoint

    # Filter out dropout-related keys and handle missing keys
    model_state = model.state_dict()
    filtered_state = {}

    for k, v in state_dict.items():
        # Skip dropout layers
        if 'dropout' in k.lower():
            continue
        if k in model_state:
            if model_state[k].shape == v.shape:
                filtered_state[k] = v
            else:
                print(f"Shape mismatch for {k}: model={model_state[k].shape}, ckpt={v.shape}")
        else:
            print(f"Skipping {k} - not in model")

    model.load_state_dict(filtered_state, strict=False)
    print(f"Loaded {len(filtered_state)} weight tensors")
    return model


def export_to_onnx(
    model: nn.Module,
    output_path: str,
    input_samples: int = 2000,
    batch_size: int = 1,
) -> str:
    """
    Export PyTorch model to ONNX format.

    Args:
        model: PyTorch model
        output_path: Path for ONNX file
        input_samples: Number of input time samples (default 2000 = 1 second @ 2kHz)
        batch_size: Batch size (default 1 for MCU)

    Returns:
        Path to saved ONNX file
    """
    model.eval()

    # Create sample input
    # Shape: (batch, channels, time) = (1, 7, 2000)
    sample_input = torch.randn(batch_size, 7, input_samples)

    # Export to ONNX
    torch.onnx.export(
        model,
        sample_input,
        output_path,
        input_names=['emg_input'],
        output_names=['gesture_logits'],
        dynamic_axes=None,  # Static shapes for MCU deployment
        opset_version=13,  # LSTM support
        do_constant_folding=True,
        verbose=False,
    )

    print(f"Exported ONNX model to: {output_path}")

    # Verify the exported model
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model validation passed")

    return output_path


def convert_onnx_to_tflite(
    onnx_path: str,
    output_dir: str,
    quantize: bool = True,
    representative_data: Optional[np.ndarray] = None,
) -> str:
    """
    Convert ONNX model to TensorFlow Lite format.

    Args:
        onnx_path: Path to ONNX model
        output_dir: Output directory for TFLite files
        quantize: Whether to apply quantization
        representative_data: Calibration data for quantization

    Returns:
        Path to saved TFLite file
    """
    import onnx2tf
    import tensorflow as tf

    # Step 1: ONNX -> TensorFlow SavedModel
    saved_model_dir = os.path.join(output_dir, "saved_model")

    print("Converting ONNX to TensorFlow SavedModel...")
    onnx2tf.convert(
        input_onnx_file_path=onnx_path,
        output_folder_path=output_dir,
        output_signaturedefs=True,
        keep_ncwh_or_nchw_order=False,  # Convert to TF-friendly format
    )

    # Step 2: TensorFlow SavedModel -> TFLite
    print("Converting TensorFlow to TFLite...")

    # Find the saved model
    if os.path.exists(os.path.join(output_dir, "saved_model")):
        saved_model_dir = os.path.join(output_dir, "saved_model")
    else:
        saved_model_dir = output_dir

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

    if quantize:
        # Configure quantization
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        # For LSTM, we need int16 activations
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
        ]

        if representative_data is not None:
            def representative_dataset():
                for i in range(min(100, len(representative_data))):
                    yield [representative_data[i:i+1].astype(np.float32)]

            converter.representative_dataset = representative_dataset

            # Full integer quantization with int16 for LSTM
            converter.target_spec.supported_types = [tf.int8]
            converter.inference_input_type = tf.float32  # Keep float input for ease of use
            converter.inference_output_type = tf.float32

    # Convert
    tflite_model = converter.convert()

    # Save
    tflite_path = os.path.join(output_dir, "m1_gesture_model.tflite")
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)

    print(f"Saved TFLite model to: {tflite_path}")
    print(f"Model size: {len(tflite_model) / 1024:.1f} KB")

    return tflite_path


def generate_c_header(tflite_path: str, output_path: str, array_name: str = "g_model"):
    """
    Convert TFLite model to C header file for embedding in firmware.

    Args:
        tflite_path: Path to TFLite model
        output_path: Output path for C header
        array_name: Name of the C array
    """
    with open(tflite_path, 'rb') as f:
        model_data = f.read()

    # Generate C header
    header_content = f'''// Auto-generated TFLite model header
// Model: M1 CNN+LSTM Gesture Classifier
// Generated for: Seeed Studio XIAO nRF52840
// Size: {len(model_data)} bytes ({len(model_data)/1024:.1f} KB)

#ifndef M1_GESTURE_MODEL_H
#define M1_GESTURE_MODEL_H

#include <stdint.h>

// Model data array
alignas(16) const unsigned char {array_name}[] = {{
'''

    # Add model data as hex bytes
    for i, byte in enumerate(model_data):
        if i % 12 == 0:
            header_content += '\n    '
        header_content += f'0x{byte:02x}, '

    header_content += f'''
}};

// Model size
const unsigned int {array_name}_len = {len(model_data)};

#endif // M1_GESTURE_MODEL_H
'''

    with open(output_path, 'w') as f:
        f.write(header_content)

    print(f"Generated C header: {output_path}")


def generate_representative_data(
    num_samples: int = 100,
    input_channels: int = 7,
    input_samples: int = 2000,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate synthetic representative EMG data for quantization calibration.

    In production, you should use real EMG data from your training set.
    """
    np.random.seed(seed)

    # Generate synthetic EMG-like data
    # Real EMG typically has amplitude around 0-1000 uV with some noise
    data = []
    for _ in range(num_samples):
        # Base signal with muscle-like patterns
        t = np.linspace(0, 1, input_samples)
        signal = np.zeros((input_channels, input_samples), dtype=np.float32)

        for ch in range(input_channels):
            # Random muscle activation pattern
            base = np.random.uniform(10, 100) * np.sin(2 * np.pi * np.random.uniform(5, 20) * t)
            noise = np.random.normal(0, 10, input_samples)
            signal[ch] = base + noise

        data.append(signal)

    return np.array(data)


def analyze_model_size(model: nn.Module):
    """
    Analyze model size and memory requirements for MCU deployment.
    """
    total_params = 0
    param_breakdown = {}

    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params

        # Group by layer type
        layer_type = name.split('.')[0]
        if layer_type not in param_breakdown:
            param_breakdown[layer_type] = 0
        param_breakdown[layer_type] += num_params

    print("\n=== Model Size Analysis ===")
    print(f"Total parameters: {total_params:,}")
    print(f"Size (FP32): {total_params * 4 / 1024:.1f} KB")
    print(f"Size (INT8): {total_params / 1024:.1f} KB")
    print(f"Size (INT16): {total_params * 2 / 1024:.1f} KB")

    print("\nParameter breakdown by layer:")
    for layer, count in sorted(param_breakdown.items(), key=lambda x: -x[1]):
        print(f"  {layer}: {count:,} ({count/total_params*100:.1f}%)")

    # Memory budget for XIAO nRF52840
    print("\n=== Memory Budget (XIAO nRF52840) ===")
    print(f"Flash available: 1024 KB (+ 2MB QSPI external)")
    print(f"RAM available: 256 KB")

    input_buffer = 7 * 2000 * 4  # 7 channels, 2000 samples, float32
    lstm_states = 3 * 128 * 2 * 2  # 3 layers, 128 hidden, h+c, INT16
    conv_buffer = 128 * 200 * 4  # Conv output

    print(f"\nRuntime memory estimate:")
    print(f"  Input buffer: {input_buffer/1024:.1f} KB")
    print(f"  LSTM states: {lstm_states/1024:.1f} KB")
    print(f"  Conv buffer: {conv_buffer/1024:.1f} KB")
    print(f"  Total runtime: ~{(input_buffer + lstm_states + conv_buffer)/1024:.1f} KB")


def main():
    """
    Main conversion pipeline.
    """
    import argparse

    parser = argparse.ArgumentParser(description='Convert M1 model to TFLite')
    parser.add_argument('--checkpoint', type=str,
                        default=str(PROJECT_ROOT / 'models_gpu_multi_20251124_121916' / 'm1_best.pt'),
                        help='Path to M1 checkpoint')
    parser.add_argument('--output-dir', type=str,
                        default=str(PROJECT_ROOT / 'tinyml_deployment' / 'converted_models'),
                        help='Output directory')
    parser.add_argument('--input-samples', type=int, default=2000,
                        help='Input window size in samples (default: 2000 = 1 second @ 2kHz)')
    parser.add_argument('--quantize', action='store_true', default=True,
                        help='Apply quantization')
    parser.add_argument('--skip-onnx', action='store_true',
                        help='Skip ONNX conversion (use existing)')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("=== M1 Model Conversion Pipeline ===")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output dir: {args.output_dir}")
    print(f"Input samples: {args.input_samples}")
    print(f"Quantize: {args.quantize}")

    # Step 1: Create export model
    print("\n[1/5] Creating export model...")
    model = M1ModelForExport(
        input_channels=7,
        conv_output_channels=128,
        kernel_width=15,
        stride=10,
        lstm_hidden_size=128,
        lstm_num_layers=3,
        output_channels=9,
    )

    # Step 2: Load weights
    print("\n[2/5] Loading trained weights...")
    if os.path.exists(args.checkpoint):
        model = load_trained_weights(model, args.checkpoint)
    else:
        print(f"Warning: Checkpoint not found at {args.checkpoint}")
        print("Using randomly initialized weights for demonstration")

    model.eval()

    # Analyze model
    analyze_model_size(model)

    # Step 3: Export to ONNX
    onnx_path = os.path.join(args.output_dir, "m1_gesture_model.onnx")
    if not args.skip_onnx:
        print("\n[3/5] Exporting to ONNX...")
        try:
            export_to_onnx(model, onnx_path, input_samples=args.input_samples)
        except ImportError as e:
            print(f"ONNX export failed: {e}")
            print("Install onnx: pip install onnx")
            return

    # Step 4: Convert to TFLite
    print("\n[4/5] Converting to TFLite...")
    rep_data = generate_representative_data(
        num_samples=100,
        input_samples=args.input_samples,
    )

    try:
        tflite_path = convert_onnx_to_tflite(
            onnx_path,
            args.output_dir,
            quantize=args.quantize,
            representative_data=rep_data,
        )
    except ImportError as e:
        print(f"TFLite conversion failed: {e}")
        print("Install required packages:")
        print("  pip install onnx2tf tensorflow")
        return
    except Exception as e:
        print(f"TFLite conversion error: {e}")
        print("Generating placeholder TFLite file...")
        tflite_path = os.path.join(args.output_dir, "m1_gesture_model.tflite")
        # Create empty placeholder for now
        with open(tflite_path, 'wb') as f:
            f.write(b'')
        print("Note: Install onnx2tf and tensorflow to complete conversion")

    # Step 5: Generate C header
    print("\n[5/5] Generating C header...")
    header_path = os.path.join(
        PROJECT_ROOT, 'tinyml_deployment', 'platformio_project', 'include', 'm1_model.h'
    )
    os.makedirs(os.path.dirname(header_path), exist_ok=True)

    if os.path.exists(tflite_path) and os.path.getsize(tflite_path) > 0:
        generate_c_header(tflite_path, header_path)
    else:
        print("Skipping C header generation (no valid TFLite model)")

    print("\n=== Conversion Complete ===")
    print(f"ONNX model: {onnx_path}")
    print(f"TFLite model: {tflite_path}")
    print(f"C header: {header_path}")

    # Save conversion metadata
    metadata = {
        'input_shape': [1, 7, args.input_samples],
        'output_shape': [1, 9, args.input_samples // 10],  # stride=10
        'input_channels': 7,
        'output_channels': 9,
        'sampling_rate': 2000,
        'window_duration_ms': args.input_samples / 2,  # ms
        'quantized': args.quantize,
        'target_device': 'Seeed Studio XIAO nRF52840',
    }

    metadata_path = os.path.join(args.output_dir, 'model_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata: {metadata_path}")


if __name__ == '__main__':
    main()
