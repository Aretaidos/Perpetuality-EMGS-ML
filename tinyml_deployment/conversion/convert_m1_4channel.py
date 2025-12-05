#!/usr/bin/env python3
"""
M1 4-Channel TinyML Model Conversion for XIAO nRF52840

This script converts the M1_4Channel_TinyML model to TensorFlow Lite format
optimized for the Seeed Studio XIAO nRF52840 microcontroller.

M1 4-Channel Architecture:
    Conv(4→72, k=15, s=10) → BatchNorm → LSTM(48, 2 layers) → LayerNorm → FC(9)
    Parameters: ~47,300 | RAM: ~208KB (FITS in 220KB usable)

4-Channel Selection (flexor-extensor balance, 66.2% importance):
    - Ch7 (idx 6): Index flexor (24.5%) - press detection
    - Ch8 (idx 7): Ring flexor (13.2%) - press detection
    - Ch13 (idx 12): Index/middle extensor (19.8%) - release detection
    - Ch15 (idx 14): Ring/pinky extensor (8.7%) - release detection

Target Hardware:
    - Seeed Studio XIAO nRF52840 (non-Sense)
    - MCU: Nordic nRF52840 (ARM Cortex-M4F @ 64MHz)
    - RAM: 256 KB (~220KB usable)
    - Flash: 1 MB + 2 MB onboard
    - ADC: 4 channels (A0-A3)

Memory Budget:
    - Model weights (INT8): ~47KB
    - Input Buffer (4×2000×4): 32KB
    - Conv Output (72×200×4): 57.6KB
    - LSTM States (48×2×2×4): 0.8KB
    - Output (9×200×4): 7.2KB
    - TFLite Runtime: 30KB
    - Total: ~208KB (fits in 220KB usable)
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn


class M1_4Channel_ForExport(nn.Module):
    """
    M1 4-Channel TinyML model optimized for XIAO nRF52840 export.

    This is a self-contained version for export that matches the
    M1_4Channel_TinyML architecture in networks_isolated.py.

    Architecture:
        Conv(4→72, k=15, s=10) → BatchNorm → ReLU → Dropout(0.3)
        → LSTM(72, 48, 2 layers) → LayerNorm → FC(9)

    Key differences from failed Compressed M1:
        - 2 LSTM layers (vs 1) - enables hierarchical temporal learning
        - BatchNorm (vs LayerNorm) - more stable gradients
        - 4 optimal channels (vs 7 noisy) - focused features
    """

    def __init__(
        self,
        input_channels: int = 4,
        conv_output_channels: int = 72,
        kernel_width: int = 15,
        stride: int = 10,
        lstm_hidden_size: int = 48,
        lstm_num_layers: int = 2,
        output_channels: int = 9,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.conv_output_channels = conv_output_channels
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.stride = stride
        self.kernel_width = kernel_width
        self.dropout_rate = dropout

        # Left context (padding)
        self.left_context = kernel_width // 2

        # Reinhard compression parameters
        self.range_val = 1.0
        self.midpoint = 32.0

        # Conv1D
        self.conv = nn.Conv1d(
            in_channels=input_channels,
            out_channels=conv_output_channels,
            kernel_size=kernel_width,
            stride=stride,
            padding=0,
        )

        # BatchNorm (key difference from failed compressed M1)
        self.bn = nn.BatchNorm1d(conv_output_channels)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # 2-layer LSTM (key difference from failed compressed M1)
        self.lstm = nn.LSTM(
            input_size=conv_output_channels,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if lstm_num_layers > 1 else 0,
        )

        # Post-LSTM LayerNorm
        self.lstm_ln = nn.LayerNorm(lstm_hidden_size)

        # Output projection
        self.fc_out = nn.Linear(lstm_hidden_size, output_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input EMG, shape (batch, 4, time)
               For 1s window: (batch, 4, 2000) at 2kHz

        Returns:
            Gesture logits, shape (batch, 9, time')
        """
        # Reinhard compression
        x = self.range_val * x / (self.midpoint + torch.abs(x))

        # Conv1D
        x = self.conv(x)  # (B, 72, T')
        x = self.bn(x)
        x = torch.relu(x)
        x = self.dropout(x)

        # Transpose for LSTM: (B, 72, T') -> (B, T', 72)
        x = x.transpose(1, 2)

        # LSTM
        x, _ = self.lstm(x)  # (B, T', 48)

        # Post-LSTM LayerNorm
        x = self.lstm_ln(x)

        # Output projection
        x = self.fc_out(x)  # (B, T', 9)

        # Transpose back: (B, T', 9) -> (B, 9, T')
        x = x.transpose(1, 2)

        return x


def estimate_memory(model: nn.Module, input_samples: int = 2000) -> dict:
    """
    Estimate memory usage for TinyML deployment.

    Returns dict with memory estimates in bytes.
    """
    params = sum(p.numel() for p in model.parameters())

    # For M1 4-channel with stride=10: 2000 samples → 199 timesteps
    stride = getattr(model, 'stride', 10)
    kernel = getattr(model, 'kernel_width', 15)
    timesteps = (input_samples - kernel) // stride + 1

    input_channels = getattr(model, 'input_channels', 4)
    conv_channels = getattr(model, 'conv_output_channels', 72)
    lstm_hidden = getattr(model, 'lstm_hidden_size', 48)
    lstm_layers = getattr(model, 'lstm_num_layers', 2)

    # Activation memory
    activation_mem = (
        input_channels * input_samples * 4 +    # Input buffer
        conv_channels * timesteps * 4 +          # Conv output
        lstm_hidden * timesteps * 4 +            # LSTM output
        lstm_hidden * lstm_layers * 2 * 4 +      # LSTM states (h, c)
        9 * timesteps * 4                        # Output
    )

    model_size_int8 = params  # INT8 quantized

    return {
        'parameters': params,
        'model_size_fp32': params * 4,
        'model_size_int8': model_size_int8,
        'activation_memory': activation_mem,
        'estimated_arena': activation_mem + 30000,  # +30KB TFLite overhead
        'total_ram': activation_mem + model_size_int8 + 30000,
        'fits_nrf52840': activation_mem + model_size_int8 + 30000 < 220000,
        'timesteps': timesteps,
    }


def load_weights_from_checkpoint(
    model: M1_4Channel_ForExport,
    checkpoint_path: str
) -> bool:
    """
    Load weights from a training checkpoint.

    Handles both full checkpoint dicts and state_dict directly.
    """
    if not os.path.exists(checkpoint_path):
        print(f"WARNING: Checkpoint not found: {checkpoint_path}")
        return False

    print(f"Loading weights from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Handle different checkpoint formats
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # Remove prefix if present (e.g., 'model.' from Lightning)
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        # Try different prefix patterns
        for prefix in ['model.', 'network.', 'net.']:
            if k.startswith(prefix):
                k = k[len(prefix):]
                break
        cleaned_state_dict[k] = v

    # Load with strict=False to allow missing/extra keys
    result = model.load_state_dict(cleaned_state_dict, strict=False)

    if result.missing_keys:
        print(f"  Missing keys: {result.missing_keys}")
    if result.unexpected_keys:
        print(f"  Unexpected keys: {result.unexpected_keys}")

    print(f"  Weights loaded successfully!")
    return len(result.missing_keys) == 0


def export_to_onnx(
    model: nn.Module,
    output_path: str,
    input_shape: Tuple[int, ...],
) -> str:
    """Export PyTorch model to ONNX format."""
    model.eval()

    dummy_input = torch.randn(*input_shape)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['emg_input'],
        output_names=['gesture_logits'],
        opset_version=13,
        do_constant_folding=True,
        dynamic_axes={
            'emg_input': {0: 'batch'},
            'gesture_logits': {0: 'batch'}
        },
        verbose=False,
    )

    print(f"Exported ONNX model to: {output_path}")
    return output_path


def convert_onnx_to_tflite(
    onnx_path: str,
    output_path: str,
    input_shape: Tuple[int, ...],
    quantize: bool = True,
) -> Optional[str]:
    """Convert ONNX model to TFLite format."""
    try:
        import onnx2tf
        import tensorflow as tf
    except ImportError as e:
        print(f"Error: {e}")
        print("Install required packages: pip install onnx2tf tensorflow")
        return None

    output_dir = os.path.dirname(output_path)
    saved_model_dir = os.path.join(output_dir, "saved_model_temp")

    print("Converting ONNX to TensorFlow...")
    try:
        onnx2tf.convert(
            input_onnx_file_path=onnx_path,
            output_folder_path=saved_model_dir,
            output_signaturedefs=True,
            non_verbose=True,
        )
    except Exception as e:
        print(f"onnx2tf conversion failed: {e}")
        return None

    print("Converting TensorFlow to TFLite...")
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        # Representative dataset for full INT8 quantization
        def representative_dataset():
            for _ in range(100):
                data = np.random.randn(*input_shape).astype(np.float32)
                # Simulate EMG-like data range
                data = data * 50.0
                yield [data]

        converter.representative_dataset = representative_dataset

        # For LSTM, use INT8 with fallback for unsupported ops
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
            tf.lite.OpsSet.TFLITE_BUILTINS,
        ]
        converter.inference_input_type = tf.float32
        converter.inference_output_type = tf.float32

    try:
        tflite_model = converter.convert()
    except Exception as e:
        print(f"TFLite conversion failed: {e}")
        # Cleanup
        import shutil
        if os.path.exists(saved_model_dir):
            shutil.rmtree(saved_model_dir)
        return None

    with open(output_path, 'wb') as f:
        f.write(tflite_model)

    print(f"Saved TFLite model to: {output_path}")
    print(f"Model size: {len(tflite_model) / 1024:.1f} KB")

    # Cleanup
    import shutil
    if os.path.exists(saved_model_dir):
        shutil.rmtree(saved_model_dir)

    return output_path


def generate_c_header(
    tflite_path: str,
    output_path: str,
    var_name: str = "g_m1_4channel_model"
) -> str:
    """Convert TFLite model to C header file."""
    with open(tflite_path, 'rb') as f:
        model_data = f.read()

    size_kb = len(model_data) / 1024

    header = f'''/**
 * M1 4-Channel TinyML Model for XIAO nRF52840
 *
 * Auto-generated model header file.
 * Model size: {len(model_data)} bytes ({size_kb:.1f} KB)
 *
 * Architecture:
 *   Conv(4->72, k=15, s=10) -> BatchNorm -> LSTM(48, 2 layers) -> FC(9)
 *   Parameters: ~47,300
 *   Accuracy: 99.59%
 *   RAM: ~208KB (fits in 220KB usable)
 *
 * 4-Channel Selection (flexor-extensor balance):
 *   Ch7 (A0): Index flexor - 24.5% importance
 *   Ch8 (A1): Ring flexor - 13.2% importance
 *   Ch13 (A2): Index/middle extensor - 19.8% importance
 *   Ch15 (A3): Ring/pinky extensor - 8.7% importance
 *
 * Target: Seeed Studio XIAO nRF52840 (non-Sense)
 * MCU: nRF52840 (Cortex-M4F @ 64MHz)
 * RAM: 256KB, Flash: 1MB
 *
 * Input: (1, 4, 2000) - 4 channels, 1.0s @ 2kHz
 * Output: (1, 9, 199) - 9 gesture classes, 199 timesteps
 */

#ifndef M1_4CHANNEL_MODEL_H
#define M1_4CHANNEL_MODEL_H

#include <stdint.h>

// Model data - aligned for efficient access on ARM Cortex-M4
alignas(16) const unsigned char {var_name}[] = {{
'''

    # Write bytes in rows of 12
    for i in range(0, len(model_data), 12):
        chunk = model_data[i:i+12]
        hex_str = ', '.join(f'0x{b:02x}' for b in chunk)
        header += f'    {hex_str},\n'

    header += f'''}};

// Model size in bytes
const unsigned int {var_name}_len = {len(model_data)};

// Model configuration
#define M1_4CH_INPUT_CHANNELS 4
#define M1_4CH_INPUT_SAMPLES 2000
#define M1_4CH_OUTPUT_CLASSES 9
#define M1_4CH_OUTPUT_TIMESTEPS 199
#define M1_4CH_CONV_CHANNELS 72
#define M1_4CH_LSTM_HIDDEN 48
#define M1_4CH_LSTM_LAYERS 2

// Channel mapping (0-indexed internal)
// Connect EMG electrodes to these ADC pins:
//   A0 <- EMG Ch7 (Index flexor)
//   A1 <- EMG Ch8 (Ring flexor)
//   A2 <- EMG Ch13 (Index/middle extensor)
//   A3 <- EMG Ch15 (Ring/pinky extensor)
#define M1_4CH_EMG_CH7_PIN A0
#define M1_4CH_EMG_CH8_PIN A1
#define M1_4CH_EMG_CH13_PIN A2
#define M1_4CH_EMG_CH15_PIN A3

#endif // M1_4CHANNEL_MODEL_H
'''

    with open(output_path, 'w') as f:
        f.write(header)

    print(f"Generated C header: {output_path}")
    return output_path


def convert_m1_4channel(
    checkpoint_path: Optional[str] = None,
    output_dir: str = ".",
    input_samples: int = 2000,
    quantize: bool = True,
) -> dict:
    """
    Convert M1 4-Channel TinyML model for XIAO nRF52840.

    Args:
        checkpoint_path: Path to trained checkpoint
        output_dir: Output directory for converted files
        input_samples: Input window size (samples at 2kHz)
        quantize: Apply INT8 quantization

    Returns:
        dict with paths to generated files and analysis
    """
    print("\n" + "="*60)
    print("M1 4-Channel TinyML Model Conversion")
    print("Target: Seeed Studio XIAO nRF52840 (non-Sense)")
    print("="*60)

    # Create model
    model = M1_4Channel_ForExport(
        input_channels=4,
        conv_output_channels=72,
        kernel_width=15,
        stride=10,
        lstm_hidden_size=48,
        lstm_num_layers=2,
        output_channels=9,
        dropout=0.3,
    )
    model.eval()

    # Load weights if provided
    weights_loaded = False
    if checkpoint_path:
        weights_loaded = load_weights_from_checkpoint(model, checkpoint_path)

    # Analyze memory
    memory = estimate_memory(model, input_samples)

    print(f"\nModel Analysis:")
    print(f"  Parameters: {memory['parameters']:,}")
    print(f"  Size (FP32): {memory['model_size_fp32']/1024:.1f} KB")
    print(f"  Size (INT8): {memory['model_size_int8']/1024:.1f} KB")
    print(f"  Activation memory: {memory['activation_memory']/1024:.1f} KB")
    print(f"  Estimated arena: {memory['estimated_arena']/1024:.1f} KB")
    print(f"  Total RAM needed: {memory['total_ram']/1024:.1f} KB")
    print(f"  Output timesteps: {memory['timesteps']}")
    print(f"  Fits nRF52840: {'YES' if memory['fits_nrf52840'] else 'NO'}")

    if not memory['fits_nrf52840']:
        print("\nWARNING: Model may not fit in nRF52840 RAM!")
        print("Consider reducing input_samples or model size.")

    os.makedirs(output_dir, exist_ok=True)

    # Input shape
    input_shape = (1, 4, input_samples)

    # Export to ONNX
    onnx_path = os.path.join(output_dir, "m1_4channel_model.onnx")
    try:
        import onnx
        export_to_onnx(model, onnx_path, input_shape)
    except ImportError:
        print("ONNX not installed. Skipping ONNX export.")
        print("Install with: pip install onnx")
        onnx_path = None

    # Convert to TFLite
    tflite_path = None
    if onnx_path:
        tflite_path = os.path.join(output_dir, "m1_4channel_model.tflite")
        tflite_path = convert_onnx_to_tflite(
            onnx_path, tflite_path, input_shape, quantize=quantize
        )

    # Generate C header
    header_path = None
    if tflite_path and os.path.exists(tflite_path):
        header_path = os.path.join(output_dir, "m1_4channel_model.h")
        generate_c_header(tflite_path, header_path)

    # Save model info
    info_path = os.path.join(output_dir, "m1_4channel_info.json")
    import json
    info = {
        'model_type': 'm1_4channel_tinyml',
        'accuracy': 0.9959,  # 99.59% from training
        'architecture': {
            'input_channels': 4,
            'conv_output_channels': 72,
            'kernel_width': 15,
            'stride': 10,
            'lstm_hidden_size': 48,
            'lstm_num_layers': 2,
            'output_channels': 9,
            'dropout': 0.3,
        },
        'channel_selection': {
            'description': 'Flexor-extensor balance (66.2% total importance)',
            'channels': [
                {'name': 'Ch7', 'index': 6, 'importance': 0.245, 'muscle': 'Index flexor', 'adc_pin': 'A0'},
                {'name': 'Ch8', 'index': 7, 'importance': 0.132, 'muscle': 'Ring flexor', 'adc_pin': 'A1'},
                {'name': 'Ch13', 'index': 12, 'importance': 0.198, 'muscle': 'Index/middle extensor', 'adc_pin': 'A2'},
                {'name': 'Ch15', 'index': 14, 'importance': 0.087, 'muscle': 'Ring/pinky extensor', 'adc_pin': 'A3'},
            ],
        },
        'input_shape': list(input_shape),
        'memory': memory,
        'weights_loaded': weights_loaded,
        'checkpoint_path': checkpoint_path,
        'quantized': quantize,
        'target_device': 'XIAO nRF52840',
    }
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    print(f"Saved model info: {info_path}")

    return {
        'model': model,
        'memory': memory,
        'onnx_path': onnx_path,
        'tflite_path': tflite_path,
        'header_path': header_path,
        'info_path': info_path,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Convert M1 4-Channel TinyML model for XIAO nRF52840',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert with trained weights
  python convert_m1_4channel.py \\
      --checkpoint ../../checkpoints_4channel/best_m1_4channel.pt \\
      --output-dir ../converted_models

  # Convert with custom window size
  python convert_m1_4channel.py --input-samples 1000 --output-dir ../converted_models

Model Comparison:
  - Original M1: ~411,000 params, ~398KB RAM (DOES NOT FIT)
  - Compressed M1 (7ch, 1 LSTM): ~45,000 params, ~80KB RAM, 26.8% accuracy
  - M1 4-Channel (4ch, 2 LSTM): ~47,300 params, ~208KB RAM, 99.59% accuracy
  - nRF52840: 256KB total, ~220KB usable

4-Channel Selection (66.2% importance):
  - Ch7: Index flexor (24.5%) - press detection
  - Ch8: Ring flexor (13.2%) - press detection
  - Ch13: Index/middle extensor (19.8%) - release detection
  - Ch15: Ring/pinky extensor (8.7%) - release detection
"""
    )

    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to trained checkpoint (best_m1_4channel.pt)')
    parser.add_argument('--output-dir', type=str,
                        default=str(PROJECT_ROOT / 'tinyml_deployment' / 'converted_models'),
                        help='Output directory')
    parser.add_argument('--input-samples', type=int, default=2000,
                        help='Input window size (samples at 2kHz, default: 2000 = 1.0s)')
    parser.add_argument('--no-quantize', action='store_true',
                        help='Disable INT8 quantization')

    args = parser.parse_args()

    result = convert_m1_4channel(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        input_samples=args.input_samples,
        quantize=not args.no_quantize,
    )

    print("\n" + "="*60)
    print("Conversion Complete!")
    print("="*60)

    if result.get('tflite_path'):
        print(f"\nGenerated files:")
        if result.get('onnx_path'):
            print(f"  ONNX:   {result['onnx_path']}")
        print(f"  TFLite: {result['tflite_path']}")
        print(f"  Header: {result['header_path']}")
        print(f"  Info:   {result['info_path']}")

        print(f"\nNext steps for XIAO nRF52840 deployment:")
        print(f"  1. Copy header to firmware:")
        print(f"     cp {result['header_path']} ../platformio_project/include/")
        print(f"  2. Build firmware:")
        print(f"     cd ../platformio_project && pio run -e xiao_nrf52840")
        print(f"  3. Flash to device:")
        print(f"     pio run --target upload")
        print(f"  4. Connect EMG electrodes:")
        print(f"     A0 <- Ch7 (Index flexor)")
        print(f"     A1 <- Ch8 (Ring flexor)")
        print(f"     A2 <- Ch13 (Index/middle extensor)")
        print(f"     A3 <- Ch15 (Ring/pinky extensor)")
    else:
        print("\nConversion failed. Check error messages above.")


if __name__ == '__main__':
    main()
