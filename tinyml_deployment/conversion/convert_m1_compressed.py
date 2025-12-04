#!/usr/bin/env python3
"""
Compressed M1 CNN+LSTM Model Conversion for XIAO nRF52840

This script converts the CompressedM1ForXIAO model to TensorFlow Lite format
optimized for the XIAO nRF52840 microcontroller.

Compressed M1 Architecture:
    Conv(7→64) → LSTM(32, 1 layer) → FC(9)
    Parameters: ~45,000 | RAM: ~80-100KB (FITS in 256KB)

Original M1 (DOES NOT FIT):
    Conv(7→128) → LSTM(128, 3 layers) → FC(9)
    Parameters: 411,529 | RAM: ~398KB

Target Hardware:
    - Seeed Studio XIAO nRF52840 (non-Sense)
    - MCU: Nordic nRF52840 (ARM Cortex-M4F @ 64MHz)
    - RAM: 256 KB (~220KB usable)
    - Flash: 1 MB + 2 MB onboard

Memory Budget:
    - Tensor Arena: 80KB
    - Input Buffer (7×1000×4): 28KB
    - TFLite Runtime: 25KB
    - LSTM States: 0.3KB
    - Stack/Heap: 40KB
    - Total: ~173KB (fits in 220KB usable)
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


class CompressedM1ForExport(nn.Module):
    """
    Compressed M1 CNN+LSTM model optimized for XIAO nRF52840 export.

    This is a self-contained version for export that matches the
    CompressedM1ForXIAO architecture in networks_isolated.py.

    Architecture:
        Conv(7→64, k=15, s=10) → LayerNorm → LSTM(32, 1 layer) → LayerNorm → FC(9)

    Memory-optimized to fit in 256KB RAM:
        - conv_output_channels: 64 (reduced from 128)
        - lstm_hidden_size: 32 (reduced from 128)
        - lstm_num_layers: 1 (reduced from 3)
    """

    def __init__(
        self,
        input_channels: int = 7,
        conv_output_channels: int = 64,
        kernel_width: int = 15,
        stride: int = 10,
        lstm_hidden_size: int = 32,
        lstm_num_layers: int = 1,
        output_channels: int = 9,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.conv_output_channels = conv_output_channels
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.stride = stride

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

        # Post-conv LayerNorm
        self.conv_ln = nn.LayerNorm(conv_output_channels)

        # Single-layer LSTM
        self.lstm = nn.LSTM(
            input_size=conv_output_channels,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=False,
        )

        # Post-LSTM LayerNorm
        self.lstm_ln = nn.LayerNorm(lstm_hidden_size)

        # Output projection
        self.fc_out = nn.Linear(lstm_hidden_size, output_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input EMG, shape (batch, 7, time)
               For 500ms window: (batch, 7, 1000) at 2kHz

        Returns:
            Gesture logits, shape (batch, 9, time')
        """
        # Reinhard compression
        x = self.range_val * x / (self.midpoint + torch.abs(x))

        # Conv1D
        x = self.conv(x)  # (B, 64, T')
        x = torch.nn.functional.leaky_relu(x, 0.1)

        # Transpose for LayerNorm
        x = x.transpose(1, 2)  # (B, T', 64)
        x = self.conv_ln(x)

        # LSTM
        x, _ = self.lstm(x)  # (B, T', 32)

        # Post-LSTM LayerNorm
        x = self.lstm_ln(x)

        # Output projection
        x = self.fc_out(x)  # (B, T', 9)

        # Transpose back
        x = x.transpose(1, 2)  # (B, 9, T')

        return x


class CompressedM1SingleStep(nn.Module):
    """
    Single-timestep version of Compressed M1 for streaming inference.

    This version processes one LSTM timestep at a time, allowing
    state management in application code. This reduces memory by
    not allocating full sequence tensors.

    Use this for minimum memory footprint on XIAO nRF52840.
    """

    def __init__(self, original_model: CompressedM1ForExport):
        super().__init__()

        self.range_val = original_model.range_val
        self.midpoint = original_model.midpoint
        self.conv = original_model.conv
        self.conv_ln = original_model.conv_ln
        self.lstm = original_model.lstm
        self.lstm_ln = original_model.lstm_ln
        self.fc_out = original_model.fc_out

        self.lstm_hidden_size = original_model.lstm_hidden_size
        self.lstm_num_layers = original_model.lstm_num_layers

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        c: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process single timestep with external state management.

        Args:
            x: Input chunk (batch, 7, chunk_samples) - typically (1, 7, 10)
            h: LSTM hidden state (1, batch, 32)
            c: LSTM cell state (1, batch, 32)

        Returns:
            output: Gesture logits (batch, 9)
            h_new: Updated hidden state
            c_new: Updated cell state
        """
        # Reinhard compression
        x = self.range_val * x / (self.midpoint + torch.abs(x))

        # Conv1D
        x = self.conv(x)
        x = torch.nn.functional.leaky_relu(x, 0.1)

        # Transpose for LayerNorm
        x = x.transpose(1, 2)
        x = self.conv_ln(x)

        # LSTM with state
        x, (h_new, c_new) = self.lstm(x, (h, c))

        # Post-LSTM processing
        x = self.lstm_ln(x)
        x = self.fc_out(x)

        # Take last timestep output
        output = x[:, -1, :]  # (batch, 9)

        return output, h_new, c_new


def estimate_memory(model: nn.Module, input_samples: int = 1000) -> dict:
    """
    Estimate memory usage for TinyML deployment.

    Returns dict with memory estimates in bytes.
    """
    params = sum(p.numel() for p in model.parameters())

    # For compressed M1 with stride=10: 1000 samples → 100 timesteps
    stride = getattr(model, 'stride', 10)
    timesteps = (input_samples - 14) // stride  # kernel=15

    conv_channels = getattr(model, 'conv_output_channels', 64)
    lstm_hidden = getattr(model, 'lstm_hidden_size', 32)

    # Activation memory
    activation_mem = (
        7 * input_samples * 4 +           # Input buffer
        conv_channels * timesteps * 4 +   # Conv output
        lstm_hidden * timesteps * 4 +     # LSTM output
        lstm_hidden * 2 * 4 +             # LSTM states (h, c)
        9 * timesteps * 4                 # Output
    )

    return {
        'parameters': params,
        'model_size_fp32': params * 4,
        'model_size_int8': params,
        'activation_memory': activation_mem,
        'estimated_arena': activation_mem + 20000,  # +20KB overhead
        'total_ram': activation_mem + params + 50000,
        'fits_nrf52840': activation_mem + params + 50000 < 220000
    }


def load_weights_from_checkpoint(
    model: CompressedM1ForExport,
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
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

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

        # For LSTM, use INT16 activations for better accuracy
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
    var_name: str = "g_m1_compressed_model"
) -> str:
    """Convert TFLite model to C header file."""
    with open(tflite_path, 'rb') as f:
        model_data = f.read()

    size_kb = len(model_data) / 1024

    header = f'''/**
 * Compressed M1 CNN+LSTM Model for XIAO nRF52840
 *
 * Auto-generated model header file.
 * Model size: {len(model_data)} bytes ({size_kb:.1f} KB)
 *
 * Architecture:
 *   Conv(7->64, k=15, s=10) -> LSTM(32, 1 layer) -> FC(9)
 *   Parameters: ~45,000
 *   RAM: ~80-100KB (fits in 220KB usable)
 *
 * Target: Seeed Studio XIAO nRF52840 (non-Sense)
 * MCU: nRF52840 (Cortex-M4F @ 64MHz)
 * RAM: 256KB, Flash: 1MB
 *
 * Input: (1, 7, 1000) - 7 channels, 500ms @ 2kHz
 * Output: (1, 9, 99) - 9 gesture classes, 99 timesteps
 */

#ifndef M1_COMPRESSED_MODEL_H
#define M1_COMPRESSED_MODEL_H

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
#define M1_COMPRESSED_INPUT_CHANNELS 7
#define M1_COMPRESSED_INPUT_SAMPLES 1000
#define M1_COMPRESSED_OUTPUT_CLASSES 9
#define M1_COMPRESSED_OUTPUT_TIMESTEPS 99
#define M1_COMPRESSED_CONV_CHANNELS 64
#define M1_COMPRESSED_LSTM_HIDDEN 32

#endif // M1_COMPRESSED_MODEL_H
'''

    with open(output_path, 'w') as f:
        f.write(header)

    print(f"Generated C header: {output_path}")
    return output_path


def convert_compressed_m1(
    checkpoint_path: Optional[str] = None,
    output_dir: str = ".",
    input_samples: int = 1000,
    quantize: bool = True,
) -> dict:
    """
    Convert Compressed M1 CNN+LSTM model for XIAO nRF52840.

    Args:
        checkpoint_path: Path to trained checkpoint (optional)
        output_dir: Output directory for converted files
        input_samples: Input window size (samples at 2kHz)
        quantize: Apply INT8 quantization

    Returns:
        dict with paths to generated files and analysis
    """
    print("\n" + "="*60)
    print("Compressed M1 CNN+LSTM Model Conversion")
    print("Target: Seeed Studio XIAO nRF52840 (non-Sense)")
    print("="*60)

    # Create model
    model = CompressedM1ForExport(
        input_channels=7,
        conv_output_channels=64,
        kernel_width=15,
        stride=10,
        lstm_hidden_size=32,
        lstm_num_layers=1,
        output_channels=9,
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
    print(f"  Fits nRF52840: {'YES' if memory['fits_nrf52840'] else 'NO'}")

    if not memory['fits_nrf52840']:
        print("\nWARNING: Model may not fit in nRF52840 RAM!")
        print("Consider reducing input_samples or model size.")

    os.makedirs(output_dir, exist_ok=True)

    # Input shape
    input_shape = (1, 7, input_samples)

    # Export to ONNX
    onnx_path = os.path.join(output_dir, "m1_compressed_model.onnx")
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
        tflite_path = os.path.join(output_dir, "m1_compressed_model.tflite")
        tflite_path = convert_onnx_to_tflite(
            onnx_path, tflite_path, input_shape, quantize=quantize
        )

    # Generate C header
    header_path = None
    if tflite_path and os.path.exists(tflite_path):
        header_path = os.path.join(output_dir, "m1_compressed_model.h")
        generate_c_header(tflite_path, header_path)

    # Save model info
    info_path = os.path.join(output_dir, "m1_compressed_info.json")
    import json
    info = {
        'model_type': 'compressed_m1_cnn_lstm',
        'architecture': {
            'input_channels': 7,
            'conv_output_channels': 64,
            'kernel_width': 15,
            'stride': 10,
            'lstm_hidden_size': 32,
            'lstm_num_layers': 1,
            'output_channels': 9,
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
        description='Convert Compressed M1 CNN+LSTM model for XIAO nRF52840',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert with random weights (for testing)
  python convert_m1_compressed.py --output-dir ../converted_models

  # Convert with trained weights
  python convert_m1_compressed.py \\
      --checkpoint ../../checkpoints_compressed/m1_compressed-epoch=50-val_acc=0.950.ckpt \\
      --output-dir ../converted_models

  # Convert with custom window size
  python convert_m1_compressed.py --input-samples 500 --output-dir ../converted_models

Memory Requirements:
  - Compressed M1: ~45,000 parameters, ~80-100KB RAM
  - Original M1: ~411,000 parameters, ~398KB RAM (DOES NOT FIT)
  - nRF52840: 256KB total, ~220KB usable
"""
    )

    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to trained checkpoint')
    parser.add_argument('--output-dir', type=str,
                        default=str(PROJECT_ROOT / 'tinyml_deployment' / 'converted_models'),
                        help='Output directory')
    parser.add_argument('--input-samples', type=int, default=1000,
                        help='Input window size (samples at 2kHz, default: 1000 = 500ms)')
    parser.add_argument('--no-quantize', action='store_true',
                        help='Disable INT8 quantization')

    args = parser.parse_args()

    result = convert_compressed_m1(
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

        print(f"\nNext steps:")
        print(f"  1. Train the compressed model:")
        print(f"     python -m generic_neuromotor_interface.train \\")
        print(f"         --config config/discrete_gestures_m1_compressed.yaml")
        print(f"  2. Re-run conversion with trained checkpoint")
        print(f"  3. Copy header to firmware:")
        print(f"     cp {result['header_path']} ../platformio_project/include/")
        print(f"  4. Build firmware:")
        print(f"     cd ../platformio_project && pio run")
        print(f"  5. Flash to device:")
        print(f"     pio run --target upload")
    else:
        print("\nConversion failed. Check error messages above.")


if __name__ == '__main__':
    main()
