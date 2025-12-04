#!/usr/bin/env python3
"""
Model Conversion for Seeed Studio XIAO nRF52840 (non-Sense)

This script converts trained PyTorch models to TensorFlow Lite format
optimized for the XIAO nRF52840 microcontroller.

CRITICAL MEMORY CONSTRAINTS:
- nRF52840 has only 256KB RAM
- LSTM models require 398KB+ RAM (unrolled) - DOES NOT FIT
- CNN models typically need 80-150KB - FITS WELL

SOLUTIONS:
1. Use M2 (CNN-only) model instead of M1 (CNN+LSTM) - RECOMMENDED
2. Use stateless LSTM: process one timestep at a time
3. Reduce model size through aggressive quantization

Target Hardware:
- MCU: Nordic nRF52840 (ARM Cortex-M4F @ 64MHz)
- RAM: 256 KB
- Flash: 1 MB + 2 MB onboard
- FPU: Single-precision floating point
"""

import os
import sys
import json
import struct
import argparse
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn


def estimate_tflite_arena_size(model: nn.Module, input_shape: Tuple[int, ...]) -> int:
    """
    Estimate tensor arena size needed for TFLite Micro inference.

    This is a rough estimate based on model architecture.
    Actual size should be measured using RecordingMicroAllocator.
    """
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())

    # Estimate intermediate tensor sizes
    # For CNN: roughly 2x the largest feature map
    # For LSTM: much larger due to unrolled states

    batch, channels, time = input_shape

    # Check for LSTM layers
    has_lstm = any(isinstance(m, nn.LSTM) for m in model.modules())

    if has_lstm:
        # LSTM requires massive memory for unrolled inference
        # 3-layer LSTM with hidden=128: approximately 398KB
        lstm_layers = sum(1 for m in model.modules() if isinstance(m, nn.LSTM))
        lstm_hidden = 128  # typical
        lstm_mem = lstm_layers * lstm_hidden * time * 4 * 2  # h and c states
        return max(200 * 1024, lstm_mem)
    else:
        # CNN-only: much more reasonable
        # Estimate based on largest intermediate tensor
        max_channels = 256  # typical max
        downsampled_time = time // 40  # typical total stride
        intermediate = max_channels * downsampled_time * 4
        return max(80 * 1024, intermediate * 3)  # 3x for safety


def check_model_fits_nrf52840(model: nn.Module, input_shape: Tuple[int, ...]) -> dict:
    """
    Check if model fits in nRF52840 memory constraints.

    Returns dict with analysis results.
    """
    NRF52840_RAM = 256 * 1024  # 256 KB
    NRF52840_FLASH = 1024 * 1024  # 1 MB (excluding SoftDevice)
    USABLE_RAM = 220 * 1024  # After SoftDevice
    USABLE_FLASH = 800 * 1024  # After bootloader, SoftDevice

    total_params = sum(p.numel() for p in model.parameters())
    model_size_fp32 = total_params * 4
    model_size_int8 = total_params

    arena_estimate = estimate_tflite_arena_size(model, input_shape)

    # Input buffer size
    batch, channels, time = input_shape
    input_buffer = channels * time * 4  # float32

    # Check LSTM
    has_lstm = any(isinstance(m, nn.LSTM) for m in model.modules())

    analysis = {
        'total_params': total_params,
        'model_size_fp32': model_size_fp32,
        'model_size_int8': model_size_int8,
        'arena_estimate': arena_estimate,
        'input_buffer': input_buffer,
        'has_lstm': has_lstm,
        'fits_flash_int8': model_size_int8 < USABLE_FLASH,
        'fits_ram': (arena_estimate + input_buffer) < USABLE_RAM,
        'recommended_action': None,
    }

    if has_lstm:
        analysis['recommended_action'] = 'LSTM_TOO_LARGE'
        analysis['fits_ram'] = False
        analysis['warning'] = (
            f"LSTM models require ~{arena_estimate//1024}KB RAM, "
            f"but nRF52840 only has {USABLE_RAM//1024}KB usable. "
            "Use CNN-only model (M2) instead."
        )
    elif not analysis['fits_ram']:
        analysis['recommended_action'] = 'REDUCE_INPUT_SIZE'
        analysis['warning'] = (
            f"Model needs ~{(arena_estimate + input_buffer)//1024}KB RAM. "
            f"Reduce input window size."
        )
    else:
        analysis['recommended_action'] = 'OK'

    return analysis


class M1StatelessWrapper(nn.Module):
    """
    Wrapper for M1 that processes one LSTM timestep at a time.

    This allows running on memory-constrained devices by:
    1. Processing Conv layer normally (output: 1 timestep)
    2. Running LSTM for 1 step with external state management
    3. Managing h/c states in application code

    For TFLite: Export only the single-step version, handle loop in C code.
    """

    def __init__(self, original_model):
        super().__init__()
        self.compression = original_model.compression
        self.conv = original_model.conv
        self.conv_ln = original_model.conv_ln
        self.relu = original_model.relu
        self.lstm = original_model.lstm
        self.lstm_ln = original_model.lstm_ln
        self.fc_out = original_model.fc_out

        self.lstm_hidden_size = original_model.lstm_hidden_size
        self.lstm_num_layers = original_model.lstm_num_layers

    def forward(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process single timestep.

        Args:
            x: Input chunk (batch, 7, chunk_samples)
            h: LSTM hidden state (num_layers, batch, hidden)
            c: LSTM cell state (num_layers, batch, hidden)

        Returns:
            output: Gesture logits (batch, 9)
            h_new: Updated hidden state
            c_new: Updated cell state
        """
        # Conv processing
        x = self.compression(x)
        x = self.conv(x)
        x = self.relu(x)
        x = x.transpose(1, 2)
        x = self.conv_ln(x)

        # Single LSTM step
        x, (h_new, c_new) = self.lstm(x, (h, c))

        # Output
        x = self.lstm_ln(x)
        x = self.fc_out(x)

        # Take last timestep output
        output = x[:, -1, :]

        return output, h_new, c_new


class M2ForExport(nn.Module):
    """
    M2 CNN-only model prepared for TFLite export.

    This is the RECOMMENDED model for XIAO nRF52840 because:
    - No LSTM = fits in 256KB RAM
    - Pure CNN = efficient inference
    - Smaller model = faster inference
    """

    def __init__(
        self,
        input_channels: int = 7,
        output_channels: int = 9,
    ):
        super().__init__()

        # Reinhard compression
        self.range_val = 1.0
        self.midpoint = 32.0

        # Conv layers (simplified for memory)
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=21, stride=10, padding=10),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )

        # Multi-scale inception-style block
        self.conv_short = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )
        self.conv_medium = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=15, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )
        self.conv_long = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=25, padding=12),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )

        # Further processing
        self.conv2 = nn.Sequential(
            nn.Conv1d(96, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )

        # Global average pooling + classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, output_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reinhard compression
        x = self.range_val * x / (self.midpoint + torch.abs(x))

        # Conv backbone
        x = self.conv1(x)

        # Multi-scale features
        short = self.conv_short(x)
        medium = self.conv_medium(x)
        long = self.conv_long(x)
        x = torch.cat([short, medium, long], dim=1)

        # Further processing
        x = self.conv2(x)
        x = self.conv3(x)

        # Classification
        x = self.classifier(x)

        return x


def export_to_onnx(
    model: nn.Module,
    output_path: str,
    input_shape: Tuple[int, ...],
    input_names: List[str] = None,
    output_names: List[str] = None,
) -> str:
    """Export PyTorch model to ONNX format."""
    model.eval()

    # Create dummy input
    dummy_inputs = [torch.randn(*input_shape)]
    if input_names is None:
        input_names = ['input']
    if output_names is None:
        output_names = ['output']

    torch.onnx.export(
        model,
        tuple(dummy_inputs),
        output_path,
        input_names=input_names,
        output_names=output_names,
        opset_version=13,
        do_constant_folding=True,
        verbose=False,
    )

    print(f"Exported ONNX model to: {output_path}")
    return output_path


def convert_onnx_to_tflite(
    onnx_path: str,
    output_path: str,
    quantize: bool = True,
    representative_data: Optional[np.ndarray] = None,
) -> str:
    """Convert ONNX model to TFLite format."""
    try:
        import onnx2tf
        import tensorflow as tf
    except ImportError as e:
        print(f"Error: {e}")
        print("Install required packages: pip install onnx2tf tensorflow")
        return None

    # Convert ONNX to TF SavedModel
    output_dir = os.path.dirname(output_path)
    saved_model_dir = os.path.join(output_dir, "saved_model_temp")

    print("Converting ONNX to TensorFlow...")
    onnx2tf.convert(
        input_onnx_file_path=onnx_path,
        output_folder_path=saved_model_dir,
        output_signaturedefs=True,
        keep_ncwh_or_nchw_order=False,
    )

    # Convert to TFLite
    print("Converting TensorFlow to TFLite...")
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        if representative_data is not None:
            def rep_dataset():
                for i in range(min(100, len(representative_data))):
                    yield [representative_data[i:i+1].astype(np.float32)]
            converter.representative_dataset = rep_dataset
            converter.target_spec.supported_types = [tf.int8]

    tflite_model = converter.convert()

    with open(output_path, 'wb') as f:
        f.write(tflite_model)

    print(f"Saved TFLite model to: {output_path}")
    print(f"Model size: {len(tflite_model) / 1024:.1f} KB")

    # Cleanup
    import shutil
    if os.path.exists(saved_model_dir):
        shutil.rmtree(saved_model_dir)

    return output_path


def generate_c_header(tflite_path: str, output_path: str, var_name: str = "g_model"):
    """Convert TFLite model to C header file."""
    with open(tflite_path, 'rb') as f:
        model_data = f.read()

    size_kb = len(model_data) / 1024

    header = f'''/**
 * TensorFlow Lite Model for XIAO nRF52840
 *
 * Auto-generated model header file.
 * Model size: {len(model_data)} bytes ({size_kb:.1f} KB)
 *
 * Target: Seeed Studio XIAO nRF52840 (non-Sense)
 * MCU: nRF52840 (Cortex-M4F @ 64MHz)
 * RAM: 256KB, Flash: 1MB
 */

#ifndef MODEL_DATA_H
#define MODEL_DATA_H

#include <stdint.h>

// Model data - aligned for efficient access
alignas(16) const unsigned char {var_name}[] = {{
'''

    # Write bytes in rows of 12
    for i in range(0, len(model_data), 12):
        chunk = model_data[i:i+12]
        hex_str = ', '.join(f'0x{b:02x}' for b in chunk)
        header += f'    {hex_str},\n'

    header += f'''}};

// Model size
const unsigned int {var_name}_len = {len(model_data)};

#endif // MODEL_DATA_H
'''

    with open(output_path, 'w') as f:
        f.write(header)

    print(f"Generated C header: {output_path}")
    return output_path


def create_m2_model_for_xiao(
    checkpoint_path: Optional[str] = None,
    output_dir: str = ".",
    input_samples: int = 1000,
) -> dict:
    """
    Create and export M2 CNN model optimized for XIAO nRF52840.

    This is the RECOMMENDED approach for XIAO nRF52840.
    """
    print("\n" + "="*60)
    print("Creating M2 CNN Model for XIAO nRF52840")
    print("="*60)

    # Create model
    model = M2ForExport(input_channels=7, output_channels=9)
    model.eval()

    # Load weights if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading weights from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = {k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items()}
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict, strict=False)

    # Analyze memory requirements
    input_shape = (1, 7, input_samples)
    analysis = check_model_fits_nrf52840(model, input_shape)

    print(f"\nModel Analysis:")
    print(f"  Parameters: {analysis['total_params']:,}")
    print(f"  Size (INT8): {analysis['model_size_int8']/1024:.1f} KB")
    print(f"  Arena estimate: {analysis['arena_estimate']/1024:.1f} KB")
    print(f"  Has LSTM: {analysis['has_lstm']}")
    print(f"  Fits in RAM: {analysis['fits_ram']}")
    print(f"  Fits in Flash: {analysis['fits_flash_int8']}")

    if not analysis['fits_ram']:
        print(f"\nWARNING: {analysis.get('warning', 'Model may not fit')}")

    os.makedirs(output_dir, exist_ok=True)

    # Export to ONNX
    onnx_path = os.path.join(output_dir, "m2_gesture_model.onnx")
    try:
        import onnx
        export_to_onnx(model, onnx_path, input_shape)
    except ImportError:
        print("ONNX not installed. Skipping ONNX export.")
        onnx_path = None

    # Convert to TFLite
    tflite_path = None
    if onnx_path:
        tflite_path = os.path.join(output_dir, "m2_gesture_model.tflite")
        rep_data = np.random.randn(100, 7, input_samples).astype(np.float32)
        try:
            convert_onnx_to_tflite(onnx_path, tflite_path, quantize=True, representative_data=rep_data)
        except Exception as e:
            print(f"TFLite conversion failed: {e}")
            tflite_path = None

    # Generate C header
    header_path = None
    if tflite_path and os.path.exists(tflite_path):
        header_path = os.path.join(output_dir, "m2_model.h")
        generate_c_header(tflite_path, header_path, "g_m2_model")

    return {
        'model': model,
        'analysis': analysis,
        'onnx_path': onnx_path,
        'tflite_path': tflite_path,
        'header_path': header_path,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Convert models for XIAO nRF52840 deployment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert M2 CNN model (recommended)
  python convert_model_xiao.py --model m2 --output-dir ../converted_models

  # Try M1 with stateless LSTM (experimental)
  python convert_model_xiao.py --model m1 --stateless --output-dir ../converted_models

  # Check if model fits
  python convert_model_xiao.py --analyze-only --checkpoint ../../models_gpu_multi_20251124_121916/m1_best.pt
"""
    )

    parser.add_argument('--model', choices=['m1', 'm2'], default='m2',
                        help='Model to convert (m2 recommended for XIAO)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to trained checkpoint')
    parser.add_argument('--output-dir', type=str,
                        default=str(PROJECT_ROOT / 'tinyml_deployment' / 'converted_models'),
                        help='Output directory')
    parser.add_argument('--input-samples', type=int, default=1000,
                        help='Input window size (samples at 2kHz)')
    parser.add_argument('--stateless', action='store_true',
                        help='Use stateless LSTM wrapper (for M1)')
    parser.add_argument('--analyze-only', action='store_true',
                        help='Only analyze model, do not convert')
    parser.add_argument('--quantize', action='store_true', default=True,
                        help='Apply INT8 quantization')

    args = parser.parse_args()

    print("\n" + "="*60)
    print("XIAO nRF52840 Model Conversion Tool")
    print("="*60)
    print(f"\nTarget: Seeed Studio XIAO nRF52840 (non-Sense)")
    print(f"RAM: 256KB, Flash: 1MB + 2MB onboard")
    print(f"Model: {args.model.upper()}")
    print(f"Input samples: {args.input_samples} ({args.input_samples/2:.0f}ms @ 2kHz)")

    if args.model == 'm2':
        result = create_m2_model_for_xiao(
            checkpoint_path=args.checkpoint,
            output_dir=args.output_dir,
            input_samples=args.input_samples,
        )
    else:
        print("\n" + "!"*60)
        print("WARNING: M1 (LSTM) model requires ~398KB RAM")
        print("XIAO nRF52840 only has 256KB RAM - WILL NOT FIT!")
        print("Use --model m2 for CNN-only model instead.")
        print("!"*60)

        if args.stateless:
            print("\nAttempting stateless LSTM conversion (experimental)...")
            # This would require additional implementation
            print("Stateless LSTM export not yet implemented.")
            print("Recommendation: Use M2 CNN model instead.")
        return

    print("\n" + "="*60)
    print("Conversion Complete!")
    print("="*60)

    if result.get('tflite_path'):
        print(f"\nFiles generated:")
        print(f"  ONNX:   {result['onnx_path']}")
        print(f"  TFLite: {result['tflite_path']}")
        print(f"  Header: {result['header_path']}")

        print(f"\nNext steps:")
        print(f"  1. Copy {result['header_path']} to platformio_project/include/")
        print(f"  2. Build: cd ../platformio_project && pio run")
        print(f"  3. Flash: pio run --target upload")


if __name__ == '__main__':
    main()
