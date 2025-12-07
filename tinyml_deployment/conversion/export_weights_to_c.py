#!/usr/bin/env python3
"""
Export M1 4-Channel TinyML Model Weights to C Header for Arduino

This script exports the trained PyTorch model weights directly to C arrays
for use in custom inference code on XIAO nRF52840.

Architecture:
    Conv1d(4→72, k=15, s=10) → BatchNorm → LSTM(48, 2 layers) → LayerNorm → FC(9)
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch


def load_checkpoint(checkpoint_path: str) -> dict:
    """Load PyTorch checkpoint and return state dict."""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # Remove prefix if present
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        for prefix in ['model.', 'network.', 'net.']:
            if k.startswith(prefix):
                k = k[len(prefix):]
                break
        cleaned_state_dict[k] = v

    return cleaned_state_dict


def tensor_to_c_array(name: str, tensor: np.ndarray, dtype: str = 'float') -> str:
    """Convert numpy array to C array declaration."""
    shape_str = ' x '.join(str(s) for s in tensor.shape)
    flat = tensor.flatten()

    if dtype == 'float':
        c_dtype = 'const float'
        values = ', '.join(f'{v:.8f}f' for v in flat)
    elif dtype == 'int8':
        # Quantize to int8
        scale = max(abs(flat.min()), abs(flat.max())) / 127.0
        quantized = np.clip(np.round(flat / scale), -127, 127).astype(np.int8)
        c_dtype = 'const int8_t'
        values = ', '.join(str(v) for v in quantized)
        # Add scale info
        return f'''// Shape: [{shape_str}], Scale: {scale:.8f}
{c_dtype} {name}[] = {{{values}}};
const float {name}_scale = {scale:.8f}f;
'''
    else:
        raise ValueError(f"Unknown dtype: {dtype}")

    return f'''// Shape: [{shape_str}]
{c_dtype} {name}[] = {{{values}}};
'''


def generate_c_header(state_dict: dict, output_path: str, quantize: bool = True) -> str:
    """Generate C header file with model weights."""

    dtype = 'int8' if quantize else 'float'

    header = f'''/**
 * M1 4-Channel TinyML Model Weights for XIAO nRF52840
 *
 * Auto-generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
 * Quantization: {'INT8' if quantize else 'FP32'}
 *
 * Architecture:
 *   Conv1d(4→72, k=15, s=10) → BatchNorm → LSTM(48, 2 layers) → LayerNorm → FC(9)
 *   Parameters: ~47,300
 *   Accuracy: 99.59%
 *
 * 4-Channel Selection (flexor-extensor balance):
 *   A0 (Ch7): Index flexor - 24.5% importance
 *   A1 (Ch8): Ring flexor - 13.2% importance
 *   A2 (Ch13): Index/middle extensor - 19.8% importance
 *   A3 (Ch15): Ring/pinky extensor - 8.7% importance
 *
 * Target: Seeed Studio XIAO nRF52840 (non-Sense)
 */

#ifndef M1_4CHANNEL_WEIGHTS_H
#define M1_4CHANNEL_WEIGHTS_H

#include <stdint.h>

// Model configuration
#define M1_INPUT_CHANNELS 4
#define M1_INPUT_SAMPLES 2000
#define M1_CONV_OUT_CHANNELS 72
#define M1_CONV_KERNEL_SIZE 15
#define M1_CONV_STRIDE 10
#define M1_LSTM_HIDDEN_SIZE 48
#define M1_LSTM_NUM_LAYERS 2
#define M1_OUTPUT_CLASSES 9
#define M1_OUTPUT_TIMESTEPS 199

// Reinhard compression parameters
#define M1_REINHARD_RANGE 1.0f
#define M1_REINHARD_MIDPOINT 32.0f

'''

    # Process each weight tensor
    weight_sections = []

    # Conv1d weights and bias
    if 'conv.weight' in state_dict:
        w = state_dict['conv.weight'].numpy()
        print(f"  conv.weight: {w.shape}")
        weight_sections.append(f"// === Conv1D Layer ===\n")
        weight_sections.append(tensor_to_c_array('conv_weight', w, dtype))

    if 'conv.bias' in state_dict:
        b = state_dict['conv.bias'].numpy()
        print(f"  conv.bias: {b.shape}")
        weight_sections.append(tensor_to_c_array('conv_bias', b, dtype))

    # BatchNorm parameters
    if 'conv_bn.weight' in state_dict:
        weight_sections.append(f"\n// === BatchNorm Layer ===\n")
        w = state_dict['conv_bn.weight'].numpy()
        print(f"  conv_bn.weight (gamma): {w.shape}")
        weight_sections.append(tensor_to_c_array('bn_gamma', w, dtype))

    if 'conv_bn.bias' in state_dict:
        b = state_dict['conv_bn.bias'].numpy()
        print(f"  conv_bn.bias (beta): {b.shape}")
        weight_sections.append(tensor_to_c_array('bn_beta', b, dtype))

    if 'conv_bn.running_mean' in state_dict:
        m = state_dict['conv_bn.running_mean'].numpy()
        print(f"  conv_bn.running_mean: {m.shape}")
        weight_sections.append(tensor_to_c_array('bn_mean', m, dtype))

    if 'conv_bn.running_var' in state_dict:
        v = state_dict['conv_bn.running_var'].numpy()
        print(f"  conv_bn.running_var: {v.shape}")
        weight_sections.append(tensor_to_c_array('bn_var', v, dtype))

    # LSTM weights (2 layers)
    weight_sections.append(f"\n// === LSTM Layer 1 ===\n")
    for layer in range(2):
        prefix = f"lstm."
        suffix = f"_l{layer}"

        weight_sections.append(f"\n// --- LSTM Layer {layer} ---\n")

        # Input-hidden weights
        key = f'{prefix}weight_ih_l{layer}'
        if key in state_dict:
            w = state_dict[key].numpy()
            print(f"  {key}: {w.shape}")
            weight_sections.append(tensor_to_c_array(f'lstm{layer}_weight_ih', w, dtype))

        # Hidden-hidden weights
        key = f'{prefix}weight_hh_l{layer}'
        if key in state_dict:
            w = state_dict[key].numpy()
            print(f"  {key}: {w.shape}")
            weight_sections.append(tensor_to_c_array(f'lstm{layer}_weight_hh', w, dtype))

        # Input-hidden bias
        key = f'{prefix}bias_ih_l{layer}'
        if key in state_dict:
            b = state_dict[key].numpy()
            print(f"  {key}: {b.shape}")
            weight_sections.append(tensor_to_c_array(f'lstm{layer}_bias_ih', b, dtype))

        # Hidden-hidden bias
        key = f'{prefix}bias_hh_l{layer}'
        if key in state_dict:
            b = state_dict[key].numpy()
            print(f"  {key}: {b.shape}")
            weight_sections.append(tensor_to_c_array(f'lstm{layer}_bias_hh', b, dtype))

    # LayerNorm
    weight_sections.append(f"\n// === LayerNorm ===\n")
    if 'lstm_ln.weight' in state_dict:
        w = state_dict['lstm_ln.weight'].numpy()
        print(f"  lstm_ln.weight: {w.shape}")
        weight_sections.append(tensor_to_c_array('ln_weight', w, dtype))

    if 'lstm_ln.bias' in state_dict:
        b = state_dict['lstm_ln.bias'].numpy()
        print(f"  lstm_ln.bias: {b.shape}")
        weight_sections.append(tensor_to_c_array('ln_bias', b, dtype))

    # Output FC layer
    weight_sections.append(f"\n// === Output FC Layer ===\n")
    if 'fc_out.weight' in state_dict:
        w = state_dict['fc_out.weight'].numpy()
        print(f"  fc_out.weight: {w.shape}")
        weight_sections.append(tensor_to_c_array('fc_weight', w, dtype))

    if 'fc_out.bias' in state_dict:
        b = state_dict['fc_out.bias'].numpy()
        print(f"  fc_out.bias: {b.shape}")
        weight_sections.append(tensor_to_c_array('fc_bias', b, dtype))

    # Combine all sections
    header += '\n'.join(weight_sections)

    header += '''

#endif // M1_4CHANNEL_WEIGHTS_H
'''

    # Write to file
    with open(output_path, 'w') as f:
        f.write(header)

    print(f"\nGenerated C header: {output_path}")

    # Calculate file size
    file_size = os.path.getsize(output_path)
    print(f"Header file size: {file_size / 1024:.1f} KB")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Export M1 4-Channel model weights to C header for Arduino'
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained checkpoint')
    parser.add_argument('--output', type=str, default='m1_4channel_weights.h',
                        help='Output header file path')
    parser.add_argument('--no-quantize', action='store_true',
                        help='Export as float32 instead of int8')

    args = parser.parse_args()

    print("="*60)
    print("M1 4-Channel Weight Export for Arduino")
    print("="*60)

    # Load checkpoint
    state_dict = load_checkpoint(args.checkpoint)

    print(f"\nFound {len(state_dict)} weight tensors:")

    # Generate header
    generate_c_header(
        state_dict,
        args.output,
        quantize=not args.no_quantize
    )

    print("\n" + "="*60)
    print("Export Complete!")
    print("="*60)
    print(f"\nNext steps for Arduino deployment:")
    print(f"  1. Copy {args.output} to your Arduino project's include/ folder")
    print(f"  2. Implement inference code using the exported weights")
    print(f"  3. Connect EMG channels:")
    print(f"     A0 <- Ch7 (Index flexor)")
    print(f"     A1 <- Ch8 (Ring flexor)")
    print(f"     A2 <- Ch13 (Index/middle extensor)")
    print(f"     A3 <- Ch15 (Ring/pinky extensor)")


if __name__ == '__main__':
    main()
