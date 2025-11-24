#!/usr/bin/env python3
"""Test script to verify isolated model configurations."""

from hydra import initialize, compose
from hydra.utils import instantiate

print("Testing handwriting_isolated config...")
with initialize(config_path='config', version_base='1.1'):
    cfg = compose(
        config_name='handwriting_isolated',
        overrides=['data_module/data_split=handwriting_mini_split']
    )
    print('✓ Config loaded successfully')
    print(f'  Transform channels: {cfg.data_module.transform.channel_indices}')
    print(f'  Network num_channels: {cfg.lightning_module.network.num_channels}')
    
    # Try to instantiate the transform
    transform = instantiate(cfg.data_module.transform)
    print(f'  Transform instantiated: {type(transform).__name__}')

print("\nTesting discrete_gestures_isolated config...")
with initialize(config_path='config', version_base='1.1'):
    cfg = compose(
        config_name='discrete_gestures_isolated',
        overrides=['data_module/data_split=discrete_gestures_mini_split']
    )
    print('✓ Config loaded successfully')
    print(f'  Transform channels: {cfg.data_module.transform.channel_indices}')
    print(f'  Network input_channels: {cfg.lightning_module.network.input_channels}')

print("\n✓ All isolated model configurations are valid!")

