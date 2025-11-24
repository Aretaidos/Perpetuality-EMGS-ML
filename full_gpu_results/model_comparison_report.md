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
| M1: CNN+LSTM      |             nan         | failed    |            nan |         nan |
| M2: CNN-Only      |             nan         | failed    |            nan |         nan |
| M3: Random Forest |               0.0113561 | completed |            nan |         nan |

## Detailed Metrics

### M1: CNN+LSTM

- **Status**: failed
- **Error**: M1 training failed: Command '['/u/usz7pc/.conda/envs/neuromotor/bin/python3', '-m', 'generic_neuromotor_interface.train', '--config-name=discrete_gestures_isolated', 'data_location=/u/usz7pc/emg_data', 'trainer.accelerator=gpu', '+trainer.devices=1', 'trainer.strategy=auto']' returned non-zero exit status 1.
STDERR:
work' is an instance of `nn.Module` and is already saved during checkpointing. It is recommended to ignore them using `self.save_hyperparameters(ignore=['network'])`.
  rank_zero_warn(
/u/usz7pc/.conda/envs/neuromotor/lib/python3.12/site-packages/torch/cuda/__init__.py:654: UserWarning: Can't initialize NVML
  warnings.warn("Can't initialize NVML")
Error executing job with overrides: ['data_location=/u/usz7pc/emg_data', 'trainer.accelerator=gpu', '+trainer.devices=1', 'trainer.strategy=auto']
Traceback (most recent call last):
  File "/u/usz7pc/Perpetuality/generic-neuromotor-interface/generic_neuromotor_interface/train.py", line 238, in cli
    train(config)
  File "/u/usz7pc/Perpetuality/generic-neuromotor-interface/generic_neuromotor_interface/train.py", line 61, in train
    trainer = pl.Trainer(
              ^^^^^^^^^^^
  File "/u/usz7pc/.conda/envs/neuromotor/lib/python3.12/site-packages/pytorch_lightning/utilities/argparse.py", line 340, in insert_env_defaults
    return fn(self, **kwargs)
           ^^^^^^^^^^^^^^^^^^
  File "/u/usz7pc/.conda/envs/neuromotor/lib/python3.12/site-packages/pytorch_lightning/trainer/trainer.py", line 414, in __init__
    self._accelerator_connector = AcceleratorConnector(
                                  ^^^^^^^^^^^^^^^^^^^^^
  File "/u/usz7pc/.conda/envs/neuromotor/lib/python3.12/site-packages/pytorch_lightning/trainer/connectors/accelerator_connector.py", line 206, in __init__
    self._accelerator_flag = self._choose_gpu_accelerator_backend()
                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/u/usz7pc/.conda/envs/neuromotor/lib/python3.12/site-packages/pytorch_lightning/trainer/connectors/accelerator_connector.py", line 512, in _choose_gpu_accelerator_backend
    raise MisconfigurationException("No supported gpu backend found!")
lightning_lite.utilities.exceptions.MisconfigurationException: No supported gpu backend found!

Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

STDOUT:
rainer:
  max_epochs: 250
  strategy: auto
  accelerator: gpu
  devices: 1
monitor_metric: val_accuracy
monitor_mode: max
callbacks:
- _target_: pytorch_lightning.callbacks.LearningRateMonitor
- _target_: pytorch_lightning.callbacks.ModelCheckpoint
  monitor: ${monitor_metric}
  mode: ${monitor_mode}
  save_last: true
  verbose: true
data_module:
  _target_: generic_neuromotor_interface.data_module.WindowedEmgDataModule
  window_length: 16000
  stride: 16000
  batch_size: 64
  num_workers: 0
  transform:
    _target_: generic_neuromotor_interface.transforms_isolated.IsolatedDiscreteGesturesTransform
    pulse_window:
    - 0.08
    - 0.12
    channel_indices:
    - 9
    - 8
    - 7
    - 6
    - 5
    - 13
    - 15
  data_location: ${data_location}
  emg_augmentation:
    _target_: generic_neuromotor_interface.augmentation.RotationAugmentation
    rotation: 2
  data_split:
    _target_: generic_neuromotor_interface.data.DataSplit
    train:
      discrete_gestures_user_000_dataset_000:
      - - 0
        - .inf
      discrete_gestures_user_001_dataset_000:
      - - 0
        - .inf
    val:
      discrete_gestures_user_002_dataset_000:
      - - 0
        - .inf
    test:
      discrete_gestures_user_002_dataset_000:
      - - 0
        - .inf
lightning_module:
  _target_: generic_neuromotor_interface.lightning.DiscreteGesturesModule
  optimizer:
    _target_: torch.optim.Adam
    lr: 0.0001
    _partial_: true
  network:
    _target_: generic_neuromotor_interface.networks.DiscreteGesturesArchitecture
    input_channels: 7
    output_channels: 9
  learning_rate: 0.0005
  lr_scheduler_milestones:
  - 25
  lr_scheduler_factor: 0.5
  warmup_start_factor: 0.001
  warmup_end_factor: 1.0
  warmup_total_epochs: 5
  gradient_clip_val: 0.5

[2025-11-17 14:39:45,977][__main__][INFO] - Instantiating LightningModule
[2025-11-17 14:39:49,600][numexpr.utils][INFO] - NumExpr defaulting to 16 threads.
[2025-11-17 14:39:51,326][__main__][INFO] - Instantiating LightningDataModule

- **Metrics**:
  - No metrics available

### M2: CNN-Only

- **Status**: failed
- **Error**: M2 training failed: Command '['/u/usz7pc/.conda/envs/neuromotor/bin/python3', '-m', 'generic_neuromotor_interface.train', '--config-name=discrete_gestures_cnn_isolated', 'data_location=/u/usz7pc/emg_data', 'trainer.accelerator=gpu', '+trainer.devices=1', 'trainer.strategy=auto']' returned non-zero exit status 1.
STDERR:
work' is an instance of `nn.Module` and is already saved during checkpointing. It is recommended to ignore them using `self.save_hyperparameters(ignore=['network'])`.
  rank_zero_warn(
/u/usz7pc/.conda/envs/neuromotor/lib/python3.12/site-packages/torch/cuda/__init__.py:654: UserWarning: Can't initialize NVML
  warnings.warn("Can't initialize NVML")
Error executing job with overrides: ['data_location=/u/usz7pc/emg_data', 'trainer.accelerator=gpu', '+trainer.devices=1', 'trainer.strategy=auto']
Traceback (most recent call last):
  File "/u/usz7pc/Perpetuality/generic-neuromotor-interface/generic_neuromotor_interface/train.py", line 238, in cli
    train(config)
  File "/u/usz7pc/Perpetuality/generic-neuromotor-interface/generic_neuromotor_interface/train.py", line 61, in train
    trainer = pl.Trainer(
              ^^^^^^^^^^^
  File "/u/usz7pc/.conda/envs/neuromotor/lib/python3.12/site-packages/pytorch_lightning/utilities/argparse.py", line 340, in insert_env_defaults
    return fn(self, **kwargs)
           ^^^^^^^^^^^^^^^^^^
  File "/u/usz7pc/.conda/envs/neuromotor/lib/python3.12/site-packages/pytorch_lightning/trainer/trainer.py", line 414, in __init__
    self._accelerator_connector = AcceleratorConnector(
                                  ^^^^^^^^^^^^^^^^^^^^^
  File "/u/usz7pc/.conda/envs/neuromotor/lib/python3.12/site-packages/pytorch_lightning/trainer/connectors/accelerator_connector.py", line 206, in __init__
    self._accelerator_flag = self._choose_gpu_accelerator_backend()
                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/u/usz7pc/.conda/envs/neuromotor/lib/python3.12/site-packages/pytorch_lightning/trainer/connectors/accelerator_connector.py", line 512, in _choose_gpu_accelerator_backend
    raise MisconfigurationException("No supported gpu backend found!")
lightning_lite.utilities.exceptions.MisconfigurationException: No supported gpu backend found!

Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

STDOUT:
ner:
  max_epochs: 250
  strategy: auto
  accelerator: gpu
  devices: 1
monitor_metric: val_accuracy
monitor_mode: max
callbacks:
- _target_: pytorch_lightning.callbacks.LearningRateMonitor
- _target_: pytorch_lightning.callbacks.ModelCheckpoint
  monitor: ${monitor_metric}
  mode: ${monitor_mode}
  save_last: true
  verbose: true
data_module:
  _target_: generic_neuromotor_interface.data_module.WindowedEmgDataModule
  window_length: 16000
  stride: 16000
  batch_size: 64
  num_workers: 0
  transform:
    _target_: generic_neuromotor_interface.transforms_isolated.IsolatedDiscreteGesturesTransform
    pulse_window:
    - 0.08
    - 0.12
    channel_indices:
    - 9
    - 8
    - 7
    - 6
    - 5
    - 13
    - 15
  data_location: ${data_location}
  emg_augmentation:
    _target_: generic_neuromotor_interface.augmentation.RotationAugmentation
    rotation: 2
  data_split:
    _target_: generic_neuromotor_interface.data.DataSplit
    train:
      discrete_gestures_user_000_dataset_000:
      - - 0
        - .inf
      discrete_gestures_user_001_dataset_000:
      - - 0
        - .inf
    val:
      discrete_gestures_user_002_dataset_000:
      - - 0
        - .inf
    test:
      discrete_gestures_user_002_dataset_000:
      - - 0
        - .inf
lightning_module:
  _target_: generic_neuromotor_interface.lightning.DiscreteGesturesModule
  optimizer:
    _target_: torch.optim.Adam
    lr: 0.0001
    _partial_: true
  network:
    _target_: generic_neuromotor_interface.networks.DiscreteGesturesCNNArchitecture
    input_channels: 7
    output_channels: 9
  learning_rate: 0.0005
  lr_scheduler_milestones:
  - 25
  lr_scheduler_factor: 0.5
  warmup_start_factor: 0.001
  warmup_end_factor: 1.0
  warmup_total_epochs: 5
  gradient_clip_val: 0.5

[2025-11-17 14:39:59,094][__main__][INFO] - Instantiating LightningModule
[2025-11-17 14:39:59,359][numexpr.utils][INFO] - NumExpr defaulting to 16 threads.
[2025-11-17 14:39:59,754][__main__][INFO] - Instantiating LightningDataModule

- **Metrics**:
  - No metrics available

### M3: Random Forest

- **Training Time**: 0.01 hours
- **Status**: completed
- **Metrics**:
  - No metrics available

## Recommendations

- **Best Accuracy**: M1: CNN+LSTM (0.0000)
