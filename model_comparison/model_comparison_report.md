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
| M1: CNN+LSTM      |            nan          | failed    |            nan |         nan |
| M2: CNN-Only      |              0.461889   | completed |            nan |         nan |
| M3: Random Forest |              0.00859429 | completed |            nan |         nan |

## Detailed Metrics

### M1: CNN+LSTM

- **Status**: failed
- **Error**: M1 training failed: Command '['/u/usz7pc/.conda/envs/neuromotor/bin/python3', '-m', 'generic_neuromotor_interface.train', '--config-name=discrete_gestures_isolated', 'data_location=/u/usz7pc/emg_data', 'trainer.accelerator=gpu', '+trainer.devices=1']' returned non-zero exit status 1.
STDERR:
tor_interface/lightning.py", line 51, in test_step
    return self._step(batch, stage="test")
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/u/usz7pc/Perpetuality/generic-neuromotor-interface/generic_neuromotor_interface/lightning.py", line 395, in _step
    preds = self.forward(emg)
            ^^^^^^^^^^^^^^^^^
  File "/u/usz7pc/Perpetuality/generic-neuromotor-interface/generic_neuromotor_interface/lightning.py", line 40, in forward
    return self.network(emg)
           ^^^^^^^^^^^^^^^^^
  File "/u/usz7pc/.conda/envs/neuromotor/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1553, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/u/usz7pc/.conda/envs/neuromotor/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1562, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/u/usz7pc/Perpetuality/generic-neuromotor-interface/generic_neuromotor_interface/networks.py", line 170, in forward
    x, _ = self.lstm(x)
           ^^^^^^^^^^^^
  File "/u/usz7pc/.conda/envs/neuromotor/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1553, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/u/usz7pc/.conda/envs/neuromotor/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1562, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/u/usz7pc/.conda/envs/neuromotor/lib/python3.12/site-packages/torch/nn/modules/rnn.py", line 917, in forward
    result = _VF.lstm(input, hx, self._flat_weights, self.bias, self.num_layers,
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: cuDNN error: CUDNN_STATUS_NOT_SUPPORTED. This error may appear if you passed in a non-contiguous input.

Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

STDOUT:
1-16/00-36-38/lightning_logs/version_0/checkpoints/epoch=88-step=890.ckpt' with best_checkpoint_score=tensor(0.2236, device='cuda:0')...
[2025-11-16 01:11:36,471][__main__][INFO] - Destroying process group...
[2025-11-16 01:11:37,027][__main__][INFO] - Destroyed process group.
[2025-11-16 01:11:37,027][__main__][INFO] - Re-instantiating LightningDataModule for evaluation...
[2025-11-16 01:11:37,175][__main__][INFO] - Running validation...

Validation: 0it [00:00, ?it/s]
Validation:   0%|          | 0/6 [00:00<?, ?it/s]
Validation DataLoader 0:   0%|          | 0/6 [00:00<?, ?it/s]
Validation DataLoader 0:  17%|█▋        | 1/6 [00:00<00:01,  3.97it/s]
Validation DataLoader 0:  33%|███▎      | 2/6 [00:00<00:01,  2.62it/s]
Validation DataLoader 0:  50%|█████     | 3/6 [00:01<00:01,  2.68it/s]
Validation DataLoader 0:  67%|██████▋   | 4/6 [00:01<00:00,  2.71it/s]
Validation DataLoader 0:  83%|████████▎ | 5/6 [00:01<00:00,  2.73it/s]
Validation DataLoader 0: 100%|██████████| 6/6 [00:01<00:00,  3.01it/s]
Validation DataLoader 0: 100%|██████████| 6/6 [00:01<00:00,  3.00it/s]
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
Runningstage.validating metric      DataLoader 0
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
      val_accuracy          0.22363415360450745
        val_loss           0.018723363056778908
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
[2025-11-16 01:12:24,005][__main__][INFO] - Validation completed! val_results=[{'val_loss': 0.018723363056778908, 'val_accuracy': 0.22363415360450745}]
[2025-11-16 01:12:24,006][__main__][INFO] - Running test...

Testing: 0it [00:00, ?it/s]
Testing:   0%|          | 0/1 [00:00<?, ?it/s]
Testing DataLoader 0:   0%|          | 0/1 [00:00<?, ?it/s]
Testing DataLoader 0:   0%|          | 0/1 [00:01<?, ?it/s]
- **Metrics**:
  - No metrics available

### M2: CNN-Only

- **Training Time**: 0.46 hours
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
