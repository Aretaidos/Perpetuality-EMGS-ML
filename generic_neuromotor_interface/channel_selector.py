# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Channel selection utilities for isolating specific EMG channels."""

import torch
import numpy as np
from typing import Sequence


class ChannelSelector:
    """Select specific channels from EMG data.
    
    Parameters
    ----------
    channel_indices : Sequence[int]
        Indices of channels to select (0-based, original 16 channels)
    """
    
    def __init__(self, channel_indices: Sequence[int]):
        self.channel_indices = tuple(sorted(channel_indices))
        self.num_channels = len(self.channel_indices)
        
    def __call__(self, emg: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
        """Select channels from EMG tensor.
        
        Parameters
        ----------
        emg : torch.Tensor or np.ndarray
            EMG data of shape (..., num_channels) or (num_channels, ...)
            where num_channels is the original number of channels (16)
            
        Returns
        -------
        torch.Tensor or np.ndarray
            EMG data with only selected channels (same type as input)
        """
        is_numpy = isinstance(emg, np.ndarray)
        if is_numpy:
            emg_tensor = torch.from_numpy(emg).float()
        else:
            emg_tensor = emg
        
        # Handle different tensor shapes
        result = None
        if emg_tensor.ndim == 2:
            # (time, channels) or (channels, time)
            if emg_tensor.shape[-1] == 16:
                # (time, channels) -> select last dim
                result = emg_tensor[:, self.channel_indices]
            elif emg_tensor.shape[0] == 16:
                # (channels, time) -> select first dim
                result = emg_tensor[self.channel_indices, :]
        elif emg_tensor.ndim == 3:
            # (batch, time, channels) or (batch, channels, time)
            if emg_tensor.shape[-1] == 16:
                # (batch, time, channels) -> select last dim
                result = emg_tensor[:, :, self.channel_indices]
            elif emg_tensor.shape[1] == 16:
                # (batch, channels, time) -> select middle dim
                result = emg_tensor[:, self.channel_indices, :]
        elif emg_tensor.ndim == 1:
            # Single channel or time point - shouldn't happen with 16 channels
            raise ValueError(f"Unexpected 1D tensor shape: {emg_tensor.shape}")
        
        if result is None:
            raise ValueError(f"Cannot determine channel dimension for shape: {emg_tensor.shape}")
        
        # Convert back to numpy if input was numpy
        if is_numpy:
            return result.numpy()
        return result

