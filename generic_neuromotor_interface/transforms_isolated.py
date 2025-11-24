# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Isolated channel transforms for specific EMG channel subsets."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch

from generic_neuromotor_interface.channel_selector import ChannelSelector
from generic_neuromotor_interface.transforms import (
    DiscreteGesturesTransform,
    HandwritingTransform,
    WristTransform,
    _to_tensor,
)


@dataclass
class IsolatedDiscreteGesturesTransform:
    """Discrete gestures transform with channel selection.
    
    Selects only specified channels before applying the standard transform.
    """
    
    pulse_window: list[float]
    channel_indices: list[int]  # Channels to use: [9, 8, 7, 6, 5, 13, 15]
    
    def __post_init__(self):
        self.channel_selector = ChannelSelector(self.channel_indices)
        self.base_transform = DiscreteGesturesTransform(pulse_window=self.pulse_window)
    
    def __call__(
        self, timeseries: np.ndarray, prompts: pd.DataFrame | None
    ) -> dict[str, torch.Tensor]:
        # Select channels from EMG data
        emg_original = timeseries["emg"]  # (time, 16)
        emg_selected = self.channel_selector(emg_original)  # (time, num_selected)
        
        # Create new timeseries with selected channels
        # timeseries is a structured array, so we need to create a new one
        timeseries_selected = np.empty(len(timeseries), dtype=[("emg", emg_selected.dtype, (len(self.channel_indices),)), ("time", timeseries["time"].dtype)])
        timeseries_selected["emg"] = emg_selected
        timeseries_selected["time"] = timeseries["time"]
        
        # Apply base transform
        return self.base_transform(timeseries_selected, prompts)


@dataclass
class IsolatedHandwritingTransform:
    """Handwriting transform with channel selection.
    
    Selects only specified channels before applying the standard transform.
    """
    
    channel_indices: list[int]  # Channels to use: [9, 8, 7, 6, 5, 13, 15]
    
    def __post_init__(self):
        self.channel_selector = ChannelSelector(self.channel_indices)
        self.base_transform = HandwritingTransform()
    
    def __call__(
        self, timeseries: np.ndarray, prompt: str | None
    ) -> dict[str, torch.Tensor | str]:
        # Select channels from EMG data
        emg_original = timeseries["emg"]  # (time, 16)
        emg_selected = self.channel_selector(emg_original)  # (time, num_selected)
        
        # Create new timeseries with selected channels
        # timeseries is a structured array, so we need to create a new one
        timeseries_selected = np.empty(len(timeseries), dtype=[("emg", emg_selected.dtype, (len(self.channel_indices),)), ("time", timeseries["time"].dtype)])
        timeseries_selected["emg"] = emg_selected
        timeseries_selected["time"] = timeseries["time"]
        
        # Apply base transform
        return self.base_transform(timeseries_selected, prompt)


@dataclass
class IsolatedWristTransform:
    """Wrist transform with channel selection.
    
    Selects only specified channels before applying the standard transform.
    """
    
    channel_indices: list[int]  # Channels to use: [9, 8, 7, 6, 5, 13, 15]
    
    def __post_init__(self):
        self.channel_selector = ChannelSelector(self.channel_indices)
        self.base_transform = WristTransform()
    
    def __call__(
        self, timeseries: np.ndarray, prompts: pd.DataFrame | None
    ) -> dict[str, torch.Tensor]:
        # Select channels from EMG data
        emg_original = timeseries["emg"]  # (time, 16)
        emg_selected = self.channel_selector(emg_original)  # (time, num_selected)
        
        # Create new timeseries with selected channels
        # timeseries is a structured array, so we need to create a new one
        timeseries_selected = np.empty(len(timeseries), dtype=[("emg", emg_selected.dtype, (len(self.channel_indices),)), ("time", timeseries["time"].dtype)])
        timeseries_selected["emg"] = emg_selected
        timeseries_selected["time"] = timeseries["time"]
        
        # Apply base transform
        return self.base_transform(timeseries_selected, prompts)

