# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
FIXED Isolated channel transforms for specific EMG channel subsets.

KEY FIXES:
1. Proper 0-based channel indexing (indices [4,5,6,7,8,12,14] for Ch5-Ch15)
2. Preserved dtype consistency (float32)
3. Direct channel selection without intermediate structured arrays
4. Compatible with base transforms
"""

from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd
import torch

from generic_neuromotor_interface.constants import GestureType
from generic_neuromotor_interface.transforms import _to_tensor


# Optimal channel indices for discrete gestures (0-BASED indexing!)
# These correspond to: Ch5, Ch6, Ch7, Ch8, Ch9, Ch13, Ch15 in 1-based terms
# Anatomically: Thumb flexor, Index, Middle, Ring, Pinky, Extensor1, Extensor2
ISOLATED_CHANNELS = [4, 5, 6, 7, 8, 12, 14]  # 0-BASED INDEXING!


@dataclass
class IsolatedDiscreteGesturesTransform:
    """
    Discrete gestures transform with proper channel selection.

    FIXES APPLIED:
    1. Uses 0-based channel indices correctly
    2. Preserves original dtype (float32)
    3. Creates properly formatted tensors
    4. Compatible with base transform expectations

    Parameters
    ----------
    pulse_window : list[float]
        [start_offset, end_offset] in seconds around each event
    channel_indices : list[int]
        0-BASED indices of channels to select. Default: [4,5,6,7,8,12,14]
        which corresponds to channels 5,6,7,8,9,13,15 in 1-based notation
    """

    pulse_window: List[float]
    channel_indices: List[int] = field(default_factory=lambda: ISOLATED_CHANNELS)

    def __post_init__(self):
        # Sort indices for consistent ordering
        self.channel_indices = sorted(self.channel_indices)
        self.num_channels = len(self.channel_indices)

    def __call__(
        self, timeseries: np.ndarray, prompts: pd.DataFrame | None
    ) -> dict[str, torch.Tensor]:
        assert prompts is not None, "Prompts required for discrete gestures"

        # Extract EMG and select channels
        # timeseries["emg"] shape: (time, 16)
        emg_full = timeseries["emg"]  # (T, 16)
        emg_selected = emg_full[:, self.channel_indices]  # (T, 7)

        # Get time array
        times = timeseries["time"]  # (T,)

        # Filter prompts to current time window
        tlim = (times[0], times[-1])
        prompts_filtered = prompts[prompts["time"].between(*tlim)]
        prompts_filtered = prompts_filtered[
            prompts_filtered["name"].isin([g.name for g in GestureType])
        ]

        # Convert EMG to tensor: (T, C) -> (C, T)
        emg_tensor = _to_tensor(emg_selected.T)  # (7, T)

        # Create target pulse matrix
        targets = self._gesture_times_to_targets(
            times=times,
            event_times=prompts_filtered["time"].values,
            event_names=prompts_filtered["name"].values,
        )

        return {
            "emg": emg_tensor,      # (7, T)
            "targets": targets,      # (9, T)
        }

    def _gesture_times_to_targets(
        self,
        times: np.ndarray,
        event_times: np.ndarray,
        event_names: np.ndarray,
    ) -> torch.Tensor:
        """
        Convert gesture event times to binary pulse target matrix.

        Parameters
        ----------
        times : np.ndarray
            Timestamps array, shape (T,)
        event_times : np.ndarray
            Event occurrence times
        event_names : np.ndarray
            Event gesture names

        Returns
        -------
        torch.Tensor
            Binary pulse matrix, shape (9, T)
        """
        num_timesteps = len(times)
        num_gestures = len(GestureType)

        # Calculate sampling frequency from timestamps
        duration = times[-1] - times[0]
        if duration <= 0:
            duration = 1.0  # Fallback
        sampling_freq = num_timesteps / duration

        # Initialize pulse matrix
        pulse = torch.zeros(num_gestures, num_timesteps, dtype=torch.float32)

        # Map gesture names to indices
        name_to_idx = {g.name: g.value for g in GestureType}

        # Calculate window offsets in samples
        start_offset = int(self.pulse_window[0] * sampling_freq)
        end_offset = int(self.pulse_window[1] * sampling_freq)

        for event_time, event_name in zip(event_times, event_names):
            # Get gesture index
            gesture_idx = name_to_idx.get(event_name)
            if gesture_idx is None or gesture_idx >= num_gestures:
                continue

            # Find event position in time array
            event_idx = np.searchsorted(times, event_time)

            # Skip if out of bounds
            if event_idx <= 0 or event_idx >= num_timesteps:
                continue

            # Calculate pulse window bounds
            pulse_start = max(0, event_idx + start_offset)
            pulse_end = min(num_timesteps, event_idx + end_offset)

            if pulse_start < pulse_end:
                pulse[gesture_idx, pulse_start:pulse_end] = 1.0

        return pulse


@dataclass
class IsolatedWristTransform:
    """Wrist transform with channel selection for velocity prediction."""

    channel_indices: List[int] = field(default_factory=lambda: ISOLATED_CHANNELS)

    def __post_init__(self):
        self.channel_indices = sorted(self.channel_indices)

    def __call__(
        self, timeseries: np.ndarray, prompts: pd.DataFrame | None
    ) -> dict[str, torch.Tensor]:
        # Select EMG channels
        emg_selected = timeseries["emg"][:, self.channel_indices]  # (T, 7)
        wrist_angles = timeseries["wrist_angles"]  # (T, 2)

        # Convert to tensors and reshape: (T, C) -> (C, T)
        emg_tensor = _to_tensor(emg_selected.T)  # (7, T)
        wrist_tensor = _to_tensor(wrist_angles[:, [0]].T)  # (1, T) - only flex/ext

        return {
            "emg": emg_tensor,
            "wrist_angles": wrist_tensor,
        }


@dataclass
class IsolatedHandwritingTransform:
    """Handwriting transform with channel selection."""

    channel_indices: List[int] = field(default_factory=lambda: ISOLATED_CHANNELS)

    def __post_init__(self):
        self.channel_indices = sorted(self.channel_indices)

    def __call__(
        self, timeseries: np.ndarray, prompt: str | None
    ) -> dict[str, torch.Tensor | str]:
        # Select EMG channels
        emg_selected = timeseries["emg"][:, self.channel_indices]  # (T, 7)

        # Convert to tensor: (T, C) -> (C, T)
        emg_tensor = _to_tensor(emg_selected.T)  # (7, T)

        return {
            "emg": emg_tensor,
            "prompt": prompt if prompt is not None else "",
        }
