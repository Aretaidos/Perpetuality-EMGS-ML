#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Training script for Random Forest discrete gesture recognition model."""

import argparse
import logging
import pickle
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from generic_neuromotor_interface.channel_selector import ChannelSelector
from generic_neuromotor_interface.constants import EMG_SAMPLE_RATE
from generic_neuromotor_interface.random_forest_model import RandomForestGestureModel
from generic_neuromotor_interface.utils import get_full_dataset_path

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def load_data(data_location: str, channel_indices: list[int], task: str = "discrete_gestures"):
    """
    Load and preprocess EMG data for Random Forest training.

    Parameters
    ----------
    data_location : str
        Path to data directory
    channel_indices : list[int]
        Indices of channels to use
    task : str
        Task name (default: "discrete_gestures")

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        X_train, y_train arrays
    """
    log.info(f"Loading data from {data_location}")
    log.info(f"Using isolated channels: {channel_indices}")

    # Channel selector
    channel_selector = ChannelSelector(channel_indices)

    # Collect all samples
    X_samples = []
    y_samples = []

    # Load corpus CSV to get train split datasets
    # Try filtered corpus first, fall back to full corpus
    csv_path = Path(data_location) / f"{task}_corpus_filtered.csv"
    if not csv_path.exists():
        csv_path = Path(data_location) / f"{task}_corpus.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Corpus CSV not found at {csv_path}")
    
    corpus_df = pd.read_csv(csv_path)
    
    # Only use datasets that actually exist
    # Corpus CSV has dataset names with .hdf5 extension
    available_datasets = [
        "discrete_gestures_user_000_dataset_000",
        "discrete_gestures_user_001_dataset_000",
        "discrete_gestures_user_002_dataset_000",
    ]
    available_datasets_with_ext = [f"{ds}.hdf5" for ds in available_datasets]
    
    # Filter to only available datasets (check both with and without extension)
    corpus_df = corpus_df[
        corpus_df["dataset"].isin(available_datasets) | 
        corpus_df["dataset"].isin(available_datasets_with_ext)
    ]
    
    # Use train split, or if no train split exists, use all available data
    train_datasets = corpus_df[corpus_df["split"] == "train"]["dataset"].unique()
    if len(train_datasets) == 0:
        # If no train split, use all available datasets
        log.warning("No train split found in corpus, using all available datasets")
        train_datasets = available_datasets[:2]  # Use first 2 for training

    # Process each dataset in train split
    for dataset_name in train_datasets:
        try:
            # Remove .hdf5 extension if present (get_full_dataset_path will add it)
            dataset_name_clean = dataset_name.replace(".hdf5", "")
            hdf5_path = get_full_dataset_path(data_location, dataset_name_clean)
            
            if not hdf5_path.exists():
                log.warning(f"HDF5 file not found: {hdf5_path}, skipping")
                continue
                
            # Use EmgRecording to load data properly
            from generic_neuromotor_interface.data import EmgRecording
            
            # Get partitions for this dataset from corpus
            dataset_rows = corpus_df[
                (corpus_df["split"] == "train") & (corpus_df["dataset"] == dataset_name)
            ]
            
            # If no corpus entries found, use full time range
            if len(dataset_rows) == 0:
                log.warning(f"No corpus entries for {dataset_name}, using full time range")
                dataset_rows = pd.DataFrame([{
                    "start": 0.0,
                    "end": float("inf"),
                    "dataset": dataset_name,
                    "split": "train"
                }])
            
            for _, row in dataset_rows.iterrows():
                start_time = float(row["start"])
                end_val = row["end"]
                # Handle inf, None, or NaN values - EmgRecording expects float, use np.inf for full range
                if pd.isna(end_val) or end_val == float("inf") or end_val is None:
                    end_time = float("inf")
                else:
                    end_time = float(end_val)
                
                with EmgRecording(hdf5_path, start_time=start_time, end_time=end_time) as recording:
                    if recording.prompts is None:
                        log.warning(f"No prompts in {dataset_name}, skipping")
                        continue
                    
                    # Get EMG data and prompts
                    timeseries = recording[:]  # Get all data in partition
                    emg_data = timeseries["emg"]  # (time, 16)
                    time_data = timeseries["time"]
                    
                    # Select channels
                    emg_selected = channel_selector(emg_data)  # (time, 7)
                    
                    # Get prompts within this partition
                    prompts = recording.prompts[
                        recording.prompts["time"].between(
                            time_data[0], time_data[-1]
                        )
                    ]
                    
                    # Extract windows around gesture events
                    window_size = 200
                    window_half = window_size // 2
                    # Sample rate is 2000 Hz, but we'll downsample to 200 Hz for windows
                    sample_rate = 200.0
                    downsample_factor = int(EMG_SAMPLE_RATE / sample_rate)
                    
                    for _, prompt_row in prompts.iterrows():
                        event_time = prompt_row["time"]
                        gesture_name = prompt_row["name"]
                        
                        # Map gesture name to class index
                        from generic_neuromotor_interface.constants import GestureType
                        try:
                            gesture_type = GestureType[gesture_name].value
                        except KeyError:
                            log.warning(f"Unknown gesture: {gesture_name}, skipping")
                            continue
                        
                        # Find event index in original data (at 2000 Hz)
                        event_idx_original = np.searchsorted(time_data, event_time)
                        
                        # Convert to downsampled index
                        event_idx = event_idx_original // downsample_factor
                        
                        # Extract window from downsampled data
                        # First downsample EMG
                        emg_downsampled = emg_selected[::downsample_factor]  # Downsample
                        
                        start_idx = max(0, event_idx - window_half)
                        end_idx = min(len(emg_downsampled), event_idx + window_half)
                        
                        if end_idx - start_idx >= window_size:
                            # Take exactly window_size samples
                            window = emg_downsampled[start_idx:start_idx + window_size]  # (200, 7)
                            # Transpose to (7, 200) for feature extraction
                            window = window.T  # (7, 200)
                            
                            X_samples.append(window)
                            y_samples.append(gesture_type)

        except Exception as e:
            log.warning(f"Error loading {dataset_name}: {e}")
            import traceback
            log.debug(traceback.format_exc())
            continue

    if len(X_samples) == 0:
        raise ValueError("No samples loaded! Check data paths and format.")

    X = np.array(X_samples)  # (n_samples, 7, 200)
    y = np.array(y_samples)  # (n_samples,)

    log.info(f"Loaded {len(X)} samples")
    if len(y) > 0:
        unique, counts = np.unique(y, return_counts=True)
        log.info(f"Class distribution: {dict(zip(unique, counts))}")

    return X, y


def main():
    parser = argparse.ArgumentParser(
        description="Train Random Forest model for discrete gesture recognition"
    )
    parser.add_argument(
        "--data-location",
        type=str,
        default="~/emg_data",
        help="Path to EMG data directory",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="discrete_gestures",
        help="Task name (default: discrete_gestures)",
    )
    parser.add_argument(
        "--channel-indices",
        type=int,
        nargs="+",
        default=[9, 8, 7, 6, 5, 13, 15],
        help="Channel indices to use (0-based)",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=200,
        help="Number of trees in Random Forest",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Maximum tree depth (None for unlimited)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./models",
        help="Directory to save trained model",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="rf_gesture_model",
        help="Name for saved model file",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    X, y = load_data(args.data_location, args.channel_indices, args.task)

    # Split into train/val (80/20)
    n_train = int(0.8 * len(X))
    indices = np.random.RandomState(42).permutation(len(X))
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    X_train, y_train = X[train_indices], y[train_indices]
    X_val, y_val = X[val_indices], y[val_indices]

    log.info(f"Training samples: {len(X_train)}")
    log.info(f"Validation samples: {len(X_val)}")

    # Initialize and train model
    model = RandomForestGestureModel(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        n_channels=len(args.channel_indices),
    )

    model.fit(X_train, y_train)

    # Evaluate on validation set
    y_pred = model.predict(X_val)
    accuracy = np.mean(y_pred == y_val)
    log.info(f"Validation Accuracy: {accuracy:.3f}")

    # Feature importance
    log.info("\nTop 10 Most Important Features:")
    for feat, importance in model.get_feature_importance(top_k=10):
        log.info(f"  {feat}: {importance:.4f}")

    # Channel importance
    log.info("\nChannel Importance:")
    channel_importance = model.get_channel_importance()
    channel_names = [
        "Ch5 (Thumb)",
        "Ch6 (Index)",
        "Ch7 (Middle)",
        "Ch8 (Ring)",
        "Ch9 (Pinky)",
        "Ch13 (Ext1)",
        "Ch15 (Ext2)",
    ]
    for i, (name, imp) in enumerate(zip(channel_names, channel_importance)):
        log.info(f"  {name}: {imp:.4f}")

    # Save model
    model_path = output_dir / f"{args.model_name}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    log.info(f"\nModel saved to {model_path}")

    log.info(f"\nEstimated inference time on Cortex-M4: {model.inference_time_estimate():.1f} ms")


if __name__ == "__main__":
    main()

