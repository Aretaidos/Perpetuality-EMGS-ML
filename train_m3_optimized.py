#!/usr/bin/env python3
"""
Optimized M3 Random Forest Training Script

KEY OPTIMIZATIONS:
1. Batch-wise feature extraction to prevent memory overflow
2. Progress tracking with tqdm
3. Incremental data loading with garbage collection
4. Memory-efficient numpy operations
5. Uses all CPU cores for parallel processing

Usage:
    python train_m3_optimized.py --data-dir /path/to/emg_data --output-dir ./models
"""

import os
import sys
import argparse
import logging
import gc
from pathlib import Path

import numpy as np
from tqdm import tqdm
import torch
from scipy import signal
from scipy.stats import kurtosis, skew
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from generic_neuromotor_interface.data import DataSplit
from generic_neuromotor_interface.data_module import WindowedEmgDataModule
from generic_neuromotor_interface.transforms_isolated import IsolatedDiscreteGesturesTransform

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger(__name__)


class BatchedFeatureExtractor:
    """
    Memory-efficient EMG feature extractor with batch processing.

    Extracts 18 features per channel:
    - 8 time-domain: MAV, RMS, WL, ZC, SSC, VAR, iEMG, DAMV
    - 6 frequency-domain: MNF, MDF, PKF, TotalPower, SpectralEntropy, FreqRatio
    - 4 higher-order: Skewness, Kurtosis, MaxAmplitude, CrestFactor
    """

    def __init__(self, fs: float = 2000.0, n_channels: int = 7):
        self.fs = fs
        self.n_channels = n_channels
        self.n_features_per_channel = 18

        # Design filters once
        nyq = fs / 2
        low = 20.0 / nyq
        high = 450.0 / nyq
        self.bp_b, self.bp_a = signal.butter(4, [low, high], btype='band')

        notch = 60.0 / nyq
        self.notch_b, self.notch_a = signal.iirnotch(notch, Q=30)

    def preprocess_batch(self, X: np.ndarray) -> np.ndarray:
        """Apply filtering to batch. X shape: (batch, channels, time)"""
        X_filtered = np.zeros_like(X)
        for i in range(X.shape[0]):
            for c in range(X.shape[1]):
                x = X[i, c, :]
                x_notched = signal.filtfilt(self.notch_b, self.notch_a, x)
                X_filtered[i, c, :] = signal.filtfilt(self.bp_b, self.bp_a, x_notched)
        return X_filtered

    def extract_single_window(self, x: np.ndarray) -> np.ndarray:
        """Extract 18 features from single-channel window."""
        eps = 1e-10
        x = np.asarray(x).flatten()

        # TIME DOMAIN (8 features)
        mav = np.mean(np.abs(x))
        rms = np.sqrt(np.mean(x**2) + eps)
        wl = np.sum(np.abs(np.diff(x)))

        threshold = 0.02 * np.max(np.abs(x) + eps)
        zc = np.sum((x[:-1] * x[1:] < 0) & (np.abs(x[:-1] - x[1:]) > threshold))

        dx = np.diff(x)
        ssc = np.sum((dx[:-1] * dx[1:] < 0) & (np.abs(dx[:-1] - dx[1:]) > threshold))

        var = np.var(x)
        iemg = np.sum(np.abs(x))
        damv = np.mean(np.abs(np.diff(x)))

        # FREQUENCY DOMAIN (6 features)
        nperseg = min(256, len(x))
        freqs, psd = signal.welch(x, fs=self.fs, nperseg=nperseg, noverlap=nperseg//2)

        mask = (freqs >= 20) & (freqs <= 450)
        freqs_valid = freqs[mask]
        psd_valid = psd[mask]

        if len(psd_valid) == 0 or np.sum(psd_valid) < eps:
            mnf, mdf, pkf, log_power, spectral_entropy, freq_ratio = 0, 0, 0, 0, 0, 0
        else:
            total_power = np.sum(psd_valid)
            psd_norm = psd_valid / (total_power + eps)

            mnf = np.sum(freqs_valid * psd_valid) / (total_power + eps)
            cumsum = np.cumsum(psd_valid)
            mdf_idx = np.searchsorted(cumsum, total_power / 2)
            mdf = freqs_valid[min(mdf_idx, len(freqs_valid)-1)]
            pkf = freqs_valid[np.argmax(psd_valid)]
            log_power = np.log10(total_power + eps)

            psd_norm_safe = np.clip(psd_norm, eps, None)
            spectral_entropy = -np.sum(psd_norm_safe * np.log2(psd_norm_safe))

            low_mask = freqs_valid < 100
            high_mask = freqs_valid >= 100
            low_power = np.sum(psd_valid[low_mask]) + eps
            high_power = np.sum(psd_valid[high_mask]) + eps
            freq_ratio = low_power / high_power

        # HIGHER ORDER (4 features)
        sk = skew(x)
        kt = kurtosis(x)
        max_amp = np.max(np.abs(x))
        crest = max_amp / (rms + eps)

        return np.array([
            mav, rms, wl, zc, ssc, var, iemg, damv,
            mnf, mdf, pkf, log_power, spectral_entropy, freq_ratio,
            sk, kt, max_amp, crest
        ])

    def extract_batch(self, X: np.ndarray, preprocess: bool = True) -> np.ndarray:
        """
        Extract features from batch.

        Parameters
        ----------
        X : np.ndarray
            Shape (n_samples, n_channels, window_length)

        Returns
        -------
        np.ndarray
            Shape (n_samples, n_channels * 18)
        """
        n_samples = X.shape[0]
        n_features = self.n_channels * self.n_features_per_channel

        features = np.zeros((n_samples, n_features), dtype=np.float32)

        if preprocess:
            X = self.preprocess_batch(X)

        for i in range(n_samples):
            for c in range(self.n_channels):
                feats = self.extract_single_window(X[i, c, :])
                start_idx = c * self.n_features_per_channel
                end_idx = start_idx + self.n_features_per_channel
                features[i, start_idx:end_idx] = feats

        return features


def main():
    parser = argparse.ArgumentParser(description='Optimized M3 Random Forest Training')
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='./models')
    parser.add_argument('--batch-size', type=int, default=5000,
                        help='Batch size for feature extraction (larger = faster but more memory)')
    parser.add_argument('--n-estimators', type=int, default=300)
    parser.add_argument('--max-depth', type=int, default=20)

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("M3 Random Forest Training - OPTIMIZED")
    log.info("=" * 60)
    log.info(f"Data directory: {args.data_dir}")
    log.info(f"Output directory: {args.output_dir}")
    log.info(f"Batch size for feature extraction: {args.batch_size}")
    log.info(f"Number of trees: {args.n_estimators}")
    log.info(f"Max depth: {args.max_depth}")

    # Initialize feature extractor
    extractor = BatchedFeatureExtractor(fs=2000.0, n_channels=7)

    # Load data module
    log.info("Setting up data module...")

    data_split = DataSplit.from_csv(
        csv_filename=os.path.join(args.data_dir, 'discrete_gestures_corpus.csv'),
        pool_test_partitions=True
    )

    transform = IsolatedDiscreteGesturesTransform(
        pulse_window=[0.0, 0.04],
        channel_indices=[4, 5, 6, 7, 8, 12, 14],
    )

    data_module = WindowedEmgDataModule(
        data_location=args.data_dir,
        data_split=data_split,
        window_length=400,  # 200ms at 2kHz
        stride=200,         # 100ms stride
        batch_size=args.batch_size,
        num_workers=4,
        transform=transform,
    )

    data_module.setup()
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    n_train_batches = len(train_loader)
    n_val_batches = len(val_loader)

    log.info(f"Training batches: {n_train_batches}")
    log.info(f"Validation batches: {n_val_batches}")

    # Extract training features in batches with progress tracking
    log.info("Extracting training features (batched with progress)...")

    all_train_features = []
    all_train_labels = []

    for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training features")):
        emg = batch['emg'].numpy()  # (B, 7, T)
        targets = batch['targets'].numpy()  # (B, 9, T)

        # Get most activated gesture per sample
        gesture_labels = np.argmax(targets.sum(axis=2), axis=1)

        # Extract features for this batch
        features = extractor.extract_batch(emg, preprocess=True)

        all_train_features.append(features)
        all_train_labels.extend(gesture_labels)

        # Garbage collect periodically to prevent memory buildup
        if batch_idx % 20 == 0:
            gc.collect()

    X_train = np.concatenate(all_train_features, axis=0)
    y_train = np.array(all_train_labels)

    log.info(f"Training feature matrix: {X_train.shape}")
    log.info(f"Training labels: {y_train.shape}")

    # Clear memory
    del all_train_features
    gc.collect()

    # Extract validation features
    log.info("Extracting validation features...")

    all_val_features = []
    all_val_labels = []

    for batch_idx, batch in enumerate(tqdm(val_loader, desc="Validation features")):
        emg = batch['emg'].numpy()
        targets = batch['targets'].numpy()
        gesture_labels = np.argmax(targets.sum(axis=2), axis=1)

        features = extractor.extract_batch(emg, preprocess=True)

        all_val_features.append(features)
        all_val_labels.extend(gesture_labels)

    X_val = np.concatenate(all_val_features, axis=0)
    y_val = np.array(all_val_labels)

    log.info(f"Validation feature matrix: {X_val.shape}")

    del all_val_features
    gc.collect()

    # Handle non-finite values
    log.info("Cleaning feature matrices...")
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)

    # Normalize features
    log.info("Normalizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Create Random Forest classifier
    log.info("Creating Random Forest classifier...")
    clf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        bootstrap=True,
        oob_score=True,
        n_jobs=-1,  # Use all CPU cores
        random_state=42,
        class_weight='balanced',
        verbose=1,
    )

    # Cross-validation (3-fold for speed)
    log.info("Running 3-fold stratified cross-validation...")
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf, X_train_scaled, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
    log.info(f"CV Accuracy: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    # Train on full dataset
    log.info("Training on full dataset...")
    clf.fit(X_train_scaled, y_train)

    # Evaluate
    train_acc = accuracy_score(y_train, clf.predict(X_train_scaled))
    val_acc = accuracy_score(y_val, clf.predict(X_val_scaled))

    log.info(f"Training Accuracy: {train_acc:.4f}")
    log.info(f"Validation Accuracy: {val_acc:.4f}")
    log.info(f"OOB Score: {clf.oob_score_:.4f}")

    # Classification report
    y_val_pred = clf.predict(X_val_scaled)
    log.info("\nClassification Report:")
    log.info(classification_report(y_val, y_val_pred))

    # Confusion matrix
    cm = confusion_matrix(y_val, y_val_pred)
    log.info("\nConfusion Matrix:")
    log.info(str(cm))

    # Save model
    model_data = {
        'clf': clf,
        'scaler': scaler,
        'cv_scores': cv_scores,
        'train_accuracy': train_acc,
        'val_accuracy': val_acc,
        'oob_score': clf.oob_score_,
        'confusion_matrix': cm,
        'feature_importances': clf.feature_importances_,
    }

    model_path = output_dir / 'm3_best.pkl'
    joblib.dump(model_data, model_path)
    log.info(f"\nModel saved to {model_path}")

    # Feature importance analysis
    log.info("\nTop 20 Most Important Features:")
    feature_names = []
    td_names = ['MAV', 'RMS', 'WL', 'ZC', 'SSC', 'VAR', 'iEMG', 'DAMV']
    fd_names = ['MNF', 'MDF', 'PKF', 'LogPower', 'SpectralEntropy', 'FreqRatio']
    ho_names = ['Skewness', 'Kurtosis', 'MaxAmp', 'CrestFactor']
    all_feat_names = td_names + fd_names + ho_names

    for c in range(7):
        for feat in all_feat_names:
            feature_names.append(f'Ch{c+5}_{feat}')

    importance_pairs = list(zip(feature_names, clf.feature_importances_))
    importance_pairs.sort(key=lambda x: x[1], reverse=True)

    for name, imp in importance_pairs[:20]:
        log.info(f"  {name}: {imp:.4f}")

    log.info("\n" + "=" * 60)
    log.info("M3 TRAINING COMPLETE")
    log.info("=" * 60)

    return {
        'cv_accuracy': cv_scores.mean(),
        'val_accuracy': val_acc,
        'model_path': str(model_path)
    }


if __name__ == '__main__':
    main()
