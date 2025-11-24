# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Random Forest model with engineered features for discrete gesture recognition."""

import logging
from typing import Any

import numpy as np
import torch
from scipy import signal
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)


class EMGFeatureExtractor:
    """
    Extract comprehensive time-domain and frequency-domain features
    from multi-channel EMG signals.

    Parameters
    ----------
    fs : float
        Sampling frequency (Hz), default: 200.0
    window_size : int
        Number of samples per window, default: 200
    """

    def __init__(self, fs: float = 200.0, window_size: int = 200) -> None:
        self.fs = fs
        self.window_size = window_size

    def time_domain_features(self, x: np.ndarray) -> np.ndarray:
        """
        Extract TD features from single channel.

        Parameters
        ----------
        x : np.ndarray
            EMG signal of shape (window_size,)

        Returns
        -------
        np.ndarray
            TD feature vector of shape (6,)
        """
        # Mean Absolute Value
        mav = np.mean(np.abs(x))

        # Root Mean Square
        rms = np.sqrt(np.mean(x**2))

        # Zero Crossings
        threshold = 0.01 * np.max(np.abs(x))
        zc = np.sum(np.diff(np.sign(x)) != 0) if threshold > 0 else 0

        # Slope Sign Changes
        ssc = 0
        for i in range(1, len(x) - 1):
            if ((x[i] - x[i - 1]) * (x[i] - x[i + 1])) > 1e-4:
                ssc += 1

        # Waveform Length
        wl = np.sum(np.abs(np.diff(x)))

        # Integrated EMG
        iemg = np.sum(np.abs(x))

        return np.array([mav, rms, zc, ssc, wl, iemg])

    def frequency_domain_features(self, x: np.ndarray) -> np.ndarray:
        """
        Extract FD features from single channel.

        Parameters
        ----------
        x : np.ndarray
            EMG signal of shape (window_size,)

        Returns
        -------
        np.ndarray
            FD feature vector of shape (4,)
        """
        # Power Spectral Density using Welch's method
        freqs, psd = signal.welch(x, fs=self.fs, nperseg=min(256, len(x)))

        # Total power
        total_power = np.sum(psd)

        if total_power < 1e-10:  # Avoid division by zero
            return np.array([0.0, 0.0, 0.0, 0.0])

        # Mean Frequency
        mean_freq = np.sum(freqs * psd) / total_power

        # Median Frequency
        cumsum_psd = np.cumsum(psd)
        median_idx = np.where(cumsum_psd >= total_power / 2)[0]
        median_freq = freqs[median_idx[0]] if len(median_idx) > 0 else 0.0

        # Peak Frequency
        peak_freq = freqs[np.argmax(psd)]

        return np.array([mean_freq, median_freq, peak_freq, total_power])

    def extract_features(self, X: np.ndarray) -> np.ndarray:
        """
        Extract features from multi-channel EMG batch.

        Parameters
        ----------
        X : np.ndarray
            EMG data of shape (n_samples, n_channels, window_size)

        Returns
        -------
        np.ndarray
            Feature matrix of shape (n_samples, n_channels * 10)
        """
        n_samples, n_channels, _ = X.shape
        features = np.zeros((n_samples, n_channels * 10))

        for i in range(n_samples):
            for c in range(n_channels):
                td_feats = self.time_domain_features(X[i, c, :])
                fd_feats = self.frequency_domain_features(X[i, c, :])
                features[i, c * 10 : (c * 10 + 6)] = td_feats
                features[i, c * 10 + 6 : c * 10 + 10] = fd_feats

        return features


class RandomForestGestureModel:
    """
    Random Forest classifier with engineered features for discrete gesture recognition.
    Classical ML approach optimized for embedded systems or as baseline.

    Parameters
    ----------
    n_estimators : int
        Number of trees in forest, default: 200
    max_depth : int | None
        Maximum tree depth (None = unlimited), default: None
    min_samples_split : int
        Minimum samples to split node, default: 5
    n_channels : int
        Number of EMG channels, default: 7
    fs : float
        Sampling frequency, default: 200.0
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int | None = None,
        min_samples_split: int = 5,
        n_channels: int = 7,
        fs: float = 200.0,
    ) -> None:
        self.n_channels = n_channels
        self.feature_extractor = EMGFeatureExtractor(fs=fs, window_size=200)

        # Random Forest classifier
        self.clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=2,
            max_features="sqrt",  # sqrt(70) ≈ 8 features per split
            bootstrap=True,
            n_jobs=-1,
            random_state=42,
            class_weight="balanced",  # Handle class imbalance
        )

        # Feature scaler
        self.scaler = StandardScaler()

        self.is_fitted = False
        self.feature_importance: dict[str, float] | None = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train the Random Forest model.

        Parameters
        ----------
        X_train : np.ndarray
            EMG training data of shape (n_samples, n_channels, window_size)
        y_train : np.ndarray
            Gesture labels of shape (n_samples,) with values [0-8]
        """
        # Extract features
        log.info("Extracting training features...")
        X_features = self.feature_extractor.extract_features(X_train)

        # Normalize features
        X_scaled = self.scaler.fit_transform(X_features)

        # Train Random Forest
        log.info(f"Training Random Forest with {self.clf.n_estimators} trees...")
        self.clf.fit(X_scaled, y_train)

        self.is_fitted = True

        # Feature importance analysis
        feature_names = []
        for c in range(self.n_channels):
            for feat in [
                "MAV",
                "RMS",
                "ZC",
                "SSC",
                "WL",
                "iEMG",
                "MNF",
                "MDF",
                "PeakFreq",
                "TotalPower",
            ]:
                feature_names.append(f"Ch{c+1}_{feat}")

        self.feature_importance = dict(
            zip(feature_names, self.clf.feature_importances_)
        )

        # Cross-validation score
        cv_scores = cross_val_score(self.clf, X_scaled, y_train, cv=5, n_jobs=-1)
        log.info(
            f"5-Fold CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}"
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict gesture classes.

        Parameters
        ----------
        X : np.ndarray
            EMG test data of shape (n_samples, n_channels, window_size)

        Returns
        -------
        np.ndarray
            Predicted class indices of shape (n_samples,)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        # Extract and scale features
        X_features = self.feature_extractor.extract_features(X)
        X_scaled = self.scaler.transform(X_features)

        # Predict
        predictions = self.clf.predict(X_scaled)
        return predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Parameters
        ----------
        X : np.ndarray
            EMG test data of shape (n_samples, n_channels, window_size)

        Returns
        -------
        np.ndarray
            Class probabilities of shape (n_samples, 9)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        X_features = self.feature_extractor.extract_features(X)
        X_scaled = self.scaler.transform(X_features)

        probabilities = self.clf.predict_proba(X_scaled)
        return probabilities

    def get_feature_importance(self, top_k: int = 20) -> list[tuple[str, float]]:
        """
        Return top-k most important features.

        Parameters
        ----------
        top_k : int
            Number of top features to return, default: 20

        Returns
        -------
        list[tuple[str, float]]
            List of (feature_name, importance) tuples
        """
        if not self.is_fitted or self.feature_importance is None:
            raise ValueError("Model must be fitted first")

        sorted_features = sorted(
            self.feature_importance.items(), key=lambda x: x[1], reverse=True
        )
        return sorted_features[:top_k]

    def get_channel_importance(self) -> np.ndarray:
        """
        Aggregate feature importance by channel.

        Returns
        -------
        np.ndarray
            Channel importance array of shape (n_channels,)
        """
        if not self.is_fitted or self.feature_importance is None:
            raise ValueError("Model must be fitted first")

        channel_importance = np.zeros(self.n_channels)

        for feat_name, importance in self.feature_importance.items():
            # Extract channel number from feature name (e.g., "Ch5_MAV" -> 4)
            channel_idx = int(feat_name.split("_")[0][2:]) - 1
            if 0 <= channel_idx < self.n_channels:
                channel_importance[channel_idx] += importance

        return channel_importance

    def inference_time_estimate(self) -> float:
        """
        Estimate inference time for embedded deployment.

        Returns
        -------
        float
            Estimated inference time in milliseconds
        """
        # Feature extraction: ~2ms (10 features * 7 channels, simple operations)
        # Tree traversal: ~3ms (200 trees * ~15 µs per tree on M4)
        return 5.0  # milliseconds

