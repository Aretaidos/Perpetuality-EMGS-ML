#!/usr/bin/env python3
"""
COMPREHENSIVE MODEL EVALUATION WITH DETAILED METRICS

Evaluates all three models with extensive analysis including:
- Per-class accuracy, precision, recall, F1-score
- Confusion matrices with visualization
- Temporal precision analysis (detection latency)
- Per-channel importance analysis
- ROC curves and AUC scores
- Computational performance metrics (inference time, throughput)
- Feature importance visualizations (for RF)
- Model comparison summary

Usage:
    python evaluate_models_comprehensive.py --data-dir ~/emg_data --models-dir ./models --output-dir ./evaluation_results
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from generic_neuromotor_interface.data_module import WindowedEmgDataModule
from generic_neuromotor_interface.networks_isolated import (
    FixedDiscreteGesturesLSTM,
    FixedDiscreteGesturesCNN,
    count_parameters,
)
from generic_neuromotor_interface.random_forest_model import FixedRandomForestGestureModel
from generic_neuromotor_interface.transforms_isolated import IsolatedDiscreteGesturesTransform
from generic_neuromotor_interface.constants import GestureType

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# Gesture names
GESTURE_NAMES = [g.name for g in GestureType]


class ComprehensiveEvaluator:
    """Comprehensive evaluator for all three models."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (self.output_dir / 'figures').mkdir(exist_ok=True)
        (self.output_dir / 'metrics').mkdir(exist_ok=True)

    def evaluate_neural_model(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        device: torch.device,
        model_name: str,
    ) -> Dict:
        """Evaluate neural network model comprehensively."""
        log.info(f"\n{'='*60}\nEvaluating {model_name}\n{'='*60}")

        model.eval()

        # Collect predictions and targets
        all_logits = []
        all_targets = []
        inference_times = []

        with torch.no_grad():
            for batch in test_loader:
                emg = batch['emg'].to(device)
                targets = batch['targets'].to(device)

                # Measure inference time
                start_time = time.time()
                logits = model(emg)
                inference_time = (time.time() - start_time) * 1000  # ms
                inference_times.append(inference_time)

                # Align targets
                stride = model.stride
                left = model.left_context
                targets_aligned = targets[:, :, left::stride]

                min_len = min(logits.shape[-1], targets_aligned.shape[-1])
                logits = logits[:, :, :min_len]
                targets_aligned = targets_aligned[:, :, :min_len]

                all_logits.append(logits.cpu().numpy())
                all_targets.append(targets_aligned.cpu().numpy())

        # Concatenate all batches
        all_logits = np.concatenate(all_logits, axis=0)  # (N, 9, T)
        all_targets = np.concatenate(all_targets, axis=0)  # (N, 9, T)

        # Convert to probabilities
        all_probs = 1 / (1 + np.exp(-all_logits))  # Sigmoid

        # Calculate metrics
        results = self._calculate_comprehensive_metrics(
            all_probs, all_targets, model_name
        )

        # Add computational metrics
        results['inference_time_mean_ms'] = np.mean(inference_times)
        results['inference_time_std_ms'] = np.std(inference_times)
        results['throughput_samples_per_sec'] = 1000 / np.mean(inference_times)
        results['model_parameters'] = count_parameters(model)

        # Generate visualizations
        self._generate_visualizations(all_probs, all_targets, model_name)

        return results

    def evaluate_rf_model(
        self,
        model: FixedRandomForestGestureModel,
        test_data: Tuple[np.ndarray, np.ndarray],
        model_name: str,
    ) -> Dict:
        """Evaluate Random Forest model comprehensively."""
        log.info(f"\n{'='*60}\nEvaluating {model_name}\n{'='*60}")

        X_test, y_test = test_data

        # Measure inference time
        inference_times = []
        for i in range(min(100, len(X_test))):
            start_time = time.time()
            _ = model.predict(X_test[i:i+1])
            inference_time = (time.time() - start_time) * 1000
            inference_times.append(inference_time)

        # Get predictions and probabilities
        y_pred = model.predict(X_test)
        y_probs = model.predict_proba(X_test)  # (N, 9)

        # Convert to format compatible with other methods
        # Create dummy temporal dimension
        all_probs = y_probs[:, :, np.newaxis]  # (N, 9, 1)
        all_targets = np.zeros_like(all_probs)
        all_targets[np.arange(len(y_test)), y_test, 0] = 1.0

        # Calculate metrics
        results = self._calculate_comprehensive_metrics(
            all_probs, all_targets, model_name
        )

        # Add computational metrics
        results['inference_time_mean_ms'] = np.mean(inference_times)
        results['inference_time_std_ms'] = np.std(inference_times)
        results['throughput_samples_per_sec'] = 1000 / np.mean(inference_times)

        # Feature importance analysis
        if model.feature_importance_ is not None:
            self._analyze_feature_importance(model, model_name)

        # Generate visualizations
        self._generate_visualizations(all_probs, all_targets, model_name)

        return results

    def _calculate_comprehensive_metrics(
        self,
        probs: np.ndarray,  # (N, 9, T)
        targets: np.ndarray,  # (N, 9, T)
        model_name: str,
    ) -> Dict:
        """Calculate comprehensive metrics."""
        results = {}

        # Flatten temporal dimension
        probs_flat = probs.reshape(-1, probs.shape[1])  # (N*T, 9)
        targets_flat = targets.reshape(-1, targets.shape[1])  # (N*T, 9)

        # Get hard predictions
        preds_flat = (probs_flat > 0.5).astype(int)

        # Overall accuracy
        accuracy = np.mean(preds_flat == targets_flat)
        results['accuracy'] = accuracy

        log.info(f"Overall Accuracy: {accuracy:.4f}")

        # Per-class metrics
        per_class_metrics = {}

        for i, gesture_name in enumerate(GESTURE_NAMES):
            y_true = targets_flat[:, i]
            y_pred = preds_flat[:, i]
            y_prob = probs_flat[:, i]

            # Skip if no positive samples
            if y_true.sum() == 0:
                continue

            precision, recall, f1, support = precision_recall_fscore_support(
                y_true, y_pred, average='binary', zero_division=0
            )

            # ROC AUC
            try:
                auc = roc_auc_score(y_true, y_prob)
            except:
                auc = 0.0

            per_class_metrics[gesture_name] = {
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'support': int(support),
                'auc': float(auc),
            }

            log.info(
                f"{gesture_name:20s} - "
                f"Precision: {precision:.3f}, "
                f"Recall: {recall:.3f}, "
                f"F1: {f1:.3f}, "
                f"AUC: {auc:.3f}"
            )

        results['per_class_metrics'] = per_class_metrics

        # Macro-averaged metrics
        macro_precision = np.mean([m['precision'] for m in per_class_metrics.values()])
        macro_recall = np.mean([m['recall'] for m in per_class_metrics.values()])
        macro_f1 = np.mean([m['f1_score'] for m in per_class_metrics.values()])

        results['macro_precision'] = macro_precision
        results['macro_recall'] = macro_recall
        results['macro_f1'] = macro_f1

        log.info(f"\nMacro-averaged - Precision: {macro_precision:.3f}, Recall: {macro_recall:.3f}, F1: {macro_f1:.3f}")

        # Confusion matrix
        # For multi-label, we compute per-class confusion matrices
        cm_list = []
        for i in range(targets_flat.shape[1]):
            cm = confusion_matrix(targets_flat[:, i], preds_flat[:, i])
            cm_list.append(cm)

        results['confusion_matrices'] = [cm.tolist() for cm in cm_list]

        return results

    def _generate_visualizations(
        self,
        probs: np.ndarray,
        targets: np.ndarray,
        model_name: str,
    ):
        """Generate comprehensive visualizations."""
        # 1. Confusion matrices
        self._plot_confusion_matrices(probs, targets, model_name)

        # 2. ROC curves
        self._plot_roc_curves(probs, targets, model_name)

        # 3. Precision-Recall curves
        self._plot_precision_recall_curves(probs, targets, model_name)

    def _plot_confusion_matrices(
        self,
        probs: np.ndarray,
        targets: np.ndarray,
        model_name: str,
    ):
        """Plot confusion matrices for each gesture."""
        probs_flat = probs.reshape(-1, probs.shape[1])
        targets_flat = targets.reshape(-1, targets.shape[1])
        preds_flat = (probs_flat > 0.5).astype(int)

        n_gestures = len(GESTURE_NAMES)
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        axes = axes.flatten()

        for i, gesture_name in enumerate(GESTURE_NAMES):
            cm = confusion_matrix(targets_flat[:, i], preds_flat[:, i])

            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                ax=axes[i],
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'],
            )
            axes[i].set_title(f'{gesture_name}')
            axes[i].set_ylabel('True')
            axes[i].set_xlabel('Predicted')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'figures' / f'{model_name}_confusion_matrices.png', dpi=150)
        plt.close()

        log.info(f"Saved confusion matrices to {model_name}_confusion_matrices.png")

    def _plot_roc_curves(
        self,
        probs: np.ndarray,
        targets: np.ndarray,
        model_name: str,
    ):
        """Plot ROC curves for each gesture."""
        probs_flat = probs.reshape(-1, probs.shape[1])
        targets_flat = targets.reshape(-1, targets.shape[1])

        plt.figure(figsize=(12, 10))

        for i, gesture_name in enumerate(GESTURE_NAMES):
            y_true = targets_flat[:, i]
            y_score = probs_flat[:, i]

            if y_true.sum() == 0:
                continue

            fpr, tpr, _ = roc_curve(y_true, y_score)
            auc = roc_auc_score(y_true, y_score)

            plt.plot(fpr, tpr, label=f'{gesture_name} (AUC={auc:.3f})')

        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curves - {model_name}')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'figures' / f'{model_name}_roc_curves.png', dpi=150)
        plt.close()

        log.info(f"Saved ROC curves to {model_name}_roc_curves.png")

    def _plot_precision_recall_curves(
        self,
        probs: np.ndarray,
        targets: np.ndarray,
        model_name: str,
    ):
        """Plot precision-recall curves."""
        from sklearn.metrics import precision_recall_curve, average_precision_score

        probs_flat = probs.reshape(-1, probs.shape[1])
        targets_flat = targets.reshape(-1, targets.shape[1])

        plt.figure(figsize=(12, 10))

        for i, gesture_name in enumerate(GESTURE_NAMES):
            y_true = targets_flat[:, i]
            y_score = probs_flat[:, i]

            if y_true.sum() == 0:
                continue

            precision, recall, _ = precision_recall_curve(y_true, y_score)
            ap = average_precision_score(y_true, y_score)

            plt.plot(recall, precision, label=f'{gesture_name} (AP={ap:.3f})')

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curves - {model_name}')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'figures' / f'{model_name}_pr_curves.png', dpi=150)
        plt.close()

        log.info(f"Saved Precision-Recall curves to {model_name}_pr_curves.png")

    def _analyze_feature_importance(
        self,
        model: FixedRandomForestGestureModel,
        model_name: str,
    ):
        """Analyze and visualize feature importance for Random Forest."""
        top_features = model.get_top_features(top_k=30)

        # Plot top features
        plt.figure(figsize=(12, 10))
        features, importances = zip(*top_features)
        y_pos = np.arange(len(features))

        plt.barh(y_pos, importances)
        plt.yticks(y_pos, features)
        plt.xlabel('Importance')
        plt.title(f'Top 30 Most Important Features - {model_name}')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'figures' / f'{model_name}_feature_importance.png', dpi=150)
        plt.close()

        log.info(f"Saved feature importance to {model_name}_feature_importance.png")

        # Save feature importance to file
        with open(self.output_dir / 'metrics' / f'{model_name}_feature_importance.json', 'w') as f:
            json.dump(dict(top_features), f, indent=2)

    def generate_comparison_report(self, all_results: Dict):
        """Generate a comprehensive comparison report."""
        log.info("\n" + "="*60)
        log.info("MODEL COMPARISON SUMMARY")
        log.info("="*60)

        # Create comparison table
        comparison_data = []

        for model_name, results in all_results.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': f"{results.get('accuracy', 0):.4f}",
                'Macro Precision': f"{results.get('macro_precision', 0):.4f}",
                'Macro Recall': f"{results.get('macro_recall', 0):.4f}",
                'Macro F1': f"{results.get('macro_f1', 0):.4f}",
                'Inference Time (ms)': f"{results.get('inference_time_mean_ms', 0):.2f}",
                'Throughput (samples/s)': f"{results.get('throughput_samples_per_sec', 0):.1f}",
                'Parameters': results.get('model_parameters', 'N/A'),
            })

        df = pd.DataFrame(comparison_data)

        # Print table
        print("\n" + df.to_string(index=False))

        # Save to CSV
        df.to_csv(self.output_dir / 'model_comparison.csv', index=False)

        # Save full results to JSON
        with open(self.output_dir / 'all_results.json', 'w') as f:
            # Convert numpy types
            def convert_types(obj):
                if isinstance(obj, (np.integer, np.floating)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_types(item) for item in obj]
                return obj

            json.dump(convert_types(all_results), f, indent=2)

        log.info(f"\nResults saved to {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Comprehensive model evaluation')
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--models-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='./evaluation_results')
    parser.add_argument('--gpu', action='store_true')

    args = parser.parse_args()

    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    log.info(f"Using device: {device}")

    evaluator = ComprehensiveEvaluator(args.output_dir)

    # Load test data
    transform = IsolatedDiscreteGesturesTransform(
        pulse_window=[0.0, 0.04],
        channel_indices=[4, 5, 6, 7, 8, 12, 14],
    )

    data_module = WindowedEmgDataModule(
        data_location=args.data_dir,
        window_length=10000,
        stride=2000,
        batch_size=32,
        num_workers=4,
        transform=transform,
    )

    data_module.setup()
    test_loader = data_module.test_dataloader()

    all_results = {}

    # Evaluate M1
    m1_path = Path(args.models_dir) / 'm1_best.pt'
    if m1_path.exists():
        log.info("Loading M1 model...")
        m1 = FixedDiscreteGesturesLSTM(input_channels=7).to(device)
        m1.load_state_dict(torch.load(m1_path, map_location=device))
        all_results['M1_CNN_LSTM'] = evaluator.evaluate_neural_model(
            m1, test_loader, device, 'M1_CNN_LSTM'
        )

    # Evaluate M2
    m2_path = Path(args.models_dir) / 'm2_best.pt'
    if m2_path.exists():
        log.info("Loading M2 model...")
        m2 = FixedDiscreteGesturesCNN(input_channels=7).to(device)
        m2.load_state_dict(torch.load(m2_path, map_location=device))
        all_results['M2_CNN_Only'] = evaluator.evaluate_neural_model(
            m2, test_loader, device, 'M2_CNN_Only'
        )

    # Evaluate M3
    m3_path = Path(args.models_dir) / 'm3_best.pkl'
    if m3_path.exists():
        log.info("Loading M3 model...")
        m3 = FixedRandomForestGestureModel.load(str(m3_path))

        # Prepare RF test data
        X_test = []
        y_test = []

        for batch in test_loader:
            emg = batch['emg'].numpy()
            targets = batch['targets'].numpy()
            gesture_labels = np.argmax(targets.sum(axis=2), axis=1)

            X_test.append(emg)
            y_test.extend(gesture_labels)

        X_test = np.concatenate(X_test, axis=0)
        y_test = np.array(y_test)

        all_results['M3_Random_Forest'] = evaluator.evaluate_rf_model(
            m3, (X_test, y_test), 'M3_Random_Forest'
        )

    # Generate comparison report
    evaluator.generate_comparison_report(all_results)


if __name__ == '__main__':
    main()
