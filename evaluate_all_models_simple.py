#!/usr/bin/env python3
"""
Fixed comprehensive model evaluation script.
Handles variable-length outputs and GPU tensor issues.
"""

import os
import sys
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
)
import joblib

sys.path.insert(0, str(Path(__file__).parent))

from generic_neuromotor_interface.data_module import WindowedEmgDataModule
from generic_neuromotor_interface.data import DataSplit
from generic_neuromotor_interface.networks_isolated import (
    FixedDiscreteGesturesLSTM,
    FixedDiscreteGesturesCNN,
    count_parameters,
)
from generic_neuromotor_interface.transforms_isolated import IsolatedDiscreteGesturesTransform
from generic_neuromotor_interface.constants import GestureType

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
log = logging.getLogger(__name__)

GESTURE_NAMES = [g.name for g in GestureType]


def evaluate_neural_model(model, test_loader, device, model_name, output_dir):
    """Evaluate a neural network model."""
    log.info(f"\n{'='*60}\nEvaluating {model_name}\n{'='*60}")

    model.eval()

    all_probs = []
    all_targets = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            emg = batch['emg'].to(device).contiguous()  # Fix: ensure contiguous
            targets = batch['targets'].to(device)

            logits = model(emg)
            probs = torch.sigmoid(logits)

            # Align targets
            stride = model.stride
            left = model.left_context
            targets_aligned = targets[:, :, left::stride]

            min_len = min(probs.shape[-1], targets_aligned.shape[-1])
            probs = probs[:, :, :min_len]
            targets_aligned = targets_aligned[:, :, :min_len]

            # Flatten and append immediately (fixes variable-length issue)
            probs_flat = probs.cpu().numpy().reshape(-1, 9)
            targets_flat = targets_aligned.cpu().numpy().reshape(-1, 9)

            all_probs.append(probs_flat)
            all_targets.append(targets_flat)

            if batch_idx % 50 == 0:
                log.info(f"  Processed batch {batch_idx}")

    all_probs = np.concatenate(all_probs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    log.info(f"Total samples: {len(all_probs)}")

    # Calculate metrics
    results = calculate_metrics(all_probs, all_targets, model_name, output_dir)
    results['model_parameters'] = count_parameters(model)

    return results


def evaluate_rf_model(model_path, output_dir):
    """Evaluate Random Forest model."""
    log.info(f"\n{'='*60}\nEvaluating M3_Random_Forest\n{'='*60}")

    # Load model
    model_data = joblib.load(model_path)

    # Get confusion matrix and compute per-class metrics
    cm = model_data.get('confusion_matrix')
    if cm is not None:
        if not isinstance(cm, np.ndarray):
            cm = np.array(cm)

        # Compute per-class precision, recall, F1 from confusion matrix
        n_classes = cm.shape[0]
        precisions = []
        recalls = []
        f1s = []

        per_class = {}
        for i in range(n_classes):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

            gesture_name = GESTURE_NAMES[i] if i < len(GESTURE_NAMES) else f"class_{i}"
            per_class[gesture_name] = {
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'support': int(cm[i, :].sum()),
            }
            log.info(f"{gesture_name:20s} - P: {precision:.3f}, R: {recall:.3f}, F1: {f1:.3f}")

        macro_precision = np.mean(precisions)
        macro_recall = np.mean(recalls)
        macro_f1 = np.mean(f1s)
    else:
        macro_precision = 0.0
        macro_recall = 0.0
        macro_f1 = 0.0
        per_class = {}

    # Get results from training
    results = {
        'accuracy': float(model_data.get('val_accuracy', 0)),
        'cv_accuracy': float(np.mean(model_data.get('cv_scores', [0]))),
        'oob_score': float(model_data.get('oob_score', 0)),
        'macro_precision': float(macro_precision),
        'macro_recall': float(macro_recall),
        'macro_f1': float(macro_f1),
        'macro_auc': 0.0,  # Not available for RF without probabilities
        'per_class': per_class,
    }

    log.info(f"\nMacro - P: {macro_precision:.3f}, R: {macro_recall:.3f}, F1: {macro_f1:.3f}")
    log.info(f"RF Validation Accuracy: {results['accuracy']:.4f}")
    log.info(f"RF CV Accuracy: {results['cv_accuracy']:.4f}")
    log.info(f"RF OOB Score: {results['oob_score']:.4f}")

    # Plot confusion matrix
    cm = model_data.get('confusion_matrix')
    if cm is not None:
        if isinstance(cm, np.ndarray):
            cm = cm
        else:
            cm = np.array(cm)

        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=GESTURE_NAMES, yticklabels=GESTURE_NAMES)
        plt.title('M3 Random Forest - Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(output_dir / 'figures' / 'M3_Random_Forest_confusion_matrix.png', dpi=150)
        plt.close()
        log.info("Saved RF confusion matrix")

    # Plot feature importance
    if 'feature_importances' in model_data:
        importances = model_data['feature_importances']

        # Create feature names
        td_names = ['MAV', 'RMS', 'WL', 'ZC', 'SSC', 'VAR', 'iEMG', 'DAMV']
        fd_names = ['MNF', 'MDF', 'PKF', 'LogPower', 'SpectralEntropy', 'FreqRatio']
        ho_names = ['Skewness', 'Kurtosis', 'MaxAmp', 'CrestFactor']
        all_feat_names = td_names + fd_names + ho_names

        feature_names = []
        for c in range(7):
            for feat in all_feat_names:
                feature_names.append(f'Ch{c+5}_{feat}')

        # Sort and plot top 30
        indices = np.argsort(importances)[-30:][::-1]
        top_names = [feature_names[i] for i in indices]
        top_importances = importances[indices]

        plt.figure(figsize=(12, 10))
        plt.barh(range(len(top_names)), top_importances[::-1])
        plt.yticks(range(len(top_names)), top_names[::-1])
        plt.xlabel('Feature Importance')
        plt.title('M3 Random Forest - Top 30 Features')
        plt.tight_layout()
        plt.savefig(output_dir / 'figures' / 'M3_Random_Forest_feature_importance.png', dpi=150)
        plt.close()
        log.info("Saved RF feature importance")

    return results


def calculate_metrics(probs, targets, model_name, output_dir):
    """Calculate comprehensive metrics."""
    results = {}

    preds = (probs > 0.5).astype(int)

    # Overall accuracy
    accuracy = np.mean(preds == targets)
    results['accuracy'] = float(accuracy)
    log.info(f"Overall Accuracy: {accuracy:.4f}")

    # Per-class metrics
    per_class = {}
    for i, name in enumerate(GESTURE_NAMES):
        y_true = targets[:, i]
        y_pred = preds[:, i]
        y_prob = probs[:, i]

        if y_true.sum() == 0:
            continue

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0
        )

        try:
            auc = roc_auc_score(y_true, y_prob)
        except:
            auc = 0.0

        per_class[name] = {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'auc': float(auc),
            'support': int(y_true.sum()),
        }

        log.info(f"{name:20s} - P: {precision:.3f}, R: {recall:.3f}, F1: {f1:.3f}, AUC: {auc:.3f}")

    results['per_class'] = per_class

    # Macro averages
    results['macro_precision'] = float(np.mean([m['precision'] for m in per_class.values()]))
    results['macro_recall'] = float(np.mean([m['recall'] for m in per_class.values()]))
    results['macro_f1'] = float(np.mean([m['f1'] for m in per_class.values()]))
    results['macro_auc'] = float(np.mean([m['auc'] for m in per_class.values()]))

    log.info(f"\nMacro - P: {results['macro_precision']:.3f}, R: {results['macro_recall']:.3f}, F1: {results['macro_f1']:.3f}, AUC: {results['macro_auc']:.3f}")

    # Generate plots
    generate_plots(probs, targets, preds, model_name, output_dir)

    return results


def generate_plots(probs, targets, preds, model_name, output_dir):
    """Generate visualization plots."""
    figures_dir = output_dir / 'figures'

    # 1. Confusion matrices (3x3 grid)
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()

    for i, name in enumerate(GESTURE_NAMES):
        cm = confusion_matrix(targets[:, i], preds[:, i])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                    xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
        axes[i].set_title(name)
        axes[i].set_ylabel('True')
        axes[i].set_xlabel('Predicted')

    plt.suptitle(f'{model_name} - Per-Gesture Confusion Matrices', y=1.02)
    plt.tight_layout()
    plt.savefig(figures_dir / f'{model_name}_confusion_matrices.png', dpi=150, bbox_inches='tight')
    plt.close()
    log.info(f"Saved {model_name} confusion matrices")

    # 2. ROC curves
    plt.figure(figsize=(12, 10))
    for i, name in enumerate(GESTURE_NAMES):
        y_true = targets[:, i]
        y_prob = probs[:, i]

        if y_true.sum() == 0:
            continue

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        plt.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} - ROC Curves')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / f'{model_name}_roc_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    log.info(f"Saved {model_name} ROC curves")

    # 3. Precision-Recall curves
    plt.figure(figsize=(12, 10))
    for i, name in enumerate(GESTURE_NAMES):
        y_true = targets[:, i]
        y_prob = probs[:, i]

        if y_true.sum() == 0:
            continue

        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        plt.plot(recall, precision, label=f'{name} (AP={ap:.3f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{model_name} - Precision-Recall Curves')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / f'{model_name}_pr_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    log.info(f"Saved {model_name} PR curves")


def generate_comparison_report(all_results, output_dir):
    """Generate final comparison report."""
    log.info("\n" + "="*60)
    log.info("MODEL COMPARISON SUMMARY")
    log.info("="*60)

    # Create comparison DataFrame
    comparison_data = []
    for model_name, results in all_results.items():
        row = {
            'Model': model_name,
            'Accuracy': f"{results.get('accuracy', 0):.4f}",
            'Macro Precision': f"{results.get('macro_precision', 0):.4f}",
            'Macro Recall': f"{results.get('macro_recall', 0):.4f}",
            'Macro F1': f"{results.get('macro_f1', 0):.4f}",
            'Macro AUC': f"{results.get('macro_auc', 0):.4f}",
            'Parameters': results.get('model_parameters', 'N/A'),
        }
        comparison_data.append(row)

    df = pd.DataFrame(comparison_data)
    print("\n" + df.to_string(index=False))

    # Save CSV
    df.to_csv(output_dir / 'model_comparison.csv', index=False)

    # Save full results JSON
    with open(output_dir / 'all_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    # Generate comparison bar chart
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    models = list(all_results.keys())
    x = np.arange(len(models))

    metrics = [
        ('Accuracy', 'accuracy'),
        ('Macro F1', 'macro_f1'),
        ('Macro Precision', 'macro_precision'),
        ('Macro Recall', 'macro_recall'),
    ]

    colors = ['#2ecc71', '#3498db', '#e74c3c']

    for ax, (title, key) in zip(axes.flatten(), metrics):
        values = [all_results[m].get(key, 0) for m in models]
        bars = ax.bar(x, values, color=colors[:len(models)])
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15)
        ax.set_ylim(0, 1)

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=10)

    plt.suptitle('Model Comparison - Key Metrics', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'figures' / 'model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    log.info(f"\nResults saved to {output_dir}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--models-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='./evaluation_results')
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'figures').mkdir(exist_ok=True)

    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    log.info(f"Using device: {device}")

    # Setup data
    transform = IsolatedDiscreteGesturesTransform(
        pulse_window=[0.0, 0.04],
        channel_indices=[4, 5, 6, 7, 8, 12, 14],
    )

    data_split = DataSplit.from_csv(
        csv_filename=os.path.join(args.data_dir, 'discrete_gestures_corpus.csv'),
        pool_test_partitions=True
    )

    data_module = WindowedEmgDataModule(
        data_location=args.data_dir,
        data_split=data_split,
        window_length=10000,
        stride=2000,
        batch_size=16,
        num_workers=4,
        transform=transform,
    )

    data_module.setup()
    test_loader = data_module.test_dataloader()

    all_results = {}

    # Evaluate M1
    m1_path = Path(args.models_dir) / 'm1_best.pt'
    if m1_path.exists():
        log.info("Loading M1...")
        m1 = FixedDiscreteGesturesLSTM(input_channels=7).to(device)
        m1.load_state_dict(torch.load(m1_path, map_location=device))
        all_results['M1_CNN_LSTM'] = evaluate_neural_model(
            m1, test_loader, device, 'M1_CNN_LSTM', output_dir
        )

    # Evaluate M2
    m2_path = Path(args.models_dir) / 'm2_best.pt'
    if m2_path.exists():
        log.info("Loading M2...")
        m2 = FixedDiscreteGesturesCNN(input_channels=7).to(device)
        m2.load_state_dict(torch.load(m2_path, map_location=device))
        all_results['M2_CNN_Only'] = evaluate_neural_model(
            m2, test_loader, device, 'M2_CNN_Only', output_dir
        )

    # Evaluate M3
    m3_path = Path(args.models_dir) / 'm3_best.pkl'
    if m3_path.exists():
        all_results['M3_Random_Forest'] = evaluate_rf_model(m3_path, output_dir)

    # Generate comparison
    if all_results:
        generate_comparison_report(all_results, output_dir)

    log.info("\n" + "="*60)
    log.info("EVALUATION COMPLETE!")
    log.info("="*60)


if __name__ == '__main__':
    main()
