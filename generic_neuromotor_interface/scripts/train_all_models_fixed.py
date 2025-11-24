#!/usr/bin/env python3
"""
COMPREHENSIVE TRAINING SCRIPT FOR ALL THREE MODELS

Trains M1 (CNN+LSTM), M2 (CNN-only), and M3 (Random Forest)
with proper GPU optimization and comprehensive metrics.

Usage:
    python train_all_models_fixed.py --model m1 --data-dir ~/emg_data --gpu
    python train_all_models_fixed.py --model all --data-dir ~/emg_data --gpu --epochs 100
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from generic_neuromotor_interface.data import DataSplit
from generic_neuromotor_interface.data_module import WindowedEmgDataModule
from generic_neuromotor_interface.networks_isolated import (
    FixedDiscreteGesturesLSTM,
    FixedDiscreteGesturesCNN,
    count_parameters,
)
from generic_neuromotor_interface.random_forest_model import FixedRandomForestGestureModel
from generic_neuromotor_interface.transforms_isolated import IsolatedDiscreteGesturesTransform

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def train_epoch_neural(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler],
    epoch: int,
) -> Dict[str, float]:
    """Train one epoch for neural network models."""
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    criterion = nn.BCEWithLogitsLoss(reduction='mean')

    for batch_idx, batch in enumerate(train_loader):
        emg = batch['emg'].to(device)
        targets = batch['targets'].to(device)

        optimizer.zero_grad(set_to_none=True)

        # Mixed precision training
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            logits = model(emg)

            # Align targets to model output
            stride = model.stride
            left = model.left_context
            targets_aligned = targets[:, :, left::stride]

            # Crop to match output size
            min_len = min(logits.shape[-1], targets_aligned.shape[-1])
            logits = logits[:, :, :min_len]
            targets_aligned = targets_aligned[:, :, :min_len]

            loss = criterion(logits, targets_aligned)

        # Backward pass
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()

        # Calculate accuracy
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            correct += (preds == targets_aligned).sum().item()
            total += targets_aligned.numel()

    return {
        'train_loss': total_loss / len(train_loader),
        'train_accuracy': correct / total if total > 0 else 0,
    }


def validate_neural(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Validate neural network model."""
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_targets = []

    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for batch in val_loader:
            emg = batch['emg'].to(device)
            targets = batch['targets'].to(device)

            logits = model(emg)

            # Align targets
            stride = model.stride
            left = model.left_context
            targets_aligned = targets[:, :, left::stride]

            min_len = min(logits.shape[-1], targets_aligned.shape[-1])
            logits = logits[:, :, :min_len]
            targets_aligned = targets_aligned[:, :, :min_len]

            loss = criterion(logits, targets_aligned)
            total_loss += loss.item()

            # Collect predictions
            probs = torch.sigmoid(logits)
            all_preds.append(probs.cpu().numpy())
            all_targets.append(targets_aligned.cpu().numpy())

    # Concatenate all batches
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Calculate accuracy
    pred_binary = (all_preds > 0.5).astype(float)
    accuracy = np.mean(pred_binary == all_targets)

    return {
        'val_loss': total_loss / len(val_loader),
        'val_accuracy': accuracy,
    }


def train_model_m1(
    data_dir: str,
    output_dir: str,
    device: torch.device,
    config: dict,
) -> Dict:
    """Train M1 (CNN+LSTM) model."""
    log.info("=" * 60)
    log.info("Training M1: CNN+LSTM Model")
    log.info("=" * 60)

    # Create data module
    transform = IsolatedDiscreteGesturesTransform(
        pulse_window=[0.0, 0.04],
        channel_indices=[4, 5, 6, 7, 8, 12, 14],
    )

    # Create data split
    data_split = DataSplit.from_csv(
        csv_filename=os.path.join(data_dir, 'discrete_gestures_corpus.csv'),
        pool_test_partitions=True
    )

    data_module = WindowedEmgDataModule(
        data_location=data_dir,
        data_split=data_split,
        window_length=10000,  # 5 seconds at 2kHz
        stride=2000,          # 1 second stride
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        transform=transform,
    )

    data_module.setup()
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    log.info(f"Training samples: {len(train_loader.dataset)}")
    log.info(f"Validation samples: {len(val_loader.dataset)}")

    # Create model
    model = FixedDiscreteGesturesLSTM(input_channels=7).to(device)
    log.info(f"Model parameters: {count_parameters(model):,}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
    )

    # Scheduler
    scheduler = OneCycleLR(
        optimizer,
        max_lr=config['learning_rate'] * 10,
        epochs=config['num_epochs'],
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
    )

    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if config['use_amp'] and device.type == 'cuda' else None

    # Training loop
    best_val_loss = float('inf')
    best_epoch = 0
    history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}

    for epoch in range(config['num_epochs']):
        train_metrics = train_epoch_neural(
            model, train_loader, optimizer, scheduler, device, scaler, epoch
        )
        val_metrics = validate_neural(model, val_loader, device)

        history['train_loss'].append(train_metrics['train_loss'])
        history['val_loss'].append(val_metrics['val_loss'])
        history['val_accuracy'].append(val_metrics['val_accuracy'])

        if val_metrics['val_loss'] < best_val_loss:
            best_val_loss = val_metrics['val_loss']
            best_epoch = epoch
            torch.save(model.state_dict(), f"{output_dir}/m1_best.pt")

        if epoch % 10 == 0:
            log.info(
                f"Epoch {epoch:3d} | "
                f"Train Loss: {train_metrics['train_loss']:.4f} | "
                f"Val Loss: {val_metrics['val_loss']:.4f} | "
                f"Val Acc: {val_metrics['val_accuracy']:.4f}"
            )

    log.info(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")

    return {
        'model_type': 'M1_CNN_LSTM',
        'best_val_loss': best_val_loss,
        'best_epoch': best_epoch,
        'final_val_accuracy': history['val_accuracy'][-1],
        'checkpoint_path': f"{output_dir}/m1_best.pt",
        'history': history,
    }


def train_model_m2(
    data_dir: str,
    output_dir: str,
    device: torch.device,
    config: dict,
) -> Dict:
    """Train M2 (CNN-only) model."""
    log.info("=" * 60)
    log.info("Training M2: CNN-only Model")
    log.info("=" * 60)

    # Create data module
    transform = IsolatedDiscreteGesturesTransform(
        pulse_window=[0.0, 0.04],
        channel_indices=[4, 5, 6, 7, 8, 12, 14],
    )

    # Create data split
    data_split = DataSplit.from_csv(
        csv_filename=os.path.join(data_dir, 'discrete_gestures_corpus.csv'),
        pool_test_partitions=True
    )

    data_module = WindowedEmgDataModule(
        data_location=data_dir,
        data_split=data_split,
        window_length=10000,
        stride=2000,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        transform=transform,
    )

    data_module.setup()
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    log.info(f"Training samples: {len(train_loader.dataset)}")
    log.info(f"Validation samples: {len(val_loader.dataset)}")

    # Create model
    model = FixedDiscreteGesturesCNN(input_channels=7).to(device)
    log.info(f"Model parameters: {count_parameters(model):,}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
    )

    # Scheduler
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)

    scaler = torch.cuda.amp.GradScaler() if config['use_amp'] and device.type == 'cuda' else None

    best_val_loss = float('inf')
    best_epoch = 0
    history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}

    for epoch in range(config['num_epochs']):
        train_metrics = train_epoch_neural(
            model, train_loader, optimizer, None, device, scaler, epoch
        )
        val_metrics = validate_neural(model, val_loader, device)

        scheduler.step()

        history['train_loss'].append(train_metrics['train_loss'])
        history['val_loss'].append(val_metrics['val_loss'])
        history['val_accuracy'].append(val_metrics['val_accuracy'])

        if val_metrics['val_loss'] < best_val_loss:
            best_val_loss = val_metrics['val_loss']
            best_epoch = epoch
            torch.save(model.state_dict(), f"{output_dir}/m2_best.pt")

        if epoch % 10 == 0:
            log.info(
                f"Epoch {epoch:3d} | "
                f"Train Loss: {train_metrics['train_loss']:.4f} | "
                f"Val Loss: {val_metrics['val_loss']:.4f} | "
                f"Val Acc: {val_metrics['val_accuracy']:.4f}"
            )

    log.info(f"Best validation loss: {best_val_loss:.4f} at epoch {best_epoch}")

    return {
        'model_type': 'M2_CNN_Only',
        'best_val_loss': best_val_loss,
        'best_epoch': best_epoch,
        'final_val_accuracy': history['val_accuracy'][-1],
        'checkpoint_path': f"{output_dir}/m2_best.pt",
        'history': history,
    }


def train_model_m3(
    data_dir: str,
    output_dir: str,
    config: dict,
) -> Dict:
    """Train M3 (Random Forest) model."""
    log.info("=" * 60)
    log.info("Training M3: Random Forest Model")
    log.info("=" * 60)

    # Load data using data module
    transform = IsolatedDiscreteGesturesTransform(
        pulse_window=[0.0, 0.04],
        channel_indices=[4, 5, 6, 7, 8, 12, 14],
    )

    # Create data split
    data_split = DataSplit.from_csv(
        csv_filename=os.path.join(data_dir, 'discrete_gestures_corpus.csv'),
        pool_test_partitions=True
    )

    data_module = WindowedEmgDataModule(
        data_location=data_dir,
        data_split=data_split,
        window_length=400,  # 200ms at 2kHz
        stride=200,         # 100ms stride
        batch_size=32,
        num_workers=4,
        transform=transform,
    )

    data_module.setup()
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    # Convert to numpy arrays for RF
    X_train = []
    y_train = []

    log.info("Loading training data...")
    for batch in train_loader:
        emg = batch['emg'].numpy()  # (B, 7, T)
        targets = batch['targets'].numpy()  # (B, 9, T)

        # Get most activated gesture per sample
        gesture_labels = np.argmax(targets.sum(axis=2), axis=1)

        X_train.append(emg)
        y_train.extend(gesture_labels)

    X_train = np.concatenate(X_train, axis=0)
    y_train = np.array(y_train)

    log.info(f"Training samples: {len(X_train)}")

    # Validation data
    X_val = []
    y_val = []

    log.info("Loading validation data...")
    for batch in val_loader:
        emg = batch['emg'].numpy()
        targets = batch['targets'].numpy()
        gesture_labels = np.argmax(targets.sum(axis=2), axis=1)

        X_val.append(emg)
        y_val.extend(gesture_labels)

    X_val = np.concatenate(X_val, axis=0)
    y_val = np.array(y_val)

    log.info(f"Validation samples: {len(X_val)}")

    # Train model
    model = FixedRandomForestGestureModel(
        n_estimators=300,
        max_depth=20,
        n_channels=7,
        fs=2000.0,
    )

    results = model.fit(X_train, y_train, X_val, y_val)

    # Save model
    model.save(f"{output_dir}/m3_best.pkl")

    log.info(f"Random Forest CV Accuracy: {results['cv_accuracy_mean']:.3f}")
    log.info(f"Validation Accuracy: {results.get('val_accuracy', 'N/A')}")

    return {
        'model_type': 'M3_Random_Forest',
        'cv_accuracy_mean': results['cv_accuracy_mean'],
        'cv_accuracy_std': results['cv_accuracy_std'],
        'val_accuracy': results.get('val_accuracy'),
        'checkpoint_path': f"{output_dir}/m3_best.pkl",
    }


def main():
    parser = argparse.ArgumentParser(description='Train gesture recognition models')
    parser.add_argument('--model', type=str, default='all', choices=['m1', 'm2', 'm3', 'all'])
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='./models')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num-workers', type=int, default=4)

    args = parser.parse_args()

    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    log.info(f"Using device: {device}")

    config = {
        'num_epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'weight_decay': 1e-4,
        'num_workers': args.num_workers,
        'use_amp': args.gpu and torch.cuda.is_available(),
    }

    results = {}

    # Train requested models
    start_time = time.time()

    if args.model in ['m1', 'all']:
        results['M1'] = train_model_m1(args.data_dir, str(output_dir), device, config)

    if args.model in ['m2', 'all']:
        results['M2'] = train_model_m2(args.data_dir, str(output_dir), device, config)

    if args.model in ['m3', 'all']:
        results['M3'] = train_model_m3(args.data_dir, str(output_dir), config)

    total_time = time.time() - start_time

    # Save results summary
    results['training_time_seconds'] = total_time
    results['config'] = config

    with open(output_dir / 'training_results.json', 'w') as f:
        # Convert numpy types to native Python types
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

        json.dump(convert_types(results), f, indent=2)

    log.info("\n" + "=" * 60)
    log.info("TRAINING COMPLETE")
    log.info("=" * 60)
    log.info(f"Total training time: {total_time/60:.2f} minutes")
    for model_name, result in results.items():
        if model_name not in ['training_time_seconds', 'config']:
            log.info(f"\n{model_name}:")
            for key, value in result.items():
                if key != 'history':
                    log.info(f"  {key}: {value}")


if __name__ == '__main__':
    main()
