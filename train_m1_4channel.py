#!/usr/bin/env python3
"""
M1 4-Channel TinyML Training Script

Trains the optimized 4-channel M1 model for Seeed Studio XIAO nRF52840 deployment.

Architecture:
    Conv1d(4, 72, k=15, s=10) -> BatchNorm -> LSTM(48, 2 layers) -> FC(9)

4-Channel Selection (flexor-extensor balance, 66.2% importance):
    - Ch7 (idx 6): Index flexor (24.5%) - press detection
    - Ch8 (idx 7): Ring flexor (13.2%) - press detection
    - Ch13 (idx 12): Index/middle extensor (19.8%) - release detection
    - Ch15 (idx 14): Ring/pinky extensor (8.7%) - release detection

Usage:
    python train_m1_4channel.py --data-dir ~/emg_data --epochs 100 --gpu
    python train_m1_4channel.py --data-dir ~/emg_data --epochs 100 --gpu --multi-gpu
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
from torch.optim.lr_scheduler import OneCycleLR

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from generic_neuromotor_interface.data import DataSplit
from generic_neuromotor_interface.data_module import WindowedEmgDataModule
from generic_neuromotor_interface.networks_isolated import M1_4Channel_TinyML, count_parameters
from generic_neuromotor_interface.transforms_isolated import Isolated4ChannelTransform

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler],
    epoch: int,
) -> Dict[str, float]:
    """Train one epoch."""
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
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == targets_aligned).sum().item()
            total += targets_aligned.numel()

        if batch_idx % 50 == 0:
            log.info(f"  Batch {batch_idx}/{len(train_loader)}: Loss={loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total if total > 0 else 0

    return {'loss': avg_loss, 'accuracy': accuracy}


@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Validate one epoch."""
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    criterion = nn.BCEWithLogitsLoss(reduction='mean')

    for batch in val_loader:
        emg = batch['emg'].to(device)
        targets = batch['targets'].to(device)

        with torch.cuda.amp.autocast():
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

        preds = (torch.sigmoid(logits) > 0.5).float()
        correct += (preds == targets_aligned).sum().item()
        total += targets_aligned.numel()

        # Store for metrics
        all_preds.append(preds.cpu())
        all_targets.append(targets_aligned.cpu())

    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total if total > 0 else 0

    # Calculate per-class recall
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    recalls = []
    for c in range(9):
        class_mask = all_targets[:, c, :] == 1
        if class_mask.sum() > 0:
            class_recall = (all_preds[:, c, :][class_mask] == 1).float().mean().item()
            recalls.append(class_recall)

    mean_recall = np.mean(recalls) if recalls else 0

    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'recall': mean_recall,
    }


def main():
    parser = argparse.ArgumentParser(description='Train M1 4-Channel TinyML Model')
    parser.add_argument('--data-dir', type=str, default=os.path.expanduser('~/emg_data'),
                        help='Path to EMG data directory')
    parser.add_argument('--output-dir', type=str, default='./checkpoints_4channel',
                        help='Output directory for checkpoints')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size (per GPU)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--gpu', action='store_true', help='Use GPU')
    parser.add_argument('--multi-gpu', action='store_true', help='Use all available GPUs')
    parser.add_argument('--window-length', type=int, default=2000, help='Window length in samples')
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup device
    if args.gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        log.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        if args.multi_gpu and torch.cuda.device_count() > 1:
            log.info(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
    else:
        device = torch.device('cpu')
        log.info("Using CPU")

    # 4-channel transform
    transform = Isolated4ChannelTransform(
        pulse_window=[0.0, 0.04],
        channel_indices=[6, 7, 12, 14],  # Ch7, Ch8, Ch13, Ch15
    )

    log.info("=" * 60)
    log.info("M1 4-Channel TinyML Training")
    log.info("=" * 60)
    log.info(f"Channels: {transform.channel_indices} (Ch7, Ch8, Ch13, Ch15)")
    log.info(f"Window: {args.window_length} samples ({args.window_length/2000:.2f}s)")
    log.info(f"Batch size: {args.batch_size}")
    log.info(f"Learning rate: {args.lr}")
    log.info(f"Epochs: {args.epochs}")
    log.info("=" * 60)

    # Create data split from CSV
    log.info("Loading data...")
    data_split = DataSplit.from_csv(
        csv_filename=os.path.join(args.data_dir, 'discrete_gestures_corpus.csv'),
        pool_test_partitions=True
    )

    # Create data module with optimized settings
    num_workers = 4 if args.gpu else 0  # Use workers for GPU training
    data_module = WindowedEmgDataModule(
        data_location=args.data_dir,
        window_length=args.window_length,
        stride=args.window_length // 2,  # 50% overlap
        batch_size=args.batch_size,
        num_workers=num_workers,
        transform=transform,
        data_split=data_split,
    )
    data_module.setup()

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    log.info(f"Train batches: {len(train_loader)}")
    log.info(f"Val batches: {len(val_loader)}")

    # Create model
    log.info("Creating M1 4-Channel TinyML model...")
    model = M1_4Channel_TinyML(
        input_channels=4,
        conv_output_channels=72,
        kernel_width=15,
        stride=10,
        lstm_hidden_size=48,
        lstm_num_layers=2,
        output_channels=9,
        dropout=0.3,
    )

    # Memory estimate
    mem = model.get_memory_estimate(window_samples=args.window_length)
    log.info(f"Parameters: {count_parameters(model):,}")
    log.info(f"Model size (INT8): {mem['model_size_int8']/1024:.1f} KB")
    log.info(f"Runtime estimate: {mem['estimated_total_runtime']/1024:.1f} KB")
    log.info(f"Fits nRF52840: {mem['fits_nrf52840']}")

    # Multi-GPU support with proper DataParallel
    if args.multi_gpu and torch.cuda.device_count() > 1:
        num_gpus = torch.cuda.device_count()
        log.info(f"Using DataParallel across {num_gpus} GPUs")
        model = nn.DataParallel(model, device_ids=list(range(num_gpus)))
        # Effective batch size = batch_size * num_gpus
        log.info(f"Effective batch size: {args.batch_size * num_gpus}")

    model = model.to(device)

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    total_steps = len(train_loader) * args.epochs
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy='cos',
    )

    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if args.gpu else None

    # Training loop
    best_val_acc = 0.0
    best_epoch = 0

    log.info("\nStarting training...")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        log.info(f"\n{'='*60}")
        log.info(f"Epoch {epoch}/{args.epochs}")
        log.info(f"{'='*60}")

        # Train
        train_metrics = train_epoch(
            model=model.module if isinstance(model, nn.DataParallel) else model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            scaler=scaler,
            epoch=epoch,
        )

        # Validate
        val_metrics = validate_epoch(
            model=model.module if isinstance(model, nn.DataParallel) else model,
            val_loader=val_loader,
            device=device,
        )

        epoch_time = time.time() - epoch_start

        log.info(f"\nEpoch {epoch} Results:")
        log.info(f"  Train Loss: {train_metrics['loss']:.4f}")
        log.info(f"  Train Acc:  {train_metrics['accuracy']*100:.2f}%")
        log.info(f"  Val Loss:   {val_metrics['loss']:.4f}")
        log.info(f"  Val Acc:    {val_metrics['accuracy']*100:.2f}%")
        log.info(f"  Val Recall: {val_metrics['recall']*100:.2f}%")
        log.info(f"  Time:       {epoch_time:.1f}s")

        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            best_epoch = epoch

            checkpoint_path = os.path.join(args.output_dir, 'best_m1_4channel.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': (model.module if isinstance(model, nn.DataParallel) else model).state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_metrics['accuracy'],
                'val_recall': val_metrics['recall'],
                'config': {
                    'input_channels': 4,
                    'conv_output_channels': 72,
                    'lstm_hidden_size': 48,
                    'lstm_num_layers': 2,
                    'window_length': args.window_length,
                    'channel_indices': [6, 7, 12, 14],
                },
            }, checkpoint_path)
            log.info(f"  ** New best model saved: {val_metrics['accuracy']*100:.2f}% **")

        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': (model.module if isinstance(model, nn.DataParallel) else model).state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_metrics['accuracy'],
            }, checkpoint_path)

    total_time = time.time() - start_time

    log.info("\n" + "=" * 60)
    log.info("Training Complete!")
    log.info("=" * 60)
    log.info(f"Best Val Accuracy: {best_val_acc*100:.2f}% (epoch {best_epoch})")
    log.info(f"Total Time: {total_time/60:.1f} minutes")
    log.info(f"Best model saved to: {os.path.join(args.output_dir, 'best_m1_4channel.pt')}")

    # Save training summary
    summary = {
        'best_val_accuracy': best_val_acc,
        'best_epoch': best_epoch,
        'total_epochs': args.epochs,
        'total_time_seconds': total_time,
        'parameters': count_parameters(model.module if isinstance(model, nn.DataParallel) else model),
        'config': {
            'input_channels': 4,
            'conv_output_channels': 72,
            'lstm_hidden_size': 48,
            'lstm_num_layers': 2,
            'window_length': args.window_length,
            'channel_indices': [6, 7, 12, 14],
        },
    }

    with open(os.path.join(args.output_dir, 'training_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    log.info("\nNext steps for XIAO nRF52840 deployment:")
    log.info("  1. Convert to TFLite: python tinyml_deployment/conversion/convert_m1_4channel.py")
    log.info("  2. Build firmware: cd tinyml_deployment/platformio_project && pio run")
    log.info("  3. Flash to device: pio run --target upload")


if __name__ == '__main__':
    main()
