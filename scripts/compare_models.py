#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Compare three ML models (M1: CNN+LSTM, M2: CNN-Only, M3: Random Forest)
on isolated EMG channels for discrete gesture recognition.

This script trains and evaluates all three models on the same dataset
and generates a comprehensive comparison report.
"""

import argparse
import json
import logging
import os
import site
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
log = logging.getLogger(__name__)


def train_m1_cnn_lstm(
    data_location: str, output_dir: Path, gpu: bool = True, gpu_device: int = 0
) -> dict[str, Any]:
    """Train M1: CNN+LSTM model using PyTorch Lightning."""
    log.info("=" * 80)
    log.info("Training M1: CNN+LSTM Model")
    log.info("=" * 80)

    start_time = time.time()

    # Expand ~ to full path
    data_location_expanded = str(Path(data_location).expanduser())
    
    # Run training command
    cmd = [
        sys.executable,
        "-m",
        "generic_neuromotor_interface.train",
        "--config-name=discrete_gestures_isolated",
        f"data_location={data_location_expanded}",
    ]

    if gpu:
        cmd.extend([
            "trainer.accelerator=gpu", 
            "+trainer.devices=1",
            "trainer.strategy=auto"  # Use auto strategy instead of ddp for single GPU
        ])
    else:
        cmd.extend(["trainer.accelerator=cpu"])

    log.info(f"Running command: {' '.join(cmd)}")

    # Set PYTHONPATH to include project directory and user site-packages
    project_dir = str(Path(__file__).parent.parent.parent)
    env = os.environ.copy()
    user_site = site.getusersitepackages()
    existing_pythonpath = env.get('PYTHONPATH', '')
    env["PYTHONPATH"] = f"{project_dir}:{user_site}:{existing_pythonpath}".rstrip(':')
    
    # CRITICAL: Ensure all CUDA environment variables are set
    # PyTorch needs these to detect GPUs in subprocesses
    cuda_lib_path = "/sw/ubuntu2204/ebu082025/software/common/core/cuda/12.8.1/lib64"
    cuda_cupti = "/sw/ubuntu2204/ebu082025/software/common/core/cuda/12.8.1/extras/CUPTI/lib64"
    cuda_nvvm = "/sw/ubuntu2204/ebu082025/software/common/core/cuda/12.8.1/nvvm/lib64"
    cuda_targets = "/sw/ubuntu2204/ebu082025/software/common/core/cuda/12.8.1/targets/x86_64-linux/lib"
    
    # Force set LD_LIBRARY_PATH with all CUDA paths first
    existing_ld = env.get('LD_LIBRARY_PATH', '')
    cuda_ld_path = f"{cuda_lib_path}:{cuda_cupti}:{cuda_nvvm}:{cuda_targets}"
    env['LD_LIBRARY_PATH'] = f"{cuda_ld_path}:{existing_ld}" if existing_ld else cuda_ld_path
    
    # Set CUDA_HOME to help PyTorch find CUDA
    env['CUDA_HOME'] = "/sw/ubuntu2204/ebu082025/software/common/core/cuda/12.8.1"
    env['CUDA_PATH'] = "/sw/ubuntu2204/ebu082025/software/common/core/cuda/12.8.1"
    
    log.info(f"Set CUDA environment - LD_LIBRARY_PATH: {env['LD_LIBRARY_PATH'][:150]}...")
    log.info(f"CUDA_HOME: {env.get('CUDA_HOME')}")
    
    # NOTE: Not setting CUDA_VISIBLE_DEVICES - PyTorch will see all GPUs
    # The training script will use devices=1 to train on first available GPU
    # If you want to select a specific GPU, the parent process should set CUDA_VISIBLE_DEVICES
    # before calling this script

    try:
        result = subprocess.run(
            cmd,
            cwd=project_dir,
            capture_output=True,
            text=True,
            check=True,
            env=env,
        )
        log.info("M1 training completed successfully")
        log.debug(f"Output: {result.stdout[-1000:]}")  # Last 1000 chars
    except subprocess.CalledProcessError as e:
        error_msg = f"M1 training failed: {e}"
        if e.stderr:
            error_msg += f"\nSTDERR:\n{e.stderr[-2000:]}"  # Last 2000 chars
        if e.stdout:
            error_msg += f"\nSTDOUT:\n{e.stdout[-2000:]}"  # Last 2000 chars
        log.error(error_msg)
        raise RuntimeError(error_msg) from e

    training_time = time.time() - start_time

    # Try to find the latest log directory and extract metrics
    log_dir = Path(__file__).parent.parent.parent / "logs"
    metrics = extract_lightning_metrics(log_dir, "discrete-gestures-isolated")

    return {
        "model": "M1: CNN+LSTM",
        "training_time": training_time,
        "metrics": metrics,
        "status": "completed",
    }


def train_m2_cnn_only(
    data_location: str, output_dir: Path, gpu: bool = True, gpu_device: int = 0
) -> dict[str, Any]:
    """Train M2: CNN-Only model using PyTorch Lightning."""
    log.info("=" * 80)
    log.info("Training M2: CNN-Only Model")
    log.info("=" * 80)

    start_time = time.time()

    # Expand ~ to full path
    data_location_expanded = str(Path(data_location).expanduser())
    
    # Run training command
    cmd = [
        sys.executable,
        "-m",
        "generic_neuromotor_interface.train",
        "--config-name=discrete_gestures_cnn_isolated",
        f"data_location={data_location_expanded}",
    ]

    if gpu:
        cmd.extend([
            "trainer.accelerator=gpu", 
            "+trainer.devices=1",
            "trainer.strategy=auto"  # Use auto strategy instead of ddp for single GPU
        ])
    else:
        cmd.extend(["trainer.accelerator=cpu"])

    log.info(f"Running command: {' '.join(cmd)}")

    # Set PYTHONPATH to include project directory and user site-packages
    project_dir = str(Path(__file__).parent.parent.parent)
    env = os.environ.copy()
    user_site = site.getusersitepackages()
    existing_pythonpath = env.get('PYTHONPATH', '')
    env["PYTHONPATH"] = f"{project_dir}:{user_site}:{existing_pythonpath}".rstrip(':')
    
    # CRITICAL: Ensure all CUDA environment variables are set
    # PyTorch needs these to detect GPUs in subprocesses
    cuda_lib_path = "/sw/ubuntu2204/ebu082025/software/common/core/cuda/12.8.1/lib64"
    cuda_cupti = "/sw/ubuntu2204/ebu082025/software/common/core/cuda/12.8.1/extras/CUPTI/lib64"
    cuda_nvvm = "/sw/ubuntu2204/ebu082025/software/common/core/cuda/12.8.1/nvvm/lib64"
    cuda_targets = "/sw/ubuntu2204/ebu082025/software/common/core/cuda/12.8.1/targets/x86_64-linux/lib"
    
    # Force set LD_LIBRARY_PATH with all CUDA paths first
    existing_ld = env.get('LD_LIBRARY_PATH', '')
    cuda_ld_path = f"{cuda_lib_path}:{cuda_cupti}:{cuda_nvvm}:{cuda_targets}"
    env['LD_LIBRARY_PATH'] = f"{cuda_ld_path}:{existing_ld}" if existing_ld else cuda_ld_path
    
    # Set CUDA_HOME to help PyTorch find CUDA
    env['CUDA_HOME'] = "/sw/ubuntu2204/ebu082025/software/common/core/cuda/12.8.1"
    env['CUDA_PATH'] = "/sw/ubuntu2204/ebu082025/software/common/core/cuda/12.8.1"
    
    log.info(f"Set CUDA environment - LD_LIBRARY_PATH: {env['LD_LIBRARY_PATH'][:150]}...")
    log.info(f"CUDA_HOME: {env.get('CUDA_HOME')}")
    
    # Set CUDA_VISIBLE_DEVICES if using GPU
    if gpu:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_device)

    try:
        result = subprocess.run(
            cmd,
            cwd=project_dir,
            capture_output=True,
            text=True,
            check=True,
            env=env,
        )
        log.info("M2 training completed successfully")
        log.debug(f"Output: {result.stdout[-1000:]}")
    except subprocess.CalledProcessError as e:
        error_msg = f"M2 training failed: {e}"
        if e.stderr:
            error_msg += f"\nSTDERR:\n{e.stderr[-2000:]}"
        if e.stdout:
            error_msg += f"\nSTDOUT:\n{e.stdout[-2000:]}"
        log.error(error_msg)
        raise RuntimeError(error_msg) from e

    training_time = time.time() - start_time

    # Try to find the latest log directory and extract metrics
    log_dir = Path(__file__).parent.parent.parent / "logs"
    metrics = extract_lightning_metrics(log_dir, "discrete-gestures-cnn-isolated")

    return {
        "model": "M2: CNN-Only",
        "training_time": training_time,
        "metrics": metrics,
        "status": "completed",
    }


def train_m3_random_forest(
    data_location: str, output_dir: Path, channel_indices: list[int]
) -> dict[str, Any]:
    """Train M3: Random Forest model."""
    log.info("=" * 80)
    log.info("Training M3: Random Forest Model")
    log.info("=" * 80)

    start_time = time.time()

    # Expand ~ to full path
    data_location_expanded = str(Path(data_location).expanduser())
    
    # Run training command
    channel_str = " ".join(map(str, channel_indices))
    cmd = [
        sys.executable,
        "-m",
        "generic_neuromotor_interface.scripts.train_random_forest",
        f"--data-location={data_location_expanded}",
        "--channel-indices",
    ] + [str(c) for c in channel_indices] + [
        "--n-estimators=200",
        f"--output-dir={output_dir / 'rf_model'}",
    ]

    log.info(f"Running command: {' '.join(cmd)}")

    # Set PYTHONPATH to include project directory
    project_dir = str(Path(__file__).parent.parent.parent)
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{project_dir}:{env.get('PYTHONPATH', '')}"

    try:
        result = subprocess.run(
            cmd,
            cwd=project_dir,
            capture_output=True,
            text=True,
            check=True,
            env=env,
        )
        log.info("M3 training completed successfully")
        log.debug(f"Output: {result.stdout[-1000:]}")
    except subprocess.CalledProcessError as e:
        error_msg = f"M3 training failed: {e}"
        if e.stderr:
            error_msg += f"\nSTDERR:\n{e.stderr[-2000:]}"
        if e.stdout:
            error_msg += f"\nSTDOUT:\n{e.stdout[-2000:]}"
        log.error(error_msg)
        raise RuntimeError(error_msg) from e

    training_time = time.time() - start_time

    # Parse metrics from output
    metrics = parse_rf_metrics(result.stdout)

    return {
        "model": "M3: Random Forest",
        "training_time": training_time,
        "metrics": metrics,
        "status": "completed",
    }


def extract_lightning_metrics(log_dir: Path, job_name: str) -> dict[str, Any]:
    """Extract metrics from PyTorch Lightning logs."""
    metrics = {}

    # Find the most recent log directory
    if not log_dir.exists():
        log.warning(f"Log directory not found: {log_dir}")
        return metrics

    # Get all directories matching the pattern
    log_dirs = sorted(log_dir.glob("*/*"), key=lambda p: p.stat().st_mtime, reverse=True)

    for log_subdir in log_dirs:
        # Look for lightning_logs
        lightning_logs = log_subdir / "lightning_logs"
        if lightning_logs.exists():
            # Find version directories
            versions = sorted(lightning_logs.glob("version_*"), key=lambda p: int(p.name.split("_")[1]))
            if versions:
                latest_version = versions[-1]
                metrics_file = latest_version / "metrics.csv"

                if metrics_file.exists():
                    try:
                        df = pd.read_csv(metrics_file)
                        # Extract final validation and test metrics
                        val_metrics = df[df["val_accuracy"].notna()].tail(1)
                        test_metrics = df[df["test_cler"].notna()].tail(1)

                        if not val_metrics.empty:
                            metrics["val_accuracy"] = float(val_metrics["val_accuracy"].iloc[-1])
                        if not test_metrics.empty:
                            metrics["test_cler"] = float(test_metrics["test_cler"].iloc[-1])

                        log.info(f"Extracted metrics from {metrics_file}")
                        break
                    except Exception as e:
                        log.warning(f"Error reading metrics file: {e}")
                        continue

    return metrics


def parse_rf_metrics(output: str) -> dict[str, Any]:
    """Parse metrics from Random Forest training output."""
    metrics = {}

    # Extract validation accuracy
    for line in output.split("\n"):
        if "Validation Accuracy:" in line:
            try:
                acc = float(line.split(":")[-1].strip())
                metrics["val_accuracy"] = acc
            except ValueError:
                pass

    return metrics


def generate_comparison_report(
    results: list[dict[str, Any]], output_dir: Path
) -> None:
    """Generate a comprehensive comparison report."""
    log.info("=" * 80)
    log.info("Generating Comparison Report")
    log.info("=" * 80)

    # Create comparison DataFrame
    comparison_data = []

    for result in results:
        row = {
            "Model": result["model"],
            "Training Time (hours)": result.get("training_time", 0) / 3600 if "training_time" in result else np.nan,
            "Status": result.get("status", "unknown"),
        }

        # Add metrics
        metrics = result.get("metrics", {})
        row["Val Accuracy"] = metrics.get("val_accuracy", np.nan)
        row["Test CLER"] = metrics.get("test_cler", np.nan)

        comparison_data.append(row)

    df = pd.DataFrame(comparison_data)

    # Generate markdown report
    report_path = output_dir / "model_comparison_report.md"
    with open(report_path, "w") as f:
        f.write("# ML Model Comparison Report\n\n")
        f.write("## Overview\n\n")
        f.write(
            "This report compares three ML models for discrete gesture recognition "
            "using isolated EMG channels {5, 6, 7, 8, 9, 13, 15}.\n\n"
        )

        f.write("## Models Compared\n\n")
        f.write("1. **M1: CNN+LSTM** - Hybrid architecture with convolutional front-end and LSTM layers\n")
        f.write("2. **M2: CNN-Only** - Pure CNN architecture with Inception blocks\n")
        f.write("3. **M3: Random Forest** - Classical ML with engineered features\n\n")

        f.write("## Results Summary\n\n")
        # Try to use markdown, fallback to CSV if tabulate not available
        try:
            f.write(df.to_markdown(index=False))
        except ImportError:
            f.write(df.to_string(index=False))
        f.write("\n\n")

        f.write("## Detailed Metrics\n\n")
        for result in results:
            f.write(f"### {result['model']}\n\n")
            if "training_time" in result:
                f.write(f"- **Training Time**: {result['training_time']/3600:.2f} hours\n")
            f.write(f"- **Status**: {result.get('status', 'unknown')}\n")
            if "error" in result:
                f.write(f"- **Error**: {result['error']}\n")
            f.write("- **Metrics**:\n")
            metrics = result.get("metrics", {})
            if metrics:
                for key, value in metrics.items():
                    f.write(f"  - {key}: {value:.4f}\n")
            else:
                f.write("  - No metrics available\n")
            f.write("\n")

        f.write("## Recommendations\n\n")
        # Find best model by accuracy
        best_acc_model = None
        best_acc = -1
        for result in results:
            acc = result.get("metrics", {}).get("val_accuracy", 0)
            if acc > best_acc:
                best_acc = acc
                best_acc_model = result["model"]

        if best_acc_model:
            f.write(f"- **Best Accuracy**: {best_acc_model} ({best_acc:.4f})\n")

        # Find best model by CLER (lower is better)
        best_cler_model = None
        best_cler = float("inf")
        for result in results:
            cler = result.get("metrics", {}).get("test_cler", float("inf"))
            if cler < best_cler:
                best_cler = cler
                best_cler_model = result["model"]

        if best_cler_model and best_cler != float("inf"):
            f.write(f"- **Best CLER**: {best_cler_model} ({best_cler:.4f})\n")

    log.info(f"Report saved to {report_path}")

    # Also save JSON for programmatic access
    json_path = output_dir / "model_comparison_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info(f"Results JSON saved to {json_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare three ML models for discrete gesture recognition"
    )
    parser.add_argument(
        "--data-location",
        type=str,
        default="~/emg_data",
        help="Path to EMG data directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./model_comparison",
        help="Directory to save comparison results",
    )
    parser.add_argument(
        "--channel-indices",
        type=int,
        nargs="+",
        default=[9, 8, 7, 6, 5, 13, 15],
        help="Channel indices to use",
    )
    parser.add_argument(
        "--skip-m1",
        action="store_true",
        help="Skip training M1 (CNN+LSTM)",
    )
    parser.add_argument(
        "--skip-m2",
        action="store_true",
        help="Skip training M2 (CNN-Only)",
    )
    parser.add_argument(
        "--skip-m3",
        action="store_true",
        help="Skip training M3 (Random Forest)",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Use CPU instead of GPU for M1 and M2",
    )
    parser.add_argument(
        "--gpu-device",
        type=int,
        default=0,
        help="GPU device ID to use (default: 0)",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    # Train M1: CNN+LSTM
    if not args.skip_m1:
        try:
            result = train_m1_cnn_lstm(
                args.data_location, output_dir, gpu=not args.no_gpu, gpu_device=args.gpu_device
            )
            results.append(result)
        except Exception as e:
            log.error(f"Failed to train M1: {e}")
            results.append(
                {
                    "model": "M1: CNN+LSTM",
                    "status": "failed",
                    "error": str(e),
                }
            )

    # Train M2: CNN-Only
    if not args.skip_m2:
        try:
            result = train_m2_cnn_only(
                args.data_location, output_dir, gpu=not args.no_gpu, gpu_device=args.gpu_device
            )
            results.append(result)
        except Exception as e:
            log.error(f"Failed to train M2: {e}")
            results.append(
                {
                    "model": "M2: CNN-Only",
                    "status": "failed",
                    "error": str(e),
                }
            )

    # Train M3: Random Forest
    if not args.skip_m3:
        try:
            result = train_m3_random_forest(
                args.data_location, output_dir, args.channel_indices
            )
            results.append(result)
        except Exception as e:
            log.error(f"Failed to train M3: {e}")
            results.append(
                {
                    "model": "M3: Random Forest",
                    "status": "failed",
                    "error": str(e),
                }
            )

    # Generate comparison report
    generate_comparison_report(results, output_dir)

    log.info("=" * 80)
    log.info("Model Comparison Complete!")
    log.info(f"Results saved to: {output_dir}")
    log.info("=" * 80)


if __name__ == "__main__":
    main()

