#!/usr/bin/env python3
"""
Benchmark Inference Speed for M1 4-Channel Variants.

This script measures and compares inference speed across all three
M1 4-channel model variants:
- Variant 1 (Desktop/Scaled): Maximum accuracy
- Variant 2 (Optimized): Attention + BiLSTM
- Variant 3 (Efficient): Pure CNN for speed

Usage:
    # Benchmark single variant
    python benchmark_inference.py --variant 1 --device cpu
    python benchmark_inference.py --variant 2 --device cuda

    # Compare all variants
    python benchmark_inference.py --compare-all --device cpu

    # Load from checkpoint
    python benchmark_inference.py --checkpoint path/to/model.ckpt --variant 1
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from tabulate import tabulate

from generic_neuromotor_interface.networks_isolated import (
    M1_4Channel_Desktop,
    M1_4Channel_Efficient,
    M1_4Channel_Optimized,
    M1_4Channel_TinyML,
    count_parameters,
)


def benchmark_model(
    model: torch.nn.Module,
    input_shape: tuple = (1, 4, 2000),
    num_warmup: int = 10,
    num_iter: int = 100,
    device: str = "cpu",
) -> dict:
    """
    Benchmark inference speed for a model.

    Parameters
    ----------
    model : torch.nn.Module
        Model to benchmark
    input_shape : tuple
        Input tensor shape (batch, channels, time)
    num_warmup : int
        Number of warmup iterations
    num_iter : int
        Number of benchmark iterations
    device : str
        Device to run on ('cpu' or 'cuda')

    Returns
    -------
    dict
        Benchmark results with timing statistics
    """
    model.eval()
    model.to(device)

    # Warmup
    dummy = torch.randn(*input_shape, device=device)
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy)

    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(num_iter):
            x = torch.randn(*input_shape, device=device)

            start = time.perf_counter()
            _ = model(x)
            if device == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()

            times.append((end - start) * 1000)  # Convert to ms

    return {
        "mean_ms": np.mean(times),
        "std_ms": np.std(times),
        "min_ms": np.min(times),
        "max_ms": np.max(times),
        "p50_ms": np.percentile(times, 50),
        "p95_ms": np.percentile(times, 95),
        "p99_ms": np.percentile(times, 99),
    }


def create_model(variant: int, checkpoint: str = None) -> torch.nn.Module:
    """
    Create model for specified variant.

    Parameters
    ----------
    variant : int
        Variant number (1, 2, or 3)
    checkpoint : str, optional
        Path to checkpoint to load

    Returns
    -------
    torch.nn.Module
        Instantiated model
    """
    if variant == 1:
        model = M1_4Channel_Desktop()
    elif variant == 2:
        model = M1_4Channel_Optimized()
    elif variant == 3:
        model = M1_4Channel_Efficient()
    else:
        raise ValueError(f"Invalid variant: {variant}. Must be 1, 2, or 3")

    if checkpoint:
        print(f"Loading checkpoint: {checkpoint}")
        state_dict = torch.load(checkpoint, map_location="cpu")
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        model.load_state_dict(state_dict)

    return model


def get_variant_name(variant: int) -> str:
    """Get descriptive name for variant."""
    names = {
        1: "Variant 1 (Desktop/Scaled)",
        2: "Variant 2 (Optimized/Attention)",
        3: "Variant 3 (Efficient/Fast CNN)",
    }
    return names.get(variant, f"Variant {variant}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark M1 4-channel model inference speed"
    )
    parser.add_argument(
        "--variant",
        type=int,
        choices=[1, 2, 3],
        help="Model variant to benchmark (1=Desktop, 2=Optimized, 3=Efficient)",
    )
    parser.add_argument(
        "--checkpoint", type=str, help="Path to model checkpoint (.ckpt or .pt)"
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device to run on",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Batch size for inference"
    )
    parser.add_argument(
        "--num-iter",
        type=int,
        default=100,
        help="Number of benchmark iterations",
    )
    parser.add_argument(
        "--compare-all",
        action="store_true",
        help="Compare all three variants",
    )
    args = parser.parse_args()

    # Validate arguments
    if not args.compare_all and args.variant is None:
        parser.error("Must specify either --variant or --compare-all")

    if args.compare_all and args.checkpoint:
        parser.error("Cannot use --checkpoint with --compare-all")

    print("\n" + "=" * 70)
    print("M1 4-Channel Model Inference Benchmark")
    print("=" * 70)

    if args.compare_all:
        # Benchmark all three variants
        variants_to_test = [1, 2, 3]
        print(f"\nComparing all variants on {args.device.upper()}")
        print(f"Batch size: {args.batch_size}")
        print(f"Iterations: {args.num_iter}")
        print(f"Input shape: ({args.batch_size}, 4, 2000)\n")

        results_table = []
        for variant in variants_to_test:
            print(f"Benchmarking {get_variant_name(variant)}...")
            model = create_model(variant)
            params = count_parameters(model)

            input_shape = (args.batch_size, 4, 2000)
            results = benchmark_model(
                model,
                input_shape,
                num_iter=args.num_iter,
                device=args.device,
            )

            # Calculate throughput
            throughput = 1000 / results["mean_ms"] * args.batch_size

            # Real-time factor (2000 samples @ 2kHz = 1000ms)
            window_duration_ms = 1000
            rtf = results["mean_ms"] / window_duration_ms

            results_table.append(
                [
                    get_variant_name(variant),
                    f"{params/1000:.1f}K",
                    f"{results['mean_ms']:.2f}",
                    f"{results['std_ms']:.2f}",
                    f"{results['p50_ms']:.2f}",
                    f"{results['p95_ms']:.2f}",
                    f"{throughput:.1f}",
                    f"{rtf:.3f}x",
                ]
            )

        # Print comparison table
        headers = [
            "Variant",
            "Params",
            "Mean (ms)",
            "Std (ms)",
            "P50 (ms)",
            "P95 (ms)",
            "Throughput\n(samples/s)",
            "Real-time\nFactor",
        ]
        print("\n" + "=" * 70)
        print("Benchmark Results")
        print("=" * 70)
        print(
            tabulate(
                results_table, headers=headers, tablefmt="grid", stralign="right"
            )
        )

        # Print relative speed comparison
        print("\n" + "=" * 70)
        print("Relative Speed Comparison (baseline = Variant 1)")
        print("=" * 70)
        baseline_ms = float(results_table[0][2])
        for i, row in enumerate(results_table):
            variant_ms = float(row[2])
            speedup = baseline_ms / variant_ms
            print(
                f"{row[0]}: {speedup:.2f}x {'faster' if speedup > 1 else 'slower'}"
            )

    else:
        # Benchmark single variant
        print(f"\nBenchmarking: {get_variant_name(args.variant)}")
        print(f"Device: {args.device.upper()}")
        print(f"Batch size: {args.batch_size}")
        print(f"Iterations: {args.num_iter}")

        model = create_model(args.variant, args.checkpoint)
        params = count_parameters(model)

        print(f"\nModel Info:")
        print(f"  Parameters: {params:,}")
        print(f"  Model size (FP32): {params * 4 / 1024 / 1024:.2f} MB")

        # Benchmark
        input_shape = (args.batch_size, 4, 2000)
        print(f"\nRunning benchmark...")
        results = benchmark_model(
            model, input_shape, num_iter=args.num_iter, device=args.device
        )

        # Print results
        print(f"\n{'='*70}")
        print("Benchmark Results")
        print("=" * 70)
        print(f"  Mean:   {results['mean_ms']:.2f} ± {results['std_ms']:.2f} ms")
        print(f"  Median: {results['p50_ms']:.2f} ms")
        print(f"  P95:    {results['p95_ms']:.2f} ms")
        print(f"  P99:    {results['p99_ms']:.2f} ms")
        print(f"  Range:  [{results['min_ms']:.2f}, {results['max_ms']:.2f}] ms")

        # Throughput
        throughput = 1000 / results["mean_ms"] * args.batch_size
        print(f"\n  Throughput: {throughput:.1f} samples/sec")

        # Real-time factor
        window_duration_ms = 1000  # 2000 samples @ 2kHz = 1000ms
        rtf = results["mean_ms"] / window_duration_ms
        rtx = window_duration_ms / results["mean_ms"]
        print(
            f"  Real-time factor: {rtf:.3f}x ({rtx:.1f}x faster than real-time)"
        )

        # Expected accuracy (from design)
        expected_acc = {
            1: "99.3-99.6%",
            2: "99.55-99.65%",
            3: "98.0-99.0%",
        }
        print(f"\nExpected Accuracy: {expected_acc[args.variant]}")

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
