#!/usr/bin/env python
"""Test GPU4PySCF with different GPU device IDs."""

import os
import sys
import glob

import time
import numpy as np

try:
    import cupy as cp
    print(f"CuPy version: {cp.__version__}")
    print(f"Number of GPUs available: {cp.cuda.runtime.getDeviceCount()}")
except Exception as e:
    print(f"CuPy error: {e}")
    sys.exit(1)

from escflow_eval_utils import init_pyscf_mf

def test_gpu_device(gpu_id):
    """Test a specific GPU device."""
    # Water molecule (H2O)
    atoms = np.array([8, 1, 1])
    pos = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.757, 0.587],
        [0.0, -0.757, 0.587]
    ])

    print(f"\n{'='*60}")
    print(f"Testing GPU {gpu_id}")
    print(f"{'='*60}")

    try:
        calc_mf = init_pyscf_mf(atoms, pos, unit="ang", xc="pbe", basis="def2svp", use_gpu=gpu_id)
        calc_mf.conv_tol = 1e-7

        start = time.time()
        energy = calc_mf.kernel()
        elapsed = time.time() - start

        print(f"✓ GPU {gpu_id} calculation successful!")
        print(f"  Energy: {energy:.6f} Hartree")
        print(f"  Time: {elapsed:.4f}s")
        return True, elapsed
    except Exception as e:
        print(f"✗ GPU {gpu_id} calculation failed: {e}")
        return False, 0.0

def test_cpu():
    """Test CPU calculation."""
    atoms = np.array([8, 1, 1])
    pos = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.757, 0.587],
        [0.0, -0.757, 0.587]
    ])

    print(f"\n{'='*60}")
    print(f"Testing CPU (use_gpu=-1)")
    print(f"{'='*60}")

    try:
        calc_mf = init_pyscf_mf(atoms, pos, unit="ang", xc="pbe", basis="def2svp", use_gpu=-1)
        calc_mf.conv_tol = 1e-7

        start = time.time()
        energy = calc_mf.kernel()
        elapsed = time.time() - start

        print(f"✓ CPU calculation successful!")
        print(f"  Energy: {energy:.6f} Hartree")
        print(f"  Time: {elapsed:.4f}s")
        return True, elapsed
    except Exception as e:
        print(f"✗ CPU calculation failed: {e}")
        return False, 0.0

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_ids", type=str, default="0", help="Comma-separated GPU IDs to test (e.g., '0,1,2')")
    parser.add_argument("--test_cpu", action="store_true", help="Also test CPU")
    args = parser.parse_args()

    gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(',')]

    results = {}

    # Test GPUs
    for gpu_id in gpu_ids:
        success, time_taken = test_gpu_device(gpu_id)
        results[f"GPU {gpu_id}"] = (success, time_taken)

    # Test CPU if requested
    if args.test_cpu:
        success, time_taken = test_cpu()
        results["CPU"] = (success, time_taken)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for device, (success, time_taken) in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        time_str = f"{time_taken:.4f}s" if success else "N/A"
        print(f"{device:15s} {status:10s} {time_str:>10s}")
