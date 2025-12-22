#!/usr/bin/env python
"""Test script to compare CPU vs GPU performance for DFT calculations."""

import time
import numpy as np
from escflow_eval_utils import init_pyscf_mf

def test_dft_performance(use_gpu=False):
    """Test DFT calculation performance."""
    # Simple water molecule (H2O)
    atoms = np.array([8, 1, 1])  # O, H, H
    pos = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.757, 0.587],
        [0.0, -0.757, 0.587]
    ])

    print(f"\n{'='*60}")
    print(f"Testing {'GPU' if use_gpu else 'CPU'} performance")
    print(f"{'='*60}")

    # Initialize mean field
    start_time = time.time()
    calc_mf = init_pyscf_mf(atoms, pos, unit="ang", xc="pbe", basis="def2svp", use_gpu=use_gpu)
    calc_mf.conv_tol = 1e-7
    calc_mf.grids.level = 3
    calc_mf.grids.prune = None
    calc_mf.init_guess = "minao"
    calc_mf.small_rho_cutoff = 1e-12
    init_time = time.time() - start_time
    print(f"Initialization time: {init_time:.4f}s")

    # Run SCF calculation
    print("Running SCF calculation...")
    scf_start = time.time()
    energy = calc_mf.kernel()
    scf_time = time.time() - scf_start
    print(f"SCF calculation time: {scf_time:.4f}s")
    print(f"Total energy: {energy:.6f} Hartree")

    # Calculate forces
    print("Computing forces...")
    grad_frame = calc_mf.nuc_grad_method()
    force_start = time.time()
    forces = -grad_frame.kernel()
    force_time = time.time() - force_start
    print(f"Force calculation time: {force_time:.4f}s")

    total_time = init_time + scf_time + force_time
    print(f"\nTotal time: {total_time:.4f}s")
    print(f"{'='*60}\n")

    return {
        'init_time': init_time,
        'scf_time': scf_time,
        'force_time': force_time,
        'total_time': total_time,
        'energy': energy
    }

if __name__ == "__main__":
    print("\n" + "="*60)
    print("GPU4PySCF Performance Comparison Test")
    print("="*60)

    # Test CPU
    cpu_results = test_dft_performance(use_gpu=False)

    # Test GPU
    try:
        gpu_results = test_dft_performance(use_gpu=True)

        # Compare results
        print("\n" + "="*60)
        print("PERFORMANCE COMPARISON")
        print("="*60)
        print(f"{'Metric':<25} {'CPU (s)':<12} {'GPU (s)':<12} {'Speedup':<10}")
        print("-"*60)

        for metric in ['init_time', 'scf_time', 'force_time', 'total_time']:
            cpu_time = cpu_results[metric]
            gpu_time = gpu_results[metric]
            speedup = cpu_time / gpu_time if gpu_time > 0 else 0
            metric_name = metric.replace('_', ' ').title()
            print(f"{metric_name:<25} {cpu_time:<12.4f} {gpu_time:<12.4f} {speedup:<10.2f}x")

        print("="*60)
        print(f"\nEnergy difference: {abs(cpu_results['energy'] - gpu_results['energy']):.2e} Hartree")
        print("(Should be very small, confirming numerical accuracy)")

    except ImportError as e:
        print(f"\nGPU test failed: {e}")
        print("Make sure gpu4pyscf is installed: pip install gpu4pyscf-cuda12x")
    except Exception as e:
        print(f"\nGPU test encountered an error: {e}")
