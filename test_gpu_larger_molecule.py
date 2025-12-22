#!/usr/bin/env python
"""Test GPU acceleration with a larger molecule (aspirin: C9H8O4, 21 atoms)."""

import time
import numpy as np
from escflow_eval_utils import init_pyscf_mf

def test_aspirin_performance(use_gpu=False):
    """Test DFT calculation for aspirin molecule."""
    # Aspirin molecule (C9H8O4) - 21 atoms
    # Atomic numbers: C=6, H=1, O=8
    atoms = np.array([6,6,6,6,6,6,6,6,6,1,1,1,1,1,1,1,1,8,8,8,8])  # C9H8O4

    # Simplified aspirin geometry (in Angstrom)
    pos = np.array([
        [ 1.206,  0.928,  0.000],  # C
        [ 0.000,  1.643,  0.000],  # C
        [-1.206,  0.928,  0.000],  # C
        [-1.206, -0.468,  0.000],  # C
        [ 0.000, -1.183,  0.000],  # C
        [ 1.206, -0.468,  0.000],  # C
        [ 2.515, -1.239,  0.000],  # C
        [-2.515,  1.704,  0.000],  # C
        [ 3.726, -0.327,  0.000],  # C
        [ 2.106,  1.531,  0.000],  # H
        [ 0.000,  2.722,  0.000],  # H
        [-2.106, -1.072,  0.000],  # H
        [ 0.000, -2.261,  0.000],  # H
        [ 2.522, -1.893,  0.880],  # H
        [ 2.522, -1.893, -0.880],  # H
        [ 3.729,  0.311,  0.890],  # H
        [ 3.729,  0.311, -0.890],  # H
        [ 4.652, -0.900,  0.000],  # O
        [-2.524,  2.914,  0.000],  # O
        [-3.621,  0.982,  0.000],  # O
        [-4.750,  1.631,  0.000],  # O
    ])

    print(f"\n{'='*60}")
    print(f"Testing {'GPU' if use_gpu else 'CPU'} - Aspirin (C9H8O4, 21 atoms)")
    print(f"{'='*60}")

    # Initialize mean field
    start_time = time.time()
    calc_mf = init_pyscf_mf(atoms, pos, unit="ang", xc="pbe", basis="def2svp", use_gpu=use_gpu)

    if not use_gpu:
        calc_mf.conv_tol = 1e-7
        calc_mf.grids.level = 3
        calc_mf.grids.prune = None
        calc_mf.init_guess = "minao"
        calc_mf.small_rho_cutoff = 1e-12
    else:
        calc_mf.conv_tol = 1e-7
        calc_mf.grids.level = 3

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
    print(f"Max force component: {np.abs(forces).max():.6f} Ha/Bohr")

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
    import os
    print("\n" + "="*60)
    print("GPU4PySCF Performance Test - Aspirin Molecule")
    print("="*60)

    # Test GPU
    try:
        gpu_results = test_aspirin_performance(use_gpu=True)
        # Test CPU
        cpu_results = test_aspirin_performance(use_gpu=False)

        # Compare results
        print("\n" + "="*60)
        print("PERFORMANCE COMPARISON - Aspirin (21 atoms)")
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

        if gpu_results['total_time'] < cpu_results['total_time']:
            speedup = cpu_results['total_time'] / gpu_results['total_time']
            print(f"\n✓ GPU is {speedup:.2f}x FASTER!")
        else:
            slowdown = gpu_results['total_time'] / cpu_results['total_time']
            print(f"\n✗ GPU is {slowdown:.2f}x slower")
            print("Note: GPU acceleration works best for molecules with >30 atoms")

    except Exception as e:
        print(f"\nGPU test encountered an error: {e}")
        import traceback
        traceback.print_exc()
