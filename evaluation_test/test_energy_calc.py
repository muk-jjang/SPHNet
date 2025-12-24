#!/usr/bin/env python
"""Test energy calculation with GPU4PySCF to verify density matrix conversion."""

import os
import sys
import glob


import time
import numpy as np
import torch
from escflow_eval_utils import init_pyscf_mf, calc_dm0_from_ham

def test_energy_calculation(use_gpu=-1):
    """Test energy calculation with custom density matrix."""
    # Water molecule (H2O)
    atoms = np.array([8, 1, 1])
    pos = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.757, 0.587],
        [0.0, -0.757, 0.587]
    ])

    device_name = f"GPU {use_gpu}" if use_gpu >= 0 else "CPU"
    print(f"\n{'='*60}")
    print(f"Testing energy calculation on {device_name}")
    print(f"{'='*60}")

    try:
        # Initialize mean field
        calc_mf = init_pyscf_mf(atoms, pos, unit="ang", xc="pbe", basis="def2svp", use_gpu=use_gpu)
        calc_mf.conv_tol = 1e-7

        # Run SCF to get reference
        print(f"Running SCF calculation...")
        start = time.time()
        energy_scf = calc_mf.kernel()
        scf_time = time.time() - start
        print(f"  SCF Energy: {energy_scf:.6f} Hartree ({scf_time:.3f}s)")

        # Get Hamiltonian and overlap from the converged calculation
        calc_ham_np = calc_mf.get_fock(dm=calc_mf.make_rdm1())
        calc_overlap_np = calc_mf.get_ovlp()

        # Convert to torch tensors
        calc_overlap = torch.tensor(calc_overlap_np, dtype=torch.float64).unsqueeze(0)
        calc_ham = torch.tensor(calc_ham_np, dtype=torch.float64).unsqueeze(0)

        # Test with return_tensor=False (NumPy mode)
        print(f"\nTesting calc_dm0_from_ham with return_tensor=False...")
        calc_density_np, calc_res_np = calc_dm0_from_ham(
            torch.tensor(atoms), calc_overlap, calc_ham,
            transform=False, return_tensor=False
        )
        print(f"  Density type: {type(calc_density_np)}")

        # Calculate energy with numpy density
        if use_gpu >= 0:
            import cupy as cp
            calc_density_gpu = cp.asarray(calc_density_np)
            energy_np = calc_mf.energy_tot(calc_density_gpu)
        else:
            energy_np = calc_mf.energy_tot(calc_density_np)
        print(f"  Energy (numpy path): {energy_np:.6f} Hartree")

        # Test with return_tensor=True (Tensor mode)
        print(f"\nTesting calc_dm0_from_ham with return_tensor=True...")
        calc_density_tensor, calc_res_tensor = calc_dm0_from_ham(
            torch.tensor(atoms), calc_overlap, calc_ham,
            transform=False, return_tensor=True
        )
        print(f"  Density type: {type(calc_density_tensor)}")

        # Calculate energy with tensor density
        if use_gpu >= 0:
            import cupy as cp
            calc_density_gpu = cp.asarray(calc_density_tensor.cpu().numpy())
            energy_tensor = calc_mf.energy_tot(calc_density_gpu)
        else:
            calc_density_np_from_tensor = calc_density_tensor.cpu().numpy()
            energy_tensor = calc_mf.energy_tot(calc_density_np_from_tensor)
        print(f"  Energy (tensor path): {energy_tensor:.6f} Hartree")

        # Compare results
        print(f"\n{'='*60}")
        print(f"Energy Comparison:")
        print(f"  SCF Energy:      {energy_scf:.8f} Hartree")
        print(f"  NumPy path:      {energy_np:.8f} Hartree (Δ = {abs(energy_np - energy_scf):.2e})")
        print(f"  Tensor path:     {energy_tensor:.8f} Hartree (Δ = {abs(energy_tensor - energy_scf):.2e})")
        print(f"{'='*60}")

        # Check if energies match (within numerical precision)
        tol = 1e-6
        if abs(energy_np - energy_scf) < tol and abs(energy_tensor - energy_scf) < tol:
            print(f"✓ {device_name}: All energy calculations agree!")
            return True
        else:
            print(f"✗ {device_name}: Energy mismatch detected!")
            return False

    except Exception as e:
        print(f"✗ {device_name} calculation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_gpu", type=int, default=-1, help="GPU device ID (-1 for CPU)")
    args = parser.parse_args()

    success = test_energy_calculation(use_gpu=args.use_gpu)
    sys.exit(0 if success else 1)
