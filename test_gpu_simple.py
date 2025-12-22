#!/usr/bin/env python
"""Simple test to verify GPU4PySCF setup with CUDA environment variables."""

import os
import sys

# Set CUDA environment variables before importing cupy
os.environ['CUDA_HOME'] = '/usr/local/cuda-12.6'
os.environ['CUDA_PATH'] = '/usr/local/cuda-12.6'
os.environ['PATH'] = '/usr/local/cuda-12.6/bin:' + os.environ.get('PATH', '')
os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda-12.6/lib64:' + os.environ.get('LD_LIBRARY_PATH', '')

# Verify CUDA setup
print(f"CUDA_HOME: {os.environ.get('CUDA_HOME')}")
print(f"CUDA_PATH: {os.environ.get('CUDA_PATH')}")

import time
import numpy as np

try:
    import cupy as cp
    print(f"CuPy version: {cp.__version__}")
    print(f"CUDA available: {cp.cuda.is_available()}")
    if cp.cuda.is_available():
        print(f"CUDA runtime version: {cp.cuda.runtime.runtimeGetVersion()}")
        print(f"Number of GPUs: {cp.cuda.runtime.getDeviceCount()}")
except Exception as e:
    print(f"CuPy error: {e}")
    sys.exit(1)

from escflow_eval_utils import init_pyscf_mf

def test_gpu_simple():
    """Test GPU with a very simple molecule."""
    # Water molecule (H2O) - 3 atoms
    atoms = np.array([8, 1, 1])  # O, H, H
    pos = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.757, 0.587],
        [0.0, -0.757, 0.587]
    ])

    print(f"\n{'='*60}")
    print("Testing GPU4PySCF with H2O molecule")
    print(f"{'='*60}")

    try:
        # Initialize with GPU
        calc_mf = init_pyscf_mf(atoms, pos, unit="ang", xc="pbe", basis="def2svp", use_gpu=True)
        calc_mf.conv_tol = 1e-7

        print("Running SCF calculation on GPU...")
        start = time.time()
        energy = calc_mf.kernel()
        elapsed = time.time() - start

        print(f"✓ GPU calculation successful!")
        print(f"  Energy: {energy:.6f} Hartree")
        print(f"  Time: {elapsed:.4f}s")

        return True
    except Exception as e:
        print(f"✗ GPU calculation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_gpu_simple()
    sys.exit(0 if success else 1)
