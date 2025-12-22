#!/usr/bin/env python
"""
Test that gpu4pyscf uses only the specified GPU device.
"""
import os
import sys

def test_gpu4pyscf_device(gpu_id):
    """Test gpu4pyscf with specific GPU device."""
    # CRITICAL: Set CUDA_VISIBLE_DEVICES BEFORE importing gpu4pyscf
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    print(f"Set CUDA_VISIBLE_DEVICES={gpu_id}")

    # Now import libraries
    from pyscf import gto
    from gpu4pyscf import dft as gpu_dft
    import cupy as cp

    print(f"CuPy sees {cp.cuda.runtime.getDeviceCount()} device(s)")

    # Create a simple molecule
    mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g')
    print(f"Created H2 molecule")

    # Initialize GPU DFT
    mf = gpu_dft.RKS(mol).density_fit()
    mf.xc = 'pbe'
    mf.verbose = 0

    print("Running GPU DFT calculation...")
    energy = mf.kernel()
    print(f"Energy: {energy:.8f} Hartree")

    # Check memory usage
    print(f"\nGPU memory info:")
    mempool = cp.get_default_memory_pool()
    print(f"  Used: {mempool.used_bytes() / 1024**2:.1f} MB")
    print(f"  Total: {mempool.total_bytes() / 1024**2:.1f} MB")

    print("\n✓ gpu4pyscf device test PASSED")
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_gpu4pyscf_device.py <gpu_id>")
        print("Example: python test_gpu4pyscf_device.py 1")
        sys.exit(1)

    gpu_id = int(sys.argv[1])
    print(f"Testing gpu4pyscf on GPU {gpu_id}\n")

    try:
        success = test_gpu4pyscf_device(gpu_id)
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
