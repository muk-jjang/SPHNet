#!/usr/bin/env python
"""
Test script to verify GPU device selection works correctly.
This script checks that only the specified GPU is used.
"""
import os
import sys

def test_gpu_selection(gpu_id):
    """Test that only the specified GPU device is used."""
    # Set CUDA_VISIBLE_DEVICES before importing any GPU libraries
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    print(f"Set CUDA_VISIBLE_DEVICES={gpu_id}")

    # Now import GPU libraries
    try:
        import cupy as cp
        print(f"CuPy version: {cp.__version__}")

        # Check which devices CuPy sees
        n_devices = cp.cuda.runtime.getDeviceCount()
        print(f"CuPy sees {n_devices} device(s)")

        if n_devices != 1:
            print(f"ERROR: Expected to see 1 device, but saw {n_devices}")
            return False

        # Get device properties
        device_id = cp.cuda.runtime.getDevice()
        props = cp.cuda.runtime.getDeviceProperties(device_id)
        print(f"Using device: {props['name'].decode()}")

        # Create a simple array to verify GPU works
        x = cp.array([1, 2, 3, 4, 5])
        y = x * 2
        print(f"Simple CuPy test: {x.get()} * 2 = {y.get()}")

        print("\n✓ GPU device selection test PASSED")
        return True

    except ImportError:
        print("ERROR: CuPy not installed")
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_gpu_device_selection.py <gpu_id>")
        print("Example: python test_gpu_device_selection.py 1")
        sys.exit(1)

    gpu_id = int(sys.argv[1])
    print(f"Testing GPU device selection for GPU {gpu_id}\n")

    success = test_gpu_selection(gpu_id)
    sys.exit(0 if success else 1)
