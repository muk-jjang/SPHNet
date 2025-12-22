# GPU Device Selection Fix

## Problem
When running `md17_evaluation_customv2.py --use_gpu=1`, the script was using **ALL GPUs (0-7)** instead of just GPU 1.

Example nvidia-smi output showing the problem:
```
|    0   N/A  N/A    774400      C   python      948MiB |
|    1   N/A  N/A    774400      C   python      758MiB |
|    2   N/A  N/A    774400      C   python      600MiB |
|    3   N/A  N/A    774400      C   python      676MiB |
|    4   N/A  N/A    774400      C   python      832MiB |
|    5   N/A  N/A    774400      C   python      718MiB |
|    6   N/A  N/A    774400      C   python      754MiB |
|    7   N/A  N/A    774400      C   python      726MiB |
```

## Root Cause
The `CUDA_VISIBLE_DEVICES` environment variable **must be set BEFORE any CUDA/GPU libraries are imported**.

Previously:
1. Script imports happened at the top of the file (before argument parsing)
2. `CUDA_VISIBLE_DEVICES` was set inside `init_pyscf_mf()` function in `escflow_eval_utils.py`
3. By that time, CuPy and other CUDA libraries had already initialized and detected all 8 GPUs

## Solution

### 1. Set `CUDA_VISIBLE_DEVICES` Early in Main Script
In [md17_evaluation_customv2.py](md17_evaluation_customv2.py:300-304), immediately after parsing command-line arguments:

```python
if __name__ == "__main__":
    # Parse arguments first
    args = parser.parse_args()

    # CRITICAL: Set CUDA_VISIBLE_DEVICES immediately after parsing arguments
    # This must happen before ANY GPU library operations (gpu4pyscf, cupy, etc.)
    if args.use_gpu >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.use_gpu)
        print(f"Setting CUDA_VISIBLE_DEVICES={args.use_gpu} before GPU initialization")
```

### 2. Remove Redundant Environment Setting
In [escflow_eval_utils.py](escflow_eval_utils.py:79-80), removed the late `CUDA_VISIBLE_DEVICES` setting since it was too late:

```python
# Removed: os.environ['CUDA_VISIBLE_DEVICES'] = str(use_gpu)
# Note: CUDA_VISIBLE_DEVICES must be set before importing gpu4pyscf
# This should be done at the script level, not here
```

## Verification

Created test scripts to verify the fix:

### Test 1: CuPy Device Selection
```bash
python test_gpu_device_selection.py 1
```

Output confirms only 1 device is visible:
```
Set CUDA_VISIBLE_DEVICES=1
CuPy sees 1 device(s)
✓ GPU device selection test PASSED
```

### Test 2: gpu4pyscf Device Selection
```bash
python test_gpu4pyscf_device.py 3
```

Verified with `nvidia-smi` that only GPU 3 shows increased memory usage during calculation.

## Usage

Now the GPU selection works correctly:

```bash
# Use GPU 0
python md17_evaluation_customv2.py --use_gpu=0 --dir_path=/your/path

# Use GPU 1
python md17_evaluation_customv2.py --use_gpu=1 --dir_path=/your/path

# Use GPU 3
python md17_evaluation_customv2.py --use_gpu=3 --dir_path=/your/path

# Use CPU (default)
python md17_evaluation_customv2.py --dir_path=/your/path
```

## Key Takeaway

**CUDA_VISIBLE_DEVICES must be set before importing any CUDA libraries (CuPy, gpu4pyscf, etc.)**

The correct order is:
1. Parse command-line arguments
2. Set `CUDA_VISIBLE_DEVICES`
3. Import/use GPU libraries
4. Perform GPU calculations

Setting it later (inside functions or after imports) will not work because CUDA runtime initializes on first import and caches the device list.
