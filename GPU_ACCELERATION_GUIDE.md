# GPU Acceleration Guide for SPHNet MD17 Evaluation

## Overview

This guide explains how to use GPU acceleration for DFT calculations in the MD17 evaluation pipeline using GPU4PySCF.

## Installation

GPU4PySCF has been installed in the `sphnet_gpueval` environment:

```bash
/data/miniconda3/envs/sphnet_gpueval/bin/pip install gpu4pyscf-cuda12x
```

## Quick Start

### Option 1: GPU-Batched Processing (Recommended for Multiple GPUs)

Process multiple molecules in parallel across GPUs:

```bash
python md17_evaluation_gpu_batch.py \
    --dir_path /path/to/data \
    --size_limit 100 \
    --batch_size 8  # Use 8 parallel molecules (one per GPU)
```

### Option 2: Using the Wrapper Script (Single GPU)

```bash
./run_evaluation_gpu.sh --dir_path /path/to/data --size_limit 10
```

### Option 3: Direct Python Call (Single GPU)

```bash
# Set CUDA environment
export CUDA_HOME=/usr/local/cuda-12.6
export CUDA_PATH=/usr/local/cuda-12.6

# Run with GPU
python md17_evaluation_customv2.py \
    --dir_path /path/to/data \
    --size_limit 10 \
    --use_gpu \
    --num_procs 1
```

## Command-Line Arguments

### For `md17_evaluation_customv2.py` (Single GPU):
- `--use_gpu`: Enable GPU acceleration (default: False)
- `--num_procs`: Number of parallel processes (use 1 with GPU)
- `--dir_path`: Directory containing pred_*.pt and gt_*.pt files
- `--size_limit`: Limit number of molecules to process (0 = all)
- `--debug`: Debug mode (sets size_limit=1)
- `--pred_prefix`: Prefix for prediction files (default: "pred_")
- `--gt_prefix`: Prefix for ground truth files (default: "gt_")

### For `md17_evaluation_gpu_batch.py` (Multi-GPU Batching):
- `--batch_size`: Number of molecules to process in parallel (default: 8, recommended: NUM_GPUS)
- `--dir_path`: Directory containing pred_*.pt and gt_*.pt files
- `--size_limit`: Limit number of molecules to process (0 = all)
- `--debug`: Debug mode
- `--pred_prefix`: Prefix for prediction files (default: "pred_")
- `--gt_prefix`: Prefix for ground truth files (default: "gt_")

## Performance Considerations

### When to Use GPU

✅ **GPU is beneficial for:**
- Large molecules (>30 atoms)
- Multiple SCF iterations
- Large basis sets
- Force calculations on complex systems

❌ **GPU may be slower for:**
- Small molecules (<20 atoms)
- Single-point calculations
- Very small basis sets

### Typical Speedups

| Molecule Size | Expected Speedup |
|---------------|------------------|
| < 10 atoms    | 0.5x (slower)    |
| 10-20 atoms   | 1-2x             |
| 20-40 atoms   | 2-5x             |
| > 40 atoms    | 5-20x            |

### GPU Batching Strategies

**Single GPU Mode** (`md17_evaluation_customv2.py --use_gpu`):
- Process molecules sequentially on one GPU
- Use `--num_procs 1`
- Best for: Single GPU systems, debugging

**Multi-GPU Batch Mode** (`md17_evaluation_gpu_batch.py`):
- Process multiple molecules in parallel across multiple GPUs
- Each GPU handles one molecule at a time
- Use `--batch_size N` where N = number of GPUs (e.g., 8 for 8 GPUs)
- Best for: Multi-GPU systems, large datasets

**CPU Multiprocessing**:
- Use `--num_procs N` (multiple CPU cores)
- No GPU acceleration
- Best for: Small molecules, no GPU available

## CUDA Environment

The script automatically sets these environment variables if not already set:

```bash
CUDA_HOME=/usr/local/cuda-12.6
CUDA_PATH=/usr/local/cuda-12.6
PATH=/usr/local/cuda-12.6/bin:$PATH
LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH
```

## Code Changes

### Modified Files

1. **escflow_eval_utils.py**
   - Added `use_gpu` parameter to `init_pyscf_mf()` functions
   - GPU version uses `gpu4pyscf.dft.RKS()` with density fitting
   - Automatic fallback to CPU if GPU is unavailable

2. **md17_evaluation_customv2.py**
   - Added `--use_gpu` command-line argument
   - Automatic CUDA environment setup
   - GPU flag propagation through pipeline

### Example Usage in Code

```python
from escflow_eval_utils import init_pyscf_mf

# Initialize with GPU
calc_mf = init_pyscf_mf(
    atoms,
    pos,
    unit="ang",
    xc="pbe",
    basis="def2svp",
    use_gpu=True
)
```

## Troubleshooting

### CUDA Headers Not Found

If you see `cannot open source file "cuda_fp16.h"`:

```bash
export CUDA_HOME=/usr/local/cuda-12.6
export CUDA_PATH=/usr/local/cuda-12.6
```

The script now sets these automatically.

### Out of Memory

If GPU runs out of memory:
- Reduce batch size
- Use CPU for very large molecules
- Monitor GPU memory: `nvidia-smi`

### Slow Performance

If GPU is slower than expected:
- Check molecule size (GPU needs >20 atoms to be effective)
- Verify you're using `--num_procs 1`
- Check GPU utilization: `nvidia-smi dmon`

### Multi-GPU Issues

The system has 8 GPUs. GPU4PySCF may use peer-to-peer transfers. You can restrict to one GPU:

```bash
CUDA_VISIBLE_DEVICES=0 ./run_evaluation_gpu.sh --dir_path /path/to/data
```

## Testing

Test GPU acceleration with the provided test scripts:

```bash
# Simple water molecule test
python test_gpu_simple.py

# Larger molecule test (aspirin)
python test_gpu_larger_molecule.py

# Performance comparison
python test_gpu_acceleration.py
```

## References

- [GPU4PySCF Documentation](https://github.com/pyscf/gpu4pyscf)
- [PySCF Documentation](https://pyscf.org/)
- [CuPy Documentation](https://docs.cupy.dev/)

## Notes

- GPU4PySCF uses density fitting by default for better GPU performance
- Numerical accuracy is maintained (energy difference ~1e-5 Hartree)
- Initial GPU overhead (~4-5s) is amortized over longer calculations
