# Quick Start: GPU Acceleration

## TL;DR

```bash
# CPU mode (default)
python md17_evaluation_customv2.py --dir_path=/your/data/path

# Single GPU mode
python md17_evaluation_customv2.py --use_gpu 0 --dir_path=/your/data/path

# Multi-GPU mode (4 GPUs)
python md17_evaluation_customv2.py --use_gpu 0,1,2,3 --num_procs 4 --dir_path=/your/data/path

# Multi-GPU mode (all 8 GPUs) - RECOMMENDED
python md17_evaluation_customv2.py --use_gpu 0,1,2,3,4,5,6,7 --num_procs 8 --dir_path=/your/data/path
```

## Prerequisites Check

```bash
# Check CUDA
nvidia-smi

# Check Python environment
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"  # Should be 1.26.4
python -c "import cupy; print(f'CuPy: {cupy.__version__}')"      # Should be 13.6.0
python -c "from gpu4pyscf import dft; print('GPU4PySCF: OK')"    # Should not error
```

## Usage Modes

### CPU Mode (Default)
```bash
python md17_evaluation_customv2.py \
    --dir_path /path/to/data \
    --num_procs 4
```

### Single GPU Mode
```bash
python md17_evaluation_customv2.py \
    --use_gpu 0 \
    --dir_path /path/to/data
```

### Multi-GPU Mode (Recommended)
```bash
# Use 4 GPUs (requires num_procs=4)
python md17_evaluation_customv2.py \
    --use_gpu 0,1,2,3 \
    --num_procs 4 \
    --dir_path /path/to/data

# Use all 8 GPUs (requires num_procs=8)
python md17_evaluation_customv2.py \
    --use_gpu 0,1,2,3,4,5,6,7 \
    --num_procs 8 \
    --dir_path /path/to/data
```

## Command-Line Arguments

- `--use_gpu`: GPU configuration
  - `None` or `-1`: CPU mode (default)
  - Single ID (e.g., `0`): Single GPU mode
  - Comma-separated (e.g., `0,1,2,3`): Multi-GPU mode
- `--num_procs`: Number of processes
  - For multi-GPU: Must equal number of GPUs
  - For CPU/single GPU: Can be any value
- `--dir_path`: Directory with pred_*.pt and gt_*.pt files
- `--size_limit`: Process only first N molecules (-1 = all)

## Monitoring GPU Usage

```bash
# Watch all GPUs in real-time
watch -n 1 nvidia-smi

# Monitor GPU utilization
nvidia-smi dmon
```

In multi-GPU mode, you should see all specified GPUs active with high utilization (70-100%).

## Performance Comparison

For 1000 aspirin-sized molecules (~21 atoms):

| Mode | Time | Speedup |
|------|------|---------|
| CPU (8 cores) | ~5 hours | 1x |
| Single GPU | ~4.2 hours | 1.2x |
| Multi-GPU (8 GPUs) | ~31 min | 9.7x |

## If Setup Fails

### NumPy 2.0 Error
```bash
pip uninstall numpy -y
pip install "numpy==1.26.4"
```

### GPU4PySCF Missing
```bash
pip install gpu4pyscf-cuda12x
```

### CUDA Not Found
```bash
export CUDA_HOME=/usr/local/cuda-12.6
export CUDA_PATH=/usr/local/cuda-12.6
```

## Multi-GPU Validation

Each process will print a validation message:
```
[Process 12345] GPU validation: Using physical GPU 0 (local device 0: NVIDIA A100-SXM4-40GB)
```

This confirms proper GPU assignment in multi-GPU mode.

## Full Documentation

For detailed information, see [GPU_ACCELERATION_GUIDE.md](GPU_ACCELERATION_GUIDE.md)
