# GPU Acceleration Guide for SPHNet

This guide explains how to use GPU acceleration via gpu4pyscf in the SPHNet evaluation code.

## Setup Summary

### 1. Dependencies Installed
- **NumPy**: 1.26.4 (required: < 2.0 for gpu4pyscf compatibility)
- **gpu4pyscf-cuda12x**: 1.5.1 (for CUDA 12.x)
- **cupy-cuda12x**: 13.6.0
- **pyscf**: 2.10.0

### 2. CUDA Detection
The code automatically detects CUDA installations at:
- `/usr/local/cuda-12.3` (current server)
- `/usr/local/cuda-12.6` (other server)
- `/usr/local/cuda` (symlink)

## Usage

### Basic Usage

#### CPU Mode (default)
```bash
python md17_evaluation_customv2.py \
    --dir_path=/path/to/data \
    --num_procs=1 \
    --size_limit=10
```

#### GPU Mode - Specific Device
```bash
python md17_evaluation_customv2.py \
    --dir_path=/path/to/data \
    --use_gpu=0 \
    --num_procs=1 \
    --size_limit=10
```

### Command-line Arguments

- `--use_gpu`: GPU device ID to use
  - `-1` (default): Use CPU only
  - `0-7`: Use specific GPU device (0 = first GPU, 1 = second GPU, etc.)

### Examples

```bash
# Use GPU 0
python md17_evaluation_customv2.py --use_gpu=0 --dir_path=/nas/data

# Use GPU 3
python md17_evaluation_customv2.py --use_gpu=3 --dir_path=/nas/data

# Use CPU
python md17_evaluation_customv2.py --use_gpu=-1 --dir_path=/nas/data
# or simply
python md17_evaluation_customv2.py --dir_path=/nas/data
```

## Testing

### Test Single GPU
```bash
python test_gpu_simple.py
```

### Test Multiple GPUs
```bash
# Test GPUs 0, 1, 2 and CPU
python test_multi_gpu.py --gpu_ids=0,1,2 --test_cpu

# Test only GPU 0
python test_multi_gpu.py --gpu_ids=0

# Test GPUs 0-7
python test_multi_gpu.py --gpu_ids=0,1,2,3,4,5,6,7
```

## Performance

Based on H2O test molecule:
- **GPU 0 (first run)**: ~3.7s
- **GPU 1 (warmed up)**: ~0.9s
- **CPU**: ~4.2s

Larger molecules will see more significant GPU speedup.

## Implementation Details

### Code Changes

1. **escflow_eval_utils.py**:
   - `init_pyscf_mf()`: Changed `use_gpu` from bool to int
   - `init_pyscf_mf_()`: Added GPU device selection via `CUDA_VISIBLE_DEVICES`

2. **md17_evaluation_customv2.py**:
   - Added dynamic CUDA detection function
   - Changed `--use_gpu` from flag to integer argument
   - Updated process function to accept GPU ID

### GPU Device Selection

The GPU device is selected using `CUDA_VISIBLE_DEVICES` environment variable:
```python
os.environ['CUDA_VISIBLE_DEVICES'] = str(use_gpu)
```

This ensures each process uses only the specified GPU device.

## Multi-Processing Notes

**WARNING**: When using GPU mode with multiple processes (`--num_procs > 1`), all processes will compete for the same GPU. For best performance:

- **Single GPU**: Use `--num_procs=1`
- **Multiple GPUs**: Manually assign different GPU IDs to different processes

Example for manual multi-GPU usage:
```bash
# Terminal 1 - GPU 0
python md17_evaluation_customv2.py --use_gpu=0 --dir_path=/data/batch1 &

# Terminal 2 - GPU 1
python md17_evaluation_customv2.py --use_gpu=1 --dir_path=/data/batch2 &

# Terminal 3 - GPU 2
python md17_evaluation_customv2.py --use_gpu=2 --dir_path=/data/batch3 &
```

## Troubleshooting

### NumPy Version Error
If you see `np.float_ was removed in NumPy 2.0`:
```bash
pip uninstall numpy -y
pip install "numpy==1.26.4"
```

### CUDA Not Found
If CUDA is not auto-detected:
```bash
export CUDA_HOME=/usr/local/cuda-12.3
export PATH=/usr/local/cuda-12.3/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.3/lib64:$LD_LIBRARY_PATH
```

### GPU Not Available
Check GPU availability:
```bash
nvidia-smi
python -c "import cupy as cp; print(f'GPUs: {cp.cuda.runtime.getDeviceCount()}')"
```

## API Usage in Code

```python
from escflow_eval_utils import init_pyscf_mf

# CPU mode
mf = init_pyscf_mf(atoms, pos, xc="pbe", basis="def2svp", use_gpu=-1)

# GPU 0
mf = init_pyscf_mf(atoms, pos, xc="pbe", basis="def2svp", use_gpu=0)

# GPU 3
mf = init_pyscf_mf(atoms, pos, xc="pbe", basis="def2svp", use_gpu=3)
```

## Server Configuration

### Current Server (G-Drstrange)
- CUDA: 12.3
- GPUs: 8 devices
- CuPy: 13.6.0

### Other Server
- CUDA: 12.6 (auto-detected)
- GPUs: (will auto-detect)

The code works on both servers without modification.
