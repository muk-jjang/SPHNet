# Quick Start: GPU Acceleration

## TL;DR

```bash
# Use GPU 0
python md17_evaluation_customv2.py --use_gpu=0 --dir_path=/your/data/path

# Use GPU 3
python md17_evaluation_customv2.py --use_gpu=3 --dir_path=/your/data/path

# Use CPU (default)
python md17_evaluation_customv2.py --dir_path=/your/data/path
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

## Testing

```bash
# Quick test
python test_gpu_simple.py

# Test multiple GPUs
python test_multi_gpu.py --gpu_ids=0,1,2,3

# Test energy accuracy
python test_energy_calc.py --use_gpu=0
```

## Common Issues

| Error | Solution |
|-------|----------|
| `np.float_` removed | `pip install "numpy==1.26.4"` |
| `gpu4pyscf not installed` | `pip install gpu4pyscf-cuda12x` |
| `nvcc not found` | Code auto-detects CUDA, should work |
| `TypeError: Unsupported type` | Already fixed in code |

## Performance Expectations

- **Small molecules (< 10 atoms)**: GPU ~2-5x speedup
- **Medium molecules (10-20 atoms)**: GPU ~5-10x speedup
- **Large molecules (> 20 atoms)**: GPU > 10x speedup
- **First run**: Slower (GPU warmup)
- **Subsequent runs**: Much faster

## What Changed

1. `--use_gpu` is now an integer (GPU ID) instead of a flag
2. Use `-1` for CPU, `0-7` for GPU devices
3. Automatic CUDA detection (works on both servers)
4. Automatic NumPy→CuPy conversion for GPU calculations

## Need Help?

See detailed documentation:
- [GPU_USAGE_GUIDE.md](GPU_USAGE_GUIDE.md) - User guide
- [GPU_IMPLEMENTATION_SUMMARY.md](GPU_IMPLEMENTATION_SUMMARY.md) - Technical details
