# GPU Batch Processing for MD17 Evaluation

## TL;DR - Quick Usage

You have **8 GPUs available**. To leverage all of them for batch processing:

```bash
/data/miniconda3/envs/sphnet_gpueval/bin/python md17_evaluation_gpu_batch.py \
    --dir_path /path/to/your/data \
    --batch_size 8 \
    --size_limit 0  # Process all molecules
```

## What's Different?

### Before (Sequential CPU/GPU):
```
Molecule 1 → Process → Done
Molecule 2 → Process → Done
Molecule 3 → Process → Done
...
```
Time = N × (time per molecule)

### After (Parallel Multi-GPU Batching):
```
GPU 0: Molecule 1 → Process → Done
GPU 1: Molecule 2 → Process → Done
GPU 2: Molecule 3 → Process → Done
GPU 3: Molecule 4 → Process → Done
GPU 4: Molecule 5 → Process → Done
GPU 5: Molecule 6 → Process → Done
GPU 6: Molecule 7 → Process → Done
GPU 7: Molecule 8 → Process → Done
(All happening simultaneously)
```
Time = N/8 × (time per molecule)

## How It Works

The new `md17_evaluation_gpu_batch.py` script:

1. **Detects available GPUs** (you have 8)
2. **Distributes molecules** across GPUs using round-robin assignment
3. **Processes in parallel** using ThreadPoolExecutor
4. **Each GPU gets one molecule** at a time to avoid memory issues
5. **Automatic GPU assignment**: Molecule 0→GPU0, Molecule 1→GPU1, ..., Molecule 8→GPU0, etc.

## Performance Estimation

Assuming each molecule takes ~15 seconds on GPU:

| Dataset Size | Sequential (1 GPU) | Batched (8 GPUs) | Speedup |
|--------------|-------------------|------------------|---------|
| 100 molecules | ~25 minutes | ~3.1 minutes | ~8x |
| 1000 molecules | ~4.2 hours | ~31 minutes | ~8x |
| 10000 molecules | ~41.7 hours | ~5.2 hours | ~8x |

## Two Scripts Available

### 1. `md17_evaluation_customv2.py` (Original + GPU support)
- **Use case**: Single GPU, sequential processing, debugging
- **Command**: `--use_gpu --num_procs 1`
- **Processes**: One molecule at a time

### 2. `md17_evaluation_gpu_batch.py` (NEW - Multi-GPU batching)
- **Use case**: Multiple GPUs, parallel processing, production runs
- **Command**: `--batch_size 8`
- **Processes**: 8 molecules simultaneously (one per GPU)

## Example Commands

### Debug mode (test with 8 molecules):
```bash
/data/miniconda3/envs/sphnet_gpueval/bin/python md17_evaluation_gpu_batch.py \
    --dir_path /path/to/data \
    --debug
```

### Process 100 molecules:
```bash
/data/miniconda3/envs/sphnet_gpueval/bin/python md17_evaluation_gpu_batch.py \
    --dir_path /path/to/data \
    --size_limit 100 \
    --batch_size 8
```

### Process entire dataset:
```bash
/data/miniconda3/envs/sphnet_gpueval/bin/python md17_evaluation_gpu_batch.py \
    --dir_path /path/to/data \
    --batch_size 8
```

### Use only 4 GPUs (if some are busy):
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 /data/miniconda3/envs/sphnet_gpueval/bin/python md17_evaluation_gpu_batch.py \
    --dir_path /path/to/data \
    --batch_size 4
```

## Implementation Details

### GPU Assignment
- Uses modulo assignment: `gpu_id = molecule_idx % NUM_GPUS`
- Example with 8 GPUs:
  - Molecule 0 → GPU 0
  - Molecule 1 → GPU 1
  - ...
  - Molecule 7 → GPU 7
  - Molecule 8 → GPU 0 (wraps around)
  - Molecule 9 → GPU 1
  - etc.

### Threading vs Multiprocessing
- Uses **ThreadPoolExecutor** instead of multiprocessing
- Threads are lighter weight for I/O-bound GPU operations
- Each thread sets its own GPU device
- Better GPU memory management

### Memory Management
- Each GPU processes one molecule at a time
- No batching within a single GPU (to avoid OOM)
- GPU4PySCF handles internal GPU memory allocation

## Monitoring

Check GPU utilization while running:
```bash
watch -n 1 nvidia-smi
```

You should see all 8 GPUs active with similar utilization.

## Important Notes

1. **CUDA Environment**: Automatically set in the script
2. **First run overhead**: ~5s GPU initialization per GPU
3. **Molecule size matters**: GPU benefits increase with molecule size
4. **Thread safety**: Each GPU calculation is independent
5. **Error handling**: Failed molecules are logged, don't stop batch

## Troubleshooting

**Problem**: Only GPU 0 is being used
- **Solution**: Check `CUDA_VISIBLE_DEVICES` is not set to restrict GPUs

**Problem**: Out of memory errors
- **Solution**: Reduce `--batch_size` (e.g., to 4 instead of 8)

**Problem**: Slower than expected
- **Solution**: Check molecule size (GPU needs >20 atoms to be efficient)

**Problem**: Thread conflicts
- **Solution**: The script handles GPU device assignment automatically

## Files Created

- `md17_evaluation_gpu_batch.py` - Main batched evaluation script
- `GPU_ACCELERATION_GUIDE.md` - Complete GPU acceleration documentation
- `GPU_BATCH_SUMMARY.md` - This file (quick reference)
- Test scripts for validation

## Next Steps

1. Test with a small batch first (`--debug`)
2. Verify all GPUs are being used (`nvidia-smi`)
3. Run on your full dataset
4. Compare timing with single GPU mode
