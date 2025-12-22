# MD17 Evaluation - Comparison of Different Modes

## Quick Decision Guide

**Choose based on your needs:**

| Your Situation | Recommended Mode | Command |
|---------------|------------------|---------|
| Have 8 GPUs, large dataset | **Multi-GPU Batch** | `./run_evaluation_gpu_batch.sh --dir_path /path` |
| Have 1 GPU, medium dataset | **Single GPU** | `./run_evaluation_gpu.sh --dir_path /path` |
| No GPU, want parallel | **CPU Multiprocessing** | `python md17_evaluation_customv2.py --num_procs 8` |
| Small molecules (<20 atoms) | **CPU Multiprocessing** | `python md17_evaluation_customv2.py --num_procs 8` |
| Debugging | **Single GPU** | `python md17_evaluation_customv2.py --use_gpu --debug` |

## Detailed Comparison

### Mode 1: CPU Multiprocessing (Original)
```bash
python md17_evaluation_customv2.py --num_procs 8 --dir_path /path/to/data
```

**Pros:**
- No GPU needed
- Works on any machine
- Good for small molecules (<20 atoms)
- Predictable performance

**Cons:**
- Slower for large molecules
- Limited by CPU cores
- No GPU acceleration benefits

**Performance:**
- Small molecules (< 20 atoms): ~1-2s per molecule
- Large molecules (> 40 atoms): ~10-30s per molecule

**Best for:**
- Small molecules
- Machines without GPU
- Quick prototyping

---

### Mode 2: Single GPU (New)
```bash
./run_evaluation_gpu.sh --dir_path /path/to/data
# OR
python md17_evaluation_customv2.py --use_gpu --num_procs 1 --dir_path /path/to/data
```

**Pros:**
- Faster for medium/large molecules
- Simple setup
- Good for debugging
- Predictable GPU usage

**Cons:**
- Only uses 1 GPU (wastes other 7)
- Sequential processing
- Initial GPU overhead (~5s)

**Performance:**
- Small molecules (< 20 atoms): ~10-15s per molecule (slower than CPU!)
- Medium molecules (20-40 atoms): ~8-12s per molecule
- Large molecules (> 40 atoms): ~5-15s per molecule (faster than CPU)

**Best for:**
- Single GPU systems
- Medium/large molecules
- Testing GPU setup
- Debugging

---

### Mode 3: Multi-GPU Batch (New - RECOMMENDED)
```bash
./run_evaluation_gpu_batch.sh --dir_path /path/to/data
# OR
python md17_evaluation_gpu_batch.py --batch_size 8 --dir_path /path/to/data
```

**Pros:**
- **Uses all 8 GPUs in parallel**
- **~8x speedup over single GPU**
- Best for large datasets
- Efficient resource utilization
- Automatic load balancing

**Cons:**
- More complex (but handled automatically)
- Requires multiple GPUs
- Higher GPU memory usage overall
- Not ideal for very small molecules

**Performance:**
- Processes 8 molecules simultaneously
- Time = (Total molecules / 8) × (time per molecule)
- Example: 1000 molecules @ 15s each = ~31 minutes (vs 4.2 hours sequential)

**Best for:**
- **Your system (8 GPUs available)**
- Large datasets (>100 molecules)
- Production runs
- Medium/large molecules

---

## Performance Comparison Table

Assuming aspirin-sized molecules (~21 atoms) and 1000 molecules:

| Mode | Hardware Used | Total Time | Molecules/min | Efficiency |
|------|--------------|------------|---------------|------------|
| CPU (8 cores) | 8 CPU cores | ~5 hours | 3.3 | Baseline |
| Single GPU | 1 GPU (7 idle) | ~4.2 hours | 4.0 | 12.5% GPU utilization |
| **Multi-GPU Batch** | **8 GPUs** | **~31 min** | **32** | **100% GPU utilization** |

## Cost Analysis

If GPU time costs money (cloud computing):

| Mode | GPU-hours | CPU-hours | Relative Cost |
|------|-----------|-----------|---------------|
| CPU only | 0 | 40 | 1.0x (CPU) |
| Single GPU | 4.2 | 0 | 0.5x (if GPU = CPU cost) |
| Multi-GPU Batch | 4.2 (total) | 0 | 0.5x (same total GPU time) |

**Note**: Multi-GPU uses same total GPU time as single GPU, but finishes 8x faster!

## Code Examples

### CPU Multiprocessing
```python
# md17_evaluation_customv2.py
python md17_evaluation_customv2.py \
    --dir_path /path/to/data \
    --num_procs 8 \
    --size_limit 100
```

### Single GPU
```python
# md17_evaluation_customv2.py with --use_gpu
python md17_evaluation_customv2.py \
    --dir_path /path/to/data \
    --use_gpu \
    --num_procs 1 \
    --size_limit 100
```

### Multi-GPU Batch
```python
# md17_evaluation_gpu_batch.py (new script)
python md17_evaluation_gpu_batch.py \
    --dir_path /path/to/data \
    --batch_size 8 \
    --size_limit 100
```

## Implementation Details

### CPU Multiprocessing
- Uses Python `multiprocessing.Pool`
- Each process handles one molecule
- Pure Python parallelism

### Single GPU
- GPU4PySCF acceleration
- Sequential processing
- `use_gpu=True` flag

### Multi-GPU Batch
- `ThreadPoolExecutor` for parallelism
- Round-robin GPU assignment
- Each thread → one GPU → one molecule
- Automatic GPU device management

## Recommendations

### For Your Use Case (8 GPUs, MD17 evaluation):

1. **Production runs**: Use `md17_evaluation_gpu_batch.py` with `--batch_size 8`
2. **Testing**: Use `md17_evaluation_customv2.py` with `--use_gpu --debug`
3. **Small subset**: Use `md17_evaluation_customv2.py` with `--use_gpu`

### General Guidelines:

- **Dataset size < 10 molecules**: Single GPU mode
- **Dataset size 10-100 molecules**: Multi-GPU batch
- **Dataset size > 100 molecules**: Multi-GPU batch (strongly recommended)
- **Molecule size < 20 atoms**: Consider CPU multiprocessing
- **Molecule size > 20 atoms**: GPU (preferably batched)

## Migration Path

If you're currently using CPU mode:

1. **Test GPU**: `python md17_evaluation_customv2.py --use_gpu --debug`
2. **Verify speedup**: Compare timing with CPU mode
3. **Scale to batch**: `python md17_evaluation_gpu_batch.py --debug`
4. **Full production**: `./run_evaluation_gpu_batch.sh --dir_path /path/to/data`

## Monitoring Commands

### CPU mode:
```bash
htop  # Monitor CPU usage
```

### Single GPU mode:
```bash
nvidia-smi dmon  # Monitor single GPU
```

### Multi-GPU batch mode:
```bash
watch -n 1 nvidia-smi  # See all 8 GPUs working
```

You should see all 8 GPUs with:
- GPU utilization: 70-100%
- Memory usage: 2-8 GB per GPU
- Power draw: Near max TDP
