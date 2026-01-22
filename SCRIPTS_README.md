# SPHNet Prediction and Plotting Scripts

This directory contains bash scripts for running SPHNet predictions and generating three-method comparison plots (DFT vs QHFlow-v2 vs SPHNet).

## Available Scripts

### 1. `run_sphnet_predictions.sh` - Run SPHNet Predictions

Runs SPHNet model predictions for all molecules and site types using DFT reference data.

**Basic Usage:**
```bash
./run_sphnet_predictions.sh --ckpt-path /path/to/sphnet/model.ckpt
```

**Full Options:**
```bash
./run_sphnet_predictions.sh \
    --ckpt-path /path/to/sphnet/model.ckpt \
    --dft-dir /path/to/dft/results \
    --save-dir ./output \
    --device cuda \
    --model-length-unit ang \
    --molecules "ethanol,aspirin" \
    --dry-run
```

**Options:**
- `--ckpt-path <path>` - Path to SPHNet checkpoint (REQUIRED)
- `--dft-dir <path>` - DFT results directory (default: QHFlow exp output)
- `--save-dir <path>` - Output directory (default: ./output)
- `--device <device>` - cuda or cpu (default: cuda)
- `--model-length-unit <unit>` - ang or bohr (default: ang)
- `--molecules <list>` - Comma-separated molecule list (default: all)
- `--dry-run` - Show commands without executing

**Supported Molecules:**
- `ethanol` (sites: primary, secondary)
- `malondialdehyde` (sites: primary)
- `naphthalene` (sites: primary)
- `salicylic_acid` (sites: primary, secondary)
- `aspirin` (sites: primary)
- `uracil` (sites: primary, secondary)

**Output:**
Creates `.pt` files with naming pattern:
```
{molecule}_{site}_{bond}_ratio-{ratio}_sphnet.pt
{molecule}_{site}_{bond}_sphnet_metadata.pt
```

### 2. `run_three_method_plots.sh` - Generate Comparison Plots

Generates three-method comparison plots for all available results.

**Basic Usage:**
```bash
./run_three_method_plots.sh \
    --dft-dir ./output \
    --qhflow-dir ./output \
    --sphnet-dir ./output
```

**Full Options:**
```bash
./run_three_method_plots.sh \
    --dft-dir /path/to/dft/results \
    --qhflow-dir /path/to/qhflow/results \
    --sphnet-dir /path/to/sphnet/results \
    --save-dir ./plots \
    --molecules "ethanol,aspirin" \
    --dry-run
```

**Options:**
- `--dft-dir <path>` - DFT results directory (REQUIRED)
- `--qhflow-dir <path>` - QHFlow-v2 results directory (REQUIRED)
- `--sphnet-dir <path>` - SPHNet results directory (REQUIRED)
- `--save-dir <path>` - Output directory for plots (default: ./plots)
- `--molecules <list>` - Comma-separated molecule list (default: all)
- `--dry-run` - Show commands without executing

**Features:**
- Automatically checks if all three methods' results exist
- Skips plots for incomplete data (missing any method)
- Provides summary of successful, skipped, and failed plots

**Output:**
Creates PNG files with naming pattern:
```
{molecule}_{site}_{bond}_comparison_three_methods.png
```

### 3. `run_all_experiments.sh` - Complete Workflow

Master script that runs the complete workflow: predictions + plots.

**Basic Usage:**
```bash
./run_all_experiments.sh --ckpt-path /path/to/sphnet/model.ckpt
```

**Full Options:**
```bash
./run_all_experiments.sh \
    --ckpt-path /path/to/sphnet/model.ckpt \
    --dft-dir /path/to/dft/results \
    --qhflow-dir /path/to/qhflow/results \
    --sphnet-save-dir ./output \
    --plots-save-dir ./plots \
    --device cuda \
    --molecules "ethanol,aspirin" \
    --skip-predictions \
    --skip-plots \
    --dry-run
```

**Options:**
- `--ckpt-path <path>` - SPHNet checkpoint (REQUIRED unless --skip-predictions)
- `--dft-dir <path>` - DFT results directory
- `--qhflow-dir <path>` - QHFlow-v2 results directory
- `--sphnet-save-dir <path>` - Output for SPHNet predictions
- `--plots-save-dir <path>` - Output for plots
- `--device <device>` - cuda or cpu
- `--molecules <list>` - Comma-separated molecule list
- `--skip-predictions` - Skip predictions, only plot
- `--skip-plots` - Skip plots, only predict
- `--dry-run` - Show commands without executing

**Workflow:**
1. Runs SPHNet predictions for all molecules/sites
2. Generates three-method comparison plots

## Quick Start Examples

### Example 1: Run Everything (Predictions + Plots)

```bash
# Complete workflow for all molecules
./run_all_experiments.sh \
    --ckpt-path /path/to/sphnet/model.ckpt \
    --dft-dir /home/chanhui-lee/QHFlow-mlff/src/electronic_structure_exp/output \
    --qhflow-dir /home/chanhui-lee/QHFlow-mlff/src/electronic_structure_exp/output
```

### Example 2: Only Run Predictions

```bash
# Run predictions, skip plotting
./run_all_experiments.sh \
    --ckpt-path /path/to/sphnet/model.ckpt \
    --skip-plots
```

### Example 3: Only Generate Plots (Using Existing Results)

```bash
# Use existing predictions to generate plots
./run_all_experiments.sh \
    --dft-dir ./output \
    --qhflow-dir ./output \
    --sphnet-save-dir ./output \
    --skip-predictions
```

### Example 4: Specific Molecules Only

```bash
# Run only for ethanol and aspirin
./run_all_experiments.sh \
    --ckpt-path /path/to/sphnet/model.ckpt \
    --molecules "ethanol,aspirin"
```

### Example 5: Dry Run (Preview Commands)

```bash
# See what would be executed without running
./run_all_experiments.sh \
    --ckpt-path /path/to/sphnet/model.ckpt \
    --dry-run
```

### Example 6: CPU-Only Mode

```bash
# Use CPU instead of GPU
./run_all_experiments.sh \
    --ckpt-path /path/to/sphnet/model.ckpt \
    --device cpu
```

## Prerequisites

Before running these scripts, ensure:

1. **DFT Reference Data** exists:
   ```bash
   # Run in QHFlow-mlff repo
   cd /home/chanhui-lee/QHFlow-mlff/src/electronic_structure_exp
   python exp_bond-stretch_dft.py --molecule ethanol --save-dir ./output
   ```

2. **QHFlow-v2 Predictions** exist:
   ```bash
   # Run in QHFlow-mlff repo
   python exp_bond-stretch_model.py \
       --molecule ethanol \
       --site-type primary \
       --load-results ./output \
       --ckpt-path /path/to/qhflow.ckpt \
       --save-dir ./output
   ```

3. **SPHNet Model Checkpoint** is available

## Directory Structure

After running the complete workflow:

```
SPHNet/
├── run_sphnet_predictions.sh       # Prediction script
├── run_three_method_plots.sh       # Plotting script
├── run_all_experiments.sh          # Master script
├── exp_bond-stretch_model.py       # Python prediction script
├── output/                         # SPHNet predictions
│   ├── ethanol_primary_O2-H8_ratio-0.80_sphnet.pt
│   ├── ethanol_primary_O2-H8_ratio-1.00_sphnet.pt
│   └── ethanol_primary_O2-H8_sphnet_metadata.pt
└── plots/                          # Comparison plots
    └── ethanol_primary_O2-H8_comparison_three_methods.png
```

## Script Features

### Error Handling
- All scripts use `set -e` to exit on first error
- Validation of required arguments before execution
- Check for file/directory existence

### Progress Tracking
- Shows current task number (e.g., [3/12])
- Displays success/failure status for each task
- Provides final summary with counts

### Dry Run Mode
- Preview all commands without execution
- Useful for debugging and verification
- Use `--dry-run` flag with any script

### Selective Execution
- Run specific molecules only with `--molecules`
- Skip predictions or plots with `--skip-*` flags
- Useful for re-running failed tasks

## Troubleshooting

### Problem: "Checkpoint file not found"
**Solution:** Verify the checkpoint path is correct
```bash
ls -lh /path/to/sphnet/model.ckpt
```

### Problem: "DFT results directory not found"
**Solution:** Check if DFT calculations were run
```bash
ls /path/to/dft/results/*dft_metadata.pt
```

### Problem: "Skipping: Missing results for: SPHNet"
**Solution:** Run SPHNet predictions first
```bash
./run_sphnet_predictions.sh --ckpt-path /path/to/model.ckpt
```

### Problem: GPU out of memory
**Solution:** Use CPU mode
```bash
./run_all_experiments.sh --ckpt-path /path/to/model.ckpt --device cpu
```

### Problem: Want to see what would run
**Solution:** Use dry-run mode
```bash
./run_all_experiments.sh --ckpt-path /path/to/model.ckpt --dry-run
```

## Output Interpretation

### Successful Run Output:
```
========================================
Summary
========================================
Total tasks:       12
Successful:        12
Failed:            0
========================================

✓ All predictions completed successfully!
```

### Partial Success Output:
```
========================================
Summary
========================================
Total tasks:       12
Successful:        10
Skipped:           2
Failed:            0
========================================

Skipped items (missing results):
  - naphthalene (primary) - missing: QHFlow-v2
  - aspirin (primary) - missing: DFT
```

### Failed Run Output:
```
========================================
Summary
========================================
Total tasks:       12
Successful:        10
Failed:            2
========================================

Failed items:
  - ethanol (secondary)
  - uracil (primary)
```

## Performance Tips

1. **Use GPU for predictions** (much faster):
   ```bash
   --device cuda
   ```

2. **Run specific molecules** to save time:
   ```bash
   --molecules "ethanol,aspirin"
   ```

3. **Use dry-run first** to verify configuration:
   ```bash
   --dry-run
   ```

4. **Parallel execution** (advanced):
   You can run multiple molecules in parallel by launching multiple script instances with different `--molecules` arguments.

## Integration with Python Scripts

These bash scripts call the following Python scripts:
- `exp_bond-stretch_model.py` (SPHNet predictions)
- `exp_bond-stretch_plot_three_methods.py` (three-method plots)

You can also call these Python scripts directly for more fine-grained control.

## Help

For detailed help on any script:
```bash
./run_sphnet_predictions.sh --help
./run_three_method_plots.sh --help
./run_all_experiments.sh --help
```

## Complete Example Workflow

```bash
# 1. Navigate to SPHNet directory
cd /home/chanhui-lee/SPHNet

# 2. Preview what will run (dry-run)
./run_all_experiments.sh \
    --ckpt-path /path/to/sphnet/model.ckpt \
    --dry-run

# 3. Run complete workflow
./run_all_experiments.sh \
    --ckpt-path /path/to/sphnet/model.ckpt \
    --dft-dir /home/chanhui-lee/QHFlow-mlff/src/electronic_structure_exp/output \
    --qhflow-dir /home/chanhui-lee/QHFlow-mlff/src/electronic_structure_exp/output

# 4. View results
ls -lh output/
ls -lh plots/
```

## Additional Notes

- **Default paths** are configured for the standard setup. Override with command-line arguments if needed.
- **All molecules and sites** from `exp_config.yaml` are processed by default.
- **Results are saved** in the same format as QHFlow-v2 for compatibility.
- **Plots show all three methods** (DFT, QHFlow-v2, SPHNet) for easy comparison.

## Support

For issues or questions:
1. Check the main [EXPERIMENT_README.md](EXPERIMENT_README.md)
2. Use `--dry-run` to debug
3. Check error messages in script output
4. Verify prerequisites are met
