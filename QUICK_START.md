# Quick Start Guide - SPHNet Predictions & Three-Method Comparison

## TL;DR - Run Everything

### Option 1: Per-Molecule Checkpoints (RECOMMENDED)

```bash
cd /home/chanhui-lee/SPHNet

# 1. First, configure your checkpoints
nano exp_config_sphnet.yaml  # Edit checkpoint paths for each molecule

# 2. Run everything with one command
./run_all_experiments.sh --config exp_config_sphnet.yaml
```

### Option 2: Single Checkpoint (For Testing Only)

```bash
cd /home/chanhui-lee/SPHNet

# NOT RECOMMENDED for production - all molecules use same checkpoint
./run_all_experiments.sh --ckpt-path /path/to/your/sphnet_model.ckpt
```

Results will be in:
- `./output/` - SPHNet prediction files
- `./plots/` - Three-method comparison plots

**Important**: Each molecule needs its own trained checkpoint for valid scientific results!

---

## Prerequisites Checklist

Before running, make sure you have:

- [ ] SPHNet model checkpoints (one per molecule) ← **IMPORTANT!**
- [ ] Config file with checkpoint paths (`exp_config_sphnet.yaml`)
- [ ] DFT reference data (from QHFlow-mlff)
- [ ] QHFlow-v2 predictions (from QHFlow-mlff)
- [ ] CUDA available (or use `--device cpu`)

**See [CONFIG_SETUP_GUIDE.md](CONFIG_SETUP_GUIDE.md) for checkpoint configuration**

---

## Common Use Cases

### Use Case 1: First Time - Run All Molecules (Recommended Way)

```bash
# Step 1: Configure checkpoints (one-time setup)
nano exp_config_sphnet.yaml

# Step 2: Run all molecules
./run_all_experiments.sh --config exp_config_sphnet.yaml
```

Each molecule uses its own trained checkpoint. See [CONFIG_SETUP_GUIDE.md](CONFIG_SETUP_GUIDE.md).

### Use Case 2: Run Specific Molecules Only

```bash
./run_all_experiments.sh \
    --config exp_config_sphnet.yaml \
    --molecules "ethanol,aspirin"
```

Only processes ethanol and aspirin using their configured checkpoints.

### Use Case 3: Already Have Predictions, Just Need Plots

```bash
./run_all_experiments.sh \
    --sphnet-save-dir ./output \
    --skip-predictions
```

### Use Case 4: Preview Without Running (Dry Run)

```bash
./run_all_experiments.sh \
    --ckpt-path /path/to/sphnet.ckpt \
    --dry-run
```

### Use Case 5: CPU-Only (No GPU)

```bash
./run_all_experiments.sh \
    --ckpt-path /path/to/sphnet.ckpt \
    --device cpu
```

---

## Step-by-Step Workflow

### Step 1: Predictions Only

```bash
# Run SPHNet predictions for all molecules
./run_sphnet_predictions.sh \
    --ckpt-path /path/to/sphnet.ckpt \
    --save-dir ./output
```

This creates:
```
output/
├── ethanol_primary_O2-H8_ratio-0.80_sphnet.pt
├── ethanol_primary_O2-H8_ratio-1.00_sphnet.pt
├── ethanol_primary_O2-H8_sphnet_metadata.pt
└── ... (more molecules)
```

### Step 2: Plots Only

```bash
# Generate three-method comparison plots
./run_three_method_plots.sh \
    --dft-dir /path/to/dft/results \
    --qhflow-dir /path/to/qhflow/results \
    --sphnet-dir ./output \
    --save-dir ./plots
```

This creates:
```
plots/
├── ethanol_primary_O2-H8_comparison_three_methods.png
├── aspirin_primary_C11-O12_comparison_three_methods.png
└── ... (more plots)
```

---

## File Naming Convention

### Prediction Files
```
{molecule}_{site_type}_{bond}_{method}.pt

Examples:
- ethanol_primary_O2-H8_ratio-0.80_sphnet.pt
- aspirin_primary_C11-O12_ratio-1.00_sphnet.pt
```

### Metadata Files
```
{molecule}_{site_type}_{bond}_{method}_metadata.pt

Examples:
- ethanol_primary_O2-H8_sphnet_metadata.pt
- aspirin_primary_C11-O12_sphnet_metadata.pt
```

### Plot Files
```
{molecule}_{site_type}_{bond}_comparison_three_methods.png

Examples:
- ethanol_primary_O2-H8_comparison_three_methods.png
- aspirin_primary_C11-O12_comparison_three_methods.png
```

---

## Supported Molecules

| Molecule | Site Types | Bond Examples |
|----------|-----------|---------------|
| ethanol | primary, secondary | O-H, C-O |
| malondialdehyde | primary | C=O |
| naphthalene | primary | C-C |
| salicylic_acid | primary, secondary | O-H (phenolic), O-H (carboxylic) |
| aspirin | primary | C=O |
| uracil | primary, secondary | N-H, C=C |

---

## Troubleshooting

### Error: "Checkpoint file not found"
```bash
# Check if file exists
ls -lh /path/to/sphnet.ckpt

# Use absolute path
./run_all_experiments.sh \
    --ckpt-path /absolute/path/to/sphnet.ckpt
```

### Error: "DFT results directory not found"
```bash
# First run DFT calculations in QHFlow-mlff repo
cd /home/chanhui-lee/QHFlow-mlff/src/electronic_structure_exp
python exp_bond-stretch_dft.py --molecule ethanol --save-dir ./output
```

### Error: GPU out of memory
```bash
# Use CPU instead
./run_all_experiments.sh \
    --ckpt-path /path/to/sphnet.ckpt \
    --device cpu
```

### Plots show "Skipped: Missing results"
This means not all three methods have results. Check:
```bash
# Check DFT results
ls /path/to/dft/*dft_metadata.pt

# Check QHFlow-v2 results
ls /path/to/qhflow/*qhflow-v2_metadata.pt

# Check SPHNet results
ls ./output/*sphnet_metadata.pt
```

---

## What Gets Produced

### After Predictions (`run_sphnet_predictions.sh`):

For each molecule and site type, you get multiple files:
- One `.pt` file per stretch ratio (e.g., 0.80, 0.90, 1.00, 1.10, 1.20)
- One metadata file summarizing all ratios

Example for ethanol (primary site):
```
ethanol_primary_O2-H8_ratio-0.80_sphnet.pt
ethanol_primary_O2-H8_ratio-0.90_sphnet.pt
ethanol_primary_O2-H8_ratio-1.00_sphnet.pt
ethanol_primary_O2-H8_ratio-1.10_sphnet.pt
ethanol_primary_O2-H8_ratio-1.20_sphnet.pt
ethanol_primary_O2-H8_sphnet_metadata.pt
```

### After Plots (`run_three_method_plots.sh`):

For each molecule and site type, you get:
- One PNG file showing DFT, QHFlow-v2, and SPHNet comparison
- Plots show HOMO, LUMO, and Gap energies vs stretch ratio
- Error metrics (MAE, Max Error) included

---

## Expected Output

### Successful Prediction Run:
```
[12/12] Running: uracil (secondary)
✓ Completed: uracil (secondary)

========================================
Summary
========================================
Total tasks:       12
Successful:        12
Failed:            0
========================================

✓ All predictions completed successfully!

Results saved to: ./output
```

### Successful Plot Generation:
```
[12/12] Plotting: uracil (secondary)
✓ Completed: uracil (secondary)

========================================
Summary
========================================
Total tasks:       12
Successful:        12
Skipped:           0
Failed:            0
========================================

✓ All plots generated successfully!

Plots saved to: ./plots
```

---

## Next Steps After Running

1. **View the plots** in `./plots/` directory
2. **Check metrics** in the console output (MAE, Max Error)
3. **Share results** - plots are publication-ready PNG files
4. **Analyze specific molecules** by examining the `.pt` files

---

## Advanced: Custom Paths

If your setup differs from defaults:

```bash
./run_all_experiments.sh \
    --ckpt-path /custom/path/sphnet.ckpt \
    --dft-dir /custom/dft/results \
    --qhflow-dir /custom/qhflow/results \
    --sphnet-save-dir /custom/sphnet/output \
    --plots-save-dir /custom/plots
```

---

## Help & Documentation

- Quick reference: This file
- Detailed script docs: [SCRIPTS_README.md](SCRIPTS_README.md)
- Full experiment guide: [EXPERIMENT_README.md](EXPERIMENT_README.md)
- Code summary: [SUMMARY.md](SUMMARY.md)

For script help:
```bash
./run_all_experiments.sh --help
./run_sphnet_predictions.sh --help
./run_three_method_plots.sh --help
```

---

## Minimal Example

The absolute minimum to get started:

```bash
cd /home/chanhui-lee/SPHNet
./run_all_experiments.sh --ckpt-path YOUR_MODEL.ckpt
```

That's it! This assumes:
- DFT and QHFlow-v2 results are in standard QHFlow-mlff output directory
- You want to process all molecules
- You have CUDA available
- Default output directories are fine

Results will be in `./output/` and `./plots/`.
