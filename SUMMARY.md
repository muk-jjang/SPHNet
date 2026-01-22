# Summary of Changes

## Created Files

### Python Scripts

#### 1. SPHNet Prediction Script
**File**: `/home/chanhui-lee/SPHNet/exp_bond-stretch_model.py`

This is the main SPHNet prediction script that:
- Loads DFT results from QHFlow-mlff experiment
- Runs SPHNet model predictions on the same geometries
- Saves results in the same format as QHFlow-v2 (for compatibility with plotting)
- Uses SPHNet's native implementation from `escflow_eval_utils.py`

**Key Features**:
- `SPHNetPredictor` class for model loading and inference
- Batch inference support for efficiency
- Matrix transformation using SPHNet's orbital conventions
- Compatible data format with QHFlow-v2 results

### 2. Three-Method Comparison Plot Script
**File**: `/home/chanhui-lee/QHFlow-mlff/src/electronic_structure_exp/exp_bond-stretch_plot_three_methods.py`

A new plotting script that:
- Loads results from all three methods (DFT, QHFlow-v2, SPHNet)
- Generates comparison plots showing all three lines
- Computes and displays metrics for both models vs DFT
- Creates publication-quality figures

#### 3. Documentation
**Files**:
- `/home/chanhui-lee/SPHNet/EXPERIMENT_README.md` - Comprehensive usage guide
- `/home/chanhui-lee/SPHNet/SUMMARY.md` - This file
- `/home/chanhui-lee/SPHNet/SCRIPTS_README.md` - Bash scripts documentation
- `/home/chanhui-lee/SPHNet/QUICK_START.md` - Quick start guide

### Bash Scripts (NEW)

#### 4. SPHNet Prediction Runner
**File**: `/home/chanhui-lee/SPHNet/run_sphnet_predictions.sh`

A bash script that:
- Runs SPHNet predictions for all molecules and site types
- Configurable molecule filtering
- Supports dry-run mode for testing
- Provides progress tracking and summary

**Usage**:
```bash
./run_sphnet_predictions.sh \
    --ckpt-path /path/to/sphnet.ckpt \
    --dft-dir /path/to/dft/results \
    --save-dir ./output
```

#### 5. Three-Method Plot Generator
**File**: `/home/chanhui-lee/SPHNet/run_three_method_plots.sh`

A bash script that:
- Generates comparison plots for all molecules with complete data
- Checks for missing results and skips incomplete datasets
- Supports selective molecule processing
- Provides detailed summary of successful/skipped/failed plots

**Usage**:
```bash
./run_three_method_plots.sh \
    --dft-dir ./output \
    --qhflow-dir ./output \
    --sphnet-dir ./output \
    --save-dir ./plots
```

#### 6. Master Experiment Runner
**File**: `/home/chanhui-lee/SPHNet/run_all_experiments.sh`

A master script that:
- Runs complete workflow (predictions + plots)
- Supports skipping either step
- Configurable for all options
- Single command to run everything

**Usage**:
```bash
./run_all_experiments.sh \
    --ckpt-path /path/to/sphnet.ckpt
```

## Modified Files

### Updated QHFlow Plot Script
**File**: `/home/chanhui-lee/QHFlow-mlff/src/electronic_structure_exp/exp_bond-stretch_plot.py`

**Changes**:
1. Added `--model-type` argument to support both 'qhflow-v2' and 'sphnet'
2. Updated `load_model_results()` to accept `model_type` parameter
3. The script can now plot DFT vs QHFlow-v2 OR DFT vs SPHNet

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Experiment Workflow                       │
└─────────────────────────────────────────────────────────────┘

1. DFT Calculations (QHFlow-mlff)
   ├── exp_bond-stretch_dft.py
   └── Output: ethanol_primary_O7-H8_ratio-X.XX_dft.pt
                ethanol_primary_O7-H8_dft_metadata.pt

2a. QHFlow-v2 Predictions (QHFlow-mlff)
   ├── exp_bond-stretch_model.py
   └── Output: ethanol_primary_O7-H8_ratio-X.XX_qhflow-v2.pt
                ethanol_primary_O7-H8_qhflow-v2_metadata.pt

2b. SPHNet Predictions (SPHNet) [NEW]
   ├── exp_bond-stretch_model.py
   └── Output: ethanol_primary_O7-H8_ratio-X.XX_sphnet.pt
                ethanol_primary_O7-H8_sphnet_metadata.pt

3. Comparison & Plotting
   ├── exp_bond-stretch_plot.py (2-method comparison)
   └── exp_bond-stretch_plot_three_methods.py (3-method) [NEW]
```

## Data Format Compatibility

All three methods save results in the same format:

```python
{
    'positions': np.ndarray,            # Atomic coordinates
    'atomic_numbers': np.ndarray,       # Z values
    'homo_energy_ev': float,            # HOMO energy
    'lumo_energy_ev': float,            # LUMO energy
    'orbital_energies_ev': np.ndarray,  # All MO energies
    'orbital_coefficients': np.ndarray, # MO coefficients
    'overlap': np.ndarray,              # S matrix
    'hamiltonian_hartree': np.ndarray,  # H matrix
    'stretch_ratio': float,             # Bond stretch ratio
    # ... metadata fields
}
```

This ensures seamless comparison across all methods.

## Key Implementation Details

### SPHNet Model Predictor

The `SPHNetPredictor` class follows the same interface as QHFlow-v2's `ModelPredictor`:

```python
class SPHNetPredictor:
    def load_model(ckpt_path)
    def forward_component(atoms, coords, ovlp, init_ham)
    def forward_batch(list_atoms, list_coords, ...)
    def calc_mo_energy_and_coeff(ham, overlap)
```

### Matrix Transformations

SPHNet uses the same orbital convention transformations:
- Input: PySCF def2svp → e3nn convention
- Output: e3nn → PySCF def2svp convention

This is handled by `matrix_transform_single()` from `escflow_eval_utils.py`.

### File Naming Convention

```
{molecule}_{site_type}_{bond}_{ratio}_{method}.pt
```

Examples:
- `ethanol_primary_O7-H8_ratio-1.00_dft.pt`
- `ethanol_primary_O7-H8_ratio-1.00_qhflow-v2.pt`
- `ethanol_primary_O7-H8_ratio-1.00_sphnet.pt`

The metadata files use:
```
{molecule}_{site_type}_{bond}_{method}_metadata.pt
```

## Usage Quick Reference

### Run SPHNet Predictions
```bash
cd /home/chanhui-lee/SPHNet
python exp_bond-stretch_model.py \
    --molecule ethanol \
    --site-type primary \
    --load-results /path/to/dft/results \
    --ckpt-path /path/to/sphnet/model.ckpt \
    --save-dir ./output
```

### Plot All Three Methods
```bash
cd /home/chanhui-lee/QHFlow-mlff/src/electronic_structure_exp
python exp_bond-stretch_plot_three_methods.py \
    --dft-results ./output \
    --qhflow-results ./output \
    --sphnet-results /home/chanhui-lee/SPHNet/output \
    --molecule ethanol \
    --site-type primary \
    --save-dir ./output
```

## Usage Quick Reference

### Option 1: Use Bash Scripts (Recommended - Automated)

**Run everything with one command**:
```bash
cd /home/chanhui-lee/SPHNet
./run_all_experiments.sh --ckpt-path /path/to/sphnet.ckpt
```

**Or run steps separately**:
```bash
# Step 1: Predictions for all molecules
./run_sphnet_predictions.sh --ckpt-path /path/to/sphnet.ckpt

# Step 2: Generate all comparison plots
./run_three_method_plots.sh \
    --dft-dir /path/to/dft \
    --qhflow-dir /path/to/qhflow \
    --sphnet-dir ./output
```

**Specific molecules only**:
```bash
./run_all_experiments.sh \
    --ckpt-path /path/to/sphnet.ckpt \
    --molecules "ethanol,aspirin"
```

**Dry run (preview without executing)**:
```bash
./run_all_experiments.sh \
    --ckpt-path /path/to/sphnet.ckpt \
    --dry-run
```

See [QUICK_START.md](QUICK_START.md) for more examples.

### Option 2: Use Python Scripts Directly (Manual)

For fine-grained control, use Python scripts directly:

## Next Steps (Manual Python Workflow)

To use the Python scripts directly:

1. **Prepare DFT reference data** (if not already done):
   ```bash
   cd /home/chanhui-lee/QHFlow-mlff/src/electronic_structure_exp
   python exp_bond-stretch_dft.py --molecule ethanol --site-type primary --save-dir ./output
   ```

2. **Run SPHNet predictions**:
   ```bash
   cd /home/chanhui-lee/SPHNet
   python exp_bond-stretch_model.py \
       --molecule ethanol \
       --site-type primary \
       --load-results /home/chanhui-lee/QHFlow-mlff/src/electronic_structure_exp/output \
       --ckpt-path YOUR_SPHNET_CHECKPOINT.ckpt \
       --save-dir ./output
   ```

3. **Generate comparison plots**:
   ```bash
   cd /home/chanhui-lee/QHFlow-mlff/src/electronic_structure_exp
   python exp_bond-stretch_plot_three_methods.py \
       --dft-results ./output \
       --qhflow-results ./output \
       --sphnet-results /home/chanhui-lee/SPHNet/output \
       --molecule ethanol \
       --site-type primary \
       --save-dir ./output
   ```

## Expected Output

After running all scripts, you will have:

1. **Data files** for each stretch ratio and method:
   - DFT results (`.pt` files)
   - QHFlow-v2 predictions (`.pt` files)
   - SPHNet predictions (`.pt` files)

2. **Metadata files** for each method:
   - Summary information
   - List of result files
   - Model configuration (for predictions)

3. **Comparison plots**:
   - Three-method comparison showing DFT, QHFlow-v2, and SPHNet
   - HOMO, LUMO, and Gap energy curves
   - Error metrics (MAE, Max Error) for each model

4. **Console output**:
   - Comparison metrics summary
   - MAE and Max Error for both models vs DFT

## Important Notes

1. **Model Checkpoints**: You need to provide the path to your trained SPHNet checkpoint
2. **GPU Memory**: Batch inference is used for efficiency; adjust batch size if needed
3. **Length Units**: SPHNet uses Angstrom by default (same as QHFlow-v2)
4. **Orbital Conventions**: Both models use the same e3nn ↔ PySCF transformations
5. **Data Compatibility**: All results use the same format for fair comparison

## Troubleshooting

If you encounter issues:

1. **Import errors**: Check that you're running from the correct directory
2. **Model loading errors**: Verify checkpoint path and model compatibility
3. **Missing results**: Ensure DFT results exist before running predictions
4. **Plot errors**: Verify all three result types are available with matching parameters

See `EXPERIMENT_README.md` for detailed troubleshooting steps.
