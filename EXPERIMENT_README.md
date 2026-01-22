# Bond Stretch Hamiltonian Comparison Experiments

This directory contains scripts for comparing Hamiltonian predictions from three methods:
1. **DFT** - Ground truth from quantum chemistry calculations
2. **QHFlow-v2** - ML model predictions from QHFlow
3. **SPHNet** - ML model predictions from SPHNet (this repository)

## Overview

The experiment workflow consists of three main steps:

1. **DFT Calculations**: Generate ground truth data using PySCF (done in QHFlow-mlff repo)
2. **Model Predictions**: Run SPHNet model predictions on the same geometries (this script)
3. **Comparison & Plotting**: Compare all three methods and generate plots

## File Structure

```
SPHNet/
├── exp_bond-stretch_model.py          # SPHNet prediction script (NEW)
└── escflow_eval_utils.py               # Utility functions for SPHNet

QHFlow-mlff/src/electronic_structure_exp/
├── exp_bond-stretch_dft.py             # DFT calculation script
├── exp_bond-stretch_model.py           # QHFlow-v2 prediction script
├── exp_bond-stretch_plot.py            # Two-method comparison plots
├── exp_bond-stretch_plot_three_methods.py  # Three-method comparison plots (NEW)
└── exp_config.yaml                     # Configuration file
```

## Usage

### Step 1: Run DFT Calculations (QHFlow-mlff)

First, generate DFT reference data:

```bash
cd /home/chanhui-lee/QHFlow-mlff/src/electronic_structure_exp

python exp_bond-stretch_dft.py \
    --molecule ethanol \
    --site-type primary \
    --save-dir ./output
```

This creates files like:
- `ethanol_primary_O7-H8_ratio-0.80_dft.pt`
- `ethanol_primary_O7-H8_ratio-1.00_dft.pt`
- `ethanol_primary_O7-H8_dft_metadata.pt`

### Step 2a: Run QHFlow-v2 Predictions

```bash
cd /home/chanhui-lee/QHFlow-mlff/src/electronic_structure_exp

python exp_bond-stretch_model.py \
    --molecule ethanol \
    --site-type primary \
    --load-results ./output \
    --ckpt-path /path/to/qhflow-v2/model.ckpt \
    --save-dir ./output
```

This creates files like:
- `ethanol_primary_O7-H8_ratio-0.80_qhflow-v2.pt`
- `ethanol_primary_O7-H8_ratio-1.00_qhflow-v2.pt`
- `ethanol_primary_O7-H8_qhflow-v2_metadata.pt`

### Step 2b: Run SPHNet Predictions (NEW)

```bash
cd /home/chanhui-lee/SPHNet

python exp_bond-stretch_model.py \
    --molecule ethanol \
    --site-type primary \
    --load-results /home/chanhui-lee/QHFlow-mlff/src/electronic_structure_exp/output \
    --ckpt-path /path/to/sphnet/model.ckpt \
    --save-dir ./output \
    --device cuda \
    --model-length-unit ang
```

This creates files like:
- `ethanol_primary_O7-H8_ratio-0.80_sphnet.pt`
- `ethanol_primary_O7-H8_ratio-1.00_sphnet.pt`
- `ethanol_primary_O7-H8_sphnet_metadata.pt`

### Step 3a: Plot Two-Method Comparison (DFT vs QHFlow-v2 OR DFT vs SPHNet)

```bash
cd /home/chanhui-lee/QHFlow-mlff/src/electronic_structure_exp

# Compare DFT vs QHFlow-v2
python exp_bond-stretch_plot.py \
    --dft-results ./output \
    --model-results ./output \
    --model-type qhflow-v2 \
    --molecule ethanol \
    --site-type primary \
    --save-dir ./output

# Or compare DFT vs SPHNet
python exp_bond-stretch_plot.py \
    --dft-results ./output \
    --model-results /home/chanhui-lee/SPHNet/output \
    --model-type sphnet \
    --molecule ethanol \
    --site-type primary \
    --save-dir ./output
```

### Step 3b: Plot Three-Method Comparison (NEW)

Compare all three methods in a single plot:

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

## Output Files

### Data Files

All results are saved as PyTorch `.pt` files with the following naming convention:

```
{molecule}_{site_type}_{bond_label}_ratio-{ratio}_{method}.pt
{molecule}_{site_type}_{bond_label}_{method}_metadata.pt
```

Where:
- `molecule`: Molecule name (e.g., "ethanol", "aspirin")
- `site_type`: Reactive site type (e.g., "primary", "secondary")
- `bond_label`: Bond identifier (e.g., "O7-H8")
- `ratio`: Stretch ratio (e.g., "0.80", "1.00", "1.20")
- `method`: One of "dft", "qhflow-v2", or "sphnet"

### Plot Files

The plotting scripts generate several types of plots:

1. **Two-method comparison** (`exp_bond-stretch_plot.py`):
   - Main comparison plot with energy curves and 3D structures
   - Parity plots
   - Individual line plots for HOMO, LUMO, and Gap

2. **Three-method comparison** (`exp_bond-stretch_plot_three_methods.py`):
   - Single plot showing all three methods for easy comparison

## Data Format

Each `.pt` file contains a dictionary with the following keys:

```python
{
    # Geometry
    'positions': np.ndarray,           # Atomic positions (Å)
    'atomic_numbers': np.ndarray,      # Atomic numbers
    'stretch_ratio': float,            # Bond stretch ratio

    # Electronic structure
    'homo_energy_ev': float,           # HOMO energy (eV)
    'lumo_energy_ev': float,           # LUMO energy (eV)
    'orbital_energies_ev': np.ndarray, # All MO energies (eV)
    'orbital_coefficients': np.ndarray,# MO coefficients

    # Matrices
    'overlap': np.ndarray,             # Overlap matrix (S)
    'hamiltonian_hartree': np.ndarray, # Hamiltonian matrix (H)
    'initial_hamiltonian': np.ndarray, # Initial guess Hamiltonian

    # Metadata
    'molecule_name': str,
    'atom_types': list,
    'site_type': str,
    'atom1_idx': int,
    'atom2_idx': int,
}
```

## Key Differences Between SPHNet and QHFlow-v2 Scripts

The SPHNet prediction script (`exp_bond-stretch_model.py`) has been designed to:

1. **Match QHFlow-v2 interface**: Uses the same function names and data format for compatibility
2. **Use SPHNet's native implementation**: Imports from `src/training/module` and `escflow_eval_utils`
3. **Save with 'sphnet' label**: Output files are labeled with `sphnet` instead of `qhflow-v2`
4. **Compatible with plotting scripts**: The updated plotting scripts can handle both model types

## Configuration

Model checkpoints and experiment settings are configured in `exp_config.yaml`:

```yaml
model_checkpoints:
  ethanol:
    ckpt_path: /path/to/sphnet/model.ckpt
    data_type: md17
    model_length_unit: ang
  # Add more molecules as needed
```

## Notes

- **Length Units**: SPHNet typically uses Angstrom by default. Adjust `--model-length-unit` if needed.
- **GPU Memory**: Batch inference is used for efficiency. Reduce batch size if GPU memory is limited.
- **Compatibility**: All three methods use the same DFT reference data, ensuring fair comparison.
- **Data Sharing**: The plot scripts expect all result files (DFT, QHFlow-v2, SPHNet) to be accessible from the same or specified directories.

## Troubleshooting

### Import Errors

If you encounter import errors, ensure:
1. You're running from the correct directory
2. The Python path includes both repositories
3. All required dependencies are installed

### Model Loading Errors

If model checkpoint loading fails:
1. Verify the checkpoint path exists
2. Check that the model architecture matches the checkpoint
3. Ensure CUDA is available if using GPU

### Missing Results

If plots fail due to missing results:
1. Verify all three result types exist in the specified directories
2. Check that molecule names and site types match exactly
3. Use `--molecule` and `--site-type` arguments to filter correctly

## Example Workflow

Complete example for ethanol molecule:

```bash
# 1. Generate DFT reference data
cd /home/chanhui-lee/QHFlow-mlff/src/electronic_structure_exp
python exp_bond-stretch_dft.py --molecule ethanol --site-type primary --save-dir ./output

# 2. Run QHFlow-v2 predictions
python exp_bond-stretch_model.py --molecule ethanol --site-type primary \
    --load-results ./output --ckpt-path /path/to/qhflow.ckpt --save-dir ./output

# 3. Run SPHNet predictions
cd /home/chanhui-lee/SPHNet
python exp_bond-stretch_model.py --molecule ethanol --site-type primary \
    --load-results /home/chanhui-lee/QHFlow-mlff/src/electronic_structure_exp/output \
    --ckpt-path /path/to/sphnet.ckpt --save-dir ./output

# 4. Generate comparison plots
cd /home/chanhui-lee/QHFlow-mlff/src/electronic_structure_exp
python exp_bond-stretch_plot_three_methods.py \
    --dft-results ./output \
    --qhflow-results ./output \
    --sphnet-results /home/chanhui-lee/SPHNet/output \
    --molecule ethanol --site-type primary --save-dir ./output
```

This will generate comprehensive comparison plots showing how SPHNet and QHFlow-v2 predictions compare to DFT reference values across different bond stretch ratios.
