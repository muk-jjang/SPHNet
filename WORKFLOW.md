# Complete Workflow Diagram

## Three-Method Hamiltonian Comparison Experiment

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       PREREQUISITE: DFT CALCULATIONS                     │
│                         (QHFlow-mlff Repository)                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │ exp_bond-stretch_dft.py       │
                    │ - Run PySCF calculations      │
                    │ - Bond stretch geometries     │
                    │ - Compute HOMO/LUMO           │
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │   DFT Reference Data (.pt)    │
                    │ - Hamiltonians                │
                    │ - Overlap matrices            │
                    │ - MO energies                 │
                    └───────────────┬───────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
                    ▼                               ▼

┌─────────────────────────────────┐  ┌─────────────────────────────────┐
│     QHFlow-v2 PREDICTIONS       │  │      SPHNet PREDICTIONS         │
│   (QHFlow-mlff Repository)      │  │      (SPHNet Repository)        │
└─────────────────────────────────┘  └─────────────────────────────────┘
                │                                    │
┌───────────────┴────────────────┐   ┌──────────────┴─────────────────┐
│ exp_bond-stretch_model.py      │   │ exp_bond-stretch_model.py      │
│ - Load DFT data                │   │ - Load DFT data                │
│ - Run QHFlow model             │   │ - Run SPHNet model             │
│ - Compute MO energies          │   │ - Compute MO energies          │
└───────────────┬────────────────┘   └──────────────┬─────────────────┘
                │                                    │
                ▼                                    ▼
┌────────────────────────────────┐   ┌────────────────────────────────┐
│  QHFlow-v2 Results (.pt)       │   │   SPHNet Results (.pt)         │
│  - qhflow-v2_metadata.pt       │   │   - sphnet_metadata.pt         │
│  - *_qhflow-v2.pt files        │   │   - *_sphnet.pt files          │
└────────────────┬───────────────┘   └──────────────┬─────────────────┘
                 │                                   │
                 └─────────────┬─────────────────────┘
                               │
                               ▼
            ┌──────────────────────────────────────────┐
            │     THREE-METHOD COMPARISON PLOTS        │
            │    (QHFlow-mlff/electronic_structure)    │
            └──────────────────────────────────────────┘
                               │
            ┌──────────────────┴──────────────────┐
            │ exp_bond-stretch_plot_three_methods │
            │ - Load all three results            │
            │ - Compute metrics                   │
            │ - Generate plots                    │
            └──────────────────┬──────────────────┘
                               │
                               ▼
            ┌──────────────────────────────────────────┐
            │      COMPARISON PLOTS (.png)             │
            │  - HOMO energy vs stretch ratio          │
            │  - LUMO energy vs stretch ratio          │
            │  - Gap energy vs stretch ratio           │
            │  - Error metrics (MAE, Max)              │
            └──────────────────────────────────────────┘
```

---

## Automated Workflow with Bash Scripts

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    AUTOMATED WORKFLOW (SPHNet Repo)                      │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │  run_all_experiments.sh       │
                    │  --ckpt-path model.ckpt       │
                    └───────────────┬───────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
                    ▼                               ▼
        ┌───────────────────────┐      ┌───────────────────────┐
        │ STEP 1: PREDICTIONS   │      │  STEP 2: PLOTS        │
        └───────────────────────┘      └───────────────────────┘
                    │                               │
                    ▼                               ▼
    ┌──────────────────────────┐      ┌──────────────────────────┐
    │ run_sphnet_predictions   │      │ run_three_method_plots   │
    │ - Loop all molecules     │      │ - Loop all molecules     │
    │ - Loop all sites         │      │ - Check all 3 results    │
    │ - Call Python script     │      │ - Generate plots         │
    └──────────────┬───────────┘      └──────────┬───────────────┘
                   │                              │
                   ▼                              ▼
    ┌──────────────────────────┐      ┌──────────────────────────┐
    │  SPHNet Results          │      │  Comparison Plots        │
    │  ./output/*.pt           │      │  ./plots/*.png           │
    └──────────────────────────┘      └──────────────────────────┘
```

---

## File Flow Diagram

```
molecules: [ethanol, aspirin, malondialdehyde, naphthalene, salicylic_acid, uracil]
           └─ sites: [primary, secondary (if applicable)]
                      └─ stretch_ratios: [0.80, 0.90, 1.00, 1.10, 1.20]

For each molecule + site combination:

    DFT Results                    QHFlow-v2 Results              SPHNet Results
    ────────────                   ─────────────────              ──────────────
    ethanol_primary_               ethanol_primary_               ethanol_primary_
    O2-H8_ratio-0.80_dft.pt       O2-H8_ratio-0.80_qhflow-v2.pt O2-H8_ratio-0.80_sphnet.pt
    ethanol_primary_               ethanol_primary_               ethanol_primary_
    O2-H8_ratio-0.90_dft.pt       O2-H8_ratio-0.90_qhflow-v2.pt O2-H8_ratio-0.90_sphnet.pt
    ethanol_primary_               ethanol_primary_               ethanol_primary_
    O2-H8_ratio-1.00_dft.pt       O2-H8_ratio-1.00_qhflow-v2.pt O2-H8_ratio-1.00_sphnet.pt
    ethanol_primary_               ethanol_primary_               ethanol_primary_
    O2-H8_ratio-1.10_dft.pt       O2-H8_ratio-1.10_qhflow-v2.pt O2-H8_ratio-1.10_sphnet.pt
    ethanol_primary_               ethanol_primary_               ethanol_primary_
    O2-H8_ratio-1.20_dft.pt       O2-H8_ratio-1.20_qhflow-v2.pt O2-H8_ratio-1.20_sphnet.pt

    Metadata:                      Metadata:                      Metadata:
    ethanol_primary_               ethanol_primary_               ethanol_primary_
    O2-H8_dft_metadata.pt         O2-H8_qhflow-v2_metadata.pt   O2-H8_sphnet_metadata.pt

                                           ↓

                            Three Method Comparison Plot
                            ────────────────────────────
                            ethanol_primary_O2-H8_comparison_three_methods.png
                            - Shows DFT, QHFlow-v2, SPHNet on same plot
                            - HOMO, LUMO, Gap curves
                            - Error metrics
```

---

## Data Structure Inside .pt Files

```python
Each .pt file contains:
{
    # Geometry
    'positions': array([...]),           # Atomic positions (Å)
    'atomic_numbers': array([...]),      # Z values
    'stretch_ratio': 1.0,                # Bond stretch ratio

    # Electronic Structure
    'homo_energy_ev': -8.234,            # HOMO energy (eV)
    'lumo_energy_ev': 2.456,             # LUMO energy (eV)
    'orbital_energies_ev': array([...]), # All MO energies
    'orbital_coefficients': array([...]),# MO coefficients

    # Matrices
    'overlap': array([[...]]),           # S matrix
    'hamiltonian_hartree': array([[...]]),# H matrix (Hartree)
    'initial_hamiltonian': array([[...]]),# H0 matrix

    # Metadata
    'molecule_name': 'ethanol',
    'atom_types': ['O', 'H'],
    'site_type': 'primary',
    'atom1_idx': 2,
    'atom2_idx': 8,
    'source': 'dft' | 'qhflow_prediction' | 'sphnet_prediction'
}
```

---

## Comparison Plot Structure

```
┌─────────────────────────────────────────────────────────────────┐
│        Three-Method Comparison: ethanol (primary, O-H)          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  HOMO Energy (eV) vs Stretch Ratio                              │
│  ─────────────────────────────────────                          │
│   0  ─── DFT (blue solid line)                                  │
│  -2  ─── QHFlow-v2 (red dashed)                                 │
│  -4  ─── SPHNet (green dotted)                                  │
│  -6                                                              │
│  -8  [Error metrics box]                                        │
│ -10  QHFlow: MAE=0.234 eV                                       │
│      SPHNet: MAE=0.187 eV                                       │
│      0.8   0.9   1.0   1.1   1.2  (stretch ratio)               │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  LUMO Energy (eV) vs Stretch Ratio                              │
│  ─────────────────────────────────────                          │
│   4  ─── DFT                                                    │
│   3  ─── QHFlow-v2                                              │
│   2  ─── SPHNet                                                 │
│   1                                                              │
│   0  [Error metrics box]                                        │
│  -1  QHFlow: MAE=0.156 eV                                       │
│      SPHNet: MAE=0.142 eV                                       │
│      0.8   0.9   1.0   1.1   1.2  (stretch ratio)               │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  HOMO-LUMO Gap (eV) vs Stretch Ratio                            │
│  ─────────────────────────────────────                          │
│  12  ─── DFT                                                    │
│  10  ─── QHFlow-v2                                              │
│   8  ─── SPHNet                                                 │
│   6                                                              │
│   4  [Error metrics box]                                        │
│   2  QHFlow: MAE=0.312 eV                                       │
│      SPHNet: MAE=0.256 eV                                       │
│      0.8   0.9   1.0   1.1   1.2  (stretch ratio)               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Script Execution Flow

### Using `run_all_experiments.sh`

```
1. Parse arguments
   └─ Validate checkpoint path
   └─ Check directory existence
   └─ Set defaults

2. STEP 1: Run Predictions (if not skipped)
   └─ Call run_sphnet_predictions.sh
       └─ For each molecule in [ethanol, aspirin, ...]
           └─ For each site in [primary, secondary]
               └─ Call exp_bond-stretch_model.py
                   └─ Load DFT data
                   └─ Run SPHNet forward pass
                   └─ Compute MO energies
                   └─ Save results
               └─ Track success/failure
       └─ Print summary

3. STEP 2: Generate Plots (if not skipped)
   └─ Call run_three_method_plots.sh
       └─ For each molecule in [ethanol, aspirin, ...]
           └─ For each site in [primary, secondary]
               └─ Check if all 3 results exist
                   ├─ Yes: Call exp_bond-stretch_plot_three_methods.py
                   │       └─ Load DFT, QHFlow, SPHNet data
                   │       └─ Compute comparison metrics
                   │       └─ Generate plot
                   └─ No: Skip (report missing method)
               └─ Track success/skip/failure
       └─ Print summary

4. Final Summary
   └─ Report total successes/failures
   └─ Show output locations
   └─ Suggest next steps
```

---

## Quick Command Reference

```bash
# Full workflow
./run_all_experiments.sh --ckpt-path model.ckpt

# Predictions only
./run_sphnet_predictions.sh --ckpt-path model.ckpt

# Plots only (using existing results)
./run_three_method_plots.sh \
    --dft-dir ./output \
    --qhflow-dir ./output \
    --sphnet-dir ./output

# Specific molecules
./run_all_experiments.sh --ckpt-path model.ckpt --molecules "ethanol,aspirin"

# Dry run
./run_all_experiments.sh --ckpt-path model.ckpt --dry-run

# CPU only
./run_all_experiments.sh --ckpt-path model.ckpt --device cpu

# Help
./run_all_experiments.sh --help
```

---

## Troubleshooting Flow

```
Issue: Script fails
   │
   ├─ Check: Checkpoint exists?
   │  └─ No: Provide correct path with --ckpt-path
   │  └─ Yes: Continue
   │
   ├─ Check: DFT results exist?
   │  └─ No: Run DFT calculations first (QHFlow-mlff repo)
   │  └─ Yes: Continue
   │
   ├─ Check: GPU available?
   │  └─ No: Use --device cpu
   │  └─ Yes: Continue
   │
   ├─ Check: Import errors?
   │  └─ Yes: Check Python path, install dependencies
   │  └─ No: Continue
   │
   └─ Still fails?
      └─ Use --dry-run to see commands
      └─ Run Python script directly for detailed errors
      └─ Check SCRIPTS_README.md for more help
```

---

## Performance Considerations

```
Typical Runtime (per molecule-site):
  DFT Calculation:    ~5-10 minutes (depends on molecule size)
  QHFlow Prediction:  ~1-2 seconds  (batch inference)
  SPHNet Prediction:  ~1-2 seconds  (batch inference)
  Plot Generation:    <1 second

Total for 6 molecules × avg 1.5 sites = ~9 molecule-site pairs:
  Predictions: ~18 seconds
  Plots:       ~9 seconds
  Total:       <1 minute

GPU vs CPU:
  GPU (CUDA):  ~2 seconds per molecule-site
  CPU:         ~10-30 seconds per molecule-site (5-15x slower)
```

---

See [QUICK_START.md](QUICK_START.md) for usage examples and [SCRIPTS_README.md](SCRIPTS_README.md) for detailed documentation.
