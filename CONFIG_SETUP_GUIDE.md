# SPHNet Checkpoint Configuration Guide

## Why Per-Molecule Checkpoints?

**Each molecule requires its own trained SPHNet model checkpoint.** This is because:
- Each molecule has different molecular structure
- Models are trained specifically on each molecule's MD trajectory
- Using the wrong checkpoint will produce incorrect predictions

## Configuration File: `exp_config_sphnet.yaml`

### 1. Locate Your Checkpoints

First, find where your SPHNet model checkpoints are stored:

```bash
# Example directory structure:
/path/to/sphnet/checkpoints/
├── ethanol/
│   └── last.ckpt
├── aspirin/
│   └── last.ckpt
├── malondialdehyde/
│   └── last.ckpt
└── ...
```

### 2. Edit the Config File

Open `exp_config_sphnet.yaml` and update the checkpoint paths:

```yaml
sphnet_checkpoints:
  ethanol:
    ckpt_path: "/actual/path/to/ethanol/last.ckpt"
    model_length_unit: "ang"

  aspirin:
    ckpt_path: "/actual/path/to/aspirin/last.ckpt"
    model_length_unit: "ang"

  # ... update all molecules
```

### 3. Three Ways to Specify Paths

#### Option A: Absolute Paths (Recommended)
```yaml
ethanol:
  ckpt_path: "/home/user/sphnet/checkpoints/ethanol/last.ckpt"
```

#### Option B: Relative Paths (relative to config file)
```yaml
ethanol:
  ckpt_path: "./checkpoints/ethanol/last.ckpt"
```

#### Option C: Home Directory Expansion
```yaml
ethanol:
  ckpt_path: "~/sphnet_models/ethanol/last.ckpt"
```

### 4. Verify Your Configuration

Check that all checkpoint files exist:

```bash
# Quick validation script
python3 << 'EOF'
import yaml
import os

config_file = "exp_config_sphnet.yaml"
with open(config_file, 'r') as f:
    config = yaml.safe_load(f)

checkpoints = config.get('sphnet_checkpoints', {})
print(f"Checking {len(checkpoints)} molecule checkpoints...\n")

for molecule, settings in checkpoints.items():
    ckpt_path = settings.get('ckpt_path', '')
    exists = os.path.exists(ckpt_path)
    status = "✓" if exists else "✗"
    print(f"{status} {molecule:20s} {ckpt_path}")

print("\nDone!")
EOF
```

## Example Config File

### Complete Example

```yaml
# SPHNet Bond Stretch Experiment Configuration
sphnet_checkpoints:
  # MD17 molecules
  ethanol:
    ckpt_path: "/home/user/sphnet/checkpoints/ethanol/last.ckpt"
    model_length_unit: "ang"

  malondialdehyde:
    ckpt_path: "/home/user/sphnet/checkpoints/malondialdehyde/last.ckpt"
    model_length_unit: "ang"

  uracil:
    ckpt_path: "/home/user/sphnet/checkpoints/uracil/last.ckpt"
    model_length_unit: "ang"

  # RMD17 molecules
  salicylic_acid:
    ckpt_path: "/home/user/sphnet/checkpoints/salicylic_acid/last.ckpt"
    model_length_unit: "ang"

  naphthalene:
    ckpt_path: "/home/user/sphnet/checkpoints/naphthalene/last.ckpt"
    model_length_unit: "ang"

  aspirin:
    ckpt_path: "/home/user/sphnet/checkpoints/aspirin/last.ckpt"
    model_length_unit: "ang"

default_settings:
  model_length_unit: "ang"
  device: "cuda"
```

### Partial Configuration (Only Some Molecules)

If you only have checkpoints for some molecules:

```yaml
sphnet_checkpoints:
  # Only configure molecules you have
  ethanol:
    ckpt_path: "/path/to/ethanol.ckpt"
    model_length_unit: "ang"

  aspirin:
    ckpt_path: "/path/to/aspirin.ckpt"
    model_length_unit: "ang"

  # Leave out molecules you don't have
  # They will be automatically skipped
```

## Using the Config File

### Run All Molecules with Config

```bash
./run_all_experiments.sh --config exp_config_sphnet.yaml
```

### Run Specific Molecules

```bash
./run_all_experiments.sh \
    --config exp_config_sphnet.yaml \
    --molecules "ethanol,aspirin"
```

### Predictions Only (No Plots)

```bash
./run_sphnet_predictions.sh --config exp_config_sphnet.yaml
```

## Single Checkpoint Mode (Not Recommended)

If you want to use the same checkpoint for all molecules (for testing):

```bash
./run_all_experiments.sh --ckpt-path /path/to/single/checkpoint.ckpt
```

**Warning**: This will use the same model for all molecules, which is **not scientifically valid** for production use. Only use for:
- Testing the workflow
- Debugging
- Demonstrations

## What Happens When Checkpoint is Missing?

The script will:
1. Check if checkpoint path is configured in YAML
2. Check if the file actually exists
3. If missing: **Skip that molecule** and continue with others
4. Print warning message
5. Show summary of skipped molecules at the end

Example output:
```
WARNING: No checkpoint configured for naphthalene, skipping all sites
⊘ Skipped: naphthalene (primary) - no checkpoint configured

========================================
Summary
========================================
Total tasks:       12
Successful:        10
Skipped:           2
Failed:            0

Skipped items (no checkpoint):
  - naphthalene (primary)
  - salicylic_acid (primary)
```

## Troubleshooting

### Problem: "Config file not found"
**Solution**: Check the path to config file
```bash
ls -l exp_config_sphnet.yaml
```

### Problem: "Checkpoint not found for molecule X"
**Solution**: Verify the path in config file
```bash
# Check what's in config
grep -A2 "ethanol:" exp_config_sphnet.yaml

# Verify file exists
ls -l /path/from/config/file
```

### Problem: All molecules skipped
**Solution**: Check YAML syntax and paths
```bash
# Validate YAML syntax
python3 -c "import yaml; yaml.safe_load(open('exp_config_sphnet.yaml'))"

# Check paths
python3 << EOF
import yaml
with open('exp_config_sphnet.yaml') as f:
    config = yaml.safe_load(f)
    for mol, settings in config['sphnet_checkpoints'].items():
        print(f"{mol}: {settings['ckpt_path']}")
EOF
```

### Problem: Wrong checkpoint being used
**Solution**: Check which checkpoint is loaded
```bash
# Use dry-run to see commands
./run_sphnet_predictions.sh --config exp_config_sphnet.yaml --dry-run
```

## Model Length Units

SPHNet typically uses **Angstrom** for coordinates. If your model uses Bohr:

```yaml
ethanol:
  ckpt_path: "/path/to/checkpoint.ckpt"
  model_length_unit: "bohr"  # Change to "bohr" if needed
```

To check what your model uses:
1. Check training logs
2. Check model configuration
3. Default for SPHNet is usually "ang"

## Best Practices

1. **Use absolute paths** for checkpoint files
2. **Verify all paths exist** before running
3. **Keep config file** in SPHNet repository root
4. **Comment out** molecules you don't have:
   ```yaml
   # ethanol:
   #   ckpt_path: "/path/not/available/yet"
   ```
5. **Use per-molecule checkpoints** (not single checkpoint) for valid results
6. **Back up** your config file after setting it up

## Quick Setup Workflow

```bash
# 1. Copy template
cp exp_config_sphnet.yaml my_config.yaml

# 2. Edit paths
nano my_config.yaml

# 3. Validate
python3 -c "import yaml; print('✓ Valid YAML'); yaml.safe_load(open('my_config.yaml'))"

# 4. Dry run
./run_all_experiments.sh --config my_config.yaml --dry-run

# 5. Run for real
./run_all_experiments.sh --config my_config.yaml
```

## Config File Template

A template `exp_config_sphnet.yaml` is provided with placeholder paths. Update it with your actual checkpoint locations.

---

For more details, see:
- [QUICK_START.md](QUICK_START.md) - Quick start guide
- [SCRIPTS_README.md](SCRIPTS_README.md) - Full script documentation
- [EXPERIMENT_README.md](EXPERIMENT_README.md) - Complete experiment guide
