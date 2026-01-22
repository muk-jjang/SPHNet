# Update: Per-Molecule Checkpoint Support

## What Changed?

The bash scripts now support **per-molecule checkpoints**, which is the correct way to run SPHNet predictions since each molecule requires its own trained model.

## Why This Matters

### ❌ Previous Approach (Incorrect)
```bash
# Using same checkpoint for all molecules - WRONG!
./run_all_experiments.sh --ckpt-path single_checkpoint.ckpt
```

**Problem**: This would use the same SPHNet model for ethanol, aspirin, malondialdehyde, etc., which is scientifically invalid since each molecule needs a model trained specifically on its MD trajectory.

### ✅ New Approach (Correct)
```bash
# Using per-molecule checkpoints - CORRECT!
./run_all_experiments.sh --config exp_config_sphnet.yaml
```

**Benefit**: Each molecule gets predictions from its own trained model, ensuring valid scientific results.

---

## What Was Updated?

### 1. New Config File: `exp_config_sphnet.yaml`
A YAML configuration file where you specify checkpoint paths for each molecule:

```yaml
sphnet_checkpoints:
  ethanol:
    ckpt_path: "/path/to/ethanol/checkpoint.ckpt"
    model_length_unit: "ang"

  aspirin:
    ckpt_path: "/path/to/aspirin/checkpoint.ckpt"
    model_length_unit: "ang"

  # ... one entry per molecule
```

### 2. Updated Script: `run_sphnet_predictions.sh`
- Added `--config` option to read checkpoint paths from YAML
- Automatically loads correct checkpoint for each molecule
- Skips molecules with missing/unconfigured checkpoints
- Reports skipped molecules in summary

### 3. Updated Script: `run_all_experiments.sh`
- Added `--config` option
- Passes config to prediction script
- Supports both single checkpoint (legacy) and per-molecule (recommended)

### 4. New Documentation: `CONFIG_SETUP_GUIDE.md`
Complete guide for setting up the configuration file

---

## How to Use

### Quick Start

```bash
cd /home/chanhui-lee/SPHNet

# 1. Edit config file with your checkpoint paths
nano exp_config_sphnet.yaml

# 2. Run predictions with config
./run_all_experiments.sh --config exp_config_sphnet.yaml
```

### What the Config Looks Like

```yaml
sphnet_checkpoints:
  ethanol:
    ckpt_path: "/home/user/sphnet/models/ethanol/last.ckpt"
    model_length_unit: "ang"

  malondialdehyde:
    ckpt_path: "/home/user/sphnet/models/malondialdehyde/last.ckpt"
    model_length_unit: "ang"

  uracil:
    ckpt_path: "/home/user/sphnet/models/uracil/last.ckpt"
    model_length_unit: "ang"

  salicylic_acid:
    ckpt_path: "/home/user/sphnet/models/salicylic_acid/last.ckpt"
    model_length_unit: "ang"

  naphthalene:
    ckpt_path: "/home/user/sphnet/models/naphthalene/last.ckpt"
    model_length_unit: "ang"

  aspirin:
    ckpt_path: "/home/user/sphnet/models/aspirin/last.ckpt"
    model_length_unit: "ang"
```

---

## Features

### Automatic Checkpoint Selection
The script automatically:
1. Reads the config file
2. Finds the checkpoint for each molecule
3. Uses the correct checkpoint when processing that molecule
4. Skips molecules with missing checkpoints

### Graceful Handling of Missing Checkpoints
If a checkpoint is not configured or doesn't exist:
```
WARNING: No checkpoint configured for naphthalene, skipping all sites
⊘ Skipped: naphthalene (primary) - no checkpoint configured
```

### Summary Report
```
========================================
Summary
========================================
Total tasks:       12
Successful:        10
Skipped:           2
Failed:            0

Skipped items (no checkpoint):
  - naphthalene (primary)
  - salicylic_acid (secondary)
```

---

## Backward Compatibility

### Old Way Still Works (For Testing)
You can still use a single checkpoint:
```bash
./run_all_experiments.sh --ckpt-path /path/to/single.ckpt
```

**But**: This uses the same model for all molecules, which is **not valid** for scientific use. Only for:
- Testing the workflow
- Debugging
- Demonstrations

### Recommended Way (Production)
```bash
./run_all_experiments.sh --config exp_config_sphnet.yaml
```

Use this for actual research/production work.

---

## Command Examples

### All Molecules with Config
```bash
./run_all_experiments.sh --config exp_config_sphnet.yaml
```

### Specific Molecules with Config
```bash
./run_all_experiments.sh \
    --config exp_config_sphnet.yaml \
    --molecules "ethanol,aspirin"
```

### Predictions Only (No Plots)
```bash
./run_sphnet_predictions.sh --config exp_config_sphnet.yaml
```

### Dry Run (Preview)
```bash
./run_all_experiments.sh \
    --config exp_config_sphnet.yaml \
    --dry-run
```

---

## Validation Script

Check if all your checkpoint paths are valid:

```bash
python3 << 'EOF'
import yaml
import os

config_file = "exp_config_sphnet.yaml"
with open(config_file, 'r') as f:
    config = yaml.safe_load(f)

checkpoints = config.get('sphnet_checkpoints', {})
print(f"Validating {len(checkpoints)} checkpoints...\n")

valid = 0
invalid = 0

for molecule, settings in checkpoints.items():
    ckpt_path = settings.get('ckpt_path', '')

    if os.path.exists(ckpt_path):
        print(f"✓ {molecule:20s} {ckpt_path}")
        valid += 1
    else:
        print(f"✗ {molecule:20s} {ckpt_path} (NOT FOUND)")
        invalid += 1

print(f"\n{valid} valid, {invalid} invalid")
EOF
```

---

## Troubleshooting

### All Molecules Skipped
**Cause**: Config file not found or paths are wrong
**Solution**:
```bash
# Check config exists
ls -l exp_config_sphnet.yaml

# Validate paths
python3 -c "import yaml; print(yaml.safe_load(open('exp_config_sphnet.yaml')))"
```

### Specific Molecule Skipped
**Cause**: Checkpoint path missing or incorrect
**Solution**: Edit config file and verify path exists

### "Config file not found"
**Cause**: Wrong path to config
**Solution**: Use absolute path or run from SPHNet directory
```bash
cd /home/chanhui-lee/SPHNet
./run_all_experiments.sh --config exp_config_sphnet.yaml
```

---

## Migration Guide

### If You Were Using Single Checkpoint

**Old command**:
```bash
./run_all_experiments.sh --ckpt-path /path/to/model.ckpt
```

**New command** (after setting up config):
```bash
# 1. Create config file
cp exp_config_sphnet.yaml my_config.yaml

# 2. Edit with your paths
nano my_config.yaml

# 3. Run with config
./run_all_experiments.sh --config my_config.yaml
```

---

## Files Changed/Added

### Added
- `exp_config_sphnet.yaml` - Per-molecule checkpoint configuration
- `CONFIG_SETUP_GUIDE.md` - Configuration guide
- `UPDATE_PER_MOLECULE_CKPT.md` - This document

### Modified
- `run_sphnet_predictions.sh` - Added `--config` option
- `run_all_experiments.sh` - Added `--config` option
- `QUICK_START.md` - Updated examples
- `SCRIPTS_README.md` - Updated documentation

---

## Summary

✅ **Use per-molecule checkpoints** for valid scientific results
✅ **Configure once** in `exp_config_sphnet.yaml`
✅ **Automatic handling** of missing checkpoints
✅ **Backward compatible** with single checkpoint mode
✅ **Clear error messages** when checkpoints missing

See [CONFIG_SETUP_GUIDE.md](CONFIG_SETUP_GUIDE.md) for detailed setup instructions.
