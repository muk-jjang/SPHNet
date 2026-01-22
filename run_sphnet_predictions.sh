#!/bin/bash
# Run SPHNet model predictions for bond stretch HOMO-LUMO experiment
#
# This script runs SPHNet predictions for all molecules and site types
# using DFT reference data from QHFlow-mlff experiments.
# The results can be used for three-method comparison plots.
#
# Usage:
#   ./run_sphnet_predictions.sh [OPTIONS]
#
# Options:
#   --ckpt-path <path>        Path to single SPHNet checkpoint for all molecules
#   --config <yaml>           Path to config file with per-molecule checkpoints
#                             (Use either --ckpt-path OR --config, not both)
#   --dft-dir <path>          Directory containing DFT results (default: QHFlow exp output)
#   --save-dir <path>         Output directory for SPHNet predictions (default: ./output)
#   --device <device>         Device for inference: cuda or cpu (default: cuda)
#   --model-length-unit <unit> Length unit: ang or bohr (default: ang)
#   --molecules <list>        Comma-separated list of molecules (default: all)
#   --dry-run                 Show what would be run without executing
#
# Example (single checkpoint):
#   ./run_sphnet_predictions.sh \
#       --ckpt-path /path/to/sphnet/model.ckpt \
#       --dft-dir /home/chanhui-lee/QHFlow-mlff/src/electronic_structure_exp/output \
#       --save-dir ./output
#
# Example (per-molecule checkpoints - RECOMMENDED):
#   ./run_sphnet_predictions.sh \
#       --config exp_config_sphnet.yaml \
#       --save-dir ./output
#
# Example (specific molecules only):
#   ./run_sphnet_predictions.sh \
#       --config exp_config_sphnet.yaml \
#       --molecules "ethanol,aspirin"

set -e  # Exit on error

# ============================================================================
# Default Configuration
# ============================================================================

# Checkpoint configuration
CKPT_PATH=""              # Single checkpoint for all molecules (optional)
CONFIG_FILE=""            # Config file with per-molecule checkpoints (optional)

# Default paths
DFT_DIR="/home/chanhui-lee/QHFlow-mlff/src/electronic_structure_exp/output"
SAVE_DIR="./output"

# Default model settings
DEVICE="cuda"
MODEL_LENGTH_UNIT="ang"

# Molecule and site configuration
# Format: molecule:site1,site2|molecule2:site1
# If a molecule has no sites specified, all sites from config will be used
MOLECULES_FILTER=""
DRY_RUN=false
USE_PER_MOLECULE_CKPT=false

# All molecules and their site types (from exp_config.yaml)
declare -A MOLECULE_SITES
MOLECULE_SITES["ethanol"]="primary secondary"
MOLECULE_SITES["malondialdehyde"]="primary"
MOLECULE_SITES["naphthalene"]="primary"
MOLECULE_SITES["salicylic_acid"]="primary secondary"
MOLECULE_SITES["aspirin"]="primary"
MOLECULE_SITES["uracil"]="primary secondary"

# ============================================================================
# Parse Command Line Arguments
# ============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --ckpt-path)
            CKPT_PATH="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            USE_PER_MOLECULE_CKPT=true
            shift 2
            ;;
        --dft-dir)
            DFT_DIR="$2"
            shift 2
            ;;
        --save-dir)
            SAVE_DIR="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --model-length-unit)
            MODEL_LENGTH_UNIT="$2"
            shift 2
            ;;
        --molecules)
            MOLECULES_FILTER="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            # Extract and display the header comment
            sed -n '2,/^$/p' "$0" | sed 's/^# //'
            exit 0
            ;;
        *)
            echo "Error: Unknown argument: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# ============================================================================
# Validate Required Arguments
# ============================================================================

# Check if either single checkpoint or config file is provided
if [ -z "$CKPT_PATH" ] && [ -z "$CONFIG_FILE" ]; then
    echo "Error: Either --ckpt-path or --config is required"
    echo ""
    echo "Usage Options:"
    echo "  1. Single checkpoint for all molecules:"
    echo "     $0 --ckpt-path <path> [OPTIONS]"
    echo ""
    echo "  2. Per-molecule checkpoints from config:"
    echo "     $0 --config <config.yaml> [OPTIONS]"
    echo ""
    echo "Use --help for more information"
    exit 1
fi

# If using single checkpoint, validate it exists
if [ -n "$CKPT_PATH" ] && [ ! -f "$CKPT_PATH" ]; then
    echo "Error: Checkpoint file not found: $CKPT_PATH"
    exit 1
fi

# If using config file, validate it exists
if [ "$USE_PER_MOLECULE_CKPT" = true ]; then
    if [ ! -f "$CONFIG_FILE" ]; then
        echo "Error: Config file not found: $CONFIG_FILE"
        exit 1
    fi
    echo "Using per-molecule checkpoints from: $CONFIG_FILE"
fi

if [ ! -d "$DFT_DIR" ]; then
    echo "Error: DFT results directory not found: $DFT_DIR"
    exit 1
fi

# ============================================================================
# Determine Which Molecules to Process
# ============================================================================

# If specific molecules are requested, filter the list
if [ -n "$MOLECULES_FILTER" ]; then
    # Split by comma
    IFS=',' read -ra MOLECULES_ARRAY <<< "$MOLECULES_FILTER"

    # Validate molecules
    for mol in "${MOLECULES_ARRAY[@]}"; do
        if [ -z "${MOLECULE_SITES[$mol]}" ]; then
            echo "Error: Unknown molecule: $mol"
            echo "Available molecules: ${!MOLECULE_SITES[@]}"
            exit 1
        fi
    done
else
    # Process all molecules
    MOLECULES_ARRAY=("${!MOLECULE_SITES[@]}")
fi

# ============================================================================
# Display Configuration
# ============================================================================

echo "=========================================="
echo "SPHNet Bond Stretch Predictions"
echo "=========================================="
if [ "$USE_PER_MOLECULE_CKPT" = true ]; then
    echo "Config file:       $CONFIG_FILE"
    echo "Checkpoint mode:   Per-molecule checkpoints"
else
    echo "Checkpoint:        $CKPT_PATH"
    echo "Checkpoint mode:   Single checkpoint for all"
fi
echo "DFT results dir:   $DFT_DIR"
echo "Save directory:    $SAVE_DIR"
echo "Device:            $DEVICE"
echo "Length unit:       $MODEL_LENGTH_UNIT"
echo "Molecules:         ${MOLECULES_ARRAY[@]}"
if [ "$DRY_RUN" = true ]; then
    echo "Mode:              DRY RUN (no execution)"
fi
echo "=========================================="
echo ""

# Create output directory
if [ "$DRY_RUN" = false ]; then
    mkdir -p "$SAVE_DIR"
fi

# ============================================================================
# Run Predictions
# ============================================================================

TOTAL_RUNS=0
SUCCESSFUL_RUNS=0
FAILED_RUNS=0
SKIPPED_RUNS=0
declare -a FAILED_ITEMS
declare -a SKIPPED_ITEMS

# Count total runs
for molecule in "${MOLECULES_ARRAY[@]}"; do
    sites="${MOLECULE_SITES[$molecule]}"
    for site in $sites; do
        ((TOTAL_RUNS++))
    done
done

echo "Total prediction tasks: $TOTAL_RUNS"
echo ""

CURRENT_RUN=0

for molecule in "${MOLECULES_ARRAY[@]}"; do
    sites="${MOLECULE_SITES[$molecule]}"

    # Determine checkpoint for this molecule
    if [ "$USE_PER_MOLECULE_CKPT" = true ]; then
        # Extract checkpoint from config file using Python
        MOLECULE_CKPT=$(python3 -c "
import yaml
import sys
try:
    with open('$CONFIG_FILE', 'r') as f:
        config = yaml.safe_load(f)
    ckpt = config.get('sphnet_checkpoints', {}).get('$molecule', {}).get('ckpt_path', '')
    print(ckpt)
except Exception as e:
    print('', file=sys.stderr)
    sys.exit(1)
" 2>/dev/null)

        if [ -z "$MOLECULE_CKPT" ] || [ "$MOLECULE_CKPT" = "None" ]; then
            echo "WARNING: No checkpoint configured for $molecule, skipping all sites"
            # Skip all sites for this molecule
            for site in $sites; do
                ((CURRENT_RUN++))
                ((SKIPPED_RUNS++))
                echo "⊘ Skipped: $molecule ($site) - no checkpoint configured"
            done
            continue
        fi

        if [ ! -f "$MOLECULE_CKPT" ]; then
            echo "WARNING: Checkpoint not found for $molecule: $MOLECULE_CKPT"
            echo "Skipping all sites for $molecule"
            # Skip all sites for this molecule
            for site in $sites; do
                ((CURRENT_RUN++))
                ((SKIPPED_RUNS++))
                echo "⊘ Skipped: $molecule ($site) - checkpoint not found"
            done
            continue
        fi
    else
        # Use single checkpoint for all molecules
        MOLECULE_CKPT="$CKPT_PATH"
    fi

    for site in $sites; do
        ((CURRENT_RUN++))

        echo "=========================================="
        echo "[$CURRENT_RUN/$TOTAL_RUNS] Running: $molecule ($site)"
        if [ "$USE_PER_MOLECULE_CKPT" = true ]; then
            echo "Checkpoint: $MOLECULE_CKPT"
        fi
        echo "=========================================="

        # Build command
        CMD="python exp_bond-stretch_model.py \
            --molecule $molecule \
            --site-type $site \
            --load-results $DFT_DIR \
            --ckpt-path $MOLECULE_CKPT \
            --save-dir $SAVE_DIR \
            --device $DEVICE \
            --model-length-unit $MODEL_LENGTH_UNIT"

        if [ "$DRY_RUN" = true ]; then
            echo "[DRY RUN] Would execute:"
            echo "$CMD"
            echo ""
            continue
        fi

        # Execute command
        echo "Command: $CMD"
        echo ""

        if eval "$CMD"; then
            ((SUCCESSFUL_RUNS++))
            echo ""
            echo "✓ Completed: $molecule ($site)"
            echo ""
        else
            ((FAILED_RUNS++))
            FAILED_ITEMS+=("$molecule ($site)")
            echo ""
            echo "✗ Failed: $molecule ($site)"
            echo ""
        fi
    done
done

# ============================================================================
# Summary
# ============================================================================

echo "=========================================="
echo "Summary"
echo "=========================================="
echo "Total tasks:       $TOTAL_RUNS"
echo "Successful:        $SUCCESSFUL_RUNS"
echo "Skipped:           $SKIPPED_RUNS"
echo "Failed:            $FAILED_RUNS"

if [ $SKIPPED_RUNS -gt 0 ]; then
    echo ""
    echo "Skipped items (no checkpoint):"
    for item in "${SKIPPED_ITEMS[@]}"; do
        echo "  - $item"
    done
fi

if [ $FAILED_RUNS -gt 0 ]; then
    echo ""
    echo "Failed items:"
    for item in "${FAILED_ITEMS[@]}"; do
        echo "  - $item"
    done
fi

echo "=========================================="
echo ""

if [ "$DRY_RUN" = true ]; then
    echo "This was a dry run. No predictions were executed."
    echo "Remove --dry-run to actually run the predictions."
    exit 0
fi

if [ $FAILED_RUNS -eq 0 ]; then
    echo "✓ All predictions completed successfully!"
    echo ""
    echo "Results saved to: $SAVE_DIR"
    echo ""
    echo "Next steps:"
    echo "1. Compare with QHFlow-v2 predictions using:"
    echo "   cd /home/chanhui-lee/QHFlow-mlff/src/electronic_structure_exp"
    echo "   python exp_bond-stretch_plot_three_methods.py \\"
    echo "       --dft-results $DFT_DIR \\"
    echo "       --qhflow-results $DFT_DIR \\"
    echo "       --sphnet-results $SAVE_DIR \\"
    echo "       --molecule <molecule_name> \\"
    echo "       --site-type <site_type> \\"
    echo "       --save-dir ./plots"
    echo ""
    echo "2. Or generate all comparison plots using the batch plot script"
    exit 0
else
    echo "✗ Some predictions failed. Check the errors above."
    exit 1
fi
