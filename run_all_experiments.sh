#!/bin/bash
# Master script to run complete three-method comparison experiment
#
# This script runs the complete workflow:
#   1. SPHNet predictions for all molecules
#   2. Three-method comparison plots (DFT vs QHFlow-v2 vs SPHNet)
#
# Prerequisites:
#   - DFT reference data must already exist
#   - QHFlow-v2 predictions must already exist
#
# Usage:
#   ./run_all_experiments.sh --ckpt-path <sphnet_checkpoint> [OPTIONS]
#
# Options:
#   --ckpt-path <path>        Path to single SPHNet checkpoint for all molecules
#   --config <yaml>           Path to config file with per-molecule checkpoints
#                             (Use either --ckpt-path OR --config, not both)
#   --dft-dir <path>          Directory containing DFT results
#   --qhflow-dir <path>       Directory containing QHFlow-v2 results
#   --sphnet-save-dir <path>  Output directory for SPHNet predictions
#   --plots-save-dir <path>   Output directory for comparison plots
#   --device <device>         Device for SPHNet inference (cuda/cpu)
#   --molecules <list>        Comma-separated list of molecules
#   --skip-predictions        Skip SPHNet predictions (only generate plots)
#   --skip-plots              Skip plot generation (only run predictions)
#   --dry-run                 Show what would be run without executing
#
# Example (single checkpoint):
#   ./run_all_experiments.sh \
#       --ckpt-path /path/to/sphnet/model.ckpt
#
# Example (per-molecule checkpoints - RECOMMENDED):
#   ./run_all_experiments.sh \
#       --config exp_config_sphnet.yaml
#
# Example (specific molecules):
#   ./run_all_experiments.sh \
#       --config exp_config_sphnet.yaml \
#       --molecules "ethanol,aspirin"

set -e  # Exit on error

# ============================================================================
# Script Directory
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================================
# Default Configuration
# ============================================================================

# Required (one of these)
CKPT_PATH=""
CONFIG_FILE=""

# Directories
DFT_DIR="/home/chanhui-lee/QHFlow-mlff/src/electronic_structure_exp/output"
QHFLOW_DIR="/home/chanhui-lee/QHFlow-mlff/src/electronic_structure_exp/output"
SPHNET_SAVE_DIR="$SCRIPT_DIR/output"
PLOTS_SAVE_DIR="$SCRIPT_DIR/plots"

# Model settings
DEVICE="cuda"

# Workflow control
SKIP_PREDICTIONS=false
SKIP_PLOTS=false
DRY_RUN=false

# Molecule filter
MOLECULES_FILTER=""

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
            shift 2
            ;;
        --dft-dir)
            DFT_DIR="$2"
            shift 2
            ;;
        --qhflow-dir)
            QHFLOW_DIR="$2"
            shift 2
            ;;
        --sphnet-save-dir)
            SPHNET_SAVE_DIR="$2"
            shift 2
            ;;
        --plots-save-dir)
            PLOTS_SAVE_DIR="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --molecules)
            MOLECULES_FILTER="$2"
            shift 2
            ;;
        --skip-predictions)
            SKIP_PREDICTIONS=true
            shift
            ;;
        --skip-plots)
            SKIP_PLOTS=true
            shift
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
# Validate Configuration
# ============================================================================

if [ "$SKIP_PREDICTIONS" = false ]; then
    if [ -z "$CKPT_PATH" ] && [ -z "$CONFIG_FILE" ]; then
        echo "Error: Either --ckpt-path or --config is required (unless --skip-predictions is used)"
        echo ""
        echo "Options:"
        echo "  1. Single checkpoint: --ckpt-path <path>"
        echo "  2. Per-molecule checkpoints: --config <config.yaml>"
        echo ""
        echo "Use --help for more information"
        exit 1
    fi
fi

if [ "$SKIP_PREDICTIONS" = true ] && [ "$SKIP_PLOTS" = true ]; then
    echo "Error: Cannot skip both predictions and plots"
    exit 1
fi

# ============================================================================
# Display Configuration
# ============================================================================

echo "=========================================="
echo "Complete Three-Method Experiment"
echo "=========================================="
echo ""
echo "Configuration:"
if [ -n "$CONFIG_FILE" ]; then
    echo "  Config file:       $CONFIG_FILE (per-molecule checkpoints)"
elif [ -n "$CKPT_PATH" ]; then
    echo "  SPHNet checkpoint: $CKPT_PATH (single for all)"
else
    echo "  SPHNet checkpoint: N/A (predictions skipped)"
fi
echo "  DFT results:       $DFT_DIR"
echo "  QHFlow results:    $QHFLOW_DIR"
echo "  SPHNet output:     $SPHNET_SAVE_DIR"
echo "  Plots output:      $PLOTS_SAVE_DIR"
echo "  Device:            $DEVICE"
if [ -n "$MOLECULES_FILTER" ]; then
    echo "  Molecules:         $MOLECULES_FILTER"
else
    echo "  Molecules:         All"
fi
echo ""

echo "Workflow:"
if [ "$SKIP_PREDICTIONS" = false ]; then
    echo "  ✓ Run SPHNet predictions"
else
    echo "  ⊘ Skip SPHNet predictions"
fi
if [ "$SKIP_PLOTS" = false ]; then
    echo "  ✓ Generate comparison plots"
else
    echo "  ⊘ Skip comparison plots"
fi
if [ "$DRY_RUN" = true ]; then
    echo ""
    echo "  Mode: DRY RUN (no execution)"
fi
echo "=========================================="
echo ""

# ============================================================================
# Step 1: Run SPHNet Predictions
# ============================================================================

if [ "$SKIP_PREDICTIONS" = false ]; then
    echo ""
    echo "=========================================="
    echo "STEP 1: Running SPHNet Predictions"
    echo "=========================================="
    echo ""

    PRED_CMD="$SCRIPT_DIR/run_sphnet_predictions.sh \
        --dft-dir \"$DFT_DIR\" \
        --save-dir \"$SPHNET_SAVE_DIR\" \
        --device $DEVICE"

    # Add checkpoint argument (either single or config)
    if [ -n "$CONFIG_FILE" ]; then
        PRED_CMD="$PRED_CMD --config \"$CONFIG_FILE\""
    elif [ -n "$CKPT_PATH" ]; then
        PRED_CMD="$PRED_CMD --ckpt-path \"$CKPT_PATH\""
    fi

    if [ -n "$MOLECULES_FILTER" ]; then
        PRED_CMD="$PRED_CMD --molecules \"$MOLECULES_FILTER\""
    fi

    if [ "$DRY_RUN" = true ]; then
        PRED_CMD="$PRED_CMD --dry-run"
    fi

    echo "Running: $PRED_CMD"
    echo ""

    if eval "$PRED_CMD"; then
        echo ""
        echo "✓ SPHNet predictions completed successfully"
    else
        echo ""
        echo "✗ SPHNet predictions failed"
        exit 1
    fi
else
    echo ""
    echo "=========================================="
    echo "STEP 1: Skipping SPHNet Predictions"
    echo "=========================================="
    echo "Using existing results from: $SPHNET_SAVE_DIR"
fi

# ============================================================================
# Step 2: Generate Comparison Plots
# ============================================================================

if [ "$SKIP_PLOTS" = false ]; then
    echo ""
    echo "=========================================="
    echo "STEP 2: Generating Comparison Plots"
    echo "=========================================="
    echo ""

    PLOT_CMD="$SCRIPT_DIR/run_three_method_plots.sh \
        --dft-dir \"$DFT_DIR\" \
        --qhflow-dir \"$QHFLOW_DIR\" \
        --sphnet-dir \"$SPHNET_SAVE_DIR\" \
        --save-dir \"$PLOTS_SAVE_DIR\""

    if [ -n "$MOLECULES_FILTER" ]; then
        PLOT_CMD="$PLOT_CMD --molecules \"$MOLECULES_FILTER\""
    fi

    if [ "$DRY_RUN" = true ]; then
        PLOT_CMD="$PLOT_CMD --dry-run"
    fi

    echo "Running: $PLOT_CMD"
    echo ""

    if eval "$PLOT_CMD"; then
        echo ""
        echo "✓ Comparison plots completed successfully"
    else
        echo ""
        echo "✗ Comparison plots failed"
        exit 1
    fi
else
    echo ""
    echo "=========================================="
    echo "STEP 2: Skipping Comparison Plots"
    echo "=========================================="
fi

# ============================================================================
# Final Summary
# ============================================================================

echo ""
echo "=========================================="
echo "Experiment Complete!"
echo "=========================================="
echo ""

if [ "$DRY_RUN" = true ]; then
    echo "This was a dry run. No operations were performed."
    echo "Remove --dry-run to execute the experiment."
else
    if [ "$SKIP_PREDICTIONS" = false ]; then
        echo "✓ SPHNet predictions saved to:"
        echo "  $SPHNET_SAVE_DIR"
        echo ""
    fi

    if [ "$SKIP_PLOTS" = false ]; then
        echo "✓ Comparison plots saved to:"
        echo "  $PLOTS_SAVE_DIR"
        echo ""
    fi

    echo "To view individual plots, check the output directory."
    echo ""
    echo "Result files follow this naming pattern:"
    echo "  - SPHNet predictions: {molecule}_{site}_{bond}_sphnet.pt"
    echo "  - Plots: {molecule}_{site}_{bond}_comparison_three_methods.png"
fi

echo ""
echo "=========================================="
