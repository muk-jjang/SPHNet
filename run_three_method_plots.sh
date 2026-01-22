#!/bin/bash
# Generate three-method comparison plots (DFT vs QHFlow-v2 vs SPHNet)
#
# This script generates comparison plots for all molecules and site types
# that have results from all three methods (DFT, QHFlow-v2, SPHNet).
#
# Usage:
#   ./run_three_method_plots.sh [OPTIONS]
#
# Options:
#   --dft-dir <path>          Directory containing DFT results (REQUIRED)
#   --qhflow-dir <path>       Directory containing QHFlow-v2 results (REQUIRED)
#   --sphnet-dir <path>       Directory containing SPHNet results (REQUIRED)
#   --save-dir <path>         Output directory for plots (default: ./plots)
#   --molecules <list>        Comma-separated list of molecules (default: all)
#   --dry-run                 Show what would be run without executing
#
# Example:
#   ./run_three_method_plots.sh \
#       --dft-dir /path/to/dft/results \
#       --qhflow-dir /path/to/qhflow/results \
#       --sphnet-dir /path/to/sphnet/results \
#       --save-dir ./plots
#
# Example (specific molecules only):
#   ./run_three_method_plots.sh \
#       --dft-dir ./output \
#       --qhflow-dir ./output \
#       --sphnet-dir ./output \
#       --molecules "ethanol,aspirin"

set -e  # Exit on error

# ============================================================================
# Default Configuration
# ============================================================================

# Required arguments (will be checked later)
DFT_DIR=""
QHFLOW_DIR=""
SPHNET_DIR=""

# Default output directory
SAVE_DIR="./plots"

# Molecule filter
MOLECULES_FILTER=""
DRY_RUN=false

# All molecules and their site types (from exp_config.yaml)
declare -A MOLECULE_SITES
MOLECULE_SITES["ethanol"]="primary secondary"
MOLECULE_SITES["malondialdehyde"]="primary"
MOLECULE_SITES["naphthalene"]="primary"
MOLECULE_SITES["salicylic_acid"]="primary secondary"
MOLECULE_SITES["aspirin"]="primary"
MOLECULE_SITES["uracil"]="primary secondary"

# Plot script path
PLOT_SCRIPT="/home/chanhui-lee/QHFlow-mlff/src/electronic_structure_exp/exp_bond-stretch_plot_three_methods.py"

# ============================================================================
# Parse Command Line Arguments
# ============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --dft-dir)
            DFT_DIR="$2"
            shift 2
            ;;
        --qhflow-dir)
            QHFLOW_DIR="$2"
            shift 2
            ;;
        --sphnet-dir)
            SPHNET_DIR="$2"
            shift 2
            ;;
        --save-dir)
            SAVE_DIR="$2"
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

MISSING_ARGS=false

if [ -z "$DFT_DIR" ]; then
    echo "Error: --dft-dir is required"
    MISSING_ARGS=true
fi

if [ -z "$QHFLOW_DIR" ]; then
    echo "Error: --qhflow-dir is required"
    MISSING_ARGS=true
fi

if [ -z "$SPHNET_DIR" ]; then
    echo "Error: --sphnet-dir is required"
    MISSING_ARGS=true
fi

if [ "$MISSING_ARGS" = true ]; then
    echo ""
    echo "Usage:"
    echo "  $0 --dft-dir <path> --qhflow-dir <path> --sphnet-dir <path> [OPTIONS]"
    echo ""
    echo "Use --help for more information"
    exit 1
fi

# Check directories exist
for dir_name in "DFT" "QHFlow" "SPHNet"; do
    case $dir_name in
        "DFT") dir_path="$DFT_DIR" ;;
        "QHFlow") dir_path="$QHFLOW_DIR" ;;
        "SPHNet") dir_path="$SPHNET_DIR" ;;
    esac

    if [ ! -d "$dir_path" ]; then
        echo "Error: $dir_name results directory not found: $dir_path"
        exit 1
    fi
done

# Check plot script exists
if [ ! -f "$PLOT_SCRIPT" ]; then
    echo "Error: Plot script not found: $PLOT_SCRIPT"
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
echo "Three-Method Comparison Plots"
echo "=========================================="
echo "DFT results:       $DFT_DIR"
echo "QHFlow results:    $QHFLOW_DIR"
echo "SPHNet results:    $SPHNET_DIR"
echo "Save directory:    $SAVE_DIR"
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
# Function to check if results exist
# ============================================================================

check_results_exist() {
    local molecule=$1
    local site=$2
    local dir=$3
    local method=$4

    # Check for metadata file
    local metadata_pattern="${molecule}_${site}_*_${method}_metadata.pt"
    local metadata_files=$(find "$dir" -maxdepth 1 -name "$metadata_pattern" 2>/dev/null)

    if [ -z "$metadata_files" ]; then
        return 1  # Not found
    fi

    return 0  # Found
}

# ============================================================================
# Generate Plots
# ============================================================================

TOTAL_PLOTS=0
SUCCESSFUL_PLOTS=0
FAILED_PLOTS=0
SKIPPED_PLOTS=0
declare -a FAILED_ITEMS
declare -a SKIPPED_ITEMS

# Count total plots
for molecule in "${MOLECULES_ARRAY[@]}"; do
    sites="${MOLECULE_SITES[$molecule]}"
    for site in $sites; do
        ((TOTAL_PLOTS++))
    done
done

echo "Total plot tasks: $TOTAL_PLOTS"
echo ""

CURRENT_PLOT=0

for molecule in "${MOLECULES_ARRAY[@]}"; do
    sites="${MOLECULE_SITES[$molecule]}"

    for site in $sites; do
        ((CURRENT_PLOT++))

        echo "=========================================="
        echo "[$CURRENT_PLOT/$TOTAL_PLOTS] Plotting: $molecule ($site)"
        echo "=========================================="

        # Check if all three results exist
        MISSING_RESULTS=false
        MISSING_METHODS=""

        if ! check_results_exist "$molecule" "$site" "$DFT_DIR" "dft"; then
            MISSING_RESULTS=true
            MISSING_METHODS="$MISSING_METHODS DFT"
        fi

        if ! check_results_exist "$molecule" "$site" "$QHFLOW_DIR" "qhflow-v2"; then
            MISSING_RESULTS=true
            MISSING_METHODS="$MISSING_METHODS QHFlow-v2"
        fi

        if ! check_results_exist "$molecule" "$site" "$SPHNET_DIR" "sphnet"; then
            MISSING_RESULTS=true
            MISSING_METHODS="$MISSING_METHODS SPHNet"
        fi

        if [ "$MISSING_RESULTS" = true ]; then
            echo "⊘ Skipping: Missing results for:$MISSING_METHODS"
            ((SKIPPED_PLOTS++))
            SKIPPED_ITEMS+=("$molecule ($site) - missing:$MISSING_METHODS")
            echo ""
            continue
        fi

        # Build command
        CMD="python $PLOT_SCRIPT \
            --dft-results $DFT_DIR \
            --qhflow-results $QHFLOW_DIR \
            --sphnet-results $SPHNET_DIR \
            --molecule $molecule \
            --site-type $site \
            --save-dir $SAVE_DIR"

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
            ((SUCCESSFUL_PLOTS++))
            echo ""
            echo "✓ Completed: $molecule ($site)"
            echo ""
        else
            ((FAILED_PLOTS++))
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
echo "Total tasks:       $TOTAL_PLOTS"
echo "Successful:        $SUCCESSFUL_PLOTS"
echo "Skipped:           $SKIPPED_PLOTS"
echo "Failed:            $FAILED_PLOTS"

if [ $SKIPPED_PLOTS -gt 0 ]; then
    echo ""
    echo "Skipped items (missing results):"
    for item in "${SKIPPED_ITEMS[@]}"; do
        echo "  - $item"
    done
fi

if [ $FAILED_PLOTS -gt 0 ]; then
    echo ""
    echo "Failed items:"
    for item in "${FAILED_ITEMS[@]}"; do
        echo "  - $item"
    done
fi

echo "=========================================="
echo ""

if [ "$DRY_RUN" = true ]; then
    echo "This was a dry run. No plots were generated."
    echo "Remove --dry-run to actually generate the plots."
    exit 0
fi

if [ $FAILED_PLOTS -eq 0 ]; then
    if [ $SUCCESSFUL_PLOTS -gt 0 ]; then
        echo "✓ All plots generated successfully!"
        echo ""
        echo "Plots saved to: $SAVE_DIR"
        echo ""
        echo "Generated $SUCCESSFUL_PLOTS comparison plots"
        if [ $SKIPPED_PLOTS -gt 0 ]; then
            echo "Note: $SKIPPED_PLOTS plots were skipped due to missing results"
        fi
    else
        echo "⊘ No plots were generated (all skipped due to missing results)"
        echo ""
        echo "Make sure you have run predictions for all three methods:"
        echo "  1. DFT calculations (exp_bond-stretch_dft.py)"
        echo "  2. QHFlow-v2 predictions (exp_bond-stretch_model.py)"
        echo "  3. SPHNet predictions (exp_bond-stretch_model.py in SPHNet repo)"
    fi
    exit 0
else
    echo "✗ Some plots failed. Check the errors above."
    exit 1
fi
