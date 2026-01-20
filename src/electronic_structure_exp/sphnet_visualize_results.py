#!/usr/bin/env python3
"""
SPHNet Results Visualization Script

Visualize bond stretch analysis results from SPHNet inference.
Supports comparison with DFT reference data.

Usage:
    python sphnet_visualize_results.py \
        --results-path /path/to/sphnet_inference_results.pt \
        --save-dir ./output
        
    # With DFT reference comparison:
    python sphnet_visualize_results.py \
        --results-path /path/to/sphnet_inference_results.pt \
        --dft-results-path /path/to/dft_results_metadata.pt \
        --save-dir ./output
"""

import os
import sys
import argparse
import numpy as np
import torch
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

# Add project root to path
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "../..")
sys.path.insert(0, PROJECT_ROOT)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ==============================================================================
# Constants
# ==============================================================================

ATOMIC_SYMBOLS = {
    1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F',
    15: 'P', 16: 'S', 17: 'Cl', 35: 'Br', 53: 'I'
}

# Unit conversions
HA2eV  = 27.211396641308  
HA2meV   = HA2eV * 1000 
BOHR2ANG = 0.5291772105638411  
HA_BOHR_2_eV_ANG = HA2eV / BOHR2ANG
HA_BOHR_2_meV_ANG = HA2meV / BOHR2ANG


# ==============================================================================
# Data Loading
# ==============================================================================

def load_sphnet_results(results_path):
    """
    Load SPHNet inference results from .pt file.
    
    Args:
        results_path (str): Path to SPHNet results file
        
    Returns:
        dict: Results dictionary containing metadata and predictions
    """
    data = torch.load(results_path, weights_only=False)
    
    print(f"Loaded SPHNet results from: {results_path}")
    
    if 'metadata' in data:
        # Bond stretch experiment results
        metadata = data['metadata']
        predictions = data['predictions']
        
        print(f"  Molecule: {metadata.get('molecule_name', 'Unknown')}")
        print(f"  Site type: {metadata.get('site_type', 'Unknown')}")
        print(f"  Number of predictions: {len(predictions)}")
        
        return {
            'type': 'bond_stretch',
            'metadata': metadata,
            'predictions': predictions
        }
    else:
        # Single prediction result
        print(f"  Single prediction result")
        return {
            'type': 'single',
            'prediction': data
        }


def load_dft_results(dft_path):
    """
    Load DFT reference results from metadata .pt file.
    
    Args:
        dft_path (str): Path to DFT metadata file (*_metadata.pt)
        
    Returns:
        dict: DFT results dictionary
    """
    # Check if it's a metadata file or directory
    if os.path.isdir(dft_path):
        # Find metadata file in directory
        metadata_files = [f for f in os.listdir(dft_path) if f.endswith('_metadata.pt')]
        if len(metadata_files) == 0:
            raise FileNotFoundError(f"No metadata file found in {dft_path}")
        dft_path = os.path.join(dft_path, metadata_files[0])
    
    metadata = torch.load(dft_path, weights_only=False)
    
    print(f"Loaded DFT metadata from: {dft_path}")
    
    # Convert tensors to numpy
    def to_numpy(obj):
        if isinstance(obj, torch.Tensor):
            return obj.numpy()
        elif isinstance(obj, dict):
            return {k: to_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [to_numpy(v) for v in obj]
        return obj
    
    metadata = to_numpy(metadata)
    
    # Load individual result files
    result_dir = os.path.dirname(dft_path)
    result_files = metadata.get('result_files', [])
    
    list_stretched_results = []
    for filename in result_files:
        result_path = os.path.join(result_dir, filename)
        if os.path.exists(result_path):
            result = torch.load(result_path, weights_only=False)
            result = to_numpy(result)
            list_stretched_results.append(result)
    
    print(f"  Molecule: {metadata.get('molecule_name', 'Unknown')}")
    print(f"  Site type: {metadata.get('site_type', 'Unknown')}")
    print(f"  Loaded {len(list_stretched_results)} DFT results")
    
    return {
        'metadata': metadata,
        'list_stretched_results': list_stretched_results
    }


# ==============================================================================
# Visualization Utilities
# ==============================================================================

def calculate_bond_distance(positions, atom1_idx, atom2_idx):
    """
    Calculate the distance between two atoms.
    """
    bond_vector = positions[atom2_idx] - positions[atom1_idx]
    bond_length = np.linalg.norm(bond_vector)
    return bond_length, bond_vector


def plot_molecule_compact(ax, atoms, positions, bond_threshold=1.8, highlight_bonds=None,
                         title=None, view_angle=(30, 45), show_labels=True):
    """
    Create a compact 3D molecular visualization for embedding in subplots.
    """
    # Element colors and sizes
    element_symbols = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 15: 'P', 16: 'S', 9: 'F', 17: 'Cl'}
    element_colors = {1: 'white', 6: 'gray', 7: 'blue', 8: 'red', 15: 'orange',
                     16: 'yellow', 9: 'green', 17: 'green'}
    element_sizes = {1: 80, 6: 150, 7: 150, 8: 150, 15: 150, 16: 150, 9: 120, 17: 150}

    # Plot atoms
    for i, (atom, pos) in enumerate(zip(atoms, positions)):
        symbol = element_symbols.get(int(atom), str(atom))
        color = element_colors.get(int(atom), 'pink')
        size = element_sizes.get(int(atom), 120)
        ax.scatter(pos[0], pos[1], pos[2], c=color, s=size,
                  edgecolors='black', linewidth=1.5, alpha=0.9)

        # Add atom labels if requested
        if show_labels:
            label = f"{symbol}{i}"
            ax.text(pos[0], pos[1], pos[2] + 0.2, label, fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none'))

    # Draw bonds
    bonds = []
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            bond_length, _ = calculate_bond_distance(positions, i, j)
            if bond_length < bond_threshold:
                bonds.append((i, j, bond_length))

    for i, j, bond_length in bonds:
        # Check if this bond should be highlighted
        is_highlighted = False
        if highlight_bonds:
            is_highlighted = (i, j) in highlight_bonds or (j, i) in highlight_bonds

        # Draw bond line
        x_coords = [positions[i][0], positions[j][0]]
        y_coords = [positions[i][1], positions[j][1]]
        z_coords = [positions[i][2], positions[j][2]]

        if is_highlighted:
            ax.plot(x_coords, y_coords, z_coords, 'r-', linewidth=4, alpha=0.9)
        else:
            ax.plot(x_coords, y_coords, z_coords, 'k-', linewidth=1.5, alpha=0.6)

    # Set equal aspect ratio
    max_range = np.array([
        positions[:, 0].max() - positions[:, 0].min(),
        positions[:, 1].max() - positions[:, 1].min(),
        positions[:, 2].max() - positions[:, 2].min()
    ]).max() / 2.0

    mid_x = (positions[:, 0].max() + positions[:, 0].min()) * 0.5
    mid_y = (positions[:, 1].max() + positions[:, 1].min()) * 0.5
    mid_z = (positions[:, 2].max() + positions[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # Remove axes for cleaner look
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)

    # Set viewing angle
    ax.view_init(elev=view_angle[0], azim=view_angle[1])

    # Set title
    if title:
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10)


def plot_bond_stretch_analysis(
    atoms,
    list_stretch_ratio,
    list_stretched_positions,
    sphnet_results,
    dft_results=None,
    atom1_idx=0,
    atom2_idx=1,
    save_path='bond_stretch_analysis.png',
    molecule_name=None,
    atom_types=None,
    site_type=None,
    role=None
):
    """
    Create comprehensive bond stretch analysis plot with energy plots and 3D structures.
    
    Args:
        atoms (array): Atomic numbers (N,)
        list_stretch_ratio (array): Array of stretch ratios
        list_stretched_positions (list): List of position arrays for each stretch ratio
        sphnet_results (list): List of SPHNet prediction results
        dft_results (list): Optional list of DFT reference results
        atom1_idx (int): Index of first atom in the bond
        atom2_idx (int): Index of second atom in the bond
        save_path (str): Path to save the figure
        molecule_name (str): Name of the molecule
        atom_types (list): List of atom type symbols, e.g. ["O", "H"]
        site_type (str): Type of reactive site
        role (str): Reactive role
    """
    measures = ['homo_energy_ev', 'lumo_energy_ev', 'homo_lumo_gap_ev']
    measure_labels = {
        'homo_energy_ev': 'HOMO Energy (eV)',
        'lumo_energy_ev': 'LUMO Energy (eV)',
        'homo_lumo_gap_ev': 'HOMO-LUMO Gap (eV)',
    }

    # Select 3 key stretch ratios for visualization
    list_stretch_ratio = np.array(list_stretch_ratio)
    middle_idx = np.argmin(np.abs(list_stretch_ratio - 1.0))
    selected_indices = [0, middle_idx, len(list_stretch_ratio) - 1]

    # Prepare title suffix
    title_suffix = ""
    if molecule_name:
        title_suffix = f" ({molecule_name.capitalize()}"
        if atom_types:
            bond_label = f"{atom_types[0]}{atom1_idx}-{atom_types[1]}{atom2_idx}"
            title_suffix += f", {bond_label}"
        if site_type:
            title_suffix += f", {site_type.capitalize()}"
        if role:
            title_suffix += f", {role}"
        title_suffix += ")"

    # Create figure
    fig, axes = plt.subplots(3, 2, figsize=(14, 15))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    # Extract values from SPHNet results
    def get_sphnet_value(result, key):
        if key == 'homo_lumo_gap_ev':
            homo = result.get('homo_energy_ev', 0)
            lumo = result.get('lumo_energy_ev', 0)
            return result.get('homo_lumo_gap_ev', lumo - homo)
        return result.get(key, 0)

    # Extract values from DFT results
    def get_dft_value(result, key):
        if key == 'homo_lumo_gap_ev':
            homo = result.get('homo_energy_ev', 0)
            lumo = result.get('lumo_energy_ev', 0)
            return result.get('homo_lumo_gap_ev', lumo - homo)
        if key == 'total_energy_ev':
            return result.get('energy_ev', 0)
        return result.get(key, 0)

    # Left column: Line plots
    for idx, measure in enumerate(measures):
        ax = axes[idx, 0]

        # SPHNet values
        sphnet_values = [get_sphnet_value(r, measure) for r in sphnet_results]
        ax.plot(list_stretch_ratio, sphnet_values, 'o-', linewidth=2, markersize=6,
                color='gray', label='SPHNet')

        # # Highlight selected points (SPHNet)
        # for sel_idx in selected_indices:
        #     if sel_idx == 0:
        #         point_label = "Compressed"
        #     elif sel_idx == middle_idx:
        #         point_label = "Original"
        #     else:
        #         point_label = "Extended"
        #     ax.scatter(list_stretch_ratio[sel_idx], sphnet_values[sel_idx],
        #                color='blue', s=60, edgecolor='black', zorder=5)
        #     ax.text(list_stretch_ratio[sel_idx], sphnet_values[sel_idx],
        #             point_label, fontsize=9, fontweight='bold', color='blue',
        #             ha='center', va='bottom',
        #             bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
        #                       edgecolor='blue', alpha=0.6))

        # DFT reference values (if available)
        if dft_results is not None:
            dft_values = [get_dft_value(r, measure) for r in dft_results]
            ax.plot(list_stretch_ratio, dft_values, 's--', linewidth=2, markersize=6,
                    color='blue', label='DFT Reference', alpha=0.7)

            # Highlight selected points (DFT)
            for sel_idx in selected_indices:
                if sel_idx == 0:
                    point_label = "Compressed"
                elif sel_idx == middle_idx:
                    point_label = "Original"
                else:
                    point_label = "Extended"

                y_offset = (max(dft_values) - min(dft_values)) * 0.03 if len(dft_values) > 1 else 0
                y_pos = dft_values[sel_idx] + (y_offset if sel_idx != middle_idx else -y_offset)

                ax.scatter(list_stretch_ratio[sel_idx], dft_values[sel_idx],
                           color='blue', s=60, edgecolor='black', zorder=5)
                ax.text(list_stretch_ratio[sel_idx], y_pos, point_label,
                        fontsize=9, fontweight='bold', color='blue',
                        ha='center', va='bottom',
                        bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                                  edgecolor='blue', alpha=0.7))

        # Add reference lines
        if middle_idx < len(sphnet_values):
            ax.axhline(y=dft_values[middle_idx], color='blue', linestyle='--', linewidth=2, alpha=0.5)
        ax.axvline(x=1.0, color='k', linestyle=':', linewidth=1, alpha=0.5)

        ax.set_xlabel('Reactive Bond Stretch Ratio', fontsize=11)
        ax.set_ylabel(measure_labels[measure], fontsize=11)
        ax.set_title(f'{measure_labels[measure]} vs Bond Stretch{title_suffix}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # Right column: 3D molecular structures
    for mol_idx, sel_idx in enumerate(selected_indices):
        fig.delaxes(axes[mol_idx, 1])
        ax_3d = fig.add_subplot(3, 2, 2 * mol_idx + 2, projection='3d')

        stretched_pos = list_stretched_positions[sel_idx]
        if isinstance(stretched_pos, torch.Tensor):
            stretched_pos = stretched_pos.numpy()
        stretched_pos = np.array(stretched_pos)
        
        stretch_ratio = list_stretch_ratio[sel_idx]
        bond_dist, _ = calculate_bond_distance(stretched_pos, atom1_idx, atom2_idx)

        if sel_idx == 0:
            structure_label = "Reactive Bond Compressed"
        elif sel_idx == middle_idx:
            structure_label = "Original"
        else:
            structure_label = "Reactive Bond Extended"

        atoms_np = np.array(atoms) if isinstance(atoms, (list, torch.Tensor)) else atoms
        
        plot_molecule_compact(
            ax_3d,
            atoms_np,
            stretched_pos,
            highlight_bonds=[(atom1_idx, atom2_idx)],
            title=f'{structure_label}\nRatio={stretch_ratio:.2f}, Bond={bond_dist:.3f}Å',
            view_angle=(20, 45),
            show_labels=True
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {save_path}")
    plt.show()



def generate_plot_filename(molecule_name=None, site_type=None, atom_types=None, 
                           atom1_idx=0, atom2_idx=1, suffix=''):
    """Generate plot filename."""
    parts = ['sphnet_bond_stretch_analysis']
    if molecule_name:
        parts.append(molecule_name)
    if site_type:
        parts.append(site_type)
    if atom_types:
        parts.append(f'{atom_types[0]}{atom1_idx}-{atom_types[1]}{atom2_idx}')
    if suffix:
        parts.append(suffix)
    parts.append('.png')
    return '_'.join(parts[:-1]) + parts[-1]


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="SPHNet Results Visualization")
    
    parser.add_argument(
        "--results-path", type=str, required=True,
        help="Path to SPHNet inference results (.pt file)"
    )
    parser.add_argument(
        "--dft-results-path", type=str, default=None,
        help="Path to DFT reference results (metadata .pt file or directory)"
    )
    parser.add_argument(
        "--save-dir", type=str, default="./output",
        help="Directory to save visualization plots"
    )
    parser.add_argument(
        "--molecule", type=str, default=None,
        help="Molecule name (for filtering and labeling)"
    )
    parser.add_argument(
        "--site-type", type=str, default=None,
        help="Site type (e.g., 'primary', 'secondary')"
    )
    parser.add_argument(
        "--no-show", action="store_true",
        help="Don't display plots (only save)"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # ===========================================================================
    # Load SPHNet Results
    # ===========================================================================
    print("=" * 80)
    print("LOADING SPHNET RESULTS")
    print("=" * 80)
    
    sphnet_data = load_sphnet_results(args.results_path)
    
    if sphnet_data['type'] != 'bond_stretch':
        print("Error: This visualization script requires bond stretch experiment results.")
        print("Please run sphnet_inference.py with a metadata file first.")
        return
    
    metadata = sphnet_data['metadata']
    sphnet_predictions = sphnet_data['predictions']
    
    # Convert metadata tensors to numpy
    def to_numpy(obj):
        if isinstance(obj, torch.Tensor):
            return obj.numpy()
        return obj
    
    atoms = to_numpy(metadata.get('atoms', []))
    positions = to_numpy(metadata.get('positions', []))
    list_stretch_ratio = to_numpy(metadata.get('list_stretch_ratio', []))
    list_stretched_positions = metadata.get('list_stretched_positions', [])
    if list_stretched_positions:
        list_stretched_positions = [to_numpy(p) for p in list_stretched_positions]
    
    molecule_name = metadata.get('molecule_name', args.molecule)
    site_type = metadata.get('site_type', args.site_type)
    atom_types = metadata.get('atom_types', None)
    atom1_idx = metadata.get('atom1_idx', 0)
    atom2_idx = metadata.get('atom2_idx', 1)
    
    # ===========================================================================
    # Load DFT Results (if provided)
    # ===========================================================================
    dft_results = None
    if args.dft_results_path:
        print("\n" + "=" * 80)
        print("LOADING DFT REFERENCE RESULTS")
        print("=" * 80)
        
        dft_data = load_dft_results(args.dft_results_path)
        dft_results = dft_data['list_stretched_results']
        
        # Use DFT metadata for positions if SPHNet doesn't have them
        if len(list_stretched_positions) == 0:
            list_stretched_positions = dft_data['metadata'].get('list_stretched_positions', [])
            if list_stretched_positions:
                list_stretched_positions = [to_numpy(p) for p in list_stretched_positions]
    
    # ===========================================================================
    # Generate Visualizations
    # ===========================================================================
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)
    
    # Main bond stretch analysis plot
    plot_filename = generate_plot_filename(
        molecule_name=molecule_name,
        site_type=site_type,
        atom_types=atom_types,
        atom1_idx=atom1_idx,
        atom2_idx=atom2_idx
    )
    plot_save_path = os.path.join(args.save_dir, plot_filename)
    
    # Use stretched positions from predictions if not available from metadata
    if len(list_stretched_positions) == 0:
        list_stretched_positions = [pred.get('positions', positions) for pred in sphnet_predictions]
    
    plot_bond_stretch_analysis(
        atoms=atoms,
        list_stretch_ratio=list_stretch_ratio,
        list_stretched_positions=list_stretched_positions,
        sphnet_results=sphnet_predictions,
        dft_results=dft_results,
        atom1_idx=atom1_idx,
        atom2_idx=atom2_idx,
        save_path=plot_save_path,
        molecule_name=molecule_name,
        atom_types=atom_types,
        site_type=site_type
    )
    
    # Comparison metrics plot (if DFT results available)
    if dft_results is not None:
        comparison_filename = generate_plot_filename(
            molecule_name=molecule_name,
            site_type=site_type,
            atom_types=atom_types,
            atom1_idx=atom1_idx,
            atom2_idx=atom2_idx,
            suffix='comparison'
        )
        comparison_save_path = os.path.join(args.save_dir, comparison_filename)
    
    # ===========================================================================
    # Print Summary Statistics
    # ===========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    print(f"\nMolecule: {molecule_name}")
    print(f"Site type: {site_type}")
    print(f"Bond: {atom_types[0] if atom_types else 'atom'}{atom1_idx}-{atom_types[1] if atom_types else 'atom'}{atom2_idx}")
    print(f"Stretch ratios: {list_stretch_ratio.min():.2f} to {list_stretch_ratio.max():.2f}")
    
    # SPHNet statistics
    print(f"\nSPHNet Predictions:")
    homo_vals = [p.get('homo_energy_ev', 0) for p in sphnet_predictions]
    lumo_vals = [p.get('lumo_energy_ev', 0) for p in sphnet_predictions]
    gap_vals = [p.get('homo_lumo_gap_ev', l-h) for p, h, l in zip(sphnet_predictions, homo_vals, lumo_vals)]
    
    print(f"  HOMO range: {min(homo_vals):.4f} to {max(homo_vals):.4f} eV")
    print(f"  LUMO range: {min(lumo_vals):.4f} to {max(lumo_vals):.4f} eV")
    print(f"  Gap range:  {min(gap_vals):.4f} to {max(gap_vals):.4f} eV")
    
    # Comparison with DFT
    if dft_results is not None:
        print(f"\nDFT Reference:")
        dft_homo = [r.get('homo_energy_ev', 0) for r in dft_results]
        dft_lumo = [r.get('lumo_energy_ev', 0) for r in dft_results]
        
        print(f"  HOMO range: {min(dft_homo):.4f} to {max(dft_homo):.4f} eV")
        print(f"  LUMO range: {min(dft_lumo):.4f} to {max(dft_lumo):.4f} eV")
        
        print(f"\nMAE (SPHNet vs DFT):")
        print(f"  HOMO: {np.mean(np.abs(np.array(homo_vals) - np.array(dft_homo))):.4f} eV")
        print(f"  LUMO: {np.mean(np.abs(np.array(lumo_vals) - np.array(dft_lumo))):.4f} eV")
    
    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
