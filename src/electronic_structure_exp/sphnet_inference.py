#!/usr/bin/env python3
"""
SPHNet Inference Script

Load a trained SPHNet model and perform inference on .pt data files.
Supports both individual .pt files and metadata+result files from bond stretch experiments.

Usage:
    python sphnet_inference.py \
        --checkpoint /path/to/checkpoint.ckpt \
        --data-path /path/to/data.pt
"""

import os
import sys
import argparse
import logging
import numpy as np
import torch
import warnings

from torch_geometric.data import Data
from torch_geometric.transforms.radius_graph import RadiusGraph

warnings.filterwarnings('ignore', category=FutureWarning)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Add project root to path
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "../..")
sys.path.insert(0, PROJECT_ROOT)

from pyscf import gto, dft
from pyscf.grad import rks as grad_rks
from omegaconf import OmegaConf

# SPHNet imports
from src.training.module import LNNP
from src.dataset.build_label import build_label
from src.dataset.utils import collate_fn_unified
from src.dataset.buildblock import get_conv_variable_lin, block2matrix, matrixtoblock_lin
from src.training.logger import get_latest_ckpt
# Evaluation utilities
from escflow_eval_utils import (
    init_pyscf_mf, matrix_transform_single,
    BOHR2ANG, HA2eV, HA2meV, HA_BOHR_2_meV_ANG, ANG2BOHR
)

# Add QHFlow-mlff path for metric import
QHFLOW_PATH = "/home/sungjun/repos/QHFlow-mlff/src"
if QHFLOW_PATH not in sys.path:
    sys.path.insert(0, QHFLOW_PATH)
from common.metric import cal_orbital_and_energies

# Additional unit conversion (eV/Å instead of meV/Å)
HA_BOHR_2_eV_ANG = HA2eV / BOHR2ANG  # Hartree/Bohr to eV/Angstrom

# ==============================================================================
# Constants
# ==============================================================================

ATOMIC_SYMBOLS = {
    1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F',
    15: 'P', 16: 'S', 17: 'Cl', 35: 'Br', 53: 'I'
}


# ==============================================================================
# Helper Functions (from sphnet_md17_eval_multiproc.py)
# ==============================================================================

def calc_density(mo_coeff, n_occ):
    """Calculate density matrix from molecular orbital coefficients."""
    sliced_mo_coeff = mo_coeff[:, :n_occ]
    density = sliced_mo_coeff @ sliced_mo_coeff.T * 2
    return density

def init_pyscf_mol(atoms, pos, unit="ang", basis="def2svp"):
    """
    Initialize PySCF Molecule object.
    
    Args:
        atoms (list): List of atomic numbers
        pos (array): Atomic positions in angstrom
        unit (str): Unit of position (default: "ang")
    """
    if unit.lower() == "ang" or unit.lower() == "angstrom" or unit.lower() == "a":
        pos_factor = 1.0
    elif unit.lower() == "bohr":
        pos_factor = BOHR2ANG
    else:
        raise ValueError(f"Invalid unit: {unit}")
    return init_pyscf_mol_(atoms, pos, pos_factor=pos_factor, basis=basis)

def init_pyscf_mol_(atoms, pos, pos_factor=1.0, basis="def2svp"):
    """
    Initialize PySCF Molecule object.
    
    Args:
        atoms (list): List of atomic numbers
        pos (array): Atomic positions in angstrom
        unit (str): Unit of position (default: "ang")
    """
    pos = pos * pos_factor
    mol = gto.Mole()
    mol_conf = [[atoms[atom_idx], pos[atom_idx]] for atom_idx in range(len(atoms))]
    mol.build(verbose=0, atom=mol_conf, basis=basis, unit="ang")
    return mol


def calc_mo_energy_and_coeff(ham_transformed, calc_overlap, tol=1e-8, pad_eigval=1):
    """Calculate molecular orbital energies and coefficients.
    
    Note: pad_eigval=1 matches the behavior in sphnet_md17_eval_multiproc.py
    """
    dtype = torch.float64
    overlap = calc_overlap.to(dtype)
    ham_transformed = ham_transformed.to(dtype)

    if overlap.dim() == 2:
        overlap = overlap.unsqueeze(0)
    if ham_transformed.dim() == 2:
        ham_transformed = ham_transformed.unsqueeze(0)

    orbital_energies, orbital_coefficients = cal_orbital_and_energies(
        overlap, ham_transformed, tol=tol, pad_eigval=pad_eigval
    )

    mo_energy = orbital_energies.squeeze().numpy()
    mo_coeff = orbital_coefficients.squeeze().numpy()

    return mo_energy, mo_coeff


def compute_energy_forces(density, mo_energy, mo_coeff, calc_mf, grad_frame):
    """Compute energy and forces from density matrix and orbital properties (CPU only)."""
    # Energy calculation (CPU)
    energy = calc_mf.energy_tot(density)

    mo_occ = calc_mf.get_occ(mo_energy, mo_coeff)

    # Force calculation (CPU)
    forces = -grad_frame.kernel(
        mo_energy=mo_energy,
        mo_coeff=mo_coeff,
        mo_occ=mo_occ
    )

    return energy, forces


# ==============================================================================
# SPHNet Inference Class
# ==============================================================================

class SPHNetInference:
    """
    SPHNet model wrapper for inference.
    """

    def __init__(self, checkpoint_path, config_path=None, device='cuda'):
        """
        Initialize SPHNet inference engine.

        Args:
            checkpoint_path (str): Path to trained model checkpoint (.ckpt)
            config_path (str): Path to config file (.yaml)
            device (str): Device for inference ('cuda' or 'cpu')
        """
        self.device = device
        self.checkpoint_path = checkpoint_path

        # Load config
        if config_path is None:
            config_dir = os.path.dirname(checkpoint_path)
            config_path = os.path.join(config_dir, "config.yaml")

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        logger.info(f"Loading config from: {config_path}")
        self.config = OmegaConf.load(config_path)

        latest_file = get_latest_ckpt(config_dir)
        # Load model
        logger.info(f"Loading model from: {latest_file}")
        self.model = LNNP.load_from_checkpoint(
            latest_file,
            hparams=self.config,
            map_location=device
        )
        self.model.eval()
        self.model.to(device)

        # Setup
        self.collate_fn = collate_fn_unified(
            long_cutoff_upper=9,
            unit=self.config.get('unit', 1)
        )

        self.basis = self.config.get('basis', 'def2-svp')
        self.xc = self.config.get('xc', 'pbe')
        self.remove_init = True

        # Get conv and mask for block operations
        self.conv, _, self.mask_lin, _ = get_conv_variable_lin(self.basis)
        self.max_block_size = self.conv.max_block_size

        logger.info("Model loaded successfully!")
        logger.info(f"  Basis: {self.basis}, XC: {self.xc}")
        logger.info(f"  Remove init: {self.remove_init}")
        logger.info(f"  Max block size: {self.max_block_size}")

    def load_pt_data(self, pt_path):
        """
        Load data from .pt file.
        
        Supports the following key mappings:
        - positions / pos
        - atomic_numbers / atoms
        - overlap / s1e
        - hamiltonian_hartree / hamiltonian / fock
        - initial_hamiltonian / init_fock / init_ham
        
        Args:
            pt_path (str): Path to .pt file
            
        Returns:
            dict: Processed data dictionary
        """
        raw_data = torch.load(pt_path, weights_only=False)
        
        # Map keys to standard format
        data = {}
        
        # Positions
        if 'positions' in raw_data:
            data['pos'] = raw_data['positions']
        elif 'pos' in raw_data:
            data['pos'] = raw_data['pos']
        else:
            raise KeyError("No position data found. Expected 'positions' or 'pos'")
        
        # Atomic numbers
        if 'atomic_numbers' in raw_data:
            data['atomic_numbers'] = raw_data['atomic_numbers']
        elif 'atoms' in raw_data:
            data['atomic_numbers'] = raw_data['atoms']
        else:
            raise KeyError("No atomic numbers found. Expected 'atomic_numbers' or 'atoms'")
        
        # Overlap matrix
        if 'overlap' in raw_data:
            data['s1e'] = raw_data['overlap']
        elif 's1e' in raw_data:
            data['s1e'] = raw_data['s1e']
        else:
            data['s1e'] = None  # Will compute if needed
        
        # Hamiltonian (ground truth)
        if 'hamiltonian_hartree' in raw_data:
            data['fock'] = raw_data['hamiltonian_hartree']
        elif 'hamiltonian' in raw_data:
            data['fock'] = raw_data['hamiltonian']
        elif 'fock' in raw_data:
            data['fock'] = raw_data['fock']
        else:
            data['fock'] = None
        
        # Initial Hamiltonian
        if 'initial_hamiltonian' in raw_data:
            data['init_fock'] = raw_data['initial_hamiltonian']
        elif 'init_fock' in raw_data:
            data['init_fock'] = raw_data['init_fock']
        elif 'init_ham' in raw_data:
            data['init_fock'] = raw_data['init_ham']
        else:
            data['init_fock'] = None  # Will compute if needed
        
        # Convert to proper types
        if isinstance(data['pos'], torch.Tensor):
            data['pos'] = data['pos'].numpy()
        if isinstance(data['atomic_numbers'], torch.Tensor):
            data['atomic_numbers'] = data['atomic_numbers'].numpy()
        
        data['pos'] = np.array(data['pos'], dtype=np.float64)
        data['atomic_numbers'] = np.array(data['atomic_numbers'], dtype=np.int64)
        
        # Convert matrices
        for key in ['s1e', 'fock', 'init_fock']:
            if data[key] is not None:
                if isinstance(data[key], torch.Tensor):
                    data[key] = data[key].numpy()
                data[key] = np.array(data[key], dtype=np.float64)
        
        # Store raw data for reference
        data['raw_data'] = raw_data
        
        return data

    def prepare_data_for_model(self, data):
        """
        Prepare data dictionary for model input.
        
        Args:
            data (dict): Data from load_pt_data()
            
        Returns:
            dict: Data ready for collate_fn
        """
        atoms = data['atomic_numbers']
        positions = data['pos']
        
        # Compute overlap and init_fock if not provided
        if data['s1e'] is None or data['init_fock'] is None:
            logger.info("  Computing overlap and init_fock via PySCF...")
            mf = init_pyscf_mf(atoms, positions, unit="ang", xc=self.xc, basis=self.basis)
            if data['s1e'] is None:
                data['s1e'] = mf.get_ovlp()
            if data['init_fock'] is None:
                dm0 = mf.get_init_guess(key='minao')
                data['init_fock'] = mf.get_fock(dm=dm0)
        
        # Build block representation
        n_orb = data['s1e'].shape[0]
        
        diag, non_diag, diag_mask, non_diag_mask = matrixtoblock_lin(
            data['fock'], atoms, self.mask_lin, self.max_block_size
        )
        
        # Create model input dictionary
        model_data = {
            'pos': positions.astype(np.float32)*ANG2BOHR,
            'atomic_numbers': data['atomic_numbers'],
            's1e': data['s1e'],
            'init_fock': data['init_fock'],
            'molecule_size': len(atoms),
            'diag_hamiltonian': diag.astype(np.float64),
            'non_diag_hamiltonian': non_diag.astype(np.float64),
            'diag_mask': diag_mask.astype(np.float64),
            'non_diag_mask': non_diag_mask.astype(np.float64),
        }
        
        data_object = Data()
        N_atom = atoms.shape[0]
        data_object.num_nodes = N_atom
        data_object.pos = torch.tensor(positions)
        neighbor_finder = RadiusGraph(r = 3)
        data_object = neighbor_finder(data_object)
        min_nodes_foreachGroup = 4

        build_label(data_object, num_labels = int(N_atom/min_nodes_foreachGroup),method = 'kmeans')

        # # Add edge_index (full graph)
        # n_atoms = len(atoms)
        # src, dst = [], []
        # for i in range(n_atoms):
        #     for j in range(n_atoms):
        #         if i != j:
        #             src.append(i)
        #             dst.append(j)
        model_data['edge_index'] = data_object.edge_index.numpy()
        model_data['labels'] = data_object.labels.numpy()
        
        return model_data

    @torch.no_grad()
    def predict(self, data):
        """
        Run inference on prepared data.
        
        Args:
            data (dict): Data from load_pt_data()
            
        Returns:
            dict: Prediction results including Hamiltonian
        """
        # Prepare data for model
        model_data = self.prepare_data_for_model(data)
        
        # Collate and move to device
        batch = self.collate_fn([model_data])
        batch = batch.to(self.device)
        
        # Run model
        output = self.model(batch)
        
        # Build full Hamiltonian from blocks
        pred_hamiltonians = self.model.model.hami_model.build_final_matrix_general(
            output,
            full_diag = output['pred_hamiltonian_diagonal_blocks'],
            full_non_diag = output['pred_hamiltonian_non_diagonal_blocks']
        )
        
        result = {
            'pred_hamiltonian': pred_hamiltonians.cpu().numpy(),
            'pred_ham_diag': output['pred_hamiltonian_diagonal_blocks'].cpu().numpy(),
            'pred_ham_non_diag': output['pred_hamiltonian_non_diagonal_blocks'].cpu().numpy(),
            'overlap': model_data['s1e'],
            'init_fock': model_data['init_fock'],
            'atoms': data['atomic_numbers'],
            'positions': data['pos'],
        }
        
        return result

    def compute_properties(self, pred_result):
        """
        Compute electronic properties from predicted Hamiltonian.
        
        Uses the same approach as sphnet_md17_eval_multiproc.py:
        1. Transform Hamiltonian to PySCF convention
        2. Calculate MO energies and coefficients using cal_orbital_and_energies
        3. Calculate density matrix from MO coefficients
        4. Compute energy and forces using PySCF
        
        Args:
            pred_result (dict): Output from predict()
            
        Returns:
            dict: Electronic properties (HOMO, LUMO, energy, forces, etc.)
        """
        atoms = pred_result['atoms']
        positions = pred_result['positions']
        pred_ham = pred_result['pred_hamiltonian']
        
        # Initialize PySCF - exactly matching sphnet_md17_eval_multiproc.py init_pyscf_optimized
        mol = init_pyscf_mol(atoms, positions, unit="ang", basis=self.basis)
        
        # CPU-only calculation for accuracy (same as init_pyscf_optimized)
        calc_mf = dft.RKS(mol, xc=self.xc)
        calc_mf.xc = self.xc
        calc_mf.basis = self.basis
        
        # CPU gradient calculation (same as init_pyscf_optimized)
        grad_frame = grad_rks.Gradients(calc_mf)
        
        calc_overlap = torch.tensor(mol.intor("int1e_ovlp"), dtype=torch.float64)
        
        # Calculate number of occupied orbitals
        
        n_occ = int(atoms.sum() / 2)
        
        # Transform Hamiltonian to PySCF convention
        pred_ham_tensor = torch.tensor(pred_ham, dtype=torch.float64)
        atoms_tensor = torch.tensor(atoms)
        pred_ham_transformed = matrix_transform_single(
            pred_ham_tensor.unsqueeze(0), atoms_tensor, convention="back2pyscf"
        ).squeeze()
        
        pred_ham_transformed = pred_ham_transformed + pred_result['init_fock']
        
        # Calculate molecular orbital energies and coefficients
        mo_energy, mo_coeff = calc_mo_energy_and_coeff(
            pred_ham_transformed, calc_overlap, tol=1e-8, pad_eigval=1
        )
        
        # Calculate density matrix from MO coefficients
        density = calc_density(mo_coeff, n_occ)
        
        # Compute energy and forces
        total_energy_hartree, forces_hartree_bohr = compute_energy_forces(
            density, mo_energy, mo_coeff, calc_mf, grad_frame
        )
        
        # Sort orbital energies for HOMO-LUMO calculation
        e_idx = np.argsort(mo_energy)
        e_sort = mo_energy[e_idx]
        
        homo_energy_ev = e_sort[n_occ - 1] * HA2eV
        lumo_energy_ev = e_sort[n_occ] * HA2eV
        
        # Extract occupied orbital info
        mo_energy_occ_ev = e_sort[:n_occ] * HA2eV
        mo_coeff_occ = mo_coeff[:, :n_occ]
        
        result = {
            'homo_energy_ev': homo_energy_ev,
            'lumo_energy_ev': lumo_energy_ev,
            'homo_lumo_gap_ev': lumo_energy_ev - homo_energy_ev,
            'total_energy_ev': total_energy_hartree * HA2eV,
            'total_energy_hartree': total_energy_hartree,
            'forces_ev_ang': forces_hartree_bohr * HA_BOHR_2_eV_ANG,  # eV/Å (not meV/Å)
            'forces_hartree_bohr': forces_hartree_bohr,
            'orbital_energies_ev': mo_energy * HA2eV,
            # 'orbital_energies_hartree': mo_energy,
            'orbital_energies_occ_ev': mo_energy_occ_ev,
            'orbital_coefficients': mo_coeff,
            'orbital_coefficients_occ': mo_coeff_occ,
            'n_occ': n_occ,
            'density_matrix': density,
            'pred_hamiltonian_hartree': pred_ham_transformed,
        }
        
        return result


# ==============================================================================
# Data Loading Utilities
# ==============================================================================

def load_metadata_file(metadata_path):
    """
    Load metadata file from bond stretch experiment.
    
    Args:
        metadata_path (str): Path to *_metadata.pt file
        
    Returns:
        dict: Metadata including atoms, positions, stretch ratios, etc.
    """
    metadata = torch.load(metadata_path, weights_only=False)
    
    logger.info(f"Loaded metadata from: {metadata_path}")
    logger.info(f"  Molecule: {metadata.get('molecule_name', 'Unknown')}")
    logger.info(f"  Site type: {metadata.get('site_type', 'Unknown')}")
    logger.info(f"  Bond: {metadata.get('atom_types', ['?', '?'])} [{metadata.get('atom1_idx')}-{metadata.get('atom2_idx')}]")
    logger.info(f"  Stretch ratios: {len(metadata.get('list_stretch_ratio', []))} points")
    logger.info(f"  Result files: {len(metadata.get('result_files', []))}")
    
    return metadata


def load_result_file(result_path):
    """
    Load individual result file from bond stretch experiment.
    
    Args:
        result_path (str): Path to result .pt file
        
    Returns:
        dict: Result data including positions, hamiltonian, etc.
    """
    data = torch.load(result_path, weights_only=False)
    
    stretch_ratio = data.get('stretch_ratio', 'Unknown')
    logger.info(f"  Loaded result: ratio={stretch_ratio}")
    
    return data


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="SPHNet Inference Script")
    
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to trained SPHNet checkpoint (.ckpt)"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to config file (.yaml)"
    )
    parser.add_argument(
        "--data-path", type=str, required=True,
        help="Path to .pt data file or metadata file"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        choices=["cuda", "cpu"],
        help="Device for inference"
    )
    parser.add_argument(
        "--save-dir", type=str, default="./output",
        help="Directory to save results"
    )
    parser.add_argument(
        "--compute-properties", action="store_true",
        help="Compute electronic properties (HOMO, LUMO, energy, forces)"
    )
    parser.add_argument(
        "--filter-ratio", type=float, default=None,
        help="Filter to process only results with this stretch ratio (for debugging). Example: --filter-ratio 1.0"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Check device
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        args.device = "cpu"
    
    # ===========================================================================
    # Load Model
    # ===========================================================================
    logger.info("=" * 80)
    logger.info("LOADING SPHNET MODEL")
    logger.info("=" * 80)
    
    sphnet = SPHNetInference(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        device=args.device
    )
    
    # ===========================================================================
    # Load and Process Data
    # ===========================================================================
    logger.info("=" * 80)
    logger.info("LOADING DATA")
    logger.info("=" * 80)
    
    data_path = args.data_path
    
    # Check if it's a metadata file
    if '_metadata.pt' in data_path:
        # Load metadata and process all result files
        metadata = load_metadata_file(data_path)
        data_dir = os.path.dirname(data_path)
        
        # Extract metadata info for filename generation
        molecule_name = metadata.get('molecule_name', 'unknown')
        site_type = metadata.get('site_type', 'unknown')
        atom1_idx = metadata.get('atom1_idx', 0)
        atom2_idx = metadata.get('atom2_idx', 0)
        atom_types = metadata.get('atom_types', ['X', 'X'])
        bond_str = f"{atom_types[0]}{atom1_idx}-{atom_types[1]}{atom2_idx}"
        
        # Create subfolder based on molecule_name and site_type: e.g., aspirin_primary
        subfolder_name = f"{molecule_name}_{site_type}"
        save_subdir = os.path.join(args.save_dir, subfolder_name)
        os.makedirs(save_subdir, exist_ok=True)
        logger.info(f"Output directory: {save_subdir}")
        
        # Log filter status if filtering is enabled
        if args.filter_ratio is not None:
            logger.info("=" * 80)
            logger.info(f"DEBUG MODE: Filtering for ratio = {args.filter_ratio:.2f}")
            logger.info("=" * 80)
        
        results = []
        for result_file in metadata.get('result_files', []):
            result_path = os.path.join(data_dir, result_file)
            
            if not os.path.exists(result_path):
                logger.warning(f"  Result file not found: {result_path}")
                continue
            
            # Load result data
            result_data = load_result_file(result_path)
            
            # Filter by ratio if --filter-ratio is specified (for debugging)
            stretch_ratio = result_data.get('stretch_ratio')
            if args.filter_ratio is not None:
                if stretch_ratio is None:
                    logger.warning(f"  Skipping {result_file}: no stretch_ratio found")
                    continue
                # Allow small floating point tolerance
                if abs(stretch_ratio - args.filter_ratio) > 1e-6:
                    logger.info(f"  Skipping ratio {stretch_ratio:.2f} (filtering for {args.filter_ratio:.2f})")
                    continue
                logger.info(f"  Processing ratio {stretch_ratio:.2f} (filtered)")
            
            # Convert to standard format for inference
            data = sphnet.load_pt_data(result_path)
            
            # Run inference
            logger.info(f"  Running inference...")
            pred_result = sphnet.predict(data)
            
            # Compute properties if requested
            if args.compute_properties:
                logger.info(f"  Computing properties...")
                props = sphnet.compute_properties(pred_result)
                pred_result.update(props)
            
            # Store reference values from DFT
            stretch_ratio = result_data.get('stretch_ratio')
            
            # Build output dictionary matching DFT result format
            output_data = {
                # Basic structure info
                'positions': pred_result['positions'],
                'atoms': pred_result['atoms'],  # For compatibility with plot scripts
                'atomic_numbers': pred_result['atoms'],
                'basis': sphnet.basis,
                'xc': sphnet.xc,
                'unit (distance)': 'angstrom',
                'unit (energy)': 'eV',
                # Electronic properties (SPHNet predictions)
                'homo_energy_ev': pred_result.get('homo_energy_ev'),
                'lumo_energy_ev': pred_result.get('lumo_energy_ev'),
                'energy_ev': pred_result.get('total_energy_ev'),
                'energy_ref_ev': result_data.get('energy_ref_ev', None),
                'energy_ha': pred_result.get('total_energy_hartree'),
                'orbital_energies_ev': pred_result.get('orbital_energies_ev'),
                'forces_ev_ang': pred_result.get('forces_ev_ang'),
                'forces_ha_bohr': pred_result.get('forces_hartree_bohr'),
                'forces_ref_ev': result_data.get('forces_ref_ev', None),
                'forces_l1_sum': np.abs(pred_result.get('forces_ev_ang', np.zeros((1,3)))).sum() if pred_result.get('forces_ev_ang') is not None else None,
                'orbital_coefficients': pred_result.get('orbital_coefficients'),
                'n_occ': pred_result.get('n_occ'),
                # Matrices
                'overlap': pred_result.get('overlap'),
                'hamiltonian_hartree': pred_result.get('pred_hamiltonian_hartree'),
                'initial_hamiltonian': pred_result.get('init_fock'),
                'density_matrix': pred_result.get('density_matrix'),
                # Metadata
                'stretch_ratio': stretch_ratio,
                'molecule_name': molecule_name,
                'atom_types': atom_types,
                'site_type': site_type,
                'atom1_idx': atom1_idx,
                'atom2_idx': atom2_idx,
            }
            
            # Also keep reference values for comparison
            pred_result['stretch_ratio'] = stretch_ratio
            pred_result['ref_energy_ev'] = result_data.get('energy_ref_ev', None)
            pred_result['ref_homo_ev'] = result_data.get('homo_energy_ev', None)
            pred_result['ref_lumo_ev'] = result_data.get('lumo_energy_ev', None)
            ref_forces = result_data.get('forces_ref_ev', None)
            if ref_forces is not None:
                if isinstance(ref_forces, torch.Tensor):
                    ref_forces = ref_forces.numpy()
                pred_result['ref_forces_ev_ang'] = np.array(ref_forces)
            else:
                pred_result['ref_forces_ev_ang'] = None
            
            # Reference orbital energies (occupied)
            ref_occ_energies = result_data.get('orbital_energies_occ_ev', None)
            if ref_occ_energies is not None:
                if isinstance(ref_occ_energies, torch.Tensor):
                    ref_occ_energies = ref_occ_energies.numpy()
                pred_result['ref_orbital_energies_occ_ev'] = np.array(ref_occ_energies)
            else:
                pred_result['ref_orbital_energies_occ_ev'] = None
            
            # Save individual result file: {molecule}_{site_type}_{bond}_ratio-{ratio}_sphnet.pt
            individual_filename = f"{molecule_name}_{site_type}_{bond_str}_ratio-{stretch_ratio:.2f}_sphnet.pt"
            individual_save_path = os.path.join(save_subdir, individual_filename)
            torch.save(output_data, individual_save_path)
            logger.info(f"  Saved: {subfolder_name}/{individual_filename}")
            
            results.append(pred_result)
        
        # Save metadata file with all results: {molecule}_{site_type}_{bond}_sphnet_metadata.pt
        metadata_filename = f"{molecule_name}_{site_type}_{bond_str}_sphnet_metadata.pt"
        metadata_save_path = os.path.join(save_subdir, metadata_filename)
        
        # Build metadata file similar to DFT format
        result_files = [f"{molecule_name}_{site_type}_{bond_str}_ratio-{r['stretch_ratio']:.2f}_sphnet.pt" for r in results]
        list_stretch_ratio = [r['stretch_ratio'] for r in results]
        list_stretched_positions = [r['positions'] for r in results]
        
        metadata_output = {
            # Original metadata info
            'molecule_name': molecule_name,
            'site_type': site_type,
            'atom_types': atom_types,
            'atom1_idx': atom1_idx,
            'atom2_idx': atom2_idx,
            # Both keys for compatibility (exp_bond-stretch_plot_three_methods.py uses 'atoms')
            'atoms': results[0]['atoms'] if results else None,
            'atomic_numbers': results[0]['atoms'] if results else None,
            'positions': metadata.get('positions'),  # Original positions
            'basis': sphnet.basis,
            'xc': sphnet.xc,
            # Stretch info
            'list_stretch_ratio': list_stretch_ratio,
            'list_stretched_positions': list_stretched_positions,
            'result_files': result_files,
        }
        
        torch.save(metadata_output, metadata_save_path)
        logger.info(f"Metadata saved to: {metadata_save_path}")
        
        # Print summary statistics
        if args.compute_properties and len(results) > 0:
            logger.info("=" * 80)
            logger.info("SPHNET PREDICTION SUMMARY")
            logger.info("=" * 80)
            
            energies = [r.get('total_energy_ev', 0) for r in results]
            forces_l1 = [np.abs(r.get('forces_ev_ang', np.zeros((1,3)))).sum() for r in results]
            homos = [r.get('homo_energy_ev', 0) for r in results]
            lumos = [r.get('lumo_energy_ev', 0) for r in results]
            
            logger.info(f"Number of predictions: {len(results)}")
            logger.info(f"Total Energy (eV): min={min(energies):.4f}, max={max(energies):.4f}, mean={np.mean(energies):.4f}")
            logger.info(f"Forces L1 (eV/Å):  min={min(forces_l1):.4f}, max={max(forces_l1):.4f}, mean={np.mean(forces_l1):.4f}")
            logger.info(f"HOMO (eV):         min={min(homos):.4f}, max={max(homos):.4f}, mean={np.mean(homos):.4f}")
            logger.info(f"LUMO (eV):         min={min(lumos):.4f}, max={max(lumos):.4f}, mean={np.mean(lumos):.4f}")
            
            # Check if reference data is available
            has_ref = results[0].get('ref_energy_ev') is not None
            
            if has_ref:
                # Print comparison with reference (DFT)
                logger.info("=" * 80)
                logger.info("COMPARISON WITH DFT REFERENCE")
                logger.info("=" * 80)
                
                # Collect errors
                energy_errors = []
                homo_errors = []
                lumo_errors = []
                force_errors = []
                orbital_energies_errors = []
                
                logger.info("-" * 120)
                logger.info(f"{'Ratio':>8} | {'Pred E':>12} | {'Ref E':>12} | {'ΔE':>10} | "
                           f"{'Pred F_L1':>10} | {'Ref F_L1':>10} | {'ΔF':>8} | "
                           f"{'ΔHOMO':>8} | {'ΔLUMO':>8}")
                logger.info("-" * 120)
                
                for r in results:
                    ratio = r.get('stretch_ratio', 0)
                    
                    # Energy
                    pred_e = r.get('total_energy_ev', 0)
                    ref_e = r.get('ref_energy_ev', 0)
                    delta_e = pred_e - ref_e if ref_e else 0
                    energy_errors.append(abs(delta_e))
                    
                    # Forces
                    pred_f = np.abs(r.get('forces_ev_ang', np.zeros((1,3)))).sum()
                    ref_f_arr = r.get('ref_forces_ev_ang')
                    ref_f = np.abs(ref_f_arr).sum() if ref_f_arr is not None else 0
                    delta_f = pred_f - ref_f
                    force_errors.append(abs(delta_f))
                    
                    # HOMO/LUMO
                    pred_homo = r.get('homo_energy_ev', 0)
                    ref_homo = r.get('ref_homo_ev', 0)
                    delta_homo = pred_homo - ref_homo if ref_homo else 0
                    homo_errors.append(abs(delta_homo))
                    
                    pred_lumo = r.get('lumo_energy_ev', 0)
                    ref_lumo = r.get('ref_lumo_ev', 0)
                    delta_lumo = pred_lumo - ref_lumo if ref_lumo else 0
                    lumo_errors.append(abs(delta_lumo))
                    
                    # Occupied orbital energies MAE
                    pred_occ = r.get('orbital_energies_occ_ev')
                    ref_occ = r.get('ref_orbital_energies_occ_ev')
                    if pred_occ is not None and ref_occ is not None:
                        occ_mae = np.mean(np.abs(pred_occ - ref_occ))
                        orbital_energies_errors.append(occ_mae)
                    
                    logger.info(f"{ratio:>8.3f} | {pred_e:>12.4f} | {ref_e:>12.4f} | {delta_e:>+10.4f} | "
                               f"{pred_f:>10.4f} | {ref_f:>10.4f} | {delta_f:>+8.4f} | "
                               f"{delta_homo:>+8.4f} | {delta_lumo:>+8.4f}")
                
                # Print MAE summary
                logger.info("-" * 120)
                logger.info("MEAN ABSOLUTE ERROR (MAE) SUMMARY:")
                logger.info(f"  Energy MAE:    {np.mean(energy_errors):.6f} eV")
                logger.info(f"  Forces L1 MAE: {np.mean(force_errors):.6f} eV/Å")
                logger.info(f"  HOMO MAE:      {np.mean(homo_errors):.6f} eV")
                logger.info(f"  LUMO MAE:      {np.mean(lumo_errors):.6f} eV")
                logger.info(f"  Gap MAE:       {np.mean([abs(h-l) for h, l in zip(homo_errors, lumo_errors)]):.6f} eV")
                if orbital_energies_errors:
                    logger.info(f"  Occ. Orbital Energies MAE: {np.mean(orbital_energies_errors):.6f} eV")
                else:
                    logger.info(f"  Occ. Orbital Energies MAE: N/A (no reference data)")
            else:
                # Print per-ratio details without reference
                logger.info("-" * 80)
                logger.info(f"{'Ratio':>8} | {'Energy (eV)':>14} | {'Forces L1':>12} | {'HOMO':>10} | {'LUMO':>10}")
                logger.info("-" * 80)
                for r in results:
                    ratio = r.get('stretch_ratio', 'N/A')
                    energy = r.get('total_energy_ev', 0)
                    force_l1 = np.abs(r.get('forces_ev_ang', np.zeros((1,3)))).sum()
                    homo = r.get('homo_energy_ev', 0)
                    lumo = r.get('lumo_energy_ev', 0)
                    logger.info(f"{ratio:>8.3f} | {energy:>14.4f} | {force_l1:>12.4f} | {homo:>10.4f} | {lumo:>10.4f}")
        
    else:
        # Single file inference
        logger.info(f"Loading: {data_path}")
        data = sphnet.load_pt_data(data_path)
        
        logger.info(f"  Atoms: {data['atomic_numbers']}")
        logger.info(f"  Positions shape: {data['pos'].shape}")
        
        # Run inference
        logger.info("=" * 80)
        logger.info("RUNNING INFERENCE")
        logger.info("=" * 80)
        
        pred_result = sphnet.predict(data)
        
        logger.info(f"Predicted Hamiltonian shape: {pred_result['pred_hamiltonian'].shape}")
        
        
        # Compute properties if requested
        if args.compute_properties:
            logger.info("=" * 80)
            logger.info("COMPUTING ELECTRONIC PROPERTIES")
            logger.info("=" * 80)
            
            props = sphnet.compute_properties(pred_result)
            pred_result.update(props)
            
            forces_l1 = np.abs(props['forces_ev_ang']).sum()
            
            logger.info(f"HOMO: {props['homo_energy_ev']:.4f} eV")
            logger.info(f"LUMO: {props['lumo_energy_ev']:.4f} eV")
            logger.info(f"Gap:  {props['homo_lumo_gap_ev']:.4f} eV")
            logger.info(f"Total Energy: {props['total_energy_ev']:.4f} eV")
            logger.info(f"Forces L1 sum: {forces_l1:.4f} eV/Å")
        
        # Save results
        data_path = os.path.basename(data_path)
        data_stem, _ = os.path.splitext(data_path)
        if data_stem.endswith("_metadata"):
            data_stem = data_stem[:-len("_metadata")]
        save_path = os.path.join(args.save_dir, f"{data_stem}_sphnet_prediction.pt")
        torch.save(pred_result, save_path)
        logger.info(f"Results saved to: {save_path}")
    
    logger.info("=" * 80)
    logger.info("DONE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
