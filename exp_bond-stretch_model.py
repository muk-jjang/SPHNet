#!/usr/bin/env python3
"""
SPHNet Model-Based Prediction for Bond Stretch Experiment

This script loads DFT experiment results (saved as .pt files from exp_bond-stretch_dft.py)
and runs SPHNet model-based predictions to compare ML-predicted HOMO/LUMO energies against DFT reference values.

Example:
    python exp_bond-stretch_model.py \
        --molecule ethanol \
        --site-type primary \
        --load-results /path/to/dft/results \
        --ckpt-path /path/to/sphnet/model.ckpt \
        --save-dir ./output
"""

import os
import sys
import argparse
import numpy as np
import torch
import yaml
from tqdm import tqdm

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['NUMEXPR_NUM_THREADS'] = '4'

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__)))

# Import SPHNet utilities
from escflow_eval_utils import (
    matrix_transform_single,
    cal_orbital_and_energies,
    BOHR2ANG,
    HA2eV
)

# Import common utilities (shared with QHFlow)
sys.path.insert(0, "/home/chanhui-lee/QHFlow-mlff/src")
from common.units import convert_length

# Mapping of atomic numbers to element symbols
ATOMIC_SYMBOLS = {
    1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F',
    15: 'P', 16: 'S', 17: 'Cl', 35: 'Br', 53: 'I'
}


# ==============================================================================
# Configuration loading (shared with QHFlow-v2)
# ==============================================================================

def load_experiment_config(config_path):
    """Load experiment configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_default_config_path():
    """Get the default config file path from QHFlow experiment directory."""
    qhflow_exp_dir = "/home/chanhui-lee/QHFlow-mlff/src/electronic_structure_exp"
    return os.path.join(qhflow_exp_dir, "exp_config.yaml")


def get_model_checkpoint_config(exp_config, molecule_name):
    """Extract model checkpoint configuration for a given molecule."""
    checkpoints = exp_config.get('model_checkpoints', {})
    if molecule_name not in checkpoints:
        raise ValueError(f"No checkpoint config for molecule '{molecule_name}'. "
                         f"Available: {list(checkpoints.keys())}")
    return checkpoints[molecule_name]


# ==============================================================================
# Data loading (reused from QHFlow-v2 exp_bond-stretch_model.py)
# ==============================================================================

def _convert_to_numpy(data):
    """Recursively convert torch tensors in a dict/list to numpy arrays."""
    if isinstance(data, dict):
        return {k: _convert_to_numpy(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_convert_to_numpy(item) for item in data]
    elif isinstance(data, torch.Tensor):
        return data.cpu().numpy()
    else:
        return data


def load_single_dft_result(load_path):
    """Load a single DFT calculation result from a torch .pt file."""
    data = torch.load(load_path, weights_only=False)
    data = _convert_to_numpy(data)
    return data


def load_dft_results(load_dir, molecule_name=None, site_type=None):
    """Load DFT calculation results from a directory containing individual .pt files."""
    # Find the metadata file
    metadata_files = [f for f in os.listdir(load_dir) if 'dft_metadata' in f and f.endswith('.pt')]
    if len(metadata_files) == 0:
        raise FileNotFoundError(f"No DFT metadata file (*dft_metadata.pt) found in {load_dir}")

    # Filter by molecule_name if provided
    if molecule_name:
        metadata_files = [f for f in metadata_files if molecule_name in f]
        if len(metadata_files) == 0:
            raise FileNotFoundError(f"No metadata file matching molecule '{molecule_name}' found in {load_dir}")

    # Filter by site_type if provided
    if site_type:
        metadata_files = [f for f in metadata_files if site_type in f]
        if len(metadata_files) == 0:
            raise FileNotFoundError(f"No metadata file matching site_type '{site_type}' found in {load_dir}")

    if len(metadata_files) > 1:
        raise ValueError(f"Multiple metadata files found in {load_dir} after filtering: {metadata_files}. "
                         f"Please specify --molecule and/or --site-type to select one.")

    metadata_path = os.path.join(load_dir, metadata_files[0])
    print(f"Loading metadata from: {metadata_path}")

    # Load metadata
    metadata = torch.load(metadata_path, weights_only=False)
    metadata = _convert_to_numpy(metadata)

    # Load individual result files
    result_files = metadata['result_files']
    list_stretched_results = []

    print(f"Loading {len(result_files)} DFT result files...")
    for filename in result_files:
        result_path = os.path.join(load_dir, filename)
        result = load_single_dft_result(result_path)
        list_stretched_results.append(result)

    # Reconstruct the full data dictionary
    data = {
        'atoms': metadata['atoms'],
        'positions': metadata['positions'],
        'list_stretch_ratio': metadata['list_stretch_ratio'],
        'list_stretched_positions': metadata['list_stretched_positions'],
        'list_stretched_results': list_stretched_results,
        'original_results': metadata['original_results'],
        'molecule_name': metadata['molecule_name'],
        'atom_types': metadata['atom_types'],
        'site_type': metadata['site_type'],
        'atom1_idx': metadata['atom1_idx'],
        'atom2_idx': metadata['atom2_idx'],
        'min_energy_idx': metadata.get('min_energy_idx', None),
    }

    print(f"Successfully loaded DFT results for {len(list_stretched_results)} stretch ratios")
    return data


# ==============================================================================
# SPHNet Model Predictor
# ==============================================================================

class SPHNetPredictor:
    """
    SPHNet model predictor for Hamiltonian prediction.
    Uses SPHNet's native implementation.
    """

    def __init__(self, device="cuda"):
        self.device = device
        self.model = None
        self.config = None

    def load_model(self, ckpt_path):
        """
        Load SPHNet model from checkpoint.

        Args:
            ckpt_path (str): Path to model checkpoint
        """
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

        print(f"Loading SPHNet model from: {ckpt_path}")

        # Import SPHNet modules
        from src.training.module import LNNP

        # Load checkpoint
        ckpt = torch.load(ckpt_path, map_location=self.device)

        # Extract configuration
        self.config = ckpt.get("hyper_parameters", {})

        # Load model using Lightning module
        self.model = LNNP.load_from_checkpoint(ckpt_path, map_location=self.device)
        self.model.eval()
        self.model.to(self.device)

        print(f"Model loaded successfully")
        if 'config' in self.config:
            dataset_name = getattr(self.config.get('config'), 'dataset_name', 'Unknown')
            print(f"Model trained on dataset: {dataset_name}")

    def forward_component(self, atoms, coords, ovlp, init_ham):
        """
        Forward pass through SPHNet to get predicted Hamiltonian.

        Args:
            atoms: Atomic numbers (array)
            coords: Atomic coordinates in Angstrom
            ovlp: Overlap matrix
            init_ham: Initial Hamiltonian

        Returns:
            np.ndarray: Predicted Hamiltonian matrix
        """
        from torch_geometric.data import Data, Batch

        # Convert to torch tensors on GPU
        pos = torch.tensor(coords, dtype=torch.float32).to(self.device)
        _atoms = torch.tensor(atoms, dtype=torch.long).squeeze().to(self.device)
        _init_ham = torch.tensor(init_ham, dtype=torch.float64).squeeze().to(self.device)
        _ovlp = torch.tensor(ovlp, dtype=torch.float64).squeeze().to(self.device)
        h_dim = init_ham.shape[0]
        num_atoms = len(_atoms)

        # Transform matrices to e3nn convention (SPHNet uses similar conventions)
        _init_ham_transformed = matrix_transform_single(
            _init_ham.unsqueeze(0), _atoms, convention="pyscf_def2svp_to_e3nn"
        ).squeeze(0)
        _ovlp_transformed = matrix_transform_single(
            _ovlp.unsqueeze(0), _atoms, convention="pyscf_def2svp_to_e3nn"
        ).squeeze(0)

        # Build edge index (full connectivity)
        edge_index = []
        for i in range(len(_atoms)):
            for j in range(len(_atoms)):
                if i != j:
                    edge_index.append([i, j])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        # Create data batch compatible with SPHNet
        one_batch = Data(
            pos=pos,
            atoms=_atoms.view(-1, 1),
            init_ham=_init_ham_transformed.unsqueeze(0),
            overlap=_ovlp_transformed.unsqueeze(0),
            edge_index=edge_index,
            h_dim=torch.tensor(h_dim, dtype=torch.long).view(-1, 1),
            num_atoms=torch.tensor(num_atoms, dtype=torch.long).view(-1, 1),
        )
        batch = Batch.from_data_list([one_batch])

        # Forward pass
        with torch.no_grad():
            batch_gpu = batch.to(self.device)
            # SPHNet forward pass
            outputs = self.model(batch_gpu)

            # Extract predicted Hamiltonian
            # The output structure depends on SPHNet's implementation
            if isinstance(outputs, dict):
                pred_ham = outputs.get('hamiltonian', outputs.get('pred_hamiltonian'))
            else:
                pred_ham = outputs

            pred_ham = pred_ham.cpu()

        # Clean up GPU memory
        del batch_gpu

        # Transform Hamiltonian back to PySCF convention
        pred_ham_transformed = matrix_transform_single(
            pred_ham,
            batch.atoms,
            convention="e3nn_to_pyscf_def2svp"
        )

        return pred_ham_transformed.squeeze(0).detach().cpu().numpy()

    def forward_batch(self, list_atoms, list_coords, list_ovlp, list_init_ham):
        """
        Batch forward pass through SPHNet to get predicted Hamiltonians.

        Args:
            list_atoms: List of atomic numbers arrays
            list_coords: List of atomic coordinates arrays
            list_ovlp: List of overlap matrices
            list_init_ham: List of initial Hamiltonians

        Returns:
            list[np.ndarray]: List of predicted Hamiltonian matrices
        """
        from torch_geometric.data import Data, Batch

        batch_size = len(list_coords)
        data_list = []

        for i in range(batch_size):
            atoms = list_atoms[i]
            coords = list_coords[i]
            ovlp = list_ovlp[i]
            init_ham = list_init_ham[i]

            # Convert to torch tensors
            pos = torch.tensor(coords, dtype=torch.float32)
            _atoms = torch.tensor(atoms, dtype=torch.long).squeeze()
            _init_ham = torch.tensor(init_ham, dtype=torch.float64).squeeze()
            _ovlp = torch.tensor(ovlp, dtype=torch.float64).squeeze()
            h_dim = init_ham.shape[0]
            num_atoms = len(_atoms)

            # Transform matrices to e3nn convention
            _init_ham_transformed = matrix_transform_single(
                _init_ham.unsqueeze(0), _atoms, convention="pyscf_def2svp_to_e3nn"
            ).squeeze(0)
            _ovlp_transformed = matrix_transform_single(
                _ovlp.unsqueeze(0), _atoms, convention="pyscf_def2svp_to_e3nn"
            ).squeeze(0)

            # Build edge index
            edge_index = []
            for j in range(len(_atoms)):
                for k in range(len(_atoms)):
                    if j != k:
                        edge_index.append([j, k])
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

            # Create data object
            one_data = Data(
                pos=pos,
                atoms=_atoms.view(-1, 1),
                init_ham=_init_ham_transformed.unsqueeze(0),
                overlap=_ovlp_transformed.unsqueeze(0),
                edge_index=edge_index,
                h_dim=torch.tensor(h_dim, dtype=torch.long).view(-1, 1),
                num_atoms=torch.tensor(num_atoms, dtype=torch.long).view(-1, 1),
            )
            data_list.append(one_data)

        # Create batched data
        batch = Batch.from_data_list(data_list)

        # Forward pass
        with torch.no_grad():
            batch_gpu = batch.to(self.device)
            outputs = self.model(batch_gpu)

            # Extract predicted Hamiltonian
            if isinstance(outputs, dict):
                all_hamiltonians = outputs.get('hamiltonian', outputs.get('pred_hamiltonian'))
            else:
                all_hamiltonians = outputs

            all_hamiltonians = all_hamiltonians.cpu()

        # Clean up GPU memory
        del batch_gpu

        # Extract and transform each predicted Hamiltonian
        pred_hams = []
        for i in range(batch_size):
            pred_ham = all_hamiltonians[i]

            # Transform Hamiltonian back to PySCF convention
            pred_ham_transformed = matrix_transform_single(
                pred_ham.unsqueeze(0),
                data_list[i].atoms,
                convention="e3nn_to_pyscf_def2svp"
            )

            pred_hams.append(pred_ham_transformed.squeeze(0).detach().cpu().numpy())

        return pred_hams

    def calc_mo_energy_and_coeff(self, ham, overlap, tol=1e-8):
        """
        Calculate MO energies and coefficients from Hamiltonian.

        Args:
            ham: Hamiltonian matrix (torch.Tensor)
            overlap: Overlap matrix (torch.Tensor)
            tol: Tolerance for eigenvalue computation

        Returns:
            tuple: (mo_energy, mo_coeff) as numpy arrays
        """
        orbital_energies, orbital_coefficients = cal_orbital_and_energies(
            overlap, ham, tol=tol
        )

        mo_energy = orbital_energies.squeeze().cpu().numpy()
        mo_coeff = orbital_coefficients.squeeze().cpu().numpy()

        return mo_energy, mo_coeff


# ==============================================================================
# Model prediction functions (shared interface with QHFlow-v2)
# ==============================================================================

def init_model_predictor(ckpt_path, device="cuda"):
    """Initialize and return a configured SPHNetPredictor."""
    predictor = SPHNetPredictor(device=device)
    predictor.load_model(ckpt_path)
    return predictor


def run_model_prediction(predictor, atoms, positions, overlap, init_hamiltonian,
                         n_occ, model_length_unit="ang"):
    """
    Run SPHNet model prediction to get Hamiltonian and compute MO energies.

    Args:
        predictor: Initialized SPHNetPredictor
        atoms (np.ndarray): Atomic numbers
        positions (np.ndarray): Atomic positions in Angstrom
        overlap (np.ndarray): Overlap matrix from DFT
        init_hamiltonian (np.ndarray): Initial Hamiltonian from DFT
        n_occ (int): Number of occupied orbitals
        model_length_unit (str): Length unit expected by model ('bohr' or 'ang')

    Returns:
        dict: Prediction results
    """
    # Convert positions to model's expected unit
    input_coords = convert_length(positions, from_unit="ang", to_unit=model_length_unit)

    # Run model forward pass
    pred_ham = predictor.forward_component(
        atoms, input_coords, overlap, init_hamiltonian
    )

    # Compute MO energies from predicted Hamiltonian
    pred_ham_tensor = torch.tensor(pred_ham, dtype=torch.float64)
    overlap_tensor = torch.tensor(overlap, dtype=torch.float64)

    pred_mo_energy, pred_mo_coeff = predictor.calc_mo_energy_and_coeff(
        pred_ham_tensor.unsqueeze(0), overlap_tensor.unsqueeze(0), tol=1e-8
    )

    # Extract HOMO/LUMO
    homo_idx = n_occ - 1
    lumo_idx = n_occ

    return {
        'pred_hamiltonian': pred_ham,
        'pred_mo_energy_ha': pred_mo_energy,
        'pred_mo_coeff': pred_mo_coeff,
        'pred_mo_energy_ev': pred_mo_energy * HA2eV,
        'pred_homo_energy_ev': pred_mo_energy[homo_idx] * HA2eV,
        'pred_lumo_energy_ev': pred_mo_energy[lumo_idx] * HA2eV,
        'pred_gap_ev': (pred_mo_energy[lumo_idx] - pred_mo_energy[homo_idx]) * HA2eV,
    }


def run_predictions_for_all_geometries(predictor, dft_data, model_length_unit="ang"):
    """Run SPHNet predictions for all stretched geometries using batch inference."""
    atoms = dft_data['atoms']
    dft_results = dft_data['list_stretched_results']
    num_geometries = len(dft_results)

    print(f"\nRunning batch SPHNet predictions for {num_geometries} geometries...")

    # Prepare batch inputs
    list_atoms = []
    list_coords = []
    list_ovlp = []
    list_init_ham = []
    list_n_occ = []

    for dft_result in dft_results:
        positions = dft_result['positions']
        input_coords = convert_length(positions, from_unit="ang", to_unit=model_length_unit)

        list_atoms.append(atoms)
        list_coords.append(input_coords)
        list_ovlp.append(dft_result['overlap'])
        list_init_ham.append(dft_result['initial_hamiltonian'])
        list_n_occ.append(dft_result['n_occ'])

    # Run batch forward pass
    print("Running single batch inference...")
    pred_hams = predictor.forward_batch(
        list_atoms=list_atoms,
        list_coords=list_coords,
        list_ovlp=list_ovlp,
        list_init_ham=list_init_ham
    )

    # Compute MO energies for each predicted Hamiltonian
    model_results = []
    print("Computing MO energies...")
    for i, (pred_ham, dft_result) in enumerate(tqdm(zip(pred_hams, dft_results), total=num_geometries, desc="MO energies")):
        overlap = dft_result['overlap']
        n_occ = list_n_occ[i]

        # Compute MO energies
        pred_ham_tensor = torch.tensor(pred_ham, dtype=torch.float64)
        overlap_tensor = torch.tensor(overlap, dtype=torch.float64)

        pred_mo_energy, pred_mo_coeff = predictor.calc_mo_energy_and_coeff(
            pred_ham_tensor.unsqueeze(0), overlap_tensor.unsqueeze(0), tol=1e-8
        )

        # Extract HOMO/LUMO
        homo_idx = n_occ - 1
        lumo_idx = n_occ

        pred_result = {
            'pred_hamiltonian': pred_ham,
            'pred_mo_energy_ha': pred_mo_energy,
            'pred_mo_coeff': pred_mo_coeff,
            'pred_mo_energy_ev': pred_mo_energy * HA2eV,
            'pred_homo_energy_ev': pred_mo_energy[homo_idx] * HA2eV,
            'pred_lumo_energy_ev': pred_mo_energy[lumo_idx] * HA2eV,
            'pred_gap_ev': (pred_mo_energy[lumo_idx] - pred_mo_energy[homo_idx]) * HA2eV,
            'stretch_ratio': dft_result.get('stretch_ratio', dft_data['list_stretch_ratio'][i]),
        }

        model_results.append(pred_result)

    return model_results


# ==============================================================================
# Comparison and saving functions (shared with QHFlow-v2)
# ==============================================================================

def compute_comparison_metrics(dft_results, model_results, stretch_ratios):
    """Compute comparison metrics between DFT and model predictions."""
    dft_homo = np.array([r['homo_energy_ev'] for r in dft_results])
    dft_lumo = np.array([r['lumo_energy_ev'] for r in dft_results])
    dft_gap = dft_lumo - dft_homo

    model_homo = np.array([r['pred_homo_energy_ev'] for r in model_results])
    model_lumo = np.array([r['pred_lumo_energy_ev'] for r in model_results])
    model_gap = model_lumo - model_homo

    # Compute errors
    homo_errors = np.abs(dft_homo - model_homo)
    lumo_errors = np.abs(dft_lumo - model_lumo)
    gap_errors = np.abs(dft_gap - model_gap)

    metrics = {
        'homo_mae_ev': np.mean(homo_errors),
        'lumo_mae_ev': np.mean(lumo_errors),
        'gap_mae_ev': np.mean(gap_errors),
        'homo_max_error_ev': np.max(homo_errors),
        'lumo_max_error_ev': np.max(lumo_errors),
        'gap_max_error_ev': np.max(gap_errors),
        'homo_mae_ev_ratio': np.mean(np.abs((dft_homo - model_homo)/dft_homo)),
        'lumo_mae_ev_ratio': np.mean(np.abs((dft_lumo - model_lumo)/dft_lumo)),
        'gap_mae_ev_ratio': np.mean(np.abs((dft_gap - model_gap)/dft_gap)),
        'dft_homo': dft_homo,
        'dft_lumo': dft_lumo,
        'dft_gap': dft_gap,
        'model_homo': model_homo,
        'model_lumo': model_lumo,
        'model_gap': model_gap,
        'stretch_ratios': stretch_ratios,
    }

    return metrics


def print_comparison_summary(metrics):
    """Print a summary of comparison metrics."""
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY: DFT vs SPHNet Predictions")
    print("=" * 60)
    print(f"HOMO MAE:      {metrics['homo_mae_ev']:.4f} eV")
    print(f"LUMO MAE:      {metrics['lumo_mae_ev']:.4f} eV")
    print(f"Gap MAE:       {metrics['gap_mae_ev']:.4f} eV")
    print("-" * 60)
    print(f"HOMO Max Err:  {metrics['homo_max_error_ev']:.4f} eV")
    print(f"LUMO Max Err:  {metrics['lumo_max_error_ev']:.4f} eV")
    print(f"Gap Max Err:   {metrics['gap_max_error_ev']:.4f} eV")
    print("=" * 60)


def _convert_to_torch(value):
    """Convert numpy arrays to torch tensors (float64)."""
    if isinstance(value, np.ndarray):
        return torch.tensor(value, dtype=torch.float64)
    elif isinstance(value, dict):
        return {k: _convert_to_torch(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_convert_to_torch(v) for v in value]
    else:
        return value


def save_single_model_result(save_path, model_result, dft_result, molecule_name=None,
                              atom_types=None, site_type=None,
                              atom1_idx=None, atom2_idx=None):
    """Save a single SPHNet prediction result to a torch .pt file."""
    # Build result dict matching DFT results format
    results = {
        # Geometry info
        "positions": dft_result['positions'],
        "atomic_numbers": dft_result['atomic_numbers'],

        # Model config
        "source": "sphnet_prediction",
        "unit (distance)": "ang",
        "unit (energy)": "ev",

        # Energy descriptors (model predicted)
        "homo_energy_ev": model_result['pred_homo_energy_ev'],
        "lumo_energy_ev": model_result['pred_lumo_energy_ev'],
        "orbital_energies_ev": model_result['pred_mo_energy_ev'],

        # Also include original key names
        "pred_homo_energy_ev": model_result['pred_homo_energy_ev'],
        "pred_lumo_energy_ev": model_result['pred_lumo_energy_ev'],
        "pred_gap_ev": model_result['pred_gap_ev'],
        "pred_mo_energy_ev": model_result['pred_mo_energy_ev'],
        "pred_mo_energy_ha": model_result['pred_mo_energy_ha'],

        # Orbital information
        "orbital_coefficients": model_result['pred_mo_coeff'],
        "n_occ": dft_result['n_occ'],

        # Matrices
        "overlap": dft_result['overlap'],
        "hamiltonian_hartree": model_result['pred_hamiltonian'],
        "initial_hamiltonian": dft_result['initial_hamiltonian'],

        # Stretch ratio
        "stretch_ratio": model_result.get('stretch_ratio', dft_result.get('stretch_ratio')),
    }

    # Convert to torch tensors
    data = _convert_to_torch(results)

    # Add metadata
    data['molecule_name'] = molecule_name
    data['atom_types'] = atom_types
    data['site_type'] = site_type
    data['atom1_idx'] = atom1_idx
    data['atom2_idx'] = atom2_idx

    torch.save(data, save_path)


def save_model_results(output_dir, base_filename, model_results, dft_data,
                        stretch_ratios, metrics=None, metadata=None):
    """Save SPHNet prediction results as individual .pt files."""
    os.makedirs(output_dir, exist_ok=True)
    saved_paths = []

    molecule_name = dft_data.get('molecule_name')
    atom_types = dft_data.get('atom_types')
    site_type = dft_data.get('site_type')
    atom1_idx = dft_data.get('atom1_idx')
    atom2_idx = dft_data.get('atom2_idx')
    dft_results = dft_data['list_stretched_results']

    # Save individual .pt files
    for i, (model_result, dft_result) in enumerate(zip(model_results, dft_results)):
        stretch_ratio = model_result.get('stretch_ratio', stretch_ratios[i])
        bond_label = f'{atom_types[0]}{atom1_idx}-{atom_types[1]}{atom2_idx}'
        filename_parts = [molecule_name]
        if site_type:
            filename_parts.append(site_type)
        filename_parts.append(bond_label)
        filename_parts.append(f'ratio-{stretch_ratio:.2f}')
        filename_parts.append('qhflow-v2')
        filename = '_'.join(filename_parts) + '.pt'
        save_path = os.path.join(output_dir, filename)

        save_single_model_result(
            save_path=save_path,
            model_result=model_result,
            dft_result=dft_result,
            molecule_name=molecule_name,
            atom_types=atom_types,
            site_type=site_type,
            atom1_idx=atom1_idx,
            atom2_idx=atom2_idx
        )
        saved_paths.append(save_path)

    # Save metadata file
    meta = {
        'atoms': dft_data['atoms'],
        'positions': dft_data['positions'],
        'list_stretch_ratio': stretch_ratios,
        'list_stretched_positions': dft_data.get('list_stretched_positions'),
        'molecule_name': molecule_name,
        'atom_types': atom_types,
        'site_type': site_type,
        'atom1_idx': atom1_idx,
        'atom2_idx': atom2_idx,
        'result_files': [os.path.basename(p) for p in saved_paths],
    }

    if metrics is not None:
        meta['metrics'] = {k: v for k, v in metrics.items()
                          if not isinstance(v, np.ndarray) or k in ['stretch_ratios']}

    if metadata is not None:
        meta['model_metadata'] = metadata

    meta = _convert_to_torch(meta)
    bond_label = f'{atom_types[0]}{atom1_idx}-{atom_types[1]}{atom2_idx}'
    meta_parts = [molecule_name]
    if site_type:
        meta_parts.append(site_type)
    meta_parts.append(bond_label)
    meta_parts.append('qhflow-v2_metadata')
    metadata_filename = '_'.join(meta_parts) + '.pt'
    metadata_path = os.path.join(output_dir, metadata_filename)
    torch.save(meta, metadata_path)

    print(f"\nSaved {len(saved_paths)} SPHNet prediction result files to: {output_dir}")
    print(f"Metadata saved to: {metadata_path}")
    return saved_paths


# ==============================================================================
# Main execution
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="SPHNet model-based prediction for bond stretch HOMO-LUMO experiment"
    )
    parser.add_argument(
        "--molecule",
        type=str,
        required=True,
        help="Molecule name (e.g., 'ethanol', 'aspirin')"
    )
    parser.add_argument(
        "--site-type",
        type=str,
        default=None,
        help="Site type to analyze (e.g., 'primary', 'secondary')"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to experiment config YAML file"
    )
    parser.add_argument(
        "--ckpt-path",
        type=str,
        required=True,
        help="Path to SPHNet model checkpoint"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./output",
        help="Output directory for results and plots"
    )
    parser.add_argument(
        "--load-results",
        type=str,
        required=True,
        help="Path to DFT results directory (required)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for model inference (default: cuda)"
    )
    parser.add_argument(
        "--model-length-unit",
        type=str,
        default="ang",
        choices=["ang", "bohr"],
        help="Length unit expected by SPHNet model (default: ang)"
    )

    args = parser.parse_args()

    # 1. Load DFT results
    print(f"\nLoading DFT results from: {args.load_results}")
    dft_data = load_dft_results(
        args.load_results,
        molecule_name=args.molecule,
        site_type=args.site_type
    )

    # 2. Initialize SPHNet model predictor
    print(f"\nInitializing SPHNet model from: {args.ckpt_path}")
    predictor = init_model_predictor(
        ckpt_path=args.ckpt_path,
        device=args.device
    )

    # 3. Run predictions for all geometries
    stretch_ratios = np.array(dft_data['list_stretch_ratio'])
    model_results = run_predictions_for_all_geometries(
        predictor=predictor,
        dft_data=dft_data,
        model_length_unit=args.model_length_unit
    )

    # 4. Compute comparison metrics
    metrics = compute_comparison_metrics(
        dft_results=dft_data['list_stretched_results'],
        model_results=model_results,
        stretch_ratios=stretch_ratios
    )

    # 5. Print summary
    print_comparison_summary(metrics)

    # 6. Save results
    output_dir = args.save_dir
    os.makedirs(output_dir, exist_ok=True)

    bond_label = ""
    if dft_data['atom_types']:
        bond_label = f"{dft_data['atom_types'][0]}{dft_data['atom1_idx']}-{dft_data['atom_types'][1]}{dft_data['atom2_idx']}"

    results_filename = f"qhflow-v2-pred_{args.molecule}"
    if dft_data['site_type']:
        results_filename += f"_{dft_data['site_type']}"
    if bond_label:
        results_filename += f"_{bond_label}"

    save_model_results(
        output_dir=output_dir,
        base_filename=results_filename,
        model_results=model_results,
        dft_data=dft_data,
        stretch_ratios=stretch_ratios,
        metrics=metrics,
        metadata={
            'molecule_name': args.molecule,
            'site_type': dft_data['site_type'],
            'atom_types': dft_data['atom_types'],
            'atom1_idx': dft_data['atom1_idx'],
            'atom2_idx': dft_data['atom2_idx'],
            'ckpt_path': args.ckpt_path,
            'model_type': 'sphnet',
            'model_length_unit': args.model_length_unit,
        }
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
