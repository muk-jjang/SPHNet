import os
import glob
import torch
import time
import math
import json
import sys
import warnings
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm

from escflow_eval_utils import init_pyscf_mf, calc_dm0_from_ham, matrix_transform_single
from escflow_eval_utils import BOHR2ANG

# Suppress FutureWarning about torch.load
warnings.filterwarnings('ignore', category=FutureWarning)

# ============================================================================
# Constants
# ============================================================================
DEFAULT_XC = "pbe"
DEFAULT_BASIS = "def2svp"
DEFAULT_UNIT = "ang"
MIN_VALID_ENERGY = -300.0  # Hartree
MAX_VALID_ENERGY = 0.0       # Hartree (positive energies are suspicious)


# ============================================================================
# Utility Functions
# ============================================================================
def ensure_numpy_float64(array):
    """Convert array to NumPy float64, handling CuPy arrays and PyTorch tensors."""
    if hasattr(array, 'get'):  # CuPy array
        array = array.get()
    elif hasattr(array, 'cpu'):  # PyTorch tensor
        array = array.cpu().numpy()
    return np.asarray(array, dtype=np.float64)


def is_invalid_energy(value, min_energy=MIN_VALID_ENERGY, max_energy=MAX_VALID_ENERGY):
    """
    Check if an energy value is invalid (NaN, inf, or out of reasonable range).
    
    Returns True if the energy should be recalculated.
    """
    if value is None:
        return True
    
    try:
        if isinstance(value, torch.Tensor):
            val = value.item() if value.numel() == 1 else float(value.mean())
        elif isinstance(value, np.ndarray):
            val = float(value.mean()) if value.size > 1 else float(value)
        else:
            val = float(value)
    except (TypeError, ValueError):
        return True
    
    if math.isnan(val) or math.isinf(val):
        return True
    
    return val > max_energy or val < min_energy


def convert_density_for_gpu(density, use_gpu):
    """Convert density matrix to CuPy array if using GPU."""
    if use_gpu >= 0:
        try:
            import cupy as cp
            density_np = density.cpu().numpy() if hasattr(density, 'cpu') else density
            return cp.asarray(density_np)
        except Exception as e:
            print(f"Warning: Failed to convert density to CuPy: {e}, using CPU")
            return density.cpu().numpy() if hasattr(density, 'cpu') else density
    return density


# ============================================================================
# Force Calculation
# ============================================================================
def compute_gradient_forces(calc_mf, grad_frame, mo_energy, mo_coeff, mo_occ, use_gpu):
    """Compute gradient forces with proper GPU handling."""
    if use_gpu >= 0:
        import cupy as cp
        original = {
            'mo_coeff': getattr(calc_mf, 'mo_coeff', None),
            'mo_energy': getattr(calc_mf, 'mo_energy', None),
            'mo_occ': getattr(calc_mf, 'mo_occ', None),
        }
        
        calc_mf.mo_coeff = cp.asarray(mo_coeff, dtype=cp.float64)
        calc_mf.mo_energy = cp.asarray(mo_energy, dtype=cp.float64)
        calc_mf.mo_occ = cp.asarray(mo_occ, dtype=cp.float64)
        
        forces = -grad_frame.kernel(
            mo_energy=calc_mf.mo_energy,
            mo_coeff=calc_mf.mo_coeff,
            mo_occ=calc_mf.mo_occ,
        )
        
        for key, val in original.items():
            if val is not None:
                setattr(calc_mf, key, val)
        
        return forces
    
    return -grad_frame.kernel(mo_energy=mo_energy, mo_coeff=mo_coeff, mo_occ=mo_occ)


# ============================================================================
# Energy and Density Calculation
# ============================================================================
def compute_energy_and_forces(calc_mf, grad_frame, atoms, overlap, hamiltonian, use_gpu, label=""):
    """Compute energy, forces, and MO data from Hamiltonian."""
    density, res = calc_dm0_from_ham(
        atoms=atoms,
        overlap=overlap,
        hamiltonian=hamiltonian,
        transform=False,
        return_tensor=(use_gpu >= 0)
    )
    
    density_converted = convert_density_for_gpu(density, use_gpu)
    energy = calc_mf.energy_tot(density_converted)
    
    mo_energy = ensure_numpy_float64(res["orbital_energies"].squeeze())
    mo_coeff = ensure_numpy_float64(res["orbital_coefficients"].squeeze())
    mo_occ = ensure_numpy_float64(calc_mf.get_occ(mo_energy, mo_coeff))
    
    if label:
        print(f"{label}: Computing forces...", flush=True)
    force_start = time.time()
    forces = compute_gradient_forces(calc_mf, grad_frame, mo_energy, mo_coeff, mo_occ, use_gpu)
    if label:
        print(f"{label}: Forces completed in {time.time() - force_start:.2f}s", flush=True)
    
    return {
        'energy': energy,
        'forces': forces,
        'mo_energy': mo_energy,
        'mo_coeff': mo_coeff,
        'mo_occ': mo_occ,
    }


def setup_pyscf_mf(atoms, pos, unit, xc, basis, use_gpu):
    """Initialize and configure PySCF mean-field object."""
    calc_mf = init_pyscf_mf(atoms, pos, unit=unit, xc=xc, basis=basis, use_gpu=use_gpu)
    calc_mf.conv_tol = 1e-7
    calc_mf.grids.level = 3
    calc_mf.grids.prune = None
    calc_mf.init_guess = "minao"
    calc_mf.small_rho_cutoff = 1e-12
    return calc_mf


# ============================================================================
# Data Loading and Caching
# ============================================================================
def load_or_compute_calc_data(calc_path, gt_data, calc_mf, grad_frame, atoms, use_gpu, debug, DO_NEW_CALC, data_index):
    """Load cached calc data or compute fresh if needed."""
    if os.path.exists(calc_path) and not DO_NEW_CALC:
        calc_data = torch.load(calc_path, weights_only=False)
        return {
            'energy': calc_data["calc_energy"],
            'forces': calc_data["calc_forces"],
            'mo_energy': calc_data["calc_mo_energy"],
            'mo_coeff': calc_data["calc_mo_coeff"],
            'ham': calc_data["hamiltonian"],
            'overlap': calc_data["overlap"],
            'data': calc_data,
        }
    
    # Compute fresh
    calc_data = gt_data.copy()
    prefix = f"[Process {os.getpid()}] Molecule {data_index}"
    
    print(f"{prefix}: Starting SCF calculation...", flush=True)
    start_time = time.time()
    calc_mf.kernel()
    scf_time = time.time() - start_time
    print(f"{prefix}: SCF completed in {scf_time:.2f}s", flush=True)
    
    calc_data["calc_time"] = scf_time
    calc_data["hamiltonian"] = torch.tensor(calc_mf.get_fock(dm=calc_mf.make_rdm1()), dtype=torch.float64)
    calc_data["overlap"] = torch.tensor(calc_mf.get_ovlp(), dtype=torch.float64)
    calc_data["density_matrix"] = torch.tensor(calc_mf.make_rdm1(), dtype=torch.float64)
    calc_data["method"] = "RKS"
    calc_data["xc"] = calc_mf.xc
    calc_data["basis"] = calc_mf.mol.basis
    
    print(f"{prefix}: Computing initial forces...", flush=True)
    force_start = time.time()
    calc_data["forces"] = torch.tensor(-grad_frame.kernel(), dtype=torch.float64)
    print(f"{prefix}: Initial forces completed in {time.time() - force_start:.2f}s", flush=True)
    
    calc_overlap = calc_data["overlap"].unsqueeze(0)
    calc_ham = calc_data["hamiltonian"].unsqueeze(0)
    
    result = compute_energy_and_forces(
        calc_mf, grad_frame, atoms, calc_overlap, calc_ham, use_gpu,
        label=f"{prefix}: calc"
    )
    
    calc_data["calc_energy"] = result['energy']
    calc_data["calc_mo_energy"] = result['mo_energy']
    calc_data["calc_mo_coeff"] = result['mo_coeff']
    calc_data["mo_occ"] = result['mo_occ']
    calc_data["calc_forces"] = result['forces']
    
    if not debug:
        torch.save(calc_data, calc_path)
    
    return {
        'energy': result['energy'],
        'forces': result['forces'],
        'mo_energy': result['mo_energy'],
        'mo_coeff': result['mo_coeff'],
        'ham': calc_ham,
        'overlap': calc_overlap,
        'data': calc_data,
    }


def prepare_hamiltonian(data, init_ham, remove_init, atoms):
    """Prepare Hamiltonian with optional init_ham addition."""
    if "pred_hamiltonian" in data:
        ham = data["pred_hamiltonian"] + remove_init * init_ham.reshape(data["pred_hamiltonian"].shape)
    else:
        ham = data["hamiltonian"] + remove_init * init_ham.reshape(data["hamiltonian"].shape)
    return matrix_transform_single(ham.unsqueeze(0), atoms, convention="back2pyscf")


# ============================================================================
# Result Generation
# ============================================================================
def compute_metrics(pred, gt, calc, atoms, pred_ham, gt_ham, calc_ham, gt_overlap, calc_overlap):
    """Compute all evaluation metrics."""
    num_occ = int(atoms.sum() / 2)
    
    # Force norms
    pred_forces_norm = np.linalg.norm(pred['forces'], axis=1)
    gt_forces_norm = np.linalg.norm(gt['forces'], axis=1)
    calc_forces_norm = np.linalg.norm(calc['forces'], axis=1)
    
    # Occupied orbital data
    pred_mo_occ_coeff = torch.tensor(pred['mo_coeff'][:, :num_occ])
    gt_mo_occ_coeff = torch.tensor(gt['mo_coeff'][:, :num_occ])
    calc_mo_occ_coeff = torch.tensor(calc['mo_coeff'][:, :num_occ])
    
    pred_mo_energy_occ = pred['mo_energy'][:num_occ]
    gt_mo_energy_occ = gt['mo_energy'][:num_occ]
    calc_mo_energy_occ = calc['mo_energy'][:num_occ]
    
    return {
        "hamiltonian_diff (pred-gt)": abs(pred_ham - gt_ham).mean(),
        "hamiltonian_diff (pred-calc)": abs(pred_ham - calc_ham).mean(),
        "hamiltonian_diff (gt-calc)": abs(gt_ham - calc_ham).mean(),

        "pred_energy": pred['energy'],
        "gt_energy": gt['energy'],
        "calc_energy": calc['energy'],
        
        "energy_diff (pred-gt)": abs(pred['energy'] - gt['energy']),
        "energy_diff (pred-calc_energy)": abs(pred['energy'] - calc['energy']),
        "energy_diff (gt-calc_energy)": abs(gt['energy'] - calc['energy']),

        "pred_force": pred['forces'],
        "gt_force": gt['forces'],
        "calc_force": calc['forces'],

        "forces_diff l2 (pred-gt)": abs(pred['forces'] - gt['forces']).mean(),
        "forces_diff l2 (pred-calc_forces)": abs(pred['forces'] - calc['forces']).mean(),
        "forces_diff l2 (gt-calc_forces)": abs(gt['forces'] - calc['forces']).mean(),

        "pred_force_norm": pred_forces_norm,
        "gt_force_norm": gt_forces_norm,
        "calc_force_norm": calc_forces_norm,

        "pred_force_norm_diff (pred-gt)": abs(pred_forces_norm - gt_forces_norm).mean(),
        "pred_force_norm_diff (pred-calc_forces)": abs(pred_forces_norm - calc_forces_norm).mean(),
        "gt_force_norm_diff (gt-calc_forces)": abs(gt_forces_norm - calc_forces_norm).mean(),

        "orbital_coeff_similarity (pred-gt)": torch.cosine_similarity(pred_mo_occ_coeff, gt_mo_occ_coeff, dim=0).abs().mean(),
        "orbital_coeff_similarity (pred-calc)": torch.cosine_similarity(pred_mo_occ_coeff, calc_mo_occ_coeff, dim=0).abs().mean(),
        "orbital_coeff_similarity (gt-calc)": torch.cosine_similarity(gt_mo_occ_coeff, calc_mo_occ_coeff, dim=0).abs().mean(),

        "occupied_orbital_energy_mae (pred-gt)": np.abs(pred_mo_energy_occ - gt_mo_energy_occ).mean(),
        "occupied_orbital_energy_mae (pred-calc)": np.abs(pred_mo_energy_occ - calc_mo_energy_occ).mean(),
        "occupied_orbital_energy_mae (gt-calc)": np.abs(gt_mo_energy_occ - calc_mo_energy_occ).mean(),

        "overlap_diff (gt-calc)": np.abs(gt_overlap - calc_overlap).mean(),
    }


# ============================================================================
# Main Processing Function
# ============================================================================
def process_single_molecule(pred_file_path, gt_file_path,
                            unit=DEFAULT_UNIT, xc=DEFAULT_XC, basis=DEFAULT_BASIS,
                            debug=False, use_gpu=-1, DO_NEW_CALC=False):
    """Process a single molecule and compute evaluation metrics."""
    calc_path = pred_file_path.replace("pred_", "calc_")
    
    # Load data
    pred_data = torch.load(pred_file_path, weights_only=False)
    gt_data = torch.load(gt_file_path, weights_only=False)
    
    data_index = gt_data["idx"]
    molecule_start_time = time.time()
    prefix = f"[Process {os.getpid()}] Molecule {data_index}"
    print(f"{prefix}: Starting at {time.strftime('%H:%M:%S')}", flush=True)
    
    # Setup
    atoms = gt_data["atoms"]
    pos = gt_data["pos"] * BOHR2ANG
    calc_mf = setup_pyscf_mf(atoms, pos, unit, xc, basis, use_gpu)
    grad_frame = calc_mf.nuc_grad_method()
    
    # Load or compute calc data
    calc_result = load_or_compute_calc_data(
        calc_path, gt_data, calc_mf, grad_frame, atoms, use_gpu, debug, DO_NEW_CALC, data_index
    )
    calc_overlap = calc_result['overlap']
    calc_ham = calc_result['ham']
    
    # Handle remove_init flag
    if "remove_init" not in gt_data:
        remove_init = True
        gt_data["remove_init"] = remove_init
        pred_data["remove_init"] = remove_init
    else:
        remove_init = gt_data["remove_init"]
    
    # Check if recalculation is needed
    pred_energy_invalid = "calc_energy" not in pred_data or is_invalid_energy(pred_data.get("calc_energy"))
    gt_energy_invalid = "calc_energy" not in gt_data or is_invalid_energy(gt_data.get("calc_energy"))
    need_recalc = pred_energy_invalid or gt_energy_invalid
    
    if need_recalc:
        print(f"{prefix}: Invalid energy detected "
              f"(pred={pred_data.get('calc_energy', 'missing')}, "
              f"gt={gt_data.get('calc_energy', 'missing')}), recalculating...", flush=True)
    
    # Check if we can use cached values
    has_cached = ("calc_forces" in pred_data and "calc_forces" in gt_data)
    
    if has_cached and not DO_NEW_CALC and not need_recalc:
        # Use cached values
        pred = {
            'energy': pred_data["calc_energy"],
            'forces': pred_data["calc_forces"],
            'mo_energy': pred_data["calc_mo_energy"],
            'mo_coeff': pred_data["calc_mo_coeff"],
            'mo_occ': pred_data["mo_occ"],
        }
        gt = {
            'energy': gt_data["calc_energy"],
            'forces': gt_data["calc_forces"],
            'mo_energy': gt_data["calc_mo_energy"],
            'mo_coeff': gt_data["calc_mo_coeff"],
            'mo_occ': gt_data["mo_occ"],
        }
        pred_ham = prepare_hamiltonian(pred_data, gt_data["init_ham"], remove_init, atoms)
        gt_ham = prepare_hamiltonian(gt_data, gt_data["init_ham"], remove_init, atoms)
        gt_overlap = torch.from_numpy(gt_data["overlap"]).reshape(calc_overlap.shape)
    else:
        # Compute fresh
        gt_overlap = torch.from_numpy(gt_data["overlap"]).reshape(calc_overlap.shape)
        gt_overlap_transformed = matrix_transform_single(gt_overlap, atoms, convention="back2pyscf")
        
        pred_ham = prepare_hamiltonian(pred_data, gt_data["init_ham"], remove_init, atoms)
        gt_ham = prepare_hamiltonian(gt_data, gt_data["init_ham"], remove_init, atoms)
        
        # Compute pred
        pred = compute_energy_and_forces(
            calc_mf, grad_frame, atoms, gt_overlap_transformed, pred_ham, use_gpu,
            label=f"{prefix}: pred"
        )
        pred_data["calc_energy"] = pred['energy']
        pred_data["calc_mo_energy"] = pred['mo_energy']
        pred_data["calc_mo_coeff"] = pred['mo_coeff']
        pred_data["mo_occ"] = pred['mo_occ']
        pred_data["calc_forces"] = pred['forces']
        
        # Compute gt
        gt = compute_energy_and_forces(
            calc_mf, grad_frame, atoms, gt_overlap_transformed, gt_ham, use_gpu,
            label=f"{prefix}: gt"
        )
        gt_data["calc_energy"] = gt['energy']
        gt_data["calc_mo_energy"] = gt['mo_energy']
        gt_data["calc_mo_coeff"] = gt['mo_coeff']
        gt_data["mo_occ"] = gt['mo_occ']
        gt_data["calc_forces"] = gt['forces']
        
        gt_overlap = gt_overlap_transformed
        
        if not debug:
            torch.save(pred_data, pred_file_path)
            torch.save(gt_data, gt_file_path)
            print(f"{prefix}: Saved updated data files", flush=True)
    
    # Compute metrics
    calc = {
        'energy': calc_result['energy'],
        'forces': calc_result['forces'],
        'mo_energy': calc_result['mo_energy'],
        'mo_coeff': calc_result['mo_coeff'],
    }
    
    result = compute_metrics(pred, gt, calc, atoms, pred_ham, gt_ham, calc_ham, gt_overlap, calc_overlap)
    result["data_index"] = data_index
    result["was_nan_fixed"] = need_recalc
    
    total_time = time.time() - molecule_start_time
    print(f"{prefix}: COMPLETED in {total_time:.2f}s total", flush=True)
    
    return result


# ============================================================================
# Main Entry Point
# ============================================================================
def aggregate_results(results):
    """Aggregate results from multiple molecules."""
    if not results:
        return {}
    
    keys_to_remove = [
        "data_index", "pred_force", "gt_force", "calc_force",
        "pred_force_norm", "gt_force_norm", "calc_force_norm", "was_nan_fixed"
    ]
    
    keys = [k for k in results[0].keys() if k not in keys_to_remove]
    keys.extend(["error_count", "nan_fixed_count"])
    
    nan_fixed_count = sum(1 for r in results if r.get("was_nan_fixed", False))
    
    evaluation_result = {key: [] for key in keys}
    for result in results:
        for key in result:
            if key not in keys_to_remove:
                evaluation_result[key].append(result[key])
    
    for key in keys:
        if key == "error_count":
            evaluation_result[key] = len(evaluation_result[key])
        elif key == "nan_fixed_count":
            evaluation_result[key] = nan_fixed_count
        else:
            evaluation_result[key] = float(np.mean(evaluation_result[key]))
    
    return evaluation_result


def filter_invalid_molecules(file_pairs):
    """Filter to only molecules with invalid calc_energy."""
    print("Filtering to only process molecules with invalid calc_energy (NaN, inf, or out of range)...")
    invalid_pairs = []
    
    for pred_path, gt_path in tqdm(file_pairs, desc="Checking for invalid energy"):
        pred_data = torch.load(pred_path, weights_only=False)
        gt_data = torch.load(gt_path, weights_only=False)
        
        pred_invalid = "calc_energy" not in pred_data or is_invalid_energy(pred_data.get("calc_energy"))
        gt_invalid = "calc_energy" not in gt_data or is_invalid_energy(gt_data.get("calc_energy"))
        
        if pred_invalid or gt_invalid:
            invalid_pairs.append((pred_path, gt_path))
            data_index = gt_data.get("idx", "unknown")
            print(f"  Found invalid energy in molecule {data_index}: "
                  f"pred={pred_data.get('calc_energy', 'missing')}, "
                  f"gt={gt_data.get('calc_energy', 'missing')}")
    
    print(f"Found {len(invalid_pairs)} molecules with invalid calc_energy")
    return invalid_pairs


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate and fix NaN energies in MD17 dataset")
    parser.add_argument("--dir_path", type=str, default="/nas/seongjun/sphnet/aspirin/output_dump_batch")
    parser.add_argument("--pred_prefix", type=str, default="pred_")
    parser.add_argument("--gt_prefix", type=str, default="gt_")
    parser.add_argument("--num_procs", type=int, default=1)
    parser.add_argument("--debug", action="store_true", help="Don't save files")
    parser.add_argument("--size_limit", type=int, default=-1, help="Limit molecules (-1 for all)")
    parser.add_argument("--only_nan", action="store_true", help="Only process invalid energy molecules")
    parser.add_argument("--do_new_calc", action="store_true", help="Force recalculation")
    args = parser.parse_args()
    
    use_gpu = -1
    print("Running in CPU mode")
    
    # Gather file pairs
    pred_paths = sorted(glob.glob(os.path.join(args.dir_path, f"{args.pred_prefix}*.pt")))
    gt_paths = sorted(glob.glob(os.path.join(args.dir_path, f"{args.gt_prefix}*.pt")))
    file_pairs = list(zip(pred_paths, gt_paths))
    
    # Filter if requested
    if args.only_nan:
        file_pairs = filter_invalid_molecules(file_pairs)
    
    # Apply size limit
    if args.size_limit > 0:
        file_pairs = file_pairs[:args.size_limit]
    
    print(f"Processing {len(file_pairs)} molecules with CPU and {args.num_procs} processes...")
    
    # Process molecules
    if args.num_procs == 1:
        results = [
            process_single_molecule(pred_path, gt_path, debug=args.debug, use_gpu=use_gpu, DO_NEW_CALC=args.do_new_calc)
            for pred_path, gt_path in tqdm(file_pairs, desc="Processing molecules")
        ]
    else:
        from functools import partial
        process_func = partial(process_single_molecule, debug=args.debug, use_gpu=use_gpu, DO_NEW_CALC=args.do_new_calc)
        with Pool(processes=args.num_procs) as pool:
            results = list(tqdm(
                pool.starmap(process_func, file_pairs),
                total=len(file_pairs),
                desc="Processing molecules"
            ))
    
    print(f"\nCompleted processing {len(results)} molecules")
    
    if not results:
        print("No molecules to process. Exiting.")
        sys.exit(0)
    
    # Aggregate and save results
    nan_fixed_count = sum(1 for r in results if r.get("was_nan_fixed", False))
    print(f"Fixed {nan_fixed_count} molecules with invalid calc_energy")
    
    evaluation_result = aggregate_results(results)
    
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    for key, value in evaluation_result.items():
        print(f"{key}: {value}")
    print("=" * 80)
    
    # Save results
    dataset_name = args.dir_path.split("/")[-2]
    os.makedirs('./outputs', exist_ok=True)
    output_file = os.path.join('./outputs', f"{dataset_name}_evaluation_results_nan_fixed.json")
    with open(output_file, "w") as f:
        json.dump(evaluation_result, f, indent=4)
    print(f"\nEvaluation results saved to: {output_file}")


if __name__ == "__main__":
    main()
