import os
import glob
import torch
from pyscf import gto, scf, dft
import time
from datetime import datetime, timedelta
from escflow_eval_utils import init_pyscf_mf, calc_dm0_from_ham, matrix_transform_single
from escflow_eval_utils import BOHR2ANG, HA2meV, HA_BOHR_2_meV_ANG
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
import warnings
import json
import sys
import logging

# Suppress FutureWarning about torch.load
warnings.filterwarnings('ignore', category=FutureWarning)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def format_time(seconds):
    """Format seconds to human readable string"""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.2f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.2f}s"

def process_single_molecule_wrapper(args):
    """Wrapper function for multiprocessing with imap_unordered"""
    return process_single_molecule(*args)

# def _convert_density_for_gpu(density, use_gpu):
#     """Convert density matrix to CuPy array if using GPU, otherwise return as-is."""
#     if use_gpu >= 0:
#         try:
#             import cupy as cp
#             # If density is a tensor, convert to numpy first
#             if hasattr(density, 'cpu'):
#                 density_np = density.cpu().numpy()
#             else:
#                 density_np = density
#             return cp.asarray(density_np)
#         except Exception as e:
#             print(f"Warning: Failed to convert density to CuPy: {e}, using CPU")
#             return density.cpu().numpy() if hasattr(density, 'cpu') else density
#     else:
#         return density

def _ensure_numpy_float64(array):
    """
    Convert array to NumPy float64, handling CuPy arrays, PyTorch tensors, etc.
    This ensures compatibility with PySCF gradient calculations.
    """
    # Handle CuPy arrays (from GPU calculations)
    if hasattr(array, 'get'):
        array = array.get()
    # Handle PyTorch tensors
    elif hasattr(array, 'cpu'):
        array = array.cpu().numpy()
    # Convert to NumPy float64
    return np.asarray(array, dtype=np.float64)


def process_single_molecule(pred_file_path, gt_file_path,
    unit="ang", xc="pbe", basis="def2svp", debug=False, use_gpu=-1,
    do_new_calc=False
):
    dir_path = os.path.dirname(pred_file_path)
    calc_path = pred_file_path.replace("pred_", "calc_")
    
    # Timing dictionary to track each step
    timing_info = {}

    load_start = time.time()
    pred_data = torch.load(pred_file_path)
    gt_data = torch.load(gt_file_path)
    timing_info['data_load'] = time.time() - load_start

    data_index = gt_data["idx"]
    molecule_start_time = time.time()
    # logger.info(f"[PID:{os.getpid()}] ========== START Molecule {data_index} ==========")

    atoms = gt_data["atoms"]
    pos = gt_data["pos"] * BOHR2ANG
    
    init_start = time.time()
    calc_mf = init_pyscf_mf(atoms, pos, unit=unit, xc=xc, basis=basis)
    grad_frame = calc_mf.nuc_grad_method()
    calc_mf.conv_tol = 1e-7
    calc_mf.grids.level = 3
    calc_mf.grids.prune = None
    calc_mf.init_guess = "minao"
    calc_mf.small_rho_cutoff = 1e-12
    timing_info['pyscf_init'] = time.time() - init_start
    logger.debug(f"[PID:{os.getpid()}] Molecule {data_index}: PySCF init in {format_time(timing_info['pyscf_init'])}")

    #try:
    # Check if calculated data exists
    if os.path.exists(calc_path) and not do_new_calc:
        # logger.info(f"[PID:{os.getpid()}] Molecule {data_index}: Loading cached calc data...")
        calc_data = torch.load(calc_path)
        calc_energy = calc_data["calc_energy"]
        calc_forces = calc_data["calc_forces"]
        calc_mo_energy = calc_data["calc_mo_energy"]
        calc_mo_coeff = calc_data["calc_mo_coeff"]
        calc_ham = calc_data["hamiltonian"]
        calc_overlap = calc_data["overlap"]
        
    else:
        calc_data = gt_data.copy()  # Use copy to avoid modifying original
        scf_start = time.time()
        # logger.info(f"[PID:{os.getpid()}] Molecule {data_index}: Starting SCF calculation...")
        calc_mf.kernel()
        scf_time = time.time() - scf_start
        timing_info['scf'] = scf_time
        calc_data["calc_time"] = scf_time
        # logger.info(f"[PID:{os.getpid()}] Molecule {data_index}: SCF completed in {format_time(scf_time)}")
        calc_data["hamiltonian"] = torch.tensor(calc_mf.get_fock(dm=calc_mf.make_rdm1()), dtype=torch.float64)
        calc_data["overlap"] = torch.tensor(calc_mf.get_ovlp(), dtype=torch.float64)
        calc_data["density_matrix"] = torch.tensor(calc_mf.make_rdm1(), dtype=torch.float64)
        calc_data["method"] = "RKS"
        calc_data["xc"] = xc
        calc_data["basis"] = basis
        # calc_data["scf_cycles"] = calc_mf.cycles
        # logger.info(f"[PID:{os.getpid()}] Molecule {data_index}: Computing initial forces...")
        force_start = time.time()
        calc_data["forces"] = torch.tensor(-grad_frame.kernel(), dtype=torch.float64)
        init_force_time = time.time() - force_start
        timing_info['initial_forces'] = init_force_time
        # logger.info(f"[PID:{os.getpid()}] Molecule {data_index}: Initial forces completed in {format_time(init_force_time)}")

        calc_overlap = calc_data["overlap"].unsqueeze(0) # (gt_overlap - calc_overlap) has float32 precision error (1e^-7)
        calc_ham = calc_data["hamiltonian"].unsqueeze(0)

        calc_density, calc_res = calc_dm0_from_ham(atoms, calc_overlap, calc_ham, transform=False, return_tensor=(use_gpu >= 0))
        calc_energy = calc_mf.energy_tot(calc_density)
        calc_data["calc_energy"] = calc_energy

        calc_mo_energy = _ensure_numpy_float64(calc_res["orbital_energies"].squeeze())
        calc_mo_coeff = _ensure_numpy_float64(calc_res["orbital_coefficients"].squeeze())
        calc_data["calc_mo_energy"] = calc_mo_energy
        calc_data["calc_mo_coeff"] = calc_mo_coeff

        mo_occ = calc_mf.get_occ(calc_mo_energy, calc_mo_coeff)
        calc_data["mo_occ"] = mo_occ
        # logger.info(f"[PID:{os.getpid()}] Molecule {data_index}: Computing calc forces with MO...")
        force_start = time.time()
        calc_forces = -grad_frame.kernel(mo_energy=calc_mo_energy, mo_coeff=calc_mo_coeff, mo_occ=mo_occ)
        calc_force_time = time.time() - force_start
        timing_info['calc_forces'] = calc_force_time
        # logger.info(f"[PID:{os.getpid()}] Molecule {data_index}: Calc forces completed in {format_time(calc_force_time)}")
        calc_data["calc_forces"] = calc_forces

        # save calc_data
        if not debug:
            torch.save(calc_data, calc_path)
    
    if "remove_init" not in gt_data.keys():
        remove_init = True
        gt_data["remove_init"] = remove_init
        pred_data["remove_init"] = remove_init
    else:
        remove_init = gt_data["remove_init"]

    if "calc_forces" in pred_data.keys() and "calc_forces" in gt_data.keys() and not do_new_calc:
        pred_energy = pred_data["calc_energy"]
        pred_forces = pred_data["calc_forces"]
        pred_mo_energy = pred_data["calc_mo_energy"]
        pred_mo_coeff = pred_data["calc_mo_coeff"]
        pred_hamiltonian = pred_data["pred_hamiltonian"] + remove_init * gt_data["init_ham"].reshape(pred_data["pred_hamiltonian"].shape)
        pred_ham = matrix_transform_single(pred_hamiltonian.unsqueeze(0), atoms, convention="back2pyscf")

        gt_energy = gt_data["calc_energy"]
        gt_forces = gt_data["calc_forces"]
        gt_mo_energy = gt_data["calc_mo_energy"]
        gt_mo_coeff = gt_data["calc_mo_coeff"]
        gt_hamiltonian = gt_data["hamiltonian"] + remove_init * gt_data["init_ham"].reshape(gt_data["hamiltonian"].shape)
        gt_ham = matrix_transform_single(gt_hamiltonian.unsqueeze(0), atoms, convention="back2pyscf")
        gt_overlap = gt_data["overlap"].reshape(calc_overlap.shape)
    else:
        gt_overlap = gt_data["overlap"]
        gt_overlap = torch.from_numpy(gt_overlap).reshape(calc_overlap.shape)
        gt_overlap = matrix_transform_single(gt_overlap, atoms, convention="back2pyscf")

        pred_hamiltonian = pred_data["pred_hamiltonian"] + remove_init * gt_data["init_ham"].reshape(pred_data["pred_hamiltonian"].shape)
        pred_ham = matrix_transform_single(pred_hamiltonian.unsqueeze(0), atoms, convention="back2pyscf")
        
        gt_hamiltonian = gt_data["hamiltonian"] + remove_init * gt_data["init_ham"].reshape(gt_data["hamiltonian"].shape)
        gt_ham = matrix_transform_single(gt_hamiltonian.unsqueeze(0), atoms, convention="back2pyscf")
        
        pred_density, pred_res = calc_dm0_from_ham(
            atoms=atoms,
            overlap=gt_overlap,
            hamiltonian=pred_ham,
            transform=False,
            return_tensor=(use_gpu >= 0)
            )
        pred_energy = calc_mf.energy_tot(pred_density)
        pred_data["calc_energy"] = pred_energy

        gt_density, gt_res = calc_dm0_from_ham(
            atoms=atoms,
            overlap=gt_overlap,
            hamiltonian=gt_ham,
            transform=False,
            return_tensor=(use_gpu >= 0)
            )
        gt_energy = calc_mf.energy_tot(gt_density)
        gt_data["calc_energy"] = gt_energy
        
        pred_mo_energy = _ensure_numpy_float64(pred_res["orbital_energies"].squeeze())
        pred_mo_coeff = _ensure_numpy_float64(pred_res["orbital_coefficients"].squeeze())
        pred_data["calc_mo_energy"] = pred_mo_energy
        pred_data["calc_mo_coeff"] = pred_mo_coeff

        gt_mo_energy = _ensure_numpy_float64(gt_res["orbital_energies"].squeeze())
        gt_mo_coeff = _ensure_numpy_float64(gt_res["orbital_coefficients"].squeeze())
        gt_data["calc_mo_energy"] = gt_mo_energy
        gt_data["calc_mo_coeff"] = gt_mo_coeff

        pred_mo_occ = _ensure_numpy_float64(calc_mf.get_occ(pred_mo_energy, pred_mo_coeff))
        pred_data["mo_occ"] = pred_mo_occ
        # logger.info(f"[PID:{os.getpid()}] Molecule {data_index}: Computing pred forces...")
        force_start = time.time()
        pred_forces = -grad_frame.kernel(mo_energy=pred_mo_energy, mo_coeff=-pred_mo_coeff, mo_occ=pred_mo_occ)
        pred_force_time = time.time() - force_start
        timing_info['pred_forces'] = pred_force_time
        # logger.info(f"[PID:{os.getpid()}] Molecule {data_index}: Pred forces completed in {format_time(pred_force_time)}")
        pred_data["calc_forces"] = pred_forces

        gt_mo_occ = _ensure_numpy_float64(calc_mf.get_occ(gt_mo_energy, gt_mo_coeff))
        gt_data["mo_occ"] = gt_mo_occ
        # logger.info(f"[PID:{os.getpid()}] Molecule {data_index}: Computing gt forces...")
        force_start = time.time()
        gt_forces = -grad_frame.kernel(mo_energy=gt_mo_energy, mo_coeff=-gt_mo_coeff, mo_occ=gt_mo_occ)
        gt_force_time = time.time() - force_start
        timing_info['gt_forces'] = gt_force_time
        # logger.info(f"[PID:{os.getpid()}] Molecule {data_index}: GT forces completed in {format_time(gt_force_time)}")
        gt_data["calc_forces"] = gt_forces

        if not debug:
            torch.save(pred_data, pred_file_path)
            torch.save(gt_data, gt_file_path)

    pred_forces_norm = np.linalg.norm(pred_forces, axis=1)
    calc_forces_norm = np.linalg.norm(calc_forces, axis=1)
    gt_forces_norm = np.linalg.norm(gt_forces, axis=1)

    num_occ = int(gt_data["atoms"].sum() / 2)

    # Extract occupied orbital energies only
    pred_mo_energy_occ = pred_mo_energy[:num_occ]
    gt_mo_energy_occ = gt_mo_energy[:num_occ]
    calc_mo_energy_occ = calc_mo_energy[:num_occ]
    pred_mo_occ_coeff = torch.tensor(pred_mo_coeff[:, :num_occ])
    gt_mo_occ_coeff = torch.tensor(gt_mo_coeff[:, :num_occ])
    calc_mo_occ_coeff = torch.tensor(calc_mo_coeff[:, :num_occ])

    result = {
        "data_index": data_index,

        "hamiltonian_diff (pred-gt)": abs(pred_ham - gt_ham).mean(),
        "hamiltonian_diff (pred-calc)": abs(pred_ham - calc_ham).mean(),
        "hamiltonian_diff (gt-calc)": abs(gt_ham - calc_ham).mean(),

        "pred_energy": pred_energy,
        "gt_energy": gt_energy,
        "calc_energy": calc_energy,
        
        "energy_diff (pred-gt)": abs(pred_energy - gt_energy),
        "energy_diff (pred-calc_energy)": abs(pred_energy - calc_energy),
        "energy_diff (gt-calc_energy)": abs(gt_energy - calc_energy),

        "pred_force": pred_forces,
        "gt_force": gt_forces,
        "calc_force": calc_forces,

        "forces_diff l2 (pred-gt)": abs(pred_forces - gt_forces).mean(),
        "forces_diff l2 (pred-calc_forces)": abs(pred_forces - calc_forces).mean(),
        "forces_diff l2 (gt-calc_forces)": abs(gt_forces - calc_forces).mean(),

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

        "overlap_diff (gt-calc)": np.abs(np.asarray(gt_overlap) - np.asarray(calc_overlap)).mean(),
    }
    # unit of pyscf calculation
    # distance in bohr
    # energy in hartree
    # forces in hartree/bohr
    # hamiltonian in hartree
    """
    except Exception as e:
        print(f"Error processing molecule {data_index}: {str(e)}")
        result = {
            "error_count": data_index,
        }
    """
    total_time = time.time() - molecule_start_time
    timing_info['total'] = total_time
    result['timing_info'] = timing_info
    
    # Log timing breakdown
    timing_summary = " | ".join([f"{k}:{format_time(v)}" for k, v in timing_info.items()])
    logger.info(f"[PID:{os.getpid()}] ========== DONE Molecule {data_index} in {format_time(total_time)} ==========")
    logger.debug(f"[PID:{os.getpid()}] Molecule {data_index} timing: {timing_summary}")
    
    return result



if __name__ == "__main__":
    import argparse

    # Record overall start time
    overall_start_time = time.time()
    script_start_datetime = datetime.now()

    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_path", type=str, default="/nas/seongjun/sphnet/aspirin/output_dump_batch")
    parser.add_argument("--pred_prefix", type=str, default="pred_")
    parser.add_argument("--gt_prefix", type=str, default="gt_")
    parser.add_argument("--num_procs", type=int, default=1)
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--size_limit", type=int, default=1)
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--do_new_calc", default=False, action="store_true")
    args = parser.parse_args()

    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    logger.info("="*80)
    logger.info("MD17 EVALUATION SCRIPT STARTED")
    logger.info(f"Start time: {script_start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*80)

    dir_path = args.dir_path
    pred_prefix = args.pred_prefix
    list_pred_paths = glob.glob(os.path.join(dir_path, f"{pred_prefix}*.pt"))
    # sort by file name
    list_pred_paths.sort()

    gt_prefix = args.gt_prefix
    list_gt_paths = glob.glob(os.path.join(dir_path, f"{gt_prefix}*.pt"))
    # sort by file name
    list_gt_paths.sort()

    # apply multiprocessing to process list_pred_paths and list_gt_paths
    num_procs = args.num_procs
    # Create list of (pred_path, gt_path) tuples
    file_pairs = list(zip(list_pred_paths, list_gt_paths))
    
    if args.size_limit > 0:
        file_pairs = file_pairs[:args.size_limit]

    logger.info(f"Configuration:")
    logger.info(f"  - Directory: {dir_path}")
    logger.info(f"  - Pred prefix: {pred_prefix}")
    logger.info(f"  - GT prefix: {gt_prefix}")
    logger.info(f"  - Number of processes: {num_procs}")
    logger.info(f"  - Total molecules to process: {len(file_pairs)}")
    logger.info(f"  - Debug mode: {args.debug}")
    logger.info("-"*80)

    processing_start_time = time.time()
    
    if num_procs == 1:
        logger.info("Processing molecules sequentially (single process)...")
        results = []
        iter_bar = tqdm(file_pairs, desc="Processing molecules", unit="mol")
        for i, (pred_path, gt_path) in enumerate(iter_bar):
            mol_start = time.time()
            result = process_single_molecule(pred_path, gt_path, debug=args.debug, do_new_calc=args.do_new_calc)
            results.append(result)
            elapsed = time.time() - processing_start_time
            avg_time = elapsed / (i + 1)
            remaining = avg_time * (len(file_pairs) - i - 1)
            iter_bar.set_postfix({
                'elapsed': format_time(elapsed),
                'avg': format_time(avg_time),
                'ETA': format_time(remaining)
            })
    else:
        # Process with multiprocessing
        logger.info(f"Processing molecules with {num_procs} parallel processes...")
        # Add debug flag to file_pairs for wrapper function
        file_pairs_with_debug = [(pred, gt, "ang", "pbe", "def2svp", args.debug, -1, args.do_new_calc) for pred, gt in file_pairs]
        with Pool(processes=num_procs) as pool:
            results = list(tqdm(
                pool.imap_unordered(process_single_molecule_wrapper, file_pairs_with_debug),
                total=len(file_pairs),
                desc="Processing molecules",
                unit="mol"
            ))

    processing_time = time.time() - processing_start_time
    logger.info("-"*80)
    logger.info(f"Completed processing {len(results)} molecules in {format_time(processing_time)}")
    logger.info(f"Average time per molecule: {format_time(processing_time / len(results))}")

    # Collect timing statistics
    logger.info("-"*80)
    logger.info("TIMING STATISTICS")
    logger.info("-"*80)
    
    all_timings = {}
    for result in results:
        if 'timing_info' in result:
            for key, value in result['timing_info'].items():
                if key not in all_timings:
                    all_timings[key] = []
                all_timings[key].append(value)
    
    timing_stats = {}
    for key, values in all_timings.items():
        timing_stats[key] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values)
        }
        logger.info(f"  {key:20s}: mean={format_time(np.mean(values)):>12s}, std={format_time(np.std(values)):>12s}, min={format_time(np.min(values)):>12s}, max={format_time(np.max(values)):>12s}")

    # Convert keys to list and remove non-numeric keys
    keys = list(results[0].keys())
    keys_to_remove = ["data_index", "pred_force", "gt_force", "calc_force", "pred_force_norm", "gt_force_norm", "calc_force_norm", "timing_info"]
    keys = [k for k in keys if k not in keys_to_remove]
    keys.append("error_count")

    evaluation_result = {key: [] for key in keys}
    for result in results:
        for key in result.keys():
            if key not in keys_to_remove:
                evaluation_result[key].append(result[key])

    for key in keys:
        if key == "error_count":
            evaluation_result[key] = len(evaluation_result[key])
        else:
            evaluation_result[key] = float(np.mean(evaluation_result[key]))

    # Calculate overall time
    overall_time = time.time() - overall_start_time
    script_end_datetime = datetime.now()

    # Print evaluation results
    logger.info("")
    logger.info("="*80)
    logger.info("EVALUATION RESULTS")
    logger.info("="*80)
    for key, value in evaluation_result.items():
        logger.info(f"{key}: {value}")
    logger.info("="*80)

    dataset_name = dir_path.split("/")[-1]
    # Save evaluation results
    output_file = os.path.join("./outputs", f"{dataset_name}_evaluation_results.json")
    
    # Add timing info to saved results
    evaluation_result['_timing'] = {
        'total_time_seconds': overall_time,
        'total_time_formatted': format_time(overall_time),
        'processing_time_seconds': processing_time,
        'processing_time_formatted': format_time(processing_time),
        'avg_time_per_molecule_seconds': processing_time / len(results),
        'avg_time_per_molecule_formatted': format_time(processing_time / len(results)),
        'start_time': script_start_datetime.strftime('%Y-%m-%d %H:%M:%S'),
        'end_time': script_end_datetime.strftime('%Y-%m-%d %H:%M:%S'),
        'num_molecules': len(results),
        'num_processes': num_procs,
        'timing_stats': {k: {sk: float(sv) for sk, sv in v.items()} for k, v in timing_stats.items()}
    }
    
    os.makedirs("./outputs", exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(evaluation_result, f, indent=4)
    logger.info(f"\nEvaluation results saved to: {output_file}")
    
    # Final summary
    logger.info("")
    logger.info("="*80)
    logger.info("EXECUTION SUMMARY")
    logger.info("="*80)
    logger.info(f"Start time:            {script_start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"End time:              {script_end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Total execution time:  {format_time(overall_time)}")
    logger.info(f"Processing time:       {format_time(processing_time)}")
    logger.info(f"Molecules processed:   {len(results)}")
    logger.info(f"Avg time per molecule: {format_time(processing_time / len(results))}")
    logger.info(f"Throughput:            {len(results) / (processing_time / 60):.2f} molecules/min")
    logger.info("="*80)
