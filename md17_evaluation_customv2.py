import os
import glob
import torch
from pyscf import gto, scf, dft
import time
from escflow_eval_utils import init_pyscf_mf, calc_dm0_from_ham, matrix_transform_single
from escflow_eval_utils import BOHR2ANG, HA2meV, HA_BOHR_2_meV_ANG
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
import warnings
import json
import sys

# Suppress FutureWarning about torch.load
warnings.filterwarnings('ignore', category=FutureWarning)

def _convert_density_for_gpu(density, use_gpu):
    """Convert density matrix to CuPy array if using GPU, otherwise return as-is."""
    if use_gpu >= 0:
        try:
            import cupy as cp
            # If density is a tensor, convert to numpy first
            if hasattr(density, 'cpu'):
                density_np = density.cpu().numpy()
            else:
                density_np = density
            return cp.asarray(density_np)
        except Exception as e:
            print(f"Warning: Failed to convert density to CuPy: {e}, using CPU")
            return density.cpu().numpy() if hasattr(density, 'cpu') else density
    else:
        return density

def process_single_molecule(pred_file_path, gt_file_path,
    unit="ang", xc="pbe", basis="def2svp", debug=False, use_gpu=-1
):
    dir_path = os.path.dirname(pred_file_path)
    calc_path = pred_file_path.replace("pred_", "calc_")

    pred_data = torch.load(pred_file_path, weights_only=False)
    gt_data = torch.load(gt_file_path, weights_only=False)

    data_index = gt_data["idx"]
    molecule_start_time = time.time()
    print(f"[Process {os.getpid()}] Starting molecule {data_index} at {time.strftime('%H:%M:%S')}", flush=True)

    atoms = gt_data["atoms"]
    pos = gt_data["pos"] * BOHR2ANG
    calc_mf = init_pyscf_mf(atoms, pos, unit=unit, xc=xc, basis=basis, use_gpu=use_gpu)
    grad_frame = calc_mf.nuc_grad_method()
    calc_mf.conv_tol = 1e-7
    calc_mf.grids.level = 3
    calc_mf.grids.prune = None
    calc_mf.init_guess = "minao"
    calc_mf.small_rho_cutoff = 1e-12

    DO_NEW_CALC = False
    #try:
    # Check if calculated data exists
    if os.path.exists(calc_path) and not DO_NEW_CALC:
        calc_data = torch.load(calc_path, weights_only=False)
        calc_energy = calc_data["calc_energy"]
        calc_forces = calc_data["calc_forces"]
        calc_mo_energy = calc_data["calc_mo_energy"]
        calc_mo_coeff = calc_data["calc_mo_coeff"]
        calc_ham = calc_data["hamiltonian"]
        calc_overlap = calc_data["overlap"]
        
    else:
        calc_data = gt_data.copy()  # Use copy to avoid modifying original
        start_time = time.time()
        print(f"[Process {os.getpid()}] Molecule {data_index}: Starting SCF calculation...", flush=True)
        calc_mf.kernel()
        scf_time = time.time() - start_time
        calc_data["calc_time"] = scf_time
        print(f"[Process {os.getpid()}] Molecule {data_index}: SCF completed in {scf_time:.2f}s", flush=True)
        calc_data["hamiltonian"] = torch.tensor(calc_mf.get_fock(dm=calc_mf.make_rdm1()), dtype=torch.float64)
        calc_data["overlap"] = torch.tensor(calc_mf.get_ovlp(), dtype=torch.float64)
        calc_data["density_matrix"] = torch.tensor(calc_mf.make_rdm1(), dtype=torch.float64)
        calc_data["method"] = "RKS"
        calc_data["xc"] = xc
        calc_data["basis"] = basis
        # calc_data["scf_cycles"] = calc_mf.cycles
        print(f"[Process {os.getpid()}] Molecule {data_index}: Computing initial forces...", flush=True)
        force_start = time.time()
        calc_data["forces"] = torch.tensor(-grad_frame.kernel(), dtype=torch.float64)
        print(f"[Process {os.getpid()}] Molecule {data_index}: Initial forces completed in {time.time() - force_start:.2f}s", flush=True)

        calc_overlap = calc_data["overlap"].unsqueeze(0) # (gt_overlap - calc_overlap) has float32 precision error (1e^-7)
        calc_ham = calc_data["hamiltonian"].unsqueeze(0)

        calc_density, calc_res = calc_dm0_from_ham(atoms, calc_overlap, calc_ham, transform=False, return_tensor=(use_gpu >= 0))
        calc_density_converted = _convert_density_for_gpu(calc_density, use_gpu)
        calc_energy = calc_mf.energy_tot(calc_density_converted)
        calc_data["calc_energy"] = calc_energy

        calc_mo_energy = calc_res["orbital_energies"].squeeze().numpy()
        calc_mo_coeff = calc_res["orbital_coefficients"].squeeze().numpy()
        calc_data["calc_mo_energy"] = calc_mo_energy
        calc_data["calc_mo_coeff"] = calc_mo_coeff

        mo_occ = calc_mf.get_occ(calc_mo_energy, calc_mo_coeff)
        calc_data["mo_occ"] = mo_occ
        print(f"[Process {os.getpid()}] Molecule {data_index}: Computing calc forces with MO...", flush=True)
        force_start = time.time()
        calc_forces = -grad_frame.kernel(mo_energy=calc_mo_energy, mo_coeff=calc_mo_coeff, mo_occ=mo_occ)
        print(f"[Process {os.getpid()}] Molecule {data_index}: Calc forces completed in {time.time() - force_start:.2f}s", flush=True)
        calc_data["calc_forces"] = calc_forces

        #save calc_data
        if not debug:
            torch.save(calc_data, calc_path)
    
    if "remove_init" not in gt_data.keys():
        remove_init = True
        gt_data["remove_init"] = remove_init
        pred_data["remove_init"] = remove_init
    else:
        remove_init = gt_data["remove_init"]

    if "calc_force" in pred_data and "calc_force" in gt_data and not DO_NEW_CALC:
        pred_energy = pred_data["calc_energy"]
        pred_forces = pred_data["calc_forces"]
        pred_mo_energy = pred_data["calc_mo_energy"]
        pred_mo_coeff = pred_data["calc_mo_coeff"]

        gt_energy = gt_data["calc_energy"]
        gt_forces = gt_data["calc_forces"]
        gt_mo_energy = gt_data["calc_mo_energy"]
        gt_mo_coeff = gt_data["calc_mo_coeff"]
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
        pred_density_converted = _convert_density_for_gpu(pred_density, use_gpu)
        pred_energy = calc_mf.energy_tot(pred_density_converted)
        pred_data["calc_energy"] = pred_energy

        gt_density, gt_res = calc_dm0_from_ham(
            atoms=atoms,
            overlap=gt_overlap,
            hamiltonian=gt_ham,
            transform=False,
            return_tensor=(use_gpu >= 0)
            )
        gt_density_converted = _convert_density_for_gpu(gt_density, use_gpu)
        gt_energy = calc_mf.energy_tot(gt_density_converted)
        gt_data["calc_energy"] = gt_energy
        
        pred_mo_energy = pred_res["orbital_energies"].squeeze().cpu().numpy() if hasattr(pred_res["orbital_energies"], 'cpu') else pred_res["orbital_energies"].squeeze().numpy()
        pred_mo_coeff = pred_res["orbital_coefficients"].squeeze().cpu().numpy() if hasattr(pred_res["orbital_coefficients"], 'cpu') else pred_res["orbital_coefficients"].squeeze().numpy()
        # Ensure correct dtype for gradient calculation
        pred_mo_energy = np.asarray(pred_mo_energy, dtype=np.float64)
        pred_mo_coeff = np.asarray(pred_mo_coeff, dtype=np.float64)
        pred_data["calc_mo_energy"] = pred_mo_energy
        pred_data["calc_mo_coeff"] = pred_mo_coeff

        gt_mo_energy = gt_res["orbital_energies"].squeeze().cpu().numpy() if hasattr(gt_res["orbital_energies"], 'cpu') else gt_res["orbital_energies"].squeeze().numpy()
        gt_mo_coeff = gt_res["orbital_coefficients"].squeeze().cpu().numpy() if hasattr(gt_res["orbital_coefficients"], 'cpu') else gt_res["orbital_coefficients"].squeeze().numpy()
        # Ensure correct dtype for gradient calculation
        gt_mo_energy = np.asarray(gt_mo_energy, dtype=np.float64)
        gt_mo_coeff = np.asarray(gt_mo_coeff, dtype=np.float64)
        gt_data["calc_mo_energy"] = gt_mo_energy
        gt_data["calc_mo_coeff"] = gt_mo_coeff

        pred_mo_occ = calc_mf.get_occ(pred_mo_energy, pred_mo_coeff)
        pred_data["mo_occ"] = pred_mo_occ
        print(f"[Process {os.getpid()}] Molecule {data_index}: Computing pred forces...", flush=True)
        force_start = time.time()
        # Convert to NumPy on CPU; handle both NumPy and CuPy arrays

        pred_forces = -grad_frame.kernel(
            mo_energy=pred_mo_energy,
            mo_coeff=-pred_mo_coeff,
            mo_occ=pred_mo_occ,
        )
        print(f"[Process {os.getpid()}] Molecule {data_index}: Pred forces completed in {time.time() - force_start:.2f}s", flush=True)
        pred_data["calc_forces"] = pred_forces

        gt_mo_occ = calc_mf.get_occ(gt_mo_energy, gt_mo_coeff)
        gt_data["mo_occ"] = gt_mo_occ
        print(f"[Process {os.getpid()}] Molecule {data_index}: Computing gt forces...", flush=True)
        force_start = time.time()

        gt_forces = -grad_frame.kernel(
            mo_energy=gt_mo_energy,
            mo_coeff=-gt_mo_coeff,
            mo_occ=gt_mo_occ,
        )
        print(f"[Process {os.getpid()}] Molecule {data_index}: GT forces completed in {time.time() - force_start:.2f}s", flush=True)
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
    # For GPU mode, bring occupied energies back to CPU numpy for metric math
    # if use_gpu:
    #     pred_mo_energy_occ = cp.asnumpy(pred_mo_energy_occ)
    #     gt_mo_energy_occ = cp.asnumpy(gt_mo_energy_occ)
    #     calc_mo_energy_occ = np.asarray(calc_mo_energy_occ, dtype=np.float64)
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

        # "orbital_coeff_similarity (pred-gt)": torch.cosine_similarity(torch.tensor(pred_mo_occ_coeff), torch.tensor(gt_mo_occ_coeff), dim=1).abs().mean(),
        # "orbital_coeff_similarity (pred-calc)": torch.cosine_similarity(torch.tensor(pred_mo_occ_coeff), torch.tensor(calc_mo_occ_coeff), dim=1).abs().mean(),
        # "orbital_coeff_similarity (gt-calc)": torch.cosine_similarity(torch.tensor(gt_mo_occ_coeff), torch.tensor(calc_mo_occ_coeff), dim=1).abs().mean(),


        "orbital_coeff_similarity (pred-gt)": torch.cosine_similarity(pred_mo_occ_coeff, gt_mo_occ_coeff, dim=0).abs().mean(),
        "orbital_coeff_similarity (pred-calc)": torch.cosine_similarity(pred_mo_occ_coeff, calc_mo_occ_coeff, dim=0).abs().mean(),
        "orbital_coeff_similarity (gt-calc)": torch.cosine_similarity(gt_mo_occ_coeff, calc_mo_occ_coeff, dim=0).abs().mean(),


        "occupied_orbital_energy_mae (pred-gt)": np.abs(pred_mo_energy_occ - gt_mo_energy_occ).mean(),
        "occupied_orbital_energy_mae (pred-calc)": np.abs(pred_mo_energy_occ - calc_mo_energy_occ).mean(),
        "occupied_orbital_energy_mae (gt-calc)": np.abs(gt_mo_energy_occ - calc_mo_energy_occ).mean(),

        # "occupied_orbital_energy_mae (pred-gt)": torch.abs(pred_mo_energy_occ - gt_mo_energy_occ).mean().item(),
        # "occupied_orbital_energy_mae (pred-calc)": torch.abs(pred_mo_energy_occ - calc_mo_energy_occ).mean().item(),
        # "occupied_orbital_energy_mae (gt-calc)": torch.abs(gt_mo_energy_occ - calc_mo_energy_occ).mean().item(),

        "overlap_diff (gt-calc)": np.abs(gt_overlap - calc_overlap).mean(),
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
    print(f"[Process {os.getpid()}] Molecule {data_index}: COMPLETED in {total_time:.2f}s total", flush=True)
    return result



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_path", type=str, default="/nas/seongjun/sphnet/aspirin/output_dump_batch")
    parser.add_argument("--pred_prefix", type=str, default="pred_")
    parser.add_argument("--gt_prefix", type=str, default="gt_")
    parser.add_argument("--num_procs", type=int, default=1)
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--size_limit", type=int, default=1)
    parser.add_argument("--use_gpu", type=int, default=-1, help="GPU device ID to use (-1 for CPU, 0-7 for specific GPU)")
    args = parser.parse_args()

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

    if args.use_gpu >= 0:
        print(f"GPU mode enabled: Using GPU {args.use_gpu}")
        if num_procs > 1:
            print("Warning: GPU mode works best with --num_procs=1. Multiple processes may compete for GPU resources.")
        print(f"Processing {len(file_pairs)} molecules with GPU {args.use_gpu} and {num_procs} processes...")
    else:
        print(f"Processing {len(file_pairs)} molecules with CPU and {num_procs} processes...")

    if  num_procs == 1:
        iter_bar = tqdm(file_pairs, desc="Processing molecules")
        results = [process_single_molecule(pred_path, gt_path, debug=args.debug, use_gpu=args.use_gpu) for pred_path, gt_path in iter_bar]
    else:
        # Process with multiprocessing
        # Note: For GPU mode, create a wrapper to pass use_gpu flag
        from functools import partial
        process_func = partial(process_single_molecule, debug=args.debug, use_gpu=args.use_gpu)
        with Pool(processes=num_procs) as pool:
            results = list(tqdm(
                pool.starmap(process_func, file_pairs),
            total=len(file_pairs),
            desc="Processing molecules"
        ))

    print(f"\nCompleted processing {len(results)} molecules")

    # Convert keys to list and remove non-numeric keys
    keys = list(results[0].keys())
    keys_to_remove = ["data_index", "pred_force", "gt_force", "calc_force", "pred_force_norm", "gt_force_norm", "calc_force_norm"]
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


    # Print evaluation results
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    for key, value in evaluation_result.items():
        print(f"{key}: {value}")
    print("="*80)

    dir_dir_path = os.path.dirname(dir_path)
    dataset_name = dir_path.split("/")[-2]
    # Save evaluation results
    output_file = os.path.join("./outputs", f"{dataset_name}_evaluation_results.json")
    with open(output_file, "w") as f:
        json.dump(evaluation_result, f, indent=4)
    print(f"\nEvaluation results saved to: {output_file}")
