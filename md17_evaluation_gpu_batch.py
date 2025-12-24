#!/usr/bin/env python
"""
GPU-optimized batch evaluation for MD17.

This version processes multiple molecules in a batch-friendly way on GPU.
Strategy: Use multiple GPU streams or distribute across available GPUs.
"""

import os
import glob
import torch
from pyscf import gto, scf, dft
import time
from escflow_eval_utils import init_pyscf_mf, calc_dm0_from_ham, matrix_transform_single
from escflow_eval_utils import BOHR2ANG, HA2meV, HA_BOHR_2_meV_ANG
import numpy as np
from tqdm import tqdm
import warnings
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set CUDA environment for GPU4PySCF before importing cupy
if 'CUDA_HOME' not in os.environ and os.path.exists('/usr/local/cuda-12.6'):
    os.environ['CUDA_HOME'] = '/usr/local/cuda-12.6'
    os.environ['CUDA_PATH'] = '/usr/local/cuda-12.6'
    os.environ['PATH'] = '/usr/local/cuda-12.6/bin:' + os.environ.get('PATH', '')
    os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda-12.6/lib64:' + os.environ.get('LD_LIBRARY_PATH', '')

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)

try:
    import cupy as cp
    GPU_AVAILABLE = cp.cuda.is_available()
    NUM_GPUS = cp.cuda.runtime.getDeviceCount() if GPU_AVAILABLE else 0
except ImportError:
    GPU_AVAILABLE = False
    NUM_GPUS = 0

def process_single_molecule_gpu(pred_file_path, gt_file_path, gpu_id=0,
                                unit="ang", xc="pbe", basis="def2svp", debug=False):
    """
    Process a single molecule on a specific GPU.

    Args:
        pred_file_path: Path to prediction file
        gt_file_path: Path to ground truth file
        gpu_id: GPU device ID to use
        unit: Unit for positions
        xc: Exchange-correlation functional
        basis: Basis set
        debug: Debug mode flag
    """
    # Set GPU device for this calculation
    if GPU_AVAILABLE and NUM_GPUS > 1:
        try:
            cp.cuda.Device(gpu_id).use()
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        except:
            pass

    dir_path = os.path.dirname(pred_file_path)
    calc_path = pred_file_path.replace("pred_", "calc_")

    pred_data = torch.load(pred_file_path, weights_only=False)
    gt_data = torch.load(gt_file_path, weights_only=False)

    data_index = gt_data["idx"]
    molecule_start_time = time.time()
    print(f"[GPU {gpu_id}] Starting molecule {data_index} at {time.strftime('%H:%M:%S')}", flush=True)

    atoms = gt_data["atoms"]
    pos = gt_data["pos"] * BOHR2ANG

    # Use GPU acceleration
    calc_mf = init_pyscf_mf(atoms, pos, unit=unit, xc=xc, basis=basis, use_gpu=True)
    grad_frame = calc_mf.nuc_grad_method()
    calc_mf.conv_tol = 1e-7
    calc_mf.grids.level = 3

    DO_NEW_CALC = True

    if os.path.exists(calc_path) and not DO_NEW_CALC:
        calc_data = torch.load(calc_path, weights_only=False)
        calc_energy = calc_data["calc_energy"]
        calc_forces = calc_data["calc_forces"]
        calc_mo_energy = calc_data["calc_mo_energy"]
        calc_mo_coeff = calc_data["calc_mo_coeff"]
        calc_ham = calc_data["hamiltonian"]
        calc_overlap = calc_data["overlap"]
    else:
        calc_data = gt_data.copy()
        start_time = time.time()
        print(f"[GPU {gpu_id}] Molecule {data_index}: Starting SCF calculation...", flush=True)
        calc_mf.kernel()
        scf_time = time.time() - start_time
        calc_data["calc_time"] = scf_time
        print(f"[GPU {gpu_id}] Molecule {data_index}: SCF completed in {scf_time:.2f}s", flush=True)

        calc_data["hamiltonian"] = torch.tensor(calc_mf.get_fock(dm=calc_mf.make_rdm1()), dtype=torch.float64)
        calc_data["overlap"] = torch.tensor(calc_mf.get_ovlp(), dtype=torch.float64)
        calc_data["density_matrix"] = torch.tensor(calc_mf.make_rdm1(), dtype=torch.float64)
        calc_data["method"] = "RKS"
        calc_data["xc"] = xc
        calc_data["basis"] = basis

        print(f"[GPU {gpu_id}] Molecule {data_index}: Computing initial forces...", flush=True)
        force_start = time.time()
        calc_data["forces"] = torch.tensor(-grad_frame.kernel(), dtype=torch.float64)
        print(f"[GPU {gpu_id}] Molecule {data_index}: Initial forces completed in {time.time() - force_start:.2f}s", flush=True)

        calc_overlap = calc_data["overlap"].unsqueeze(0)
        calc_ham = calc_data["hamiltonian"].unsqueeze(0)

        calc_density, calc_res = calc_dm0_from_ham(atoms, calc_overlap, calc_ham, transform=False)
        calc_energy = calc_mf.energy_tot(calc_density)
        calc_data["calc_energy"] = calc_energy

        calc_mo_energy = calc_res["orbital_energies"].squeeze().numpy()
        calc_mo_coeff = calc_res["orbital_coefficients"].squeeze().numpy()
        calc_data["calc_mo_energy"] = calc_mo_energy
        calc_data["calc_mo_coeff"] = calc_mo_coeff

        mo_occ = calc_mf.get_occ(calc_mo_energy, calc_mo_coeff)
        calc_data["mo_occ"] = mo_occ
        print(f"[GPU {gpu_id}] Molecule {data_index}: Computing calc forces with MO...", flush=True)
        force_start = time.time()
        calc_forces = -grad_frame.kernel(mo_energy=calc_mo_energy, mo_coeff=calc_mo_coeff, mo_occ=mo_occ)
        print(f"[GPU {gpu_id}] Molecule {data_index}: Calc forces completed in {time.time() - force_start:.2f}s", flush=True)
        calc_data["calc_forces"] = calc_forces

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
            transform=False
        )
        pred_energy = calc_mf.energy_tot(pred_density)
        pred_data["calc_energy"] = pred_energy

        gt_density, gt_res = calc_dm0_from_ham(
            atoms=atoms,
            overlap=gt_overlap,
            hamiltonian=gt_ham,
            transform=False
        )
        gt_energy = calc_mf.energy_tot(gt_density)
        gt_data["calc_energy"] = gt_energy

        pred_mo_energy = pred_res["orbital_energies"].squeeze().numpy()
        pred_mo_coeff = pred_res["orbital_coefficients"].squeeze().numpy()
        pred_data["calc_mo_energy"] = pred_mo_energy
        pred_data["calc_mo_coeff"] = pred_mo_coeff

        gt_mo_energy = gt_res["orbital_energies"].squeeze().numpy()
        gt_mo_coeff = gt_res["orbital_coefficients"].squeeze().numpy()
        gt_data["calc_mo_energy"] = gt_mo_energy
        gt_data["calc_mo_coeff"] = gt_mo_coeff

        pred_mo_occ = calc_mf.get_occ(pred_mo_energy, pred_mo_coeff)
        pred_data["mo_occ"] = pred_mo_occ
        print(f"[GPU {gpu_id}] Molecule {data_index}: Computing pred forces...", flush=True)
        force_start = time.time()
        pred_forces = -grad_frame.kernel(mo_energy=pred_mo_energy, mo_coeff=-pred_mo_coeff, mo_occ=pred_mo_occ)
        print(f"[GPU {gpu_id}] Molecule {data_index}: Pred forces completed in {time.time() - force_start:.2f}s", flush=True)
        pred_data["calc_forces"] = pred_forces

        gt_mo_occ = calc_mf.get_occ(gt_mo_energy, gt_mo_coeff)
        gt_data["mo_occ"] = gt_mo_occ
        print(f"[GPU {gpu_id}] Molecule {data_index}: Computing gt forces...", flush=True)
        force_start = time.time()
        gt_forces = -grad_frame.kernel(mo_energy=gt_mo_energy, mo_coeff=-gt_mo_coeff, mo_occ=gt_mo_occ)
        print(f"[GPU {gpu_id}] Molecule {data_index}: GT forces completed in {time.time() - force_start:.2f}s", flush=True)
        gt_data["calc_forces"] = gt_forces

        if not debug:
            torch.save(pred_data, pred_file_path)
            torch.save(gt_data, gt_file_path)

    pred_forces_norm = np.linalg.norm(pred_forces, axis=1)
    calc_forces_norm = np.linalg.norm(calc_forces, axis=1)
    gt_forces_norm = np.linalg.norm(gt_forces, axis=1)

    num_occ = int(gt_data["atoms"].sum() / 2)

    pred_mo_energy_occ = pred_mo_energy[:num_occ]
    gt_mo_energy_occ = gt_mo_energy[:num_occ]
    calc_mo_energy_occ = calc_mo_energy[:num_occ]
    pred_mo_occ_coeff = pred_res["sliced_orbital_coefficients"]
    gt_mo_occ_coeff = gt_res["sliced_orbital_coefficients"]
    calc_mo_occ_coeff = calc_res["sliced_orbital_coefficients"]

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
        "orbital_coeff_similarity (pred-gt)": torch.cosine_similarity(pred_mo_occ_coeff, gt_mo_occ_coeff, dim=1).abs().mean(),
        "orbital_coeff_similarity (pred-calc)": torch.cosine_similarity(pred_mo_occ_coeff, calc_mo_occ_coeff, dim=1).abs().mean(),
        "orbital_coeff_similarity (gt-calc)": torch.cosine_similarity(gt_mo_occ_coeff, calc_mo_occ_coeff, dim=1).abs().mean(),
        "occupied_orbital_energy_mae (pred-gt)": np.abs(pred_mo_energy_occ - gt_mo_energy_occ).mean(),
        "occupied_orbital_energy_mae (pred-calc)": np.abs(pred_mo_energy_occ - calc_mo_energy_occ).mean(),
        "occupied_orbital_energy_mae (gt-calc)": np.abs(gt_mo_energy_occ - calc_mo_energy_occ).mean(),
        "overlap_diff (gt-calc)": np.abs(gt_overlap - calc_overlap).mean(),
    }

    total_time = time.time() - molecule_start_time
    print(f"[GPU {gpu_id}] Molecule {data_index}: COMPLETED in {total_time:.2f}s total", flush=True)
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GPU-batched MD17 evaluation")
    parser.add_argument("--dir_path", type=str, default="/nas/seongjun/sphnet/aspirin/output_dump_batch")
    parser.add_argument("--pred_prefix", type=str, default="pred_")
    parser.add_argument("--gt_prefix", type=str, default="gt_")
    parser.add_argument("--batch_size", type=int, default=8, help="Number of molecules to process in parallel (recommended: num_gpus)")
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--size_limit", type=int, default=0)
    args = parser.parse_args()

    print(f"GPU Available: {GPU_AVAILABLE}")
    print(f"Number of GPUs: {NUM_GPUS}")

    if not GPU_AVAILABLE:
        print("ERROR: No GPU available. Use md17_evaluation_customv2.py with --use_gpu for single GPU.")
        sys.exit(1)

    dir_path = args.dir_path
    pred_prefix = args.pred_prefix
    list_pred_paths = glob.glob(os.path.join(dir_path, f"{pred_prefix}*.pt"))
    list_pred_paths.sort()

    gt_prefix = args.gt_prefix
    list_gt_paths = glob.glob(os.path.join(dir_path, f"{gt_prefix}*.pt"))
    list_gt_paths.sort()

    file_pairs = list(zip(list_pred_paths, list_gt_paths))

    if args.debug:
        args.size_limit = args.batch_size
    if args.size_limit > 0:
        file_pairs = file_pairs[:args.size_limit]

    batch_size = min(args.batch_size, NUM_GPUS) if NUM_GPUS > 0 else 1

    print(f"\n{'='*80}")
    print(f"GPU-Batched Evaluation Configuration")
    print(f"{'='*80}")
    print(f"Total molecules: {len(file_pairs)}")
    print(f"Batch size: {batch_size} (parallel molecules)")
    print(f"GPUs used: {batch_size if NUM_GPUS > 0 else 1}")
    print(f"{'='*80}\n")

    # Process in batches using ThreadPoolExecutor
    # Each thread handles one molecule on a specific GPU
    results = []
    total_start = time.time()

    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        futures = []
        pbar = tqdm(total=len(file_pairs), desc="Processing molecules")

        for idx, (pred_path, gt_path) in enumerate(file_pairs):
            gpu_id = idx % NUM_GPUS if NUM_GPUS > 1 else 0
            future = executor.submit(
                process_single_molecule_gpu,
                pred_path,
                gt_path,
                gpu_id,
                debug=args.debug
            )
            futures.append(future)

            # Process completed futures
            if len(futures) >= batch_size or idx == len(file_pairs) - 1:
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        results.append(result)
                        pbar.update(1)
                    except Exception as e:
                        print(f"Error processing molecule: {e}")
                futures = []

        pbar.close()

    total_time = time.time() - total_start
    print(f"\nCompleted processing {len(results)} molecules in {total_time:.2f}s")
    print(f"Average time per molecule: {total_time/len(results):.2f}s")

    # Aggregate results
    keys = list(results[0].keys())
    keys_to_remove = ["data_index", "pred_force", "gt_force", "calc_force",
                      "pred_force_norm", "gt_force_norm", "calc_force_norm"]
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

    # Print results
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    for key, value in evaluation_result.items():
        print(f"{key}: {value}")
    print("="*80)

    dataset_name = dir_path.split("/")[-2]
    output_file = os.path.join("./outputs", f"{dataset_name}_gpu_batch_evaluation_results.json")
    os.makedirs("./outputs", exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(evaluation_result, f, indent=4)
    print(f"\nEvaluation results saved to: {output_file}")
