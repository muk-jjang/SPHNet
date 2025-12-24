#!/usr/bin/env python
"""
특정 분자 ID로 개별 평가를 수행하는 스크립트.

사용법:
    # 단일 분자 평가
    python md17_evaluate_single.py --dir_path outputs/uracil_split_25000_500_4500_pbe0/output_dump --mol_id mol10193
    
    # 여러 분자 평가
    python md17_evaluate_single.py --dir_path outputs/uracil_split_25000_500_4500_pbe0/output_dump --mol_ids mol10193 mol10194 mol10195
    
    # 숫자만 입력해도 자동으로 mol prefix 추가
    python md17_evaluate_single.py --dir_path outputs/uracil_split_25000_500_4500_pbe0/output_dump --mol_id 10193
"""

import os
import glob
import torch
from pyscf import gto, scf, dft
import time
from datetime import datetime
from escflow_eval_utils import init_pyscf_mf, calc_dm0_from_ham, matrix_transform_single
from escflow_eval_utils import BOHR2ANG, HA2meV, HA_BOHR_2_meV_ANG
import numpy as np
import warnings
import json
import argparse
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

# Dataset lists
md17_dataset_list = ['ethanol', 'malondialdehyde', 'uracil']
rmd17_dataset_list = ['aspirin', 'naphthalene', 'salicylic_acid']


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


def get_file_paths(dir_path, mol_id):
    """Get pred, gt, calc file paths for a molecule"""
    dir_name = dir_path.rstrip('/').split("/")[-2]
    
    # 부분 문자열로 매칭
    is_md17 = any(ds in dir_name for ds in md17_dataset_list)
    is_rmd17 = any(ds in dir_name for ds in rmd17_dataset_list)
    
    if is_md17:
        pred_path = os.path.join(dir_path, f"pred_{mol_id}.pt")
        gt_path = os.path.join(dir_path, f"gt_{mol_id}.pt")
        calc_path = os.path.join(dir_path, f"calc_{mol_id}.pt")
    elif is_rmd17:
        pred_path = os.path.join(dir_path, f"pred_batch0_{mol_id}.pt")
        gt_path = os.path.join(dir_path, f"gt_batch0_{mol_id}.pt")
        calc_path = os.path.join(dir_path, f"calc_batch0_{mol_id}.pt")
    else:
        raise ValueError(f"Unknown dataset type in path: {dir_name}. Expected one of {md17_dataset_list + rmd17_dataset_list}")
    
    return pred_path, gt_path, calc_path


def evaluate_single_molecule(dir_path, mol_id, unit="ang", xc="pbe", basis="def2svp", 
                              force_recalc=False, verbose=True):
    """
    단일 분자에 대한 평가 수행
    
    Args:
        dir_path: output_dump 디렉토리 경로
        mol_id: 분자 ID (예: mol10193)
        unit: 거리 단위
        xc: exchange-correlation functional
        basis: basis set
        force_recalc: 캐시된 계산 무시하고 재계산
        verbose: 상세 출력 여부
    
    Returns:
        result: 평가 결과 딕셔너리
    """
    pred_path, gt_path, calc_path = get_file_paths(dir_path, mol_id)
    
    # Check if files exist
    if not os.path.exists(pred_path):
        logger.error(f"Pred file not found: {pred_path}")
        return None
    if not os.path.exists(gt_path):
        logger.error(f"GT file not found: {gt_path}")
        return None
    
    timing_info = {}
    molecule_start_time = time.time()
    
    # Load data
    load_start = time.time()
    pred_data = torch.load(pred_path, weights_only=False)
    gt_data = torch.load(gt_path, weights_only=False)
    timing_info['data_load'] = time.time() - load_start
    
    data_index = gt_data.get("idx", mol_id)
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"🔬 분자 평가: {mol_id} (index: {data_index})")
        print(f"{'='*80}")
    
    atoms = gt_data["atoms"]
    pos = gt_data["pos"] * BOHR2ANG
    
    # Initialize PySCF
    init_start = time.time()
    calc_mf = init_pyscf_mf(atoms, pos, unit=unit, xc=xc, basis=basis)
    grad_frame = calc_mf.nuc_grad_method()
    calc_mf.conv_tol = 1e-7
    calc_mf.grids.level = 3
    calc_mf.grids.prune = None
    calc_mf.init_guess = "minao"
    calc_mf.small_rho_cutoff = 1e-12
    timing_info['pyscf_init'] = time.time() - init_start
    
    # Load or compute calc data
    if os.path.exists(calc_path) and not force_recalc:
        if verbose:
            print(f"\n📂 캐시된 calc 데이터 로드: {calc_path}")
        calc_data = torch.load(calc_path, weights_only=False)
        calc_energy = calc_data["calc_energy"]
        calc_forces = calc_data["calc_forces"]
        calc_mo_energy = calc_data["calc_mo_energy"]
        calc_mo_coeff = calc_data["calc_mo_coeff"]
        calc_ham = calc_data["hamiltonian"]
        calc_overlap = calc_data["overlap"]
    else:
        if verbose:
            print(f"\n⚙️ SCF 계산 수행 중...")
        calc_data = gt_data.copy()
        scf_start = time.time()
        calc_mf.kernel()
        timing_info['scf'] = time.time() - scf_start
        
        calc_data["hamiltonian"] = torch.tensor(calc_mf.get_fock(dm=calc_mf.make_rdm1()), dtype=torch.float64)
        calc_data["overlap"] = torch.tensor(calc_mf.get_ovlp(), dtype=torch.float64)
        calc_data["density_matrix"] = torch.tensor(calc_mf.make_rdm1(), dtype=torch.float64)
        
        force_start = time.time()
        calc_data["forces"] = torch.tensor(-grad_frame.kernel(), dtype=torch.float64)
        timing_info['initial_forces'] = time.time() - force_start
        
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
        
        force_start = time.time()
        calc_forces = -grad_frame.kernel(mo_energy=calc_mo_energy, mo_coeff=calc_mo_coeff, mo_occ=mo_occ)
        timing_info['calc_forces'] = time.time() - force_start
        calc_data["calc_forces"] = calc_forces
        
        # Save calc_data
        # torch.save(calc_data, calc_path)
        if verbose:
            print(f"   💾 calc 데이터 저장: {calc_path}")
    
    # Get remove_init flag
    remove_init = gt_data.get("remove_init", True)
    
    # Always compute hamiltonian transforms for metrics (needed for hamiltonian_diff, orbital_coeff_similarity)
    gt_overlap_raw = gt_data["overlap"]
    if isinstance(calc_overlap, torch.Tensor):
        gt_overlap = torch.from_numpy(gt_overlap_raw).reshape(calc_overlap.shape)
    else:
        gt_overlap = torch.from_numpy(gt_overlap_raw).reshape(calc_overlap.shape)
    gt_overlap = matrix_transform_single(gt_overlap, atoms, convention="back2pyscf")
    
    pred_hamiltonian = pred_data["pred_hamiltonian"] + remove_init * gt_data["init_ham"].reshape(pred_data["pred_hamiltonian"].shape)
    pred_ham = matrix_transform_single(pred_hamiltonian.unsqueeze(0), atoms, convention="back2pyscf")
    
    gt_hamiltonian = gt_data["hamiltonian"] + remove_init * gt_data["init_ham"].reshape(gt_data["hamiltonian"].shape)
    gt_ham = matrix_transform_single(gt_hamiltonian.unsqueeze(0), atoms, convention="back2pyscf")
    
    # Ensure calc_ham has correct shape
    if isinstance(calc_ham, torch.Tensor) and calc_ham.dim() == 2:
        calc_ham = calc_ham.unsqueeze(0)
    elif not isinstance(calc_ham, torch.Tensor):
        calc_ham = torch.tensor(calc_ham).unsqueeze(0)
    
    # Check if forces already calculated
    if "calc_forces" in pred_data and "calc_forces" in gt_data and not force_recalc:
        pred_energy = pred_data["calc_energy"]
        pred_forces = pred_data["calc_forces"]
        pred_mo_energy = pred_data["calc_mo_energy"]
        pred_mo_coeff = pred_data["calc_mo_coeff"]
        
        gt_energy = gt_data["calc_energy"]
        gt_forces = gt_data["calc_forces"]
        gt_mo_energy = gt_data["calc_mo_energy"]
        gt_mo_coeff = gt_data["calc_mo_coeff"]
        
        # Compute orbital coefficients for similarity metrics
        _, pred_res = calc_dm0_from_ham(atoms=atoms, overlap=gt_overlap, hamiltonian=pred_ham, transform=False)
        _, gt_res = calc_dm0_from_ham(atoms=atoms, overlap=gt_overlap, hamiltonian=gt_ham, transform=False)
        _, calc_res = calc_dm0_from_ham(atoms=atoms, overlap=calc_overlap if isinstance(calc_overlap, torch.Tensor) else torch.tensor(calc_overlap).unsqueeze(0), hamiltonian=calc_ham, transform=False)
        
        if verbose:
            print(f"\n📂 캐시된 pred/gt forces 사용")
    else:
        if verbose:
            print(f"\n⚙️ Pred/GT forces 계산 중...")
        
        # Pred
        pred_density, pred_res = calc_dm0_from_ham(atoms=atoms, overlap=gt_overlap, hamiltonian=pred_ham, transform=False)
        pred_energy = calc_mf.energy_tot(pred_density)
        pred_data["calc_energy"] = pred_energy
        
        pred_mo_energy = pred_res["orbital_energies"].squeeze().numpy()
        pred_mo_coeff = pred_res["orbital_coefficients"].squeeze().numpy()
        pred_data["calc_mo_energy"] = pred_mo_energy
        pred_data["calc_mo_coeff"] = pred_mo_coeff
        
        pred_mo_occ = calc_mf.get_occ(pred_mo_energy, pred_mo_coeff)
        pred_data["mo_occ"] = pred_mo_occ
        
        force_start = time.time()
        pred_forces = -grad_frame.kernel(mo_energy=pred_mo_energy, mo_coeff=-pred_mo_coeff, mo_occ=pred_mo_occ)
        timing_info['pred_forces'] = time.time() - force_start
        pred_data["calc_forces"] = pred_forces
        
        # GT
        gt_density, gt_res = calc_dm0_from_ham(atoms=atoms, overlap=gt_overlap, hamiltonian=gt_ham, transform=False)
        gt_energy = calc_mf.energy_tot(gt_density)
        gt_data["calc_energy"] = gt_energy
        
        gt_mo_energy = gt_res["orbital_energies"].squeeze().numpy()
        gt_mo_coeff = gt_res["orbital_coefficients"].squeeze().numpy()
        gt_data["calc_mo_energy"] = gt_mo_energy
        gt_data["calc_mo_coeff"] = gt_mo_coeff
        
        gt_mo_occ = calc_mf.get_occ(gt_mo_energy, gt_mo_coeff)
        gt_data["mo_occ"] = gt_mo_occ
        
        force_start = time.time()
        gt_forces = -grad_frame.kernel(mo_energy=gt_mo_energy, mo_coeff=-gt_mo_coeff, mo_occ=gt_mo_occ)
        timing_info['gt_forces'] = time.time() - force_start
        gt_data["calc_forces"] = gt_forces
        
        # Calc orbital coefficients
        _, calc_res = calc_dm0_from_ham(atoms=atoms, overlap=calc_overlap if isinstance(calc_overlap, torch.Tensor) else torch.tensor(calc_overlap).unsqueeze(0), hamiltonian=calc_ham, transform=False)
        
        # Save updated data
        # torch.save(pred_data, pred_path)
        # torch.save(gt_data, gt_path)
        if verbose:
            print(f"   💾 pred/gt 데이터 저장 완료")
    
    # Compute metrics
    pred_forces_norm = np.linalg.norm(pred_forces, axis=1)
    calc_forces_norm = np.linalg.norm(calc_forces, axis=1)
    gt_forces_norm = np.linalg.norm(gt_forces, axis=1)
    
    num_occ = int(gt_data["atoms"].sum() / 2)
    
    # Extract occupied orbital energies
    pred_mo_energy_occ = pred_mo_energy[:num_occ]
    gt_mo_energy_occ = gt_mo_energy[:num_occ]
    calc_mo_energy_occ = calc_mo_energy[:num_occ]
    
    # Extract occupied orbital coefficients for similarity
    pred_mo_occ_coeff = pred_res["sliced_orbital_coefficients"]
    gt_mo_occ_coeff = gt_res["sliced_orbital_coefficients"]
    calc_mo_occ_coeff = calc_res["sliced_orbital_coefficients"]
    
    # Compute orbital coefficient similarity
    orbital_coeff_sim_pred_gt = torch.cosine_similarity(pred_mo_occ_coeff, gt_mo_occ_coeff, dim=0).abs().mean().item()
    orbital_coeff_sim_pred_calc = torch.cosine_similarity(pred_mo_occ_coeff, calc_mo_occ_coeff, dim=0).abs().mean().item()
    orbital_coeff_sim_gt_calc = torch.cosine_similarity(gt_mo_occ_coeff, calc_mo_occ_coeff, dim=0).abs().mean().item()
    
    # Compute hamiltonian differences
    ham_diff_pred_gt = float(abs(pred_ham - gt_ham).mean())
    ham_diff_pred_calc = float(abs(pred_ham - calc_ham).mean())
    ham_diff_gt_calc = float(abs(gt_ham - calc_ham).mean())
    
    # Compute overlap difference
    if isinstance(calc_overlap, torch.Tensor):
        calc_overlap_for_diff = calc_overlap
    else:
        calc_overlap_for_diff = torch.tensor(calc_overlap)
    overlap_diff_gt_calc = float(np.abs(gt_overlap.numpy() - calc_overlap_for_diff.numpy()).mean())
    
    total_time = time.time() - molecule_start_time
    timing_info['total'] = total_time
    
    result = {
        "mol_id": mol_id,
        "data_index": data_index,
        
        # Hamiltonian differences
        "hamiltonian_diff_pred_gt": ham_diff_pred_gt,
        "hamiltonian_diff_pred_calc": ham_diff_pred_calc,
        "hamiltonian_diff_gt_calc": ham_diff_gt_calc,
        
        # Energies
        "pred_energy": float(pred_energy),
        "gt_energy": float(gt_energy),
        "calc_energy": float(calc_energy),
        
        # Energy differences (in Hartree)
        "energy_diff_pred_gt": float(abs(pred_energy - gt_energy)),
        "energy_diff_pred_calc": float(abs(pred_energy - calc_energy)),
        "energy_diff_gt_calc": float(abs(gt_energy - calc_energy)),
        
        # Energy differences (in meV)
        "energy_diff_pred_gt_meV": float(abs(pred_energy - gt_energy) * HA2meV),
        "energy_diff_pred_calc_meV": float(abs(pred_energy - calc_energy) * HA2meV),
        "energy_diff_gt_calc_meV": float(abs(gt_energy - calc_energy) * HA2meV),
        
        # Forces MAE (L2)
        "forces_mae_pred_gt": float(np.abs(pred_forces - gt_forces).mean()),
        "forces_mae_pred_calc": float(np.abs(pred_forces - calc_forces).mean()),
        "forces_mae_gt_calc": float(np.abs(gt_forces - calc_forces).mean()),
        
        # Forces MAE (in meV/Ang)
        "forces_mae_pred_gt_meV_Ang": float(np.abs(pred_forces - gt_forces).mean() * 1000),
        "forces_mae_pred_calc_meV_Ang": float(np.abs(pred_forces - calc_forces).mean() * 1000),
        "forces_mae_gt_calc_meV_Ang": float(np.abs(gt_forces - calc_forces).mean() * 1000),
        
        # Force norms (mean)
        "pred_force_norm_mean": float(pred_forces_norm.mean()),
        "gt_force_norm_mean": float(gt_forces_norm.mean()),
        "calc_force_norm_mean": float(calc_forces_norm.mean()),
        
        # Force norm differences
        "force_norm_diff_pred_gt": float(abs(pred_forces_norm - gt_forces_norm).mean()),
        "force_norm_diff_pred_calc": float(abs(pred_forces_norm - calc_forces_norm).mean()),
        "force_norm_diff_gt_calc": float(abs(gt_forces_norm - calc_forces_norm).mean()),
        
        # Orbital coefficient similarity
        "orbital_coeff_similarity_pred_gt": orbital_coeff_sim_pred_gt,
        "orbital_coeff_similarity_pred_calc": orbital_coeff_sim_pred_calc,
        "orbital_coeff_similarity_gt_calc": orbital_coeff_sim_gt_calc,
        
        # Occupied orbital energy MAE
        "occ_orbital_energy_mae_pred_gt": float(np.abs(pred_mo_energy_occ - gt_mo_energy_occ).mean()),
        "occ_orbital_energy_mae_pred_calc": float(np.abs(pred_mo_energy_occ - calc_mo_energy_occ).mean()),
        "occ_orbital_energy_mae_gt_calc": float(np.abs(gt_mo_energy_occ - calc_mo_energy_occ).mean()),
        
        # Overlap difference
        "overlap_diff_gt_calc": overlap_diff_gt_calc,
        
        # Timing
        "timing_info": timing_info,
    }
    
    if verbose:
        print(f"\n{'─'*40}")
        print(f"📊 평가 결과")
        print(f"{'─'*40}")
        
        print(f"\n📐 Hamiltonian Diff (MAE):")
        print(f"   Pred-GT:   {ham_diff_pred_gt:.6e}")
        print(f"   Pred-Calc: {ham_diff_pred_calc:.6e}")
        print(f"   GT-Calc:   {ham_diff_gt_calc:.6e}")
        
        print(f"\n🔋 에너지 (Hartree):")
        print(f"   Pred:  {pred_energy:.10f}")
        print(f"   GT:    {gt_energy:.10f}")
        print(f"   Calc:  {calc_energy:.10f}")
        print(f"\n   Diff (pred-gt):   {abs(pred_energy - gt_energy):.10e} Ha = {abs(pred_energy - gt_energy) * HA2meV:.4f} meV")
        print(f"   Diff (pred-calc): {abs(pred_energy - calc_energy):.10e} Ha = {abs(pred_energy - calc_energy) * HA2meV:.4f} meV")
        print(f"   Diff (gt-calc):   {abs(gt_energy - calc_energy):.10e} Ha = {abs(gt_energy - calc_energy) * HA2meV:.4f} meV")
        
        print(f"\n⚡ Forces MAE:")
        print(f"   Pred-GT:   {np.abs(pred_forces - gt_forces).mean():.6e} Ha/Bohr = {np.abs(pred_forces - gt_forces).mean() * HA_BOHR_2_meV_ANG:.4f} meV/Å")
        print(f"   Pred-Calc: {np.abs(pred_forces - calc_forces).mean():.6e} Ha/Bohr = {np.abs(pred_forces - calc_forces).mean() * HA_BOHR_2_meV_ANG:.4f} meV/Å")
        print(f"   GT-Calc:   {np.abs(gt_forces - calc_forces).mean():.6e} Ha/Bohr = {np.abs(gt_forces - calc_forces).mean() * HA_BOHR_2_meV_ANG:.4f} meV/Å")
        
        print(f"\n📏 Force Norm Diff:")
        print(f"   Pred-GT:   {abs(pred_forces_norm - gt_forces_norm).mean():.6e}")
        print(f"   Pred-Calc: {abs(pred_forces_norm - calc_forces_norm).mean():.6e}")
        print(f"   GT-Calc:   {abs(gt_forces_norm - calc_forces_norm).mean():.6e}")
        
        print(f"\n🔗 Orbital Coeff Similarity (cosine):")
        print(f"   Pred-GT:   {orbital_coeff_sim_pred_gt:.6f}")
        print(f"   Pred-Calc: {orbital_coeff_sim_pred_calc:.6f}")
        print(f"   GT-Calc:   {orbital_coeff_sim_gt_calc:.6f}")
        
        print(f"\n🔮 Occupied Orbital Energy MAE:")
        print(f"   Pred-GT:   {np.abs(pred_mo_energy_occ - gt_mo_energy_occ).mean():.6e}")
        print(f"   Pred-Calc: {np.abs(pred_mo_energy_occ - calc_mo_energy_occ).mean():.6e}")
        print(f"   GT-Calc:   {np.abs(gt_mo_energy_occ - calc_mo_energy_occ).mean():.6e}")
        
        print(f"\n🔄 Overlap Diff (gt-calc): {overlap_diff_gt_calc:.6e}")
        
        print(f"\n⏱️ 소요 시간: {format_time(total_time)}")
        print(f"{'='*80}")
    
    return result


def main():
    parser = argparse.ArgumentParser(description="특정 분자 평가")
    parser.add_argument("--dir_path", type=str, required=True, help="output_dump 디렉토리 경로")
    parser.add_argument("--mol_id", type=str, help="평가할 단일 분자 ID (예: mol10193 또는 10193)")
    parser.add_argument("--mol_ids", nargs="+", type=str, help="평가할 분자 ID 목록")
    parser.add_argument("--force_recalc", action="store_true", help="캐시 무시하고 재계산")
    parser.add_argument("--save_json", action="store_true", help="결과를 JSON으로 저장")
    parser.add_argument("--quiet", action="store_true", help="상세 출력 끄기")
    args = parser.parse_args()
    
    mol_ids = []
    
    if args.mol_id:
        mol_id = args.mol_id
        # 숫자만 입력한 경우 mol prefix 추가
        if mol_id.isdigit():
            mol_id = f"mol{mol_id}"
        mol_ids.append(mol_id)
    
    if args.mol_ids:
        for mid in args.mol_ids:
            if mid.isdigit():
                mid = f"mol{mid}"
            if mid not in mol_ids:
                mol_ids.append(mid)
    
    if not mol_ids:
        print("❌ 평가할 분자 ID가 지정되지 않았습니다.")
        print("   --mol_id 또는 --mol_ids 옵션을 사용하세요.")
        return
    
    print(f"\n📋 총 {len(mol_ids)}개 분자 평가 예정: {mol_ids}")
    
    all_results = []
    for mol_id in mol_ids:
        result = evaluate_single_molecule(
            args.dir_path, 
            mol_id, 
            force_recalc=args.force_recalc,
            verbose=not args.quiet
        )
        if result:
            all_results.append(result)
    
    # Save results
    if args.save_json and all_results:
        path_parts = [p for p in args.dir_path.rstrip('/').split('/') if p]
        dataset_name = path_parts[-2] if len(path_parts) >= 2 else "unknown"
        
        os.makedirs('./outputs2/evaluate', exist_ok=True)
        if len(mol_ids) == 1:
            output_file = f"./outputs2/evaluate/{dataset_name}_{mol_ids[0]}_eval.json"
        else:
            output_file = f"./outputs2/evaluate/{dataset_name}_multi_eval.json"
        
        with open(output_file, "w") as f:
            json.dump(all_results, f, indent=4)
        print(f"\n💾 결과 저장됨: {output_file}")
    
    # Summary
    if len(all_results) > 1:
        print(f"\n{'='*80}")
        print(f"📋 요약 (평균)")
        print(f"{'='*80}")
        
        # Hamiltonian
        avg_ham_diff = np.mean([r["hamiltonian_diff_pred_gt"] for r in all_results])
        print(f"Hamiltonian MAE (pred-gt): {avg_ham_diff:.6e}")
        
        # Energy
        avg_energy_diff = np.mean([r["energy_diff_pred_gt_meV"] for r in all_results])
        print(f"Energy MAE (pred-gt): {avg_energy_diff:.4f} meV")
        
        # Forces
        avg_forces_mae = np.mean([r["forces_mae_pred_gt_meV_Ang"] for r in all_results])
        print(f"Forces MAE (pred-gt): {avg_forces_mae:.4f} meV/Å")
        
        # Force norm
        avg_force_norm_diff = np.mean([r["force_norm_diff_pred_gt"] for r in all_results])
        print(f"Force Norm Diff (pred-gt): {avg_force_norm_diff:.6e}")
        
        # Orbital similarity
        avg_orbital_sim = np.mean([r["orbital_coeff_similarity_pred_gt"] for r in all_results])
        print(f"Orbital Coeff Similarity (pred-gt): {avg_orbital_sim:.6f}")
        
        # Orbital energy
        avg_orbital_energy_mae = np.mean([r["occ_orbital_energy_mae_pred_gt"] for r in all_results])
        print(f"Orbital Energy MAE (pred-gt): {avg_orbital_energy_mae:.6e}")


if __name__ == "__main__":
    main()

