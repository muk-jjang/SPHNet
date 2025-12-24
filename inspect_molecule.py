#!/usr/bin/env python
"""
개별 분자 파일을 상세하게 검사하는 스크립트.
이상치로 탐지된 파일들의 데이터를 확인할 수 있습니다.

사용법:
    # 단일 파일 검사
    python inspect_molecule.py --dir_path /path/to/output_dump --mol_id mol22434
    
    # 여러 파일 검사
    python inspect_molecule.py --dir_path /path/to/output_dump --mol_ids mol22434 mol1789 mol20201
    
    # JSON 파일에서 이상치 목록 읽어서 검사
    python inspect_molecule.py --dir_path /path/to/output_dump --from_json ./outputs/malondialdehyde_evaluate_without_nan.json --top_n 10
"""

import os
import argparse
import json
import numpy as np
import torch

md17_dataset_list = ['ethanol', 'malondialdehyde', 'uracil']
rmd17_dataset_list = ['aspirin', 'naphthalene', 'salicylic_acid']

def inspect_single_file(dir_path, mol_id, verbose=True):
    """단일 분자 파일 상세 검사"""
    if dir_path.split("/")[-2] in md17_dataset_list:
        pred_path = os.path.join(dir_path, f"pred_{mol_id}.pt")
        gt_path = os.path.join(dir_path, f"gt_{mol_id}.pt")
        calc_path = os.path.join(dir_path, f"calc_{mol_id}.pt")
    elif dir_path.split("/")[-2] in rmd17_dataset_list:
        pred_path = os.path.join(dir_path, f"pred_batch0_{mol_id}.pt")
        gt_path = os.path.join(dir_path, f"gt_batch0_{mol_id}.pt")
        calc_path = os.path.join(dir_path, f"calc_batch0_{mol_id}.pt")
    
    result = {
        "mol_id": mol_id,
        "files_exist": {
            "pred": os.path.exists(pred_path),
            "gt": os.path.exists(gt_path),
            "calc": os.path.exists(calc_path),
        }
    }
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"🔍 분자 검사: {mol_id}")
        print(f"{'='*80}")
        print(f"\n📁 파일 경로:")
        print(f"  pred: {pred_path} {'✅' if result['files_exist']['pred'] else '❌'}")
        print(f"  gt:   {gt_path} {'✅' if result['files_exist']['gt'] else '❌'}")
        print(f"  calc: {calc_path} {'✅' if result['files_exist']['calc'] else '❌'}")
    
    # Load files
    pred_data = None
    gt_data = None
    calc_data = None
    
    if result['files_exist']['pred']:
        pred_data = torch.load(pred_path, weights_only=False)
    if result['files_exist']['gt']:
        gt_data = torch.load(gt_path, weights_only=False)
    if result['files_exist']['calc']:
        calc_data = torch.load(calc_path, weights_only=False)
    
    # ===== PRED 파일 분석 =====
    if pred_data is not None:
        if verbose:
            print(f"\n{'─'*40}")
            print(f"📄 PRED 파일 분석")
            print(f"{'─'*40}")
            print(f"  Keys: {list(pred_data.keys())}")
        
        result["pred"] = {}
        
        # Energy
        if "calc_energy" in pred_data:
            energy = pred_data["calc_energy"]
            is_nan = np.isnan(energy) if isinstance(energy, (float, int, np.floating)) else torch.isnan(torch.tensor(energy)).item()
            result["pred"]["calc_energy"] = float(energy) if not is_nan else "NaN"
            if verbose:
                status = "⚠️ NaN!" if is_nan else ("🚨 비정상!" if abs(energy) > 1000 else "✅")
                print(f"  calc_energy: {energy:.10e} {status}")
        
        # Forces
        if "calc_forces" in pred_data:
            forces = pred_data["calc_forces"]
            if isinstance(forces, torch.Tensor):
                forces = forces.numpy()
            result["pred"]["calc_forces_shape"] = list(forces.shape)
            result["pred"]["calc_forces_mean"] = float(np.mean(forces))
            result["pred"]["calc_forces_std"] = float(np.std(forces))
            result["pred"]["calc_forces_has_nan"] = bool(np.isnan(forces).any())
            if verbose:
                print(f"  calc_forces: shape={forces.shape}, mean={np.mean(forces):.6f}, std={np.std(forces):.6f}, has_nan={np.isnan(forces).any()}")
        
        # Hamiltonian
        if "pred_hamiltonian" in pred_data:
            ham = pred_data["pred_hamiltonian"]
            if isinstance(ham, torch.Tensor):
                ham = ham.numpy()
            result["pred"]["pred_hamiltonian_shape"] = list(ham.shape)
            result["pred"]["pred_hamiltonian_mean"] = float(np.mean(ham))
            result["pred"]["pred_hamiltonian_std"] = float(np.std(ham))
            result["pred"]["pred_hamiltonian_has_nan"] = bool(np.isnan(ham).any())
            if verbose:
                print(f"  pred_hamiltonian: shape={ham.shape}, mean={np.mean(ham):.6f}, std={np.std(ham):.6f}, has_nan={np.isnan(ham).any()}")
        
        # MO Energy
        if "calc_mo_energy" in pred_data:
            mo_e = pred_data["calc_mo_energy"]
            if isinstance(mo_e, torch.Tensor):
                mo_e = mo_e.numpy()
            result["pred"]["calc_mo_energy_shape"] = list(mo_e.shape)
            result["pred"]["calc_mo_energy_min"] = float(np.min(mo_e))
            result["pred"]["calc_mo_energy_max"] = float(np.max(mo_e))
            result["pred"]["calc_mo_energy_has_nan"] = bool(np.isnan(mo_e).any())
            if verbose:
                print(f"  calc_mo_energy: shape={mo_e.shape}, min={np.min(mo_e):.6f}, max={np.max(mo_e):.6f}, has_nan={np.isnan(mo_e).any()}")
        
        # MO Coeff
        if "calc_mo_coeff" in pred_data:
            mo_c = pred_data["calc_mo_coeff"]
            if isinstance(mo_c, torch.Tensor):
                mo_c = mo_c.numpy()
            result["pred"]["calc_mo_coeff_shape"] = list(mo_c.shape)
            result["pred"]["calc_mo_coeff_has_nan"] = bool(np.isnan(mo_c).any())
            if verbose:
                print(f"  calc_mo_coeff: shape={mo_c.shape}, has_nan={np.isnan(mo_c).any()}")
    
    # ===== GT 파일 분석 =====
    if gt_data is not None:
        if verbose:
            print(f"\n{'─'*40}")
            print(f"📄 GT 파일 분석")
            print(f"{'─'*40}")
            print(f"  Keys: {list(gt_data.keys())}")
        
        result["gt"] = {}
        
        # Atoms
        if "atoms" in gt_data:
            atoms = gt_data["atoms"]
            if isinstance(atoms, torch.Tensor):
                atoms = atoms.numpy()
            result["gt"]["atoms"] = atoms.tolist() if isinstance(atoms, np.ndarray) else list(atoms)
            result["gt"]["num_atoms"] = len(atoms)
            result["gt"]["total_electrons"] = int(np.sum(atoms))
            if verbose:
                print(f"  atoms: {atoms}, num_atoms={len(atoms)}, total_electrons={int(np.sum(atoms))}")
        
        # Position
        if "pos" in gt_data:
            pos = gt_data["pos"]
            if isinstance(pos, torch.Tensor):
                pos = pos.numpy()
            result["gt"]["pos_shape"] = list(pos.shape)
            if verbose:
                print(f"  pos: shape={pos.shape}")
                print(f"    min={pos.min():.6f}, max={pos.max():.6f}")
        
        # Energy
        if "calc_energy" in gt_data:
            energy = gt_data["calc_energy"]
            is_nan = np.isnan(energy) if isinstance(energy, (float, int, np.floating)) else torch.isnan(torch.tensor(energy)).item()
            result["gt"]["calc_energy"] = float(energy) if not is_nan else "NaN"
            if verbose:
                status = "⚠️ NaN!" if is_nan else ("🚨 비정상!" if abs(energy) > 1000 else "✅")
                print(f"  calc_energy: {energy:.10e} {status}")
        
        # Forces
        if "calc_forces" in gt_data:
            forces = gt_data["calc_forces"]
            if isinstance(forces, torch.Tensor):
                forces = forces.numpy()
            result["gt"]["calc_forces_shape"] = list(forces.shape)
            result["gt"]["calc_forces_mean"] = float(np.mean(forces))
            result["gt"]["calc_forces_has_nan"] = bool(np.isnan(forces).any())
            if verbose:
                print(f"  calc_forces: shape={forces.shape}, mean={np.mean(forces):.6f}, has_nan={np.isnan(forces).any()}")
        
        # Hamiltonian
        if "hamiltonian" in gt_data:
            ham = gt_data["hamiltonian"]
            if isinstance(ham, torch.Tensor):
                ham = ham.numpy()
            result["gt"]["hamiltonian_shape"] = list(ham.shape)
            result["gt"]["hamiltonian_mean"] = float(np.mean(ham))
            result["gt"]["hamiltonian_std"] = float(np.std(ham))
            result["gt"]["hamiltonian_has_nan"] = bool(np.isnan(ham).any())
            if verbose:
                print(f"  hamiltonian: shape={ham.shape}, mean={np.mean(ham):.6f}, std={np.std(ham):.6f}, has_nan={np.isnan(ham).any()}")
        
        # Init Ham
        if "init_ham" in gt_data:
            init_ham = gt_data["init_ham"]
            if isinstance(init_ham, torch.Tensor):
                init_ham = init_ham.numpy()
            result["gt"]["init_ham_shape"] = list(init_ham.shape)
            result["gt"]["init_ham_mean"] = float(np.mean(init_ham))
            if verbose:
                print(f"  init_ham: shape={init_ham.shape}, mean={np.mean(init_ham):.6f}")
        
        # Overlap
        if "overlap" in gt_data:
            ovlp = gt_data["overlap"]
            if isinstance(ovlp, torch.Tensor):
                ovlp = ovlp.numpy()
            result["gt"]["overlap_shape"] = list(ovlp.shape)
            result["gt"]["overlap_has_nan"] = bool(np.isnan(ovlp).any())
            if verbose:
                print(f"  overlap: shape={ovlp.shape}, has_nan={np.isnan(ovlp).any()}")
        
        # remove_init flag
        if "remove_init" in gt_data:
            result["gt"]["remove_init"] = bool(gt_data["remove_init"])
            if verbose:
                print(f"  remove_init: {gt_data['remove_init']}")
    
    # ===== CALC 파일 분석 =====
    if calc_data is not None:
        if verbose:
            print(f"\n{'─'*40}")
            print(f"📄 CALC 파일 분석")
            print(f"{'─'*40}")
            print(f"  Keys: {list(calc_data.keys())}")
        
        result["calc"] = {}
        
        # Energy
        if "calc_energy" in calc_data:
            energy = calc_data["calc_energy"]
            is_nan = np.isnan(energy) if isinstance(energy, (float, int, np.floating)) else torch.isnan(torch.tensor(energy)).item()
            result["calc"]["calc_energy"] = float(energy) if not is_nan else "NaN"
            if verbose:
                status = "⚠️ NaN!" if is_nan else ("🚨 비정상!" if abs(energy) > 1000 else "✅")
                print(f"  calc_energy: {energy:.10e} {status}")
        
        # Forces
        if "calc_forces" in calc_data:
            forces = calc_data["calc_forces"]
            if isinstance(forces, torch.Tensor):
                forces = forces.numpy()
            result["calc"]["calc_forces_shape"] = list(forces.shape)
            result["calc"]["calc_forces_mean"] = float(np.mean(forces))
            if verbose:
                print(f"  calc_forces: shape={forces.shape}, mean={np.mean(forces):.6f}")
        
        # Hamiltonian
        if "hamiltonian" in calc_data:
            ham = calc_data["hamiltonian"]
            if isinstance(ham, torch.Tensor):
                ham = ham.numpy()
            result["calc"]["hamiltonian_shape"] = list(ham.shape)
            result["calc"]["hamiltonian_mean"] = float(np.mean(ham))
            if verbose:
                print(f"  hamiltonian: shape={ham.shape}, mean={np.mean(ham):.6f}")
    
    # ===== 비교 분석 =====
    if pred_data is not None and gt_data is not None and calc_data is not None:
        if verbose:
            print(f"\n{'─'*40}")
            print(f"📊 비교 분석")
            print(f"{'─'*40}")
        
        result["comparison"] = {}
        
        # Energy comparison
        pred_e = pred_data.get("calc_energy")
        gt_e = gt_data.get("calc_energy")
        calc_e = calc_data.get("calc_energy")
        
        if pred_e is not None and gt_e is not None and calc_e is not None:
            pred_nan = np.isnan(pred_e) if isinstance(pred_e, (float, int, np.floating)) else False
            gt_nan = np.isnan(gt_e) if isinstance(gt_e, (float, int, np.floating)) else False
            calc_nan = np.isnan(calc_e) if isinstance(calc_e, (float, int, np.floating)) else False
            
            result["comparison"]["energy_nan_status"] = {
                "pred": bool(pred_nan),
                "gt": bool(gt_nan),
                "calc": bool(calc_nan),
            }
            
            if not pred_nan and not gt_nan and not calc_nan:
                diff_pred_gt = abs(pred_e - gt_e)
                diff_pred_calc = abs(pred_e - calc_e)
                diff_gt_calc = abs(gt_e - calc_e)
                
                result["comparison"]["energy_diff"] = {
                    "pred-gt": float(diff_pred_gt),
                    "pred-calc": float(diff_pred_calc),
                    "gt-calc": float(diff_gt_calc),
                }
                
                if verbose:
                    print(f"  Energy diff (pred-gt):   {diff_pred_gt:.10e}")
                    print(f"  Energy diff (pred-calc): {diff_pred_calc:.10e}")
                    print(f"  Energy diff (gt-calc):   {diff_gt_calc:.10e}")
            else:
                if verbose:
                    print(f"  ⚠️ NaN 존재로 인해 에너지 비교 불가")
                    print(f"     pred_nan={pred_nan}, gt_nan={gt_nan}, calc_nan={calc_nan}")
    
    if verbose:
        print(f"\n{'='*80}")
    
    return result


def main():
    parser = argparse.ArgumentParser(description="개별 분자 파일 검사")
    parser.add_argument("--dir_path", type=str, required=True, help="데이터 디렉토리 경로")
    parser.add_argument("--mol_id", type=str, help="검사할 단일 분자 ID (예: mol22434)")
    parser.add_argument("--mol_ids", nargs="+", type=str, help="검사할 분자 ID 목록")
    parser.add_argument("--from_json", type=str, help="이상치 목록이 저장된 JSON 파일 경로")
    parser.add_argument("--top_n", type=int, default=10, help="JSON에서 상위 N개 이상치 검사")
    args = parser.parse_args()
    
    mol_ids_to_inspect = []
    
    # Collect mol_ids from arguments
    if args.mol_id:
        mol_ids_to_inspect.append(args.mol_id)
    
    if args.mol_ids:
        mol_ids_to_inspect.extend(args.mol_ids)
    
    # Collect mol_ids from JSON file
    if args.from_json:
        with open(args.from_json, "r") as f:
            json_data = json.load(f)
        
        # Get abnormal energy files
        abnormal_files = json_data.get("abnormal_energy_files", [])
        if abnormal_files:
            print(f"\n📋 JSON 파일에서 {len(abnormal_files)}개 비정상 에너지 파일 발견")
            print(f"   상위 {args.top_n}개 검사 예정")
            for item in abnormal_files[:args.top_n]:
                mol_id = item.get("mol_id")
                if mol_id and mol_id not in mol_ids_to_inspect:
                    mol_ids_to_inspect.append(mol_id)
    
    if not mol_ids_to_inspect:
        print("❌ 검사할 분자 ID가 지정되지 않았습니다.")
        print("   --mol_id, --mol_ids, 또는 --from_json 옵션을 사용하세요.")
        return
    
    print(f"\n총 {len(mol_ids_to_inspect)}개 분자 검사 예정: {mol_ids_to_inspect}")
    
    all_results = []
    for mol_id in mol_ids_to_inspect:
        result = inspect_single_file(args.dir_path, mol_id, verbose=True)
        all_results.append(result)
    
    # Save results to JSON (md17_evaluation_customv2.py 스타일)
    # dataset_name = dir_path.split("/")[-2]
    path_parts = [p for p in args.dir_path.rstrip('/').split('/') if p]
    dataset_name = path_parts[-2] if len(path_parts) >= 2 else path_parts[-1] if path_parts else "unknown"
    
    os.makedirs('./outputs2/molecule', exist_ok=True)
    output_file = os.path.join('./outputs2/molecule', f"{dataset_name}_{mol_ids_to_inspect[0]}_inspect_results.json")
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=4)
    print(f"\n결과 저장됨: {output_file}")
    
    # Summary
    print(f"\n{'='*80}")
    print(f"📋 요약")
    print(f"{'='*80}")
    print(f"총 검사 파일: {len(all_results)}")
    
    # Count abnormal energies
    abnormal_pred = 0
    abnormal_gt = 0
    nan_pred = 0
    nan_gt = 0
    
    for r in all_results:
        pred_e = r.get("pred", {}).get("calc_energy")
        gt_e = r.get("gt", {}).get("calc_energy")
        
        if pred_e == "NaN":
            nan_pred += 1
        elif pred_e is not None and abs(pred_e) > 1000:
            abnormal_pred += 1
        
        if gt_e == "NaN":
            nan_gt += 1
        elif gt_e is not None and abs(gt_e) > 1000:
            abnormal_gt += 1
    
    print(f"  PRED 비정상 에너지: {abnormal_pred}, NaN: {nan_pred}")
    print(f"  GT 비정상 에너지: {abnormal_gt}, NaN: {nan_gt}")


if __name__ == "__main__":
    main()

