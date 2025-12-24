"""
NaN 값 비율 확인 테스트 스크립트
사용법: python test_nan.py --dir_path /path/to/output_dump_batch
"""

import os
import glob
import torch
import numpy as np
import argparse
import time
import json
from collections import defaultdict
from tqdm import tqdm


def check_nan_in_tensor(tensor, name="tensor"):
    """텐서에서 NaN 값 확인"""
    if tensor is None:
        return {"has_nan": False, "nan_count": 0, "total_count": 0, "nan_ratio": 0.0}
    
    if isinstance(tensor, (int, float)):
        is_nan = np.isnan(tensor)
        return {
            "has_nan": is_nan,
            "nan_count": 1 if is_nan else 0,
            "total_count": 1,
            "nan_ratio": 1.0 if is_nan else 0.0
        }
    
    # Convert to numpy if tensor
    if hasattr(tensor, 'numpy'):
        arr = tensor.numpy()
    elif hasattr(tensor, 'get'):  # CuPy array
        arr = tensor.get()
    else:
        arr = np.asarray(tensor)
    
    nan_mask = np.isnan(arr)
    nan_count = np.sum(nan_mask)
    total_count = arr.size
    
    return {
        "has_nan": nan_count > 0,
        "nan_count": int(nan_count),
        "total_count": int(total_count),
        "nan_ratio": float(nan_count / total_count) if total_count > 0 else 0.0
    }


def analyze_single_file(file_path):
    """단일 .pt 파일의 NaN 분석"""
    try:
        data = torch.load(file_path, weights_only=False)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None
    
    results = {}
    
    # 확인할 키 목록
    keys_to_check = [
        # Energy 관련
        "calc_energy", "pred_energy", "gt_energy",
        # Forces 관련  
        "forces", "calc_forces", "pred_forces", "gt_forces",
        # Hamiltonian 관련
        "hamiltonian", "pred_hamiltonian", "init_ham",
        # Density matrix 관련
        "density_matrix",
        # Overlap 관련
        "overlap",
        # MO 관련
        "calc_mo_energy", "calc_mo_coeff", "mo_occ"
    ]
    
    for key in keys_to_check:
        if key in data:
            results[key] = check_nan_in_tensor(data[key], key)
        else:
            results[key] = None  # Key doesn't exist
    
    return results


def format_time(seconds):
    """초를 읽기 쉬운 형식으로 변환"""
    if seconds < 60:
        return f"{seconds:.1f}초"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}분"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}시간"


def analyze_directory(dir_path, pred_prefix="pred_", gt_prefix="gt_", calc_prefix="calc_"):
    """디렉토리 내 모든 파일 분석"""
    
    total_start_time = time.time()
    
    # 데이터셋 이름 추출 (경로에서 -2 위치, 예: /nas/.../malondialdehyde/output_dump -> malondialdehyde)
    path_parts = [p for p in dir_path.rstrip('/').split('/') if p]
    dataset_name = path_parts[-2] if len(path_parts) >= 2 else path_parts[-1] if path_parts else "unknown"
    
    # 각 prefix별로 파일 찾기
    pred_files = sorted(glob.glob(os.path.join(dir_path, f"{pred_prefix}*.pt")))
    gt_files = sorted(glob.glob(os.path.join(dir_path, f"{gt_prefix}*.pt")))
    calc_files = sorted(glob.glob(os.path.join(dir_path, f"{calc_prefix}*.pt")))
    
    print(f"\n{'='*80}")
    print(f"📂 데이터셋: {dataset_name.upper()}")
    print(f"{'='*80}")
    print(f"NaN 분석 시작: {dir_path}")
    print(f"시작 시간: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")
    print(f"발견된 파일 수: pred={len(pred_files)}, gt={len(gt_files)}, calc={len(calc_files)}")
    
    # 통계 수집
    stats = defaultdict(lambda: {"files_with_nan": 0, "total_files": 0, "total_nan_count": 0, "total_element_count": 0})
    nan_file_indices = defaultdict(list)
    
    all_files = []
    for f in pred_files:
        all_files.append(("pred", f))
    for f in gt_files:
        all_files.append(("gt", f))
    for f in calc_files:
        all_files.append(("calc", f))
    
    total_files = len(all_files)
    print(f"\n총 {total_files}개 파일 분석 중...")
    
    # 진행률 표시를 위한 tqdm 사용
    pbar = tqdm(all_files, desc="파일 분석", unit="file", 
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    
    for file_type, file_path in pbar:
        results = analyze_single_file(file_path)
        if results is None:
            continue
        
        file_idx = os.path.basename(file_path)
        pbar.set_postfix_str(f"현재: {file_idx[:20]}...")
        
        for key, result in results.items():
            if result is None:
                continue
            
            stat_key = f"{file_type}_{key}"
            stats[stat_key]["total_files"] += 1
            stats[stat_key]["total_element_count"] += result["total_count"]
            stats[stat_key]["total_nan_count"] += result["nan_count"]
            
            if result["has_nan"]:
                stats[stat_key]["files_with_nan"] += 1
                nan_file_indices[stat_key].append(file_idx)
    
    pbar.close()
    
    # 총 소요 시간 계산
    total_elapsed = time.time() - total_start_time
    avg_time_per_file = total_elapsed / total_files if total_files > 0 else 0
    
    print(f"\n{'='*80}")
    print(f"분석 완료!")
    print(f"총 소요 시간: {format_time(total_elapsed)}")
    print(f"파일당 평균 시간: {avg_time_per_file*1000:.2f}ms")
    print(f"처리 속도: {total_files/total_elapsed:.1f} files/sec")
    print(f"{'='*80}")
    
    # 파일 타입별로 그룹화
    file_types = ["pred", "gt", "calc"]
    type_stats = {ft: {"total_files": 0, "files_with_any_nan": set(), "keys": {}} for ft in file_types}
    
    for key, s in stats.items():
        for ft in file_types:
            if key.startswith(f"{ft}_"):
                field_name = key[len(ft)+1:]  # Remove prefix
                type_stats[ft]["keys"][field_name] = s
                type_stats[ft]["total_files"] = max(type_stats[ft]["total_files"], s["total_files"])
                break
    
    # NaN 파일 인덱스로부터 각 타입별 NaN 파일 수 계산
    for key, indices in nan_file_indices.items():
        for ft in file_types:
            if key.startswith(f"{ft}_"):
                type_stats[ft]["files_with_any_nan"].update(indices)
                break
    
    # ============ 전체 요약 ============
    print(f"\n{'='*80}")
    print(f"📊 [{dataset_name.upper()}] 전체 NaN 요약 (파일 타입별)")
    print(f"{'='*80}")
    print(f"{'파일 타입':<15} {'NaN 있는 파일':<20} {'총 파일수':<15} {'NaN 비율':<15}")
    print(f"{'-'*80}")
    
    for ft in file_types:
        ts = type_stats[ft]
        nan_files = len(ts["files_with_any_nan"])
        total = ts["total_files"]
        ratio = nan_files / total * 100 if total > 0 else 0
        status = "⚠️ " if nan_files > 0 else "✅ "
        print(f"{status}{ft.upper():<13} {nan_files:<20} {total:<15} {ratio:.2f}%")
    
    # ============ 각 타입별 상세 ============
    for ft in file_types:
        ts = type_stats[ft]
        if ts["total_files"] == 0:
            continue
            
        print(f"\n{'='*80}")
        print(f"📁 {ft.upper()} 파일 상세 분석")
        print(f"{'='*80}")
        print(f"{'필드명':<30} {'NaN 파일수':<15} {'총 파일수':<12} {'NaN 비율':<12} {'상태'}")
        print(f"{'-'*80}")
        
        has_nan_in_type = False
        for field_name in sorted(ts["keys"].keys()):
            s = ts["keys"][field_name]
            if s["total_files"] > 0:
                file_ratio = s["files_with_nan"] / s["total_files"] * 100
                status = "⚠️ NaN!" if s["files_with_nan"] > 0 else "✅ OK"
                if s["files_with_nan"] > 0:
                    has_nan_in_type = True
                print(f"{field_name:<30} {s['files_with_nan']:<15} {s['total_files']:<12} {file_ratio:>6.2f}%      {status}")
        
        if not has_nan_in_type:
            print(f"\n  ✅ {ft.upper()} 파일에서 NaN이 발견되지 않았습니다!")
    
    # ============ NaN 파일 목록 ============
    any_nan_found = any(len(ts["files_with_any_nan"]) > 0 for ts in type_stats.values())
    
    if any_nan_found:
        print(f"\n{'='*80}")
        print("🔍 NaN이 발견된 파일 목록 (타입별, 최대 10개)")
        print(f"{'='*80}")
        
        for ft in file_types:
            nan_files = sorted(type_stats[ft]["files_with_any_nan"])
            if nan_files:
                print(f"\n[{ft.upper()}] NaN 파일 ({len(nan_files)}개):")
                for idx in nan_files[:10]:
                    print(f"  - {idx}")
                if len(nan_files) > 10:
                    print(f"  ... 그 외 {len(nan_files) - 10}개 파일")
        
        # 상세: 어떤 필드에서 NaN이 발생했는지
        print(f"\n{'='*80}")
        print("🔬 NaN 발생 필드별 상세")
        print(f"{'='*80}")
        for key, indices in sorted(nan_file_indices.items()):
            print(f"\n{key} ({len(indices)}개 파일):")
            for idx in indices[:5]:
                print(f"  - {idx}")
            if len(indices) > 5:
                print(f"  ... 그 외 {len(indices) - 5}개 파일")
    else:
        print(f"\n{'='*80}")
        print(f"✅ [{dataset_name.upper()}] 모든 파일에서 NaN이 발견되지 않았습니다!")
        print(f"{'='*80}")
    
    # 최종 요약 한 줄
    print(f"\n{'='*80}")
    print(f"📋 [{dataset_name.upper()}] 최종 요약")
    print(f"{'='*80}")
    total_nan_files = sum(len(ts["files_with_any_nan"]) for ts in type_stats.values())
    total_all_files = sum(ts["total_files"] for ts in type_stats.values())
    print(f"  - PRED: {len(type_stats['pred']['files_with_any_nan'])}/{type_stats['pred']['total_files']} 파일에서 NaN 발견 ({len(type_stats['pred']['files_with_any_nan'])/type_stats['pred']['total_files']*100 if type_stats['pred']['total_files'] > 0 else 0:.2f}%)")
    print(f"  - GT:   {len(type_stats['gt']['files_with_any_nan'])}/{type_stats['gt']['total_files']} 파일에서 NaN 발견 ({len(type_stats['gt']['files_with_any_nan'])/type_stats['gt']['total_files']*100 if type_stats['gt']['total_files'] > 0 else 0:.2f}%)")
    print(f"  - CALC: {len(type_stats['calc']['files_with_any_nan'])}/{type_stats['calc']['total_files']} 파일에서 NaN 발견 ({len(type_stats['calc']['files_with_any_nan'])/type_stats['calc']['total_files']*100 if type_stats['calc']['total_files'] > 0 else 0:.2f}%)")
    print(f"{'='*80}")
    
    return stats, nan_file_indices, type_stats, dataset_name, total_elapsed


def save_results_to_json(stats, nan_file_indices, type_stats, dataset_name, elapsed_time, dir_path):
    """분석 결과를 JSON 파일로 저장 (md17_evaluation_customv2.py 스타일)"""
    
    # JSON 직렬화 가능한 형태로 변환
    json_results = {
        "dataset_name": dataset_name,
        "analysis_timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "elapsed_time_seconds": elapsed_time,
        "summary": {},
        "detailed_stats": {},
        "nan_file_indices": {}
    }
    
    # 파일 타입별 요약
    file_types = ["pred", "gt", "calc"]
    for ft in file_types:
        ts = type_stats[ft]
        nan_files_count = len(ts["files_with_any_nan"])
        total_files = ts["total_files"]
        json_results["summary"][ft] = {
            "nan_files_count": nan_files_count,
            "total_files": total_files,
            "nan_ratio_percent": nan_files_count / total_files * 100 if total_files > 0 else 0,
            "nan_file_list": sorted(list(ts["files_with_any_nan"]))
        }
    
    # 상세 통계 (stats를 직렬화 가능하게 변환)
    for key, s in stats.items():
        json_results["detailed_stats"][key] = {
            "files_with_nan": s["files_with_nan"],
            "total_files": s["total_files"],
            "total_nan_count": s["total_nan_count"],
            "total_element_count": s["total_element_count"],
            "nan_ratio_percent": s["files_with_nan"] / s["total_files"] * 100 if s["total_files"] > 0 else 0
        }
    
    # NaN 파일 인덱스 (리스트로 변환)
    for key, indices in nan_file_indices.items():
        json_results["nan_file_indices"][key] = sorted(indices)
    
    # md17_evaluation_customv2.py 스타일로 저장
    # dataset_name = dir_path.split("/")[-2]
    os.makedirs('./outputs2', exist_ok=True)
    output_file = os.path.join('./outputs2', f"{dataset_name}_nan_analysis.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, ensure_ascii=False, indent=4)
    
    print(f"\n💾 결과가 저장되었습니다: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="NaN 값 비율 확인 테스트")
    parser.add_argument("--dir_path", type=str, required=True, 
                        help="분석할 디렉토리 경로 (예: /nas/seongjun/sphnet/aspirin/output_dump_batch)")
    parser.add_argument("--pred_prefix", type=str, default="pred_")
    parser.add_argument("--gt_prefix", type=str, default="gt_")
    parser.add_argument("--calc_prefix", type=str, default="calc_")
    parser.add_argument("--single_file", type=str, default=None,
                        help="단일 파일만 분석하려면 파일 경로 지정")
    
    args = parser.parse_args()
    
    if args.single_file:
        print(f"\n단일 파일 분석: {args.single_file}")
        start_time = time.time()
        results = analyze_single_file(args.single_file)
        elapsed = time.time() - start_time
        if results:
            print(f"\n{'Key':<30} {'Has NaN':<10} {'NaN Count':<15} {'Total':<15} {'Ratio':<10}")
            print(f"{'-'*80}")
            for key, result in results.items():
                if result is not None:
                    print(f"{key:<30} {str(result['has_nan']):<10} {result['nan_count']:<15} {result['total_count']:<15} {result['nan_ratio']:.4f}")
            print(f"\n소요 시간: {elapsed*1000:.2f}ms")
    else:
        stats, nan_file_indices, type_stats, dataset_name, elapsed_time = analyze_directory(
            args.dir_path,
            pred_prefix=args.pred_prefix,
            gt_prefix=args.gt_prefix,
            calc_prefix=args.calc_prefix
        )
        
        # JSON 파일 저장 (md17_evaluation_customv2.py 스타일)
        save_results_to_json(stats, nan_file_indices, type_stats, dataset_name, elapsed_time, args.dir_path)


if __name__ == "__main__":
    main()

