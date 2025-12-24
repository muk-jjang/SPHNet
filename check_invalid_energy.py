#!/usr/bin/env python
"""
Check for invalid calc_energy values in prediction/ground-truth files.

Usage:
    python check_invalid_energy.py --dir_path /path/to/data
    python check_invalid_energy.py --dir_path /path/to/data --verbose
    python check_invalid_energy.py --dir_path /path/to/data --save_report
"""

import os
import glob
import torch
import math
import json
import argparse
import warnings
import numpy as np
from tqdm import tqdm
from collections import defaultdict

warnings.filterwarnings('ignore', category=FutureWarning)

# ============================================================================
# Constants (same as md17_evaluation_fix_nan.py)
# ============================================================================
MIN_VALID_ENERGY = -10000.0  # Hartree
MAX_VALID_ENERGY = 0.0       # Hartree


# ============================================================================
# Validation Functions
# ============================================================================
def classify_energy(value, min_energy=MIN_VALID_ENERGY, max_energy=MAX_VALID_ENERGY):
    """
    Classify energy value and return reason if invalid.
    
    Returns:
        (is_valid, reason)
    """
    if value is None:
        return False, "missing"
    
    try:
        if isinstance(value, torch.Tensor):
            val = value.item() if value.numel() == 1 else float(value.mean())
        elif isinstance(value, np.ndarray):
            val = float(value.mean()) if value.size > 1 else float(value)
        else:
            val = float(value)
    except (TypeError, ValueError) as e:
        return False, f"conversion_error ({e})"
    
    if math.isnan(val):
        return False, "nan"
    
    if math.isinf(val):
        return False, "inf"
    
    if val > max_energy:
        return False, f"too_positive ({val:.4f})"
    
    if val < min_energy:
        return False, f"too_negative ({val:.4f})"
    
    return True, "valid"


def check_file_pair(pred_path, gt_path):
    """Check a single pred/gt file pair for invalid energies."""
    pred_data = torch.load(pred_path, weights_only=False)
    gt_data = torch.load(gt_path, weights_only=False)
    
    data_index = gt_data.get("idx", "unknown")
    
    # Check pred calc_energy
    pred_energy = pred_data.get("calc_energy")
    pred_valid, pred_reason = classify_energy(pred_energy)
    
    # Check gt calc_energy
    gt_energy = gt_data.get("calc_energy")
    gt_valid, gt_reason = classify_energy(gt_energy)
    
    # Check if calc_forces exist
    pred_has_forces = "calc_forces" in pred_data
    gt_has_forces = "calc_forces" in gt_data
    
    return {
        "data_index": data_index,
        "pred_path": pred_path,
        "gt_path": gt_path,
        "pred_energy": pred_energy,
        "pred_valid": pred_valid,
        "pred_reason": pred_reason,
        "gt_energy": gt_energy,
        "gt_valid": gt_valid,
        "gt_reason": gt_reason,
        "pred_has_forces": pred_has_forces,
        "gt_has_forces": gt_has_forces,
    }


def check_all_files(dir_path, pred_prefix="pred_", gt_prefix="gt_", verbose=False):
    """Check all file pairs in directory."""
    pred_paths = sorted(glob.glob(os.path.join(dir_path, f"{pred_prefix}*.pt")))
    gt_paths = sorted(glob.glob(os.path.join(dir_path, f"{gt_prefix}*.pt")))
    
    if len(pred_paths) != len(gt_paths):
        print(f"Warning: Mismatch in file counts - pred: {len(pred_paths)}, gt: {len(gt_paths)}")
    
    file_pairs = list(zip(pred_paths, gt_paths))
    print(f"Found {len(file_pairs)} file pairs to check")
    
    results = []
    stats = defaultdict(int)
    invalid_details = defaultdict(list)
    
    for pred_path, gt_path in tqdm(file_pairs, desc="Checking files"):
        result = check_file_pair(pred_path, gt_path)
        results.append(result)
        
        # Update stats
        if result["pred_valid"] and result["gt_valid"]:
            stats["both_valid"] += 1
        else:
            stats["has_invalid"] += 1
            
            if not result["pred_valid"]:
                stats[f"pred_invalid_{result['pred_reason'].split()[0]}"] += 1
                invalid_details["pred_invalid"].append(result)
            
            if not result["gt_valid"]:
                stats[f"gt_invalid_{result['gt_reason'].split()[0]}"] += 1
                invalid_details["gt_invalid"].append(result)
        
        # Check forces
        if not result["pred_has_forces"]:
            stats["pred_missing_forces"] += 1
        if not result["gt_has_forces"]:
            stats["gt_missing_forces"] += 1
    
    return results, stats, invalid_details


def print_summary(stats, total_files, invalid_details, verbose=False):
    """Print summary of results."""
    print("\n" + "=" * 80)
    print("INVALID ENERGY CHECK SUMMARY")
    print("=" * 80)
    
    print(f"\nTotal files checked: {total_files}")
    print(f"Valid files (both pred & gt): {stats['both_valid']} ({100*stats['both_valid']/total_files:.1f}%)")
    print(f"Files with invalid energy: {stats['has_invalid']} ({100*stats['has_invalid']/total_files:.1f}%)")
    
    print("\n--- Breakdown by Reason ---")
    
    # Pred stats
    pred_invalid_keys = [k for k in stats if k.startswith("pred_invalid_")]
    if pred_invalid_keys:
        print("\nPred file issues:")
        for key in sorted(pred_invalid_keys):
            reason = key.replace("pred_invalid_", "")
            print(f"  - {reason}: {stats[key]}")
    
    # GT stats
    gt_invalid_keys = [k for k in stats if k.startswith("gt_invalid_")]
    if gt_invalid_keys:
        print("\nGT file issues:")
        for key in sorted(gt_invalid_keys):
            reason = key.replace("gt_invalid_", "")
            print(f"  - {reason}: {stats[key]}")
    
    # Forces stats
    if stats["pred_missing_forces"] > 0 or stats["gt_missing_forces"] > 0:
        print("\n--- Missing Forces ---")
        print(f"Pred missing calc_forces: {stats['pred_missing_forces']}")
        print(f"GT missing calc_forces: {stats['gt_missing_forces']}")
    
    # Verbose: list all invalid molecules
    if verbose and invalid_details:
        print("\n--- Invalid Molecules Detail ---")
        
        seen_indices = set()
        all_invalid = invalid_details.get("pred_invalid", []) + invalid_details.get("gt_invalid", [])
        
        for item in all_invalid:
            idx = item["data_index"]
            if idx in seen_indices:
                continue
            seen_indices.add(idx)
            
            print(f"\nMolecule {idx}:")
            print(f"  pred: {item['pred_reason']} (energy={item['pred_energy']})")
            print(f"  gt:   {item['gt_reason']} (energy={item['gt_energy']})")
    
    print("\n" + "=" * 80)


def save_report(results, stats, output_path):
    """Save detailed report to JSON."""
    invalid_molecules = []
    for r in results:
        if not r["pred_valid"] or not r["gt_valid"]:
            invalid_molecules.append({
                "data_index": r["data_index"],
                "pred_energy": float(r["pred_energy"]) if r["pred_energy"] is not None else None,
                "pred_reason": r["pred_reason"],
                "gt_energy": float(r["gt_energy"]) if r["gt_energy"] is not None else None,
                "gt_reason": r["gt_reason"],
            })
    
    report = {
        "summary": {
            "total_files": len(results),
            "valid_count": stats["both_valid"],
            "invalid_count": stats["has_invalid"],
            "valid_percentage": 100 * stats["both_valid"] / len(results) if results else 0,
        },
        "stats": dict(stats),
        "invalid_molecules": invalid_molecules,
    }
    
    # with open(output_path, "w") as f:
    #     json.dump(report, f, indent=2)
    
    # print(f"\nReport saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Check for invalid calc_energy values")
    parser.add_argument("--dir_path", type=str, required=True, help="Directory containing pred/gt files")
    parser.add_argument("--pred_prefix", type=str, default="pred_", help="Prefix for prediction files")
    parser.add_argument("--gt_prefix", type=str, default="gt_", help="Prefix for ground truth files")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show details for each invalid molecule")
    parser.add_argument("--save_report", "-s", action="store_true", help="Save detailed report to JSON")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output path for report (default: outputs/<dataset>_invalid_check.json)")
    args = parser.parse_args()
    
    # Check files
    results, stats, invalid_details = check_all_files(
        args.dir_path, 
        args.pred_prefix, 
        args.gt_prefix,
        args.verbose
    )
    
    # Print summary
    print_summary(stats, len(results), invalid_details, args.verbose)
    
    # Save report if requested
    if args.save_report:
        if args.output:
            output_path = args.output
        else:
            dataset_name = args.dir_path.rstrip("/").split("/")[-2]
            os.makedirs("./outputs", exist_ok=True)
            output_path = f"./outputs/{dataset_name}_invalid_check.json"
        
        save_report(results, stats, output_path)
    
    # Return exit code based on invalid count
    return 1 if stats["has_invalid"] > 0 else 0


if __name__ == "__main__":
    exit(main())

