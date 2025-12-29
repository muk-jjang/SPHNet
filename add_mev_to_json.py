#!/usr/bin/env python
"""
Script to add meV versions of energy and force differences to existing JSON evaluation results.

Usage:
    python add_mev_to_json.py --json_file outputs/your_evaluation_results.json
    python add_mev_to_json.py --json_dir outputs/  # Process all JSON files in directory
"""

import os
import json
import glob
import argparse

# Conversion constants (same as escflow_eval_utils)
HA2eV = 27.211386245988  # Hartree to eV conversion
HA2meV = HA2eV * 1000    # Hartree to meV conversion
BOHR2ANG = 0.529177210903  # Bohr to Angstrom conversion
HA_BOHR_2_meV_ANG = HA2meV / BOHR2ANG  # Hartree/Bohr to meV/Angstrom


def add_mev_to_results(data: dict) -> dict:
    """
    Add meV versions of energy and force differences to evaluation results.
    
    Energy: Hartree -> meV (multiply by HA2meV)
    Forces: Hartree/Bohr -> meV/Angstrom (multiply by HA_BOHR_2_meV_ANG)
    """
    # Energy differences (Hartree -> meV)
    energy_keys = [
        ("energy_diff (pred-gt)", "energy_diff_meV (pred-gt)"),
        ("energy_diff (pred-calc_energy)", "energy_diff_meV (pred-calc_energy)"),
        ("energy_diff (gt-calc_energy)", "energy_diff_meV (gt-calc_energy)"),
    ]
    
    for ha_key, mev_key in energy_keys:
        if ha_key in data:
            data[mev_key] = data[ha_key] * HA2meV
    
    # Force differences (Hartree/Bohr -> meV/Angstrom)
    force_keys = [
        ("forces_diff l2 (pred-gt)", "forces_diff_meV/A (pred-gt)"),
        ("forces_diff l2 (pred-calc_forces)", "forces_diff_meV/A (pred-calc_forces)"),
        ("forces_diff l2 (gt-calc_forces)", "forces_diff_meV/A (gt-calc_forces)"),
    ]
    
    # Forces are already in Hartree/Angstrom (since pos was converted to Ang before PySCF)
    # So we just multiply by HA2meV to get meV/Angstrom
    for ha_ang_key, mev_ang_key in force_keys:
        if ha_ang_key in data:
            data[mev_ang_key] = data[ha_ang_key] * HA2meV
    
    # Force norm differences (Hartree/Angstrom -> meV/Angstrom)
    force_norm_keys = [
        ("pred_force_norm_diff (pred-gt)", "pred_force_norm_diff_meV/A (pred-gt)"),
        ("pred_force_norm_diff (pred-calc_forces)", "pred_force_norm_diff_meV/A (pred-calc_forces)"),
        ("gt_force_norm_diff (gt-calc_forces)", "gt_force_norm_diff_meV/A (gt-calc_forces)"),
    ]
    
    for ha_ang_key, mev_ang_key in force_norm_keys:
        if ha_ang_key in data:
            data[mev_ang_key] = data[ha_ang_key] * HA2meV
    
    return data


def process_json_file(json_path: str, dry_run: bool = False) -> bool:
    """
    Process a single JSON file and add meV versions.
    
    Args:
        json_path: Path to the JSON file
        dry_run: If True, print changes but don't save
    
    Returns:
        True if successful, False otherwise
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Add meV versions
        updated_data = add_mev_to_results(data)
        
        if dry_run:
            print(f"\n[DRY RUN] Would update: {json_path}")
            print("New meV fields:")
            for key in updated_data:
                if 'meV' in key:
                    print(f"  {key}: {updated_data[key]:.6f}")
        else:
            with open(json_path, 'w') as f:
                json.dump(updated_data, f, indent=4)
            print(f"Updated: {json_path}")
        
        return True
    
    except Exception as e:
        print(f"Error processing {json_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Add meV versions of energy and force differences to JSON evaluation results."
    )
    parser.add_argument(
        "--json_file", 
        type=str, 
        default=None,
        help="Path to a single JSON file to process"
    )
    parser.add_argument(
        "--json_dir", 
        type=str, 
        default=None,
        help="Path to a directory containing JSON files to process"
    )
    parser.add_argument(
        "--dry_run", 
        action="store_true",
        help="Print changes without saving"
    )
    args = parser.parse_args()
    
    if args.json_file is None and args.json_dir is None:
        # Default to outputs directory
        args.json_dir = "./outputs"
    
    json_files = []
    
    if args.json_file:
        json_files.append(args.json_file)
    
    if args.json_dir:
        pattern = os.path.join(args.json_dir, "*_evaluation_results.json")
        json_files.extend(glob.glob(pattern))
    
    if not json_files:
        print("No JSON files found to process.")
        return
    
    print(f"Found {len(json_files)} JSON file(s) to process")
    print("=" * 60)
    print(f"Conversion factors:")
    print(f"  HA2meV = {HA2meV:.6f} (Hartree to meV)")
    print(f"  HA_BOHR_2_meV_ANG = {HA_BOHR_2_meV_ANG:.6f} (Ha/Bohr to meV/Å)")
    print("=" * 60)
    
    success_count = 0
    for json_path in json_files:
        if process_json_file(json_path, dry_run=args.dry_run):
            success_count += 1
    
    print("=" * 60)
    print(f"Processed {success_count}/{len(json_files)} files successfully")


if __name__ == "__main__":
    main()

