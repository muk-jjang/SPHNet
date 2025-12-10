#!/usr/bin/env python3
"""
LMDB 파일의 엔트리 개수를 확인하는 스크립트
"""

import os
import sys
import lmdb
import pickle
import argparse


def check_lmdb(lmdb_path, show_sample=False):
    """LMDB 파일의 엔트리 개수 확인
    
    Args:
        lmdb_path: LMDB 파일 경로
        show_sample: 샘플 데이터를 보여줄지 여부
    """
    if not os.path.exists(lmdb_path):
        print(f"❌ Error: LMDB file not found: {lmdb_path}")
        return
    
    print(f"📂 Opening LMDB: {lmdb_path}")
    
    try:
        env = lmdb.open(lmdb_path, readonly=True, lock=False, max_readers=1024, readahead=False)
        
        with env.begin() as txn:
            # "length" 키로 저장된 길이 확인
            length_bytes = txn.get("length".encode("ascii"))
            if length_bytes:
                stored_length = pickle.loads(length_bytes)
                print(f"📊 Stored length: {stored_length:,}")
            else:
                print("⚠️  No 'length' key found")
                stored_length = None
            
            # 실제 키 개수 확인
            stat = txn.stat()
            total_entries = stat['entries']
            print(f"📊 Total entries (stat): {total_entries:,}")
            
            # 실제 데이터 키 개수 확인 (length 제외)
            cursor = txn.cursor()
            data_keys = []
            for key, _ in cursor:
                if key != b"length":
                    data_keys.append(key)
            
            actual_data_count = len(data_keys)
            print(f"📊 Actual data entries: {actual_data_count:,}")
            
            # 비교
            if stored_length is not None:
                if stored_length == actual_data_count:
                    print(f"✅ Length matches: {stored_length:,} entries")
                else:
                    print(f"⚠️  Length mismatch: stored={stored_length:,}, actual={actual_data_count:,}")
            
            # 샘플 데이터 확인
            if show_sample and data_keys:
                print(f"\n📋 Sample data (first entry):")
                first_key = data_keys[0]
                data_bytes = txn.get(first_key)
                if data_bytes:
                    data = pickle.loads(data_bytes)
                    print(f"   Key: {int.from_bytes(first_key, byteorder='big')}")
                    print(f"   Keys in data: {list(data.keys())}")
                    if 'id' in data:
                        print(f"   ID: {data['id']}")
                    if 'num_nodes' in data:
                        print(f"   Num nodes: {data['num_nodes']}")
                    if 'pos' in data:
                        print(f"   Pos shape: {data['pos'].shape}")
                    if 'atoms' in data:
                        print(f"   Atoms shape: {data['atoms'].shape}")
        
        env.close()
        print(f"\n✅ Check completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check LMDB file entry count")
    parser.add_argument("lmdb_path", type=str, help="Path to LMDB file")
    parser.add_argument("--sample", action="store_true", help="Show sample data")
    
    args = parser.parse_args()
    
    check_lmdb(args.lmdb_path, show_sample=args.sample)

