import os
import sys
import shutil
import time
import lmdb
import pickle
import numpy as np

"""
Matrix Transforms Module

This module provides utility functions for Hamiltonian and overlap matrix transformations,
including orbital transformations and matrix transformations.
"""

import torch
import numpy as np
from argparse import Namespace
from torch_geometric.data import Data  
from torch_geometric.transforms.radius_graph import RadiusGraph
from torch_scatter import scatter
from sklearn.cluster import KMeans,SpectralClustering
import argparse

# Periodic Table of Elements
# -----------------------------------------------------------------------------------------------
#   │ 1  │ 2  │ 3  │ 4  │ 5  │ 6  │ 7  │ 8  │ 9  │ 10 │ 11 │ 12 │ 13 │ 14 │ 15 │ 16 │ 17 │ 18 │
#   ┌────┐                                                                               ┌────┐
# 1 │ H  │ 2                                                      13   14   15   16   17 │ He │
#   │ 1  │                                                                               │ 2  │
#   ├────┼────┐                                                 ┌────┬────┬────┬────┬────┼────┤
# 2 │ Li │ Be │                                                 │ B  │ C  │ N  │ O  │ F  │ Ne │
#   │ 3  │ 4  │                                                 │ 5  │ 6  │ 7  │ 8  │ 9  │ 10 │
#   ├────┼────┤                                                 ├────┼────┼────┼────┼────┼────┤
# 3 │ Na │ Mg │ 3    4    5    6    7    8    9    10   11   12 │ Al │ Si │ P  │ S  │ Cl │ Ar │
#   │ 11 │ 12 │                                                 │ 13 │ 14 │ 15 │ 16 │ 17 │ 18 │
#   ├────┼────┼────┬────┬────┬────┬────┬────┬────┬────┬────┬────┼────┼────┼────┼────┼────┼────┤
# 4 │ K  │ Ca │ Sc │ Ti │ V  │ Cr │ Mn │ Fe │ Co │ Ni │ Cu │ Zn │ Ga │ Ge │ As │ Se │ Br │ Kr │
#   │ 19 │ 20 │ 21 │ 22 │ 23 │ 24 │ 25 │ 26 │ 27 │ 28 │ 29 │ 30 │ 31 │ 32 │ 33 │ 34 │ 35 │ 36 │
#   ├────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┤
# 5 │ Rb │ Sr │ Y  │ Zr │ Nb │ Mo │ Tc │ Ru │ Rh │ Pd │ Ag │ Cd │ In │ Sn │ Sb │ Te │ I  │ Xe │
#   │ 37 │ 38 │ 39 │ 40 │ 41 │ 42 │ 43 │ 44 │ 45 │ 46 │ 47 │ 48 │ 49 │ 50 │ 51 │ 52 │ 53 │ 54 │
#   ├────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┤
# 6 │ Cs │ Ba │ L* │ Hf │ Ta │ W  │ Re │ Os │ Ir │ Pt │ Au │ Hg │ Tl │ Pb │ Bi │ Po │ At │ Rn │
#   │ 55 │ 56 │ -- │ 72 │ 73 │ 74 │ 75 │ 76 │ 77 │ 78 │ 79 │ 80 │ 81 │ 82 │ 83 │ 84 │ 85 │ 86 │
#   ├────┼────┼────┼────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┘
# 7 │ Fr │ Ra │ A* │
#   │ 87 │ 88 │ -- │
#   └────┴────┴────┘
# ----------------------------------------------------------------------------------------------
# L* (Lanthanide)
#   ┌────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┐
# 6 │ La │ Ce │ Pr │ Nd │ Pm │ Sm │ Eu │ Gd │ Tb │ Dy │ Ho │ Er │ Tm │ Yb │ Lu │
#   │ 57 │ 58 │ 59 │ 60 │ 61 │ 62 │ 63 │ 64 │ 65 │ 66 │ 67 │ 68 │ 69 │ 70 │ 71 │
#   └────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┘
# A* (Actinide)
#   ┌────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┐
# 7 │ Ac │ Th │ Pa │ U  │ Np │ Pu │ Am │ Cm │ Bk │ Cf │ Es │ Fm │ Md │ No │ Lr │
#   │ 89 │ 90 │ 91 │ 92 │ 93 │ 94 │ 95 │ 96 │ 97 │ 98 │ 99 │ 100│ 101│ 102│ 103│
#   └────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┘


# Atomic numbers 1 to 103
CHEMICAL_SYMBOLS = [
    "n",
    "H", "He",
    "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe",
    "Cs", "Ba",
    "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At","Rn",
    "Fr", "Ra",
    "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr"
    ] 

convention_dict = {
    'pyscf_def2-tzvp_to_e3nn': Namespace(
        atom_to_orbitals_map={
            1:  'sssp', # H
            6:  'ssssspppddf', # C
            7:  'ssssspppddf', # N
            8:  'ssssspppddf', # O
            9:  'ssssspppddf', # F
            15: 'ssssspppppddf', # P
            16: 'ssssspppppddf', # S
            17: 'ssssspppppddf', # Cl
        },
        orbital_idx_map={
            's': [0],
            'p': [1, 2, 0],
            'd': [0, 1, 2, 3, 4],
            'f': [0, 1, 2, 3, 4, 5, 6],
        },
        orbital_sign_map={
            's': [1],
            'p': [1, 1, 1],
            'd': [1, 1, 1, 1, 1],
            'f': [1, 1, 1, 1, 1, 1, 1],
        },
        orbital_order_map={
            1:  [0, 1, 2, 3],   
            6:  list(range(11)),
            7:  list(range(11)),
            8:  list(range(11)),
            9:  list(range(11)),
            15: list(range(13)),
            16: list(range(13)),
            17: list(range(13)),
        },
        max_block_size= 37, # 5s + 5p + 2d + 1f = 5 + 15 + 10 + 7 = 37
    ),
    "pyscf_631G_to_e3nn": Namespace(
        # 6-31G basis set convention used by PySCF
        # p orbitals: [px, py, pz] -> reordered to [pz, px, py] for compatibility
        atom_to_orbitals_map={
            1: "ss",
            6: "ssspp",
            7: "ssspp",
            8: "ssspp",
            9: "ssspp",
        },
        orbital_idx_map={
            "s": [0],
            "p": [1, 2, 0],   # p: [pz, px, py]
            "d": [0, 1, 2, 3, 4]
            },
        orbital_sign_map={
            "s": [1],
            "p": [1, 1, 1],
            "d": [1, 1, 1, 1, 1],
        },
        orbital_order_map={
            1: [0, 1],           # H: 2 orbitals (s, s)
            6: [0, 1, 2, 3, 4],  # C: 5 orbitals (s, s, s, p, p)
            7: [0, 1, 2, 3, 4],  # N: 5 orbitals (s, s, s, p, p)
            8: [0, 1, 2, 3, 4],  # O: 5 orbitals (s, s, s, p, p)
            9: [0, 1, 2, 3, 4],  # F: 5 orbitals (s, s, s, p, p)
        },
        max_block_size=9, # 3s + 2p = 3 + 6 = 9
    ),
    "pyscf_def2svp_to_e3nn": Namespace(
        # def2-SVP basis set convention used by PySCF
        # p orbitals: [px, py, pz] -> reordered to [py, pz, px] for compatibility
        atom_to_orbitals_map={
            1: "ssp",      # H: 3 orbitals (s, s, p)
            6: "sssppd",   # C: 6 orbitals (s, s, s, p, p, d)
            7: "sssppd",   # N: 6 orbitals (s, s, s, p, p, d)
            8: "sssppd",   # O: 6 orbitals (s, s, s, p, p, d)
            9: "sssppd",   # F: 6 orbitals (s, s, s, p, p, d)
        },
        orbital_idx_map={
            "s": [0],
            "p": [1, 2, 0],
            "d": [0, 1, 2, 3, 4],
        },  # p: [py, pz, px]
        orbital_sign_map={
            "s": [1],
            "p": [1, 1, 1],
            "d": [1, 1, 1, 1, 1],
        },
        orbital_order_map={
            1: [0, 1, 2],      # H: 3 orbitals (s, s, p)
            6: [0, 1, 2, 3, 4, 5],  # C: 6 orbitals (s, s, s, p, p, d)
            7: [0, 1, 2, 3, 4, 5],  # N: 6 orbitals (s, s, s, p, p, d)
            8: [0, 1, 2, 3, 4, 5],  # O: 6 orbitals (s, s, s, p, p, d)
            9: [0, 1, 2, 3, 4, 5],  # F: 6 orbitals (s, s, s, p, p, d)
        },
        max_block_size= 14, # 3s + 2p + 1d = 3 + 6 + 5 = 14
    ),
    'e3nn_to_pyscf_def2-tzvp': Namespace(
        atom_to_orbitals_map={
            1:  'sssp', # H
            6:  'ssssspppddf', # C
            7:  'ssssspppddf', # N
            8:  'ssssspppddf', # O
            9:  'ssssspppddf', # F
            15: 'ssssspppppddf', # P
            16: 'ssssspppppddf', # S
            17: 'ssssspppppddf', # Cl
        },
        orbital_idx_map={
            's': [0],
            'p': [2, 0, 1],
            'd': [0, 1, 2, 3, 4],
            'f': [0, 1, 2, 3, 4, 5, 6],
        },
        orbital_sign_map={
            's': [1],
            'p': [1, 1, 1],
            'd': [1, 1, 1, 1, 1],
            'f': [1, 1, 1, 1, 1, 1, 1],
        },
        orbital_order_map={
            1:  [0, 1, 2, 3],   
            6:  list(range(11)),
            7:  list(range(11)),
            8:  list(range(11)),
            9:  list(range(11)),
            15: list(range(13)),
            16: list(range(13)),
            17: list(range(13)),
        },
        max_block_size= 37, # 5s + 5p + 2d + 1f = 5 + 15 + 10 + 7 = 37
    ),
    "e3nn_to_pyscf_def2svp": Namespace(
        # Special convention to convert back to PySCF's native orbital ordering
        # This is used when you have matrices in a different convention and need to
        # convert them back to PySCF's expected format for further processing
        # 
        # Key differences from pyscf_def2svp:
        # - Same orbital types (def2-SVP basis) but different p-orbital ordering
        # - p orbitals: [py, pz, px] -> reordered to [px, py, pz] (PySCF native)
        # - This allows seamless integration with PySCF calculations
        #
        # Use case: Convert matrices from other software (e.g., ORCA, Gaussian) 
        # back to PySCF format for density matrix calculations or further analysis
        atom_to_orbitals_map={
            1: "ssp",      # H: 3 orbitals (s, s, p)
            6: "sssppd",   # C: 6 orbitals (s, s, s, p, p, d)
            7: "sssppd",   # N: 6 orbitals (s, s, s, p, p, d)
            8: "sssppd",   # O: 6 orbitals (s, s, s, p, p, d)
            9: "sssppd",   # F: 6 orbitals (s, s, s, p, p, d)
        },
        orbital_idx_map={
            "s": [0],
            "p": [2, 0, 1],
            "d": [0, 1, 2, 3, 4],
        },  # p: [px, py, pz] (PySCF native)
        orbital_sign_map={
            "s": [1],
            "p": [1, 1, 1],
            "d": [1, 1, 1, 1, 1],
        },
        orbital_order_map={
            1: [0, 1, 2],           # H: 3 orbitals (s, s, p)
            6: [0, 1, 2, 3, 4, 5],  # C: 6 orbitals (s, s, s, p, p, d)
            7: [0, 1, 2, 3, 4, 5],  # N: 6 orbitals (s, s, s, p, p, d)
            8: [0, 1, 2, 3, 4, 5],  # O: 6 orbitals (s, s, s, p, p, d)
            9: [0, 1, 2, 3, 4, 5],  # F: 6 orbitals (s, s, s, p, p, d)
        },
        max_block_size= 14,
    ),
}
convention_dict["pyscf_def2svp"]   = convention_dict["pyscf_def2svp_to_e3nn"]

def _get_orbital_mask(basis = "def2-svp"):
    """Get orbital masks for different atomic numbers.
    
    Args:
        ORBITAL_1S_2S_INDICES (torch.Tensor, optional): Indices for 1s and 2s orbitals.
            Defaults to [0, 1].
        ORBITAL_2P_INDICES (torch.Tensor, optional): Indices for 2p orbitals.
            Defaults to [3, 4, 5].
        ORBITAL_MASK_SIZE_LINE2 (int, optional): Size of orbital mask for line 2 elements.
            Defaults to 14.
    
    Returns:
        dict: Dictionary mapping atomic numbers to their orbital masks.
    """
    assert basis in ["def2-svp", "def2-tzvp"], f"Invalid basis: {basis}, only def2-svp and def2-tzvp are supported now"
    orbital_mask = {}
    
    if basis == "631G":
        pass

    elif basis == "def2-svp":        
        MAX_ORBITAL_LENGTH = 14
        MAX_ATOMIC_NUMBER = 9
        DEFAULT_ORBITAL_INDICES = torch.arange(MAX_ORBITAL_LENGTH)
        orbital_mask[1] = torch.tensor([0, 1, 3, 4, 5]) # ssp
        orbital_mask[6] = DEFAULT_ORBITAL_INDICES
        orbital_mask[7] = DEFAULT_ORBITAL_INDICES
        orbital_mask[8] = DEFAULT_ORBITAL_INDICES
        orbital_mask[9] = DEFAULT_ORBITAL_INDICES
        
    elif basis == "def2-tzvp":
        MAX_ORBITAL_LENGTH = 37
        MAX_ATOMIC_NUMBER = 17
        DEFAULT_ORBITAL_INDICES_1 = torch.tensor([
            0, 1, 2, 3, 4,      # 1s-5s
            5, 6, 7,            # 2p
            8, 9, 10,           # 3p
            11, 12, 13,         # 4p (skip 5p,6p)
            20, 21, 22, 23, 24, # 3d
            25, 26, 27, 28, 29, # 4d
            30, 31, 32, 33, 34, 35, 36 # 4f
        ])
        DEFAULT_ORBITAL_INDICES_2 = torch.arange(MAX_ORBITAL_LENGTH)
        orbital_mask[1] = torch.tensor([0, 1, 2, 5, 6, 7]) # sssp
        orbital_mask[6] = DEFAULT_ORBITAL_INDICES_1
        orbital_mask[7] = DEFAULT_ORBITAL_INDICES_1
        orbital_mask[8] = DEFAULT_ORBITAL_INDICES_1
        orbital_mask[9] = DEFAULT_ORBITAL_INDICES_1
        orbital_mask[15] = DEFAULT_ORBITAL_INDICES_2
        orbital_mask[16] = DEFAULT_ORBITAL_INDICES_2
        orbital_mask[17] = DEFAULT_ORBITAL_INDICES_2

    return orbital_mask


def data_to_lmdb(data_path, name, shard_num, prefix="_shard"):
    if shard_num == -1:
        if name == "water":
            shard_num = 2
        elif name == "ethanol":
            shard_num = 8
        elif name == "malondialdehyde":
            shard_num = 8
        elif name == "uracil":
            shard_num = 16
        elif name == "aspirin":
            shard_num = 21
        elif name == "naphthalene":
            shard_num = 19
        elif name == "salicylic_acid":
            shard_num = 16

    folder = os.path.join(data_path, name + prefix)
    processed_path = os.path.join(folder, 'processed')
    print('processed_path: ', processed_path)
    shard_dir_name = 'lmdbs'
    lmdb_path_list = [os.path.join(processed_path, shard_dir_name, f"shard_{i:03d}.lmdb") for i in range(shard_num)]
    print('lmdb_path_list: ', lmdb_path_list)
    return lmdb_path_list, shard_num

def get_env_from_lmdb(lmdb_path_list):
    data ={}
    for idx, lmdb_path in enumerate(lmdb_path_list):
        env = lmdb.open(lmdb_path, readonly=True,lock=False,max_readers=1024, readahead=False)
        data[idx] = env
        print('env: ', env)
    return data

def get_data_from_env(idx, env, print_data=False):    
    with env.begin() as txn:
        key = int(idx).to_bytes(length=4, byteorder="big")
        data = txn.get(key)
        if data is None:
            raise KeyError(f"Key {idx} not found in LMDB")        
        data = pickle.loads(data)
        if print_data:
            print('data: ', data)
        
        #data들 np.frombuffer변환해서 반환
    # print('data: ', data.keys())  # 불필요한 print 제거
    data['atoms'] = np.frombuffer(data['atoms'], np.int32)
    data['pos'] = np.frombuffer(data['pos'], np.float64).reshape(-1,3)
    data['energy'] = np.array(data['energy'], dtype=np.float32)
    data['force'] = np.frombuffer(data['force'], np.float32).reshape(-1,3)
    data['dft_energy'] = np.frombuffer(data['dft_energy'], np.float64)
    data['dft_forces'] = np.frombuffer(data['dft_forces'], np.float64).reshape(-1,3)
    data['h_dim'] = data['h_dim']
    data['orbital_energies'] = np.frombuffer(data['orbital_energies'], np.float64)
    data['hamiltonian'] = unpack_upper_triangle(np.frombuffer(data['packed_hamiltonian'], np.float64), data['h_dim'])
    data['overlap'] = unpack_upper_triangle(np.frombuffer(data['packed_overlap'], np.float64), data['h_dim'])
    data['initial_hamiltonian'] = unpack_upper_triangle(np.frombuffer(data['packed_initial_hamiltonian'], np.float64), data['h_dim'])
    return data

## qhflow func
def unpack_upper_triangle(packed: np.ndarray, n: int):
    """Unpack upper triangle array back into a symmetric matrix.
    
    Args:
        packed (np.ndarray): 1D array containing upper triangle elements.
        n (int): Size of the original square matrix.
    
    Returns:
        np.ndarray: Reconstructed symmetric matrix.
    """
    M = np.zeros((n,n), dtype=packed.dtype)
    iu = np.triu_indices(n)
    M[iu] = packed
    M[(iu[1], iu[0])] = packed  # mirror
    return M

def construct_orbital_l_index(AO_lm_index):
    idx = 0
    AO_l_index = []
    while True:
        if idx >= len(AO_lm_index):
            break
        AO_l_index.append(AO_lm_index[idx].item())
        idx += 2 * AO_lm_index[idx] + 1
    return torch.tensor(AO_l_index)

def post_process_data(data):
    """Post-process data dictionary to add full edge index.
    
    Args:
        data (dict): Dictionary containing molecular data
        
    Returns:
        dict: Dictionary with added full_edge_index field
    """
    # Convert atoms to tensor if needed
    atoms = torch.from_numpy(data['atoms']) if isinstance(data['atoms'], np.ndarray) else data['atoms']
    
    # Create full edge index (all pairs except self-loops)
    num_atoms = len(atoms)
    edge_index = []
    for i in range(num_atoms):
        for j in range(num_atoms):
            if i != j:
                edge_index.append([i, j])
    edge_index = torch.tensor(edge_index, dtype=torch.int64).t().contiguous()
    
    # Add to data dictionary
    data['full_edge_index'] = edge_index
    
    return data

def build_label(g, num_labels = 16,method = 'kmeans'):
    if num_labels == g.pos.shape[0]:
        g.labels = torch.arange(g.pos.shape[0]).long()
    elif num_labels == 1:
        g.labels = torch.zeros(g.pos.shape[0]).long()
    else:
        try:
            if method == 'kmeans':
                g.labels = torch.tensor(KMeans(n_clusters=num_labels, random_state=0).fit_predict(g.pos.numpy())).long()
            elif method == 'spectral':
                g.labels = torch.tensor(SpectralClustering(
                                                    n_clusters=num_labels,
                                                    eigen_solver="arpack",
                                                    affinity="nearest_neighbors",
                                                ).fit_predict(g.pos.numpy())).long()
        except:
            # Except for # nodes < num_labels
            g.labels = torch.arange(g.num_nodes).long()
    g.num_labels = num_labels

def write_single_to_lmdb(env, data_dict, current_length):
    """Write a single data entry to an already-opened LMDB environment.
    
    Args:
        env: Already-opened LMDB environment
        data_dict: Single data dictionary to write
        current_length: Current number of entries (for append mode)
    
    Returns:
        int: New total length after writing
    """
    with env.begin(write=True) as txn:
        key = current_length.to_bytes(length=4, byteorder='big')
        value = pickle.dumps(data_dict)
        txn.put(key, value)
        
        # Update length
        new_length = current_length + 1
        txn.put("length".encode("ascii"), pickle.dumps(new_length))
        return new_length  

if __name__ == "__main__":
    # ethanol
    parser = argparse.ArgumentParser(description="Data transformation")
    parser.add_argument("--name", type=str, help="Name of the dataset")
    args = parser.parse_args()
    data_path = '/ssd1/qhflow-mlff/dataset'
    name = args.name if args.name else 'malondialdehyde'
    lmdb_path_list, shard_num = data_to_lmdb(data_path, name, -1)
    
    # 출력 경로 설정
    output_path = '/home/sungjun/repos/SPHNet/dataset2/' + name
    os.makedirs(output_path, exist_ok=True)
    lmdb_file_path = os.path.join(output_path, 'data.lmdb')
    
    # 변수 초기화
    start_time = time.time()
    total_processed = 0
    global_idx_offset = 0
    PROGRESS_INTERVAL = 100  # 진행 상황 출력 간격
    
    def format_time(seconds):
        """시간을 읽기 쉬운 형식으로 변환"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours}시간 {minutes}분 {secs}초"
    
    def get_timestamp():
        """현재 시간을 문자열로 반환"""
        return time.strftime("%H:%M:%S", time.localtime())
    
    # 기존 파일이 있으면 삭제 (새로 시작)
    print(f"[{get_timestamp()}] Checking for existing LMDB file...")
    if os.path.exists(lmdb_file_path):
        print(f"[{get_timestamp()}] Removing existing LMDB file (this may take a while for large files)...")
        remove_start = time.time()
        shutil.rmtree(lmdb_file_path)
        remove_time = time.time() - remove_start
        print(f"[{get_timestamp()}] ✓ Removed existing LMDB file in {remove_time:.1f}초: {lmdb_file_path}")
    else:
        print(f"[{get_timestamp()}] No existing LMDB file found.")
    
    print(f"[{get_timestamp()}] Initializing configuration...")
    conv = convention_dict['pyscf_def2svp']
    mask = _get_orbital_mask(basis = "def2-svp")
    chemical_symbols = CHEMICAL_SYMBOLS
    
    # 출력 LMDB 환경을 한 번만 열기 (매번 열고 닫는 것보다 훨씬 효율적)
    print(f"[{get_timestamp()}] Opening output LMDB file...")
    output_env = lmdb.open(lmdb_file_path, map_size=80 * 1024 * 1024 * 1024)
    current_length = 0
    
    # 초기 length 설정 (새 파일인 경우)
    with output_env.begin(write=True) as txn:
        length_bytes = txn.get("length".encode("ascii"))
        if length_bytes is None:
            txn.put("length".encode("ascii"), pickle.dumps(0))
            current_length = 0
        else:
            current_length = pickle.loads(length_bytes)
    
    print(f"[{get_timestamp()}] 시작: {shard_num}개 샤드 처리 예정")
    
    try:
        for shard_idx in range(shard_num): # shard_num
            shard_start_time = time.time()
            print(f"[{get_timestamp()}] Processing shard {shard_idx+1}/{shard_num}...")
            
            # 샤드별로 LMDB 환경 열기 (한 번에 하나만) - 동적으로 크기 확인
            print(f"[{get_timestamp()}]   Opening shard {shard_idx+1}/{shard_num}: {lmdb_path_list[shard_idx]}...")
            open_start = time.time()
            env_data = lmdb.open(lmdb_path_list[shard_idx], readonly=True, lock=False, max_readers=1024, readahead=False)
            open_time = time.time() - open_start
            if open_time > 5:
                print(f"[{get_timestamp()}]   ⚠️  Opening shard took {open_time:.2f}초 (slow I/O)")
            
            # 샤드 크기 동적으로 확인
            with env_data.begin() as txn:
                shard_size = txn.stat()['entries']
            print(f"[{get_timestamp()}]   ✓ Shard {shard_idx+1}/{shard_num}: {shard_size} entries")
            
            for local_idx in range(shard_size): # shard_size
                try:
                    # 저장할 때는 전체 데이터셋에서의 전역 인덱스 사용
                    global_idx = global_idx_offset + local_idx # 전체 데이터셋에서의 전역 인덱스
                    data_processed = Data()
                    # 각 shard는 0부터 시작하는 로컬 인덱스를 사용
                    data = get_data_from_env(global_idx, env_data)
                    atoms_num = data['atoms'].shape[0]
                    # print('data keys: ', data.keys())
                    # print('dft_energy: ', data['dft_energy'].shape)
                    # print('forces: ', data['force'].shape)
                    # print('orbital_energies: ', data['orbital_energies'].shape)
                    # print('data overlap: ', data['overlap'].shape)
                    
                    data_processed.num_nodes = atoms_num
                    data_processed.pos = torch.tensor(data['pos'])
                    neighbor_finder = RadiusGraph(r=3)
                    data_processed = neighbor_finder(data_processed)
                    min_nodes_foreachGroup = 4
                    build_label(data_processed, num_labels=int(atoms_num/min_nodes_foreachGroup), method='kmeans')

                    data_dict = {
                        "id": global_idx,
                        "pos": np.array(data['pos']),
                        "atoms": data['atoms'],
                        "edge_index": data_processed['edge_index'],
                        "labels": data_processed['labels'],
                        "num_nodes": atoms_num,
                        "Ham": data['hamiltonian'],
                        "Ham_init": data['initial_hamiltonian'],
                        "energy": data['dft_energy'],
                        "forces": data['dft_forces'],
                        "overlap": data['overlap'],
                        "orbital_energies": data['orbital_energies'],
                    }
                    
                    # 각 엔트리마다 바로 저장
                    current_length = write_single_to_lmdb(output_env, data_dict, current_length)
                    total_processed += 1
                    
                    # 진행 상황 출력 (PROGRESS_INTERVAL개마다)
                    if (local_idx + 1) % PROGRESS_INTERVAL == 0:
                        progress_pct = ((local_idx + 1) / shard_size) * 100
                        elapsed = time.time() - start_time
                        avg_time_per_entry = elapsed / total_processed if total_processed > 0 else 0
                        print(f"[{get_timestamp()}] Progress: {local_idx + 1}/{shard_size} "
                              f"({progress_pct:.1f}%) | Shard {shard_idx+1}/{shard_num} | "
                              f"Total: {total_processed} | Avg: {avg_time_per_entry:.3f}초/entry")
                        
                except Exception as e:
                    print(f"  Error processing entry {local_idx}: {e}")
                    continue
            
            shard_elapsed = time.time() - shard_start_time
            shard_elapsed_str = format_time(shard_elapsed)
            print(f"[{get_timestamp()}] ✓ Shard {shard_idx+1} completed in {shard_elapsed_str} | "
                  f"Total processed: {total_processed} entries")
            sys.stdout.flush()
            
            # 메모리 정리
            env_data.close()
            import gc
            gc.collect()
            
            # 다음 샤드를 위해 오프셋 업데이트
            global_idx_offset += shard_size
        
        total_elapsed = time.time() - start_time
        total_elapsed_str = format_time(total_elapsed)
        avg_time_per_entry = total_elapsed / total_processed if total_processed > 0 else 0
        
        print(f"\n[{get_timestamp()}] ✓ All {shard_num} shards have been processed successfully!")
        print(f"[{get_timestamp()}] ✓ Total entries saved: {total_processed}")
        print(f"[{get_timestamp()}] ✓ Total time: {total_elapsed_str} (avg {avg_time_per_entry:.3f}초/entry)")
    finally:
        # 모든 처리가 끝난 후 출력 LMDB 환경 닫기
        output_env.close()