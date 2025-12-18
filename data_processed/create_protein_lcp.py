#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import lmdb
import pickle
from tqdm import tqdm
import copy
import logging

import torch
import numpy as np
import networkx as nx
from torch_geometric.data import Data
from prody import parsePDB, calcDistance
import Bio.PDB as PDB  


amino_acid_index = {
    'ALA': 0, 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4,
    'GLN': 5, 'GLU': 6, 'GLY': 7, 'HIS': 8, 'ILE': 9,
    'LEU': 10, 'LYS': 11, 'MET': 12, 'PHE': 13, 'PRO': 14,
    'SER': 15, 'THR': 16, 'TRP': 17, 'TYR': 18, 'VAL': 19
}

def parse_protein_data(protein_node_xyz, protein_seq, threshold=8.0):
    try:

        residues = []
        num_residues = len(protein_seq)
        for i in range(num_residues):
            class MockResidue:
                def __init__(self, resname, coords):
                    self.resname = resname
                    self.coords = coords

                def getResname(self):
                    return self.resname

                def getCoords(self):
                    return self.coords

            residue = MockResidue(triple_letter, protein_node_xyz[i])
            residues.append(residue)

        node_features = []
        for i, res in enumerate(residues):
            aa_type = np.zeros(20)
            res_name = res.getResname()
            if res_name in amino_acid_index:
                aa_type[amino_acid_index[res_name]] = 1
            node_features.append(aa_type)

        positions = np.array([res.getCoords() for res in residues])
        edge_index = []
        dist_matrix = np.sqrt(((positions[:, None] - positions) ** 2).sum(-1))

        for i in range(len(residues)):
            for j in range(i + 1, len(residues)):
                if dist_matrix[i, j] <= threshold:
                    edge_index.append([i, j])

        data = Data(
            x=torch.tensor(node_features, dtype=torch.float),
            edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        )

        return data

data_path = r"./pdbbind2020/dataset/processed"
protein_db = lmdb.open(os.path.join(data_path, 'protein_1d_3d.lmdb'), readonly=True)

PDB = []
all_data = []

with protein_db.begin(write=False) as txn:
    cursor = txn.cursor()
    for index, (key, value) in enumerate(tqdm(cursor, total=count), start=1):

        pdb_id = key.decode()
        PDB.append(pdb_id)
        seq_in_id = pickle.loads(value)[1].tolist()
        seq_in_str = ''.join([num_to_letter[a] for a in seq_in_id])
        protein_node_xyz, protein_seq = pickle.loads(txn.get(pdb_id.encode()))
        graph = parse_protein_data(protein_node_xyz.numpy(), protein_seq.tolist(),threshold=8.0)
        if graph is not None:   
            lcp = LocalCurvatureProfile()
            data_with_lcp, * = lcp.compute_orc(graph)
            all_data.append(data_with_lcp)

output_file = './processed_protein.pkl'
with open(output_file, 'wb') as f:
    pickle.dump(all_data, f)

