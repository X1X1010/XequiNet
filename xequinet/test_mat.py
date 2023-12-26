import os 
import argparse 
from typing import Callable

import numpy as np 
import torch
from pyscf import gto 
from torch_geometric.loader import DataLoader

from xequinet.utils import BuildMatPerMole, cal_orbital_and_energies


@torch.no_grad()
def test_over_dataset(test_loader:DataLoader, model, device:torch.device, default_dtype:torch.dtype, build_matrix:Callable):
    model.eval() 

    for test_batch_idx, data in enumerate(test_loader):
        data = data.to(device) 
        res_node, res_edge = model(data) 
        node_mask, edge_mask = data.onsite_mask, data.offsite_mask

        mol = gto.Mole()
        t = [
            [data.at_no[atom_idx].cpu().item(), data.pos[atom_idx].cpu().numpy()]
            for atom_idx in range(data.num_nodes)
        ]
        mol.build(verbose=0, atom=t, basis='def2svp', unit='ang')
        overlap = torch.from_numpy(mol.intor("int1e_ovlp")).unsqueeze(0)
        overlap = overlap.to(default_dtype).to(device)
        pred_fock = build_matrix(res_node, res_edge, node_mask, edge_mask, data.at_no, data.fc_edge_index)
        real_fock = data.hamiltonian 
        pred_e, pred_coeffs = cal_orbital_and_energies(overlap, pred_fock)
        real_e, real_coeffs = cal_orbital_and_energies(overlap, real_fock) 
        # orbital_energies, orbital_coeffs = cal_orbital_and_energies(overlap, pred_fock) 
        # norb_occ = torch.sum(data.at_no) // 2 
        mae_e = torch.nn.functional.l1_loss(real_e, pred_e) 
        similar_c = torch.cosine_similarity(pred_coeffs, real_coeffs, dim=0).abs().mean()
        




