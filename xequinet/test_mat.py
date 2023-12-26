import os 
import argparse 
from typing import Callable, Union

import numpy as np 
import torch
import torch.nn.functional as F
from pyscf import gto 
from torch_geometric.loader import DataLoader

from xequinet.utils import BuildMatPerMole, cal_orbital_and_energies


@torch.no_grad()
def test_over_dataset(test_loader:DataLoader, model, device:torch.device, default_dtype:torch.dtype, build_matrix:Callable):
    model.eval() 
    total_error_dict = {"total_mole": 0, "total_orb": 0, "mae_e": 0.0, "cosine_similarity": 0.0} 
    
    for test_batch_idx, data in enumerate(test_loader):
        data = data.to(device) 
        res_node, res_edge = model(data.at_no, data.pos, data.edge_index, data.fc_edge_index) 
        node_mask, edge_mask = data.onsite_mask, data.offsite_mask
        mol = gto.Mole()
        t = [
            [data.at_no[atom_idx].cpu().item(), data.pos[atom_idx].cpu().numpy()]
            for atom_idx in range(data.num_nodes)
        ]
        mol.build(verbose=0, atom=t, basis='def2svp', unit='ang')
        overlap = torch.from_numpy(mol.intor("int1e_ovlp"))
        overlap = overlap.to(default_dtype).to(device)
        pred_fock = build_matrix(res_node, res_edge, node_mask, edge_mask, data.at_no, data.fc_edge_index)
        real_fock = data.hamiltonian 
        pred_e, pred_coeffs = cal_orbital_and_energies(overlap, pred_fock)
        real_e, real_coeffs = cal_orbital_and_energies(overlap, real_fock) 
        # orbital_energies, orbital_coeffs = cal_orbital_and_energies(overlap, pred_fock) 
        # norb_occ = torch.sum(data.at_no) // 2 
        mae_e = F.l1_loss(pred_e, real_e) 
        cos_similarity = torch.cosine_similarity(pred_coeffs, real_coeffs, dim=0).abs().mean()
        total_error_dict["mae_e"] += mae_e.item() * real_e.shape[0] 
        total_error_dict["cos_similarity"] += cos_similarity.item()
        total_error_dict["total_mole"] += 1 
        total_error_dict["total_orb"] += real_e.shape[0] 

    total_error_dict["mae_e"] = total_error_dict["mae_e"] / total_error_dict["total_orb"] 
    total_error_dict["cos_similarity"] = total_error_dict["cos_similarity"] / total_error_dict["total_mole"]

    return total_error_dict

        
@torch.no_grad()
def test_mae(test_loader:DataLoader, model, device:torch.device):
    model.eval()
    total_error_dict = {"total_loss": 0.0, "node_loss": 0.0, "edge_loss":0.0, "node_item": 0, "edge_item": 0, "total_item": 0} 
    for data in test_loader:
        data = data.to(device)
        pred_pad_node, pred_pad_edge = model(data.at_no, data.pos, data.edge_index, data.fc_edge_index)
        batch_mask_node, batch_mask_edge = data.onsite_mask, data.offsite_mask
        pred_node, pred_edge = pred_pad_node[batch_mask_node], pred_pad_edge[batch_mask_edge]
        real_node, real_edge = data.node_label[batch_mask_node], data.edge_label[batch_mask_edge] 
        batch_pred = torch.cat([pred_node, pred_edge], dim=0)
        batch_real = torch.cat([real_node, real_edge], dim=0)
        node_l1loss = F.l1_loss(pred_node, real_node, reduction="sum")
        edge_l1loss = F.l1_loss(pred_edge, real_edge, reduction="sum")
        total_l1loss = F.l1_loss(batch_pred, batch_real, reduction="sum")
        total_error_dict["node_loss"] += node_l1loss.item() * real_node.numel() 
        total_error_dict["edge_loss"] += edge_l1loss.item() * real_edge.numel()
        total_error_dict["total_loss"] += total_l1loss.item() * batch_real.numel()
        total_error_dict["node_item"] += real_node.numel()
        total_error_dict["edge_item"] += real_edge.numel()
        total_error_dict["total_item"] += batch_real.numel()
    
    total_error_dict["node_loss"] /= total_error_dict["node_item"]
    total_error_dict["edge_loss"] /= total_error_dict["edge_item"]
    total_error_dict["total_loss"] /= total_error_dict["total_item"]
    return total_error_dict




