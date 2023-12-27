import os 
import argparse 
from typing import Callable, Union

import numpy as np 
import torch
import torch.nn.functional as F
from pyscf import gto 
from torch_geometric.loader import DataLoader

from xequinet.utils import NetConfig, set_default_unit
from xequinet.utils import BuildMatPerMole, cal_orbital_and_energies
from xequinet.data import create_dataset 
from xequinet.nn import xQHNet


@torch.no_grad()
def test_over_dataset(test_loader:DataLoader, model:torch.nn.Module, device:torch.device, default_dtype:torch.dtype, build_matrix:Callable, output_file:str):
    model.eval() 
    # total_error_dict = {"total_mole": 0, "total_orb": 0, "mae_e": 0.0, "cosine_similarity": 0.0} 
    sum_mae_e, sum_cosine_similarity = 0.0, 0.0
    num_orbs, num_moles = 0, 0 
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
        real_fock = data.target_matrice 
        pred_e, pred_coeffs = cal_orbital_and_energies(overlap, pred_fock)
        real_e, real_coeffs = cal_orbital_and_energies(overlap, real_fock) 
        # orbital_energies, orbital_coeffs = cal_orbital_and_energies(overlap, pred_fock) 
        # orbital_coeffs are of shape (nbasis, norb)
        norb_occ = torch.sum(data.at_no) // 2 
        mae_e = F.l1_loss(pred_e, real_e)  
        pred_occ_orbs = pred_coeffs[:, :norb_occ]
        real_occ_orbs = real_coeffs[:, :norb_occ]
        # cos_similarity = torch.cosine_similarity(pred_coeffs, real_coeffs, dim=0).abs().mean()
        cosine_similarity = torch.cosine_similarity(pred_occ_orbs, real_occ_orbs, dim=0).abs().mean()
        # accumulation
        sum_mae_e += mae_e.item() * real_e.shape[0] 
        sum_cosine_similarity += cosine_similarity.item()
        num_moles += 1 
        num_orbs += norb_occ
        # write to log file 
        with open(output_file, 'a') as wf:
            wf.write(f"Test error for mole {test_batch_idx:6d}: \n")
            wf.write(f"mae_e: {mae_e.item():10.4f}, cosine similarity: {cosine_similarity.item():10.4f}\n")
        
    with open(output_file, 'a') as wf:
        wf.write(f"\nAverage test error: \n")
        wf.write(f"Orbital Energy Absolute Deviation: {sum_mae_e/num_orbs:10.4f}.\n")
        wf.write(f"Wavefunction Cosine Similarity: {sum_cosine_similarity/num_moles:10.4f}.\n")

        
@torch.no_grad()
def test_mae(test_loader:DataLoader, model, device:torch.device, output_file:str):
    model.eval()
    sum_loss_node, sum_loss_edge, sum_loss_total = 0.0, 0.0, 0.0
    num_node, num_edge, num_total = 0, 0, 0
    for data in test_loader:
        data = data.to(device)
        pred_pad_node, pred_pad_edge = model(data.at_no, data.pos, data.edge_index, data.fc_edge_index)
        batch_mask_node, batch_mask_edge = data.onsite_mask, data.offsite_mask
        pred_node, pred_edge = pred_pad_node[batch_mask_node], pred_pad_edge[batch_mask_edge]
        real_node, real_edge = data.node_label[batch_mask_node], data.edge_label[batch_mask_edge] 
        batch_pred = torch.cat([pred_node, pred_edge], dim=0)
        batch_real = torch.cat([real_node, real_edge], dim=0)
        node_l1loss = F.l1_loss(pred_node, real_node, reduce=False)
        edge_l1loss = F.l1_loss(pred_edge, real_edge, reduce=False)
        total_l1loss = F.l1_loss(batch_pred, batch_real, reduce=False)
        # loss accumulation 
        sum_loss_node += node_l1loss.sum().item()
        sum_loss_edge += edge_l1loss.sum().item()
        sum_loss_total += total_l1loss.sum().item()
        # count numel 
        num_node += real_node.numel()
        num_edge += real_edge.numel()
        num_total += batch_real.numel()
    
    with open(output_file, 'a') as wf:
        wf.write(f"Test MAE: node {sum_loss_node / num_node:10.8f}, edge {sum_loss_edge / num_edge:10.8f}, total {sum_loss_total / num_total:10.8f}.\n")
    

def main():
    # parse config 
    parser = argparse.ArgumentParser(description="XequiNet test script") 
    parser.add_argument(
        "--config", "-C", type=str, default="config.json",
        help="Configuration file path (default: config.json).",
    )
    parser.add_argument(
        "--ckpt", "-c", type=str, required=True,
        help="Xequinet checkpoint file. (XXX.pt containing 'model' and 'config')",
    )
    parser.add_argument(
        "--diag", default=False, action="store_true",
        help="Whether to diagonalize the Fock matrix for orbital energy and wavefunction.",
    )
    args = parser.parse_args() 

    # set device 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load checkpoint and config
    config = NetConfig.parse_file(args.config)
    ckpt = torch.load(args.ckpt, map_location=device)
    config.parse_obj(ckpt["config"])

    # set default unit
    set_default_unit(config.default_property_unit, config.default_length_unit)

    test_dataset = create_dataset(config, "test")
    batch_size = 1 if args.diag else config.vbatch_size
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=True, drop_last=False,
    )

    # adjust some configurations
    config.node_mean = 0.0; config.graph_mean = 0.0

    # build model 
    model = xQHNet(config) 
    model.load_state_dict(ckpt["model"])
    model.eval() 

    # test
    output_file = f"{config.run_name}_test.log"

    with open(output_file, 'w') as wf:
        wf.write("Error statistics of XequiNet testing for Matrix\n")
        wf.write(f"Unit: {config.default_property_unit} {config.default_length_unit}\n")
    
    if args.diag:
        matrix_builder = BuildMatPerMole(config.irreps_out, config.possible_elements, config.target_basisname) 
        default_dtype = torch.get_default_dtype() 
        test_over_dataset(test_loader, model, device, default_dtype, matrix_builder, output_file)
    else:
        test_mae(test_loader, model, device, output_file)
    

if __name__ == "__main__":
    main()

