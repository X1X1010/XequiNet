import argparse

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_scatter import scatter_sum 

from xequinet.data import create_dataset
from xequinet.nn import resolve_model
from xequinet.utils import (
    NetConfig,
    unit_conversion, set_default_unit, get_default_unit,
    gen_3Dinfo_str,
)


@torch.no_grad()
def test_scalar(model, test_loader, device, outfile, output_dim=1, verbose=0):
    p_unit, l_unit = get_default_unit()
    sum_loss = torch.zeros(output_dim, device=device)
    num_mol = 0
    wf = open(outfile, 'a')
    for data in test_loader:
        data = data.to(device)
        pred = model(data)
        if hasattr(data, "base_y"):
            pred += data.base_y
        real = data.y
        error = real - pred
        sum_loss += error.abs().sum(dim=0)
        if verbose >= 1:
            for imol in range(len(data.y)):
                at_no = data.at_no[data.batch == imol]
                coord = data.pos[data.batch == imol] * unit_conversion(l_unit, "Angstrom")
                wf.write(f"mol {num_mol + imol + 1}\n")
                if verbose >= 2:  # print atom coordinates
                    wf.write(gen_3Dinfo_str(at_no, coord, title="Coordinates (Angstrom)"))
                    wf.write(f"Charge {int(data.charge[imol].item())}   Multiplicity {int(data.spin[imol].item()) + 1}\n")
                wf.write(f"Real:")
                wf.write("".join([f"{r.item():15.9f} " for r in real[imol]]))
                wf.write(f"    Predict:")
                wf.write("".join([f"{p.item():15.9f}" for p in pred[imol]]))
                wf.write(f"    Error:")
                wf.write("".join([f"{l.item():15.9f}" for l in error[imol]]))
                wf.write(f"    ({p_unit})\n\n")
                wf.flush()
        num_mol += len(data.y)
    avg_loss = sum_loss / num_mol
    wf.write(f"Test MAE:")
    wf.write("".join([f"{l:15.9f}" for l in avg_loss]))
    wf.write(f"  {p_unit}\n")
    wf.close()


def test_grad(model, test_loader, device, outfile, verbose=0):
    p_unit, l_unit = get_default_unit()
    sum_lossE, sum_lossF, num_mol, num_atom = 0.0, 0.0, 0, 0
    wf = open(outfile, 'a')
    for data in test_loader:
        data = data.to(device)
        data.pos.requires_grad = True
        predE, predF = model(data)
        with torch.no_grad():
            if hasattr(data, "base_y"):
                predE += data.base_y
            if hasattr(data, "base_force"):
                predF += data.base_force
            realE, realF = data.y, data.force
            errorE = realE - predE
            errorF = realF - predF
            sum_lossE += errorE.abs().sum()
            sum_lossF += errorF.abs().sum()
        if verbose >= 1:
            for imol in range(len(data.y)):
                idx = (data.batch == imol)
                at_no = data.at_no[idx]
                coord = data.pos[idx] * unit_conversion(l_unit, "Angstrom")
                wf.write(f"mol {num_mol + imol + 1}\n")
                if verbose >= 2:  # print atom coordinates
                    info_3ds = [coord, predF[idx], realF[idx], errorF[idx]]
                    titles = [
                        "Coordinates (Angstrom)",
                        f"Predicted Forces ({p_unit}/{l_unit})",
                        f"Real Forces ({p_unit}/{l_unit})",
                        f"Error Forces ({p_unit}/{l_unit})"
                    ]
                    precisions = [6, 9, 9, 9]
                    wf.write(gen_3Dinfo_str(at_no, info_3ds, titles, precisions))
                    wf.write(f"Charge {int(data.charge[imol].item())}   Multiplicity {int(data.spin[imol].item()) + 1}\n")
                wf.write(f"Energy | Real: {realE[imol].item():15.9f}    ")
                wf.write(f"Predict: {predE[imol].item():15.9f}    ")
                wf.write(f"Error: {errorE[imol].item():15.9f}    {p_unit}\n")
                wf.write(f"Force  | MAE : {errorF[idx].abs().mean():15.9f}   {p_unit}/{l_unit}\n\n")
                wf.flush()
        num_mol += data.y.numel()
        num_atom += data.at_no.numel()
    wf.write(f"Energy MAE : {sum_lossE / num_mol:15.9f}    {p_unit}\n")
    wf.write(f"Force  MAE : {sum_lossF / (3*num_atom):15.9f}    {p_unit}/{l_unit}\n")
    wf.close()


@torch.no_grad()
def test_vector(model, test_loader, device, outfile, verbose=0):
    p_unit, l_unit = get_default_unit()
    sum_loss = 0.0
    num_mol = 0
    wf = open(outfile, 'a')
    for data in test_loader:
        data = data.to(device)
        pred = model(data)
        real = data.y
        error = real - pred
        sum_loss += error.abs().sum().item()
        if verbose >= 1:
            for imol in range(len(data.y)):
                at_no = data.at_no[data.batch == imol]
                coord = data.pos[data.batch == imol] * unit_conversion(l_unit, "Angstrom")
                wf.write(f"mol {num_mol + imol + 1}\n")
                if verbose >= 2:  # print atom coordinates
                    wf.write(gen_3Dinfo_str(at_no, coord, title="Coordinates (Angstrom)"))
                    wf.write(f"Charge {int(data.charge[imol].item())}   Multiplicity {int(data.spin[imol].item()) + 1}\n")
                values = [
                    f"X{vec[imol][0].item():12.6f}  Y{vec[imol][1].item():12.6f}  Z{vec[imol][2].item():12.6f}"
                    for vec in [real, pred, error]
                ]
                titles = [f"Real ({p_unit})", f"Predict ({p_unit})", f"Error ({p_unit})"]
                filled_t = [f"{t: <{len(v)}}" for t, v in zip(titles, values)]
                wf.write("    ".join(filled_t) + "\n")
                wf.write("    ".join(values) + "\n\n")
                wf.flush()
        num_mol += len(data.y)
    wf.write(f"Test MAE: {sum_loss / num_mol / 3 :12.6f} {p_unit}\n")
    wf.close()


@torch.no_grad()
def test_polar(model, test_loader, device, outfile, verbose=0):
    p_unit, l_unit = get_default_unit()
    sum_loss = 0.0
    num_mol = 0
    wf = open(outfile, 'a')
    for data in test_loader:
        data = data.to(device)
        pred = model(data)
        real = data.y
        error = real - pred
        sum_loss += error.abs().sum().item()
        if verbose >= 1:
            for imol in range(len(data.y)):
                at_no = data.at_no[data.batch == imol]
                coord = data.pos[data.batch == imol] * unit_conversion(l_unit, "Angstrom")
                wf.write(f"mol {num_mol + imol + 1}\n")
                if verbose >= 2:  # print atom coordinates
                    wf.write(gen_3Dinfo_str(at_no, coord, title="Coordinates (Angstrom)"))
                    wf.write(f"Charge {int(data.charge[imol].item())}   Multiplicity {int(data.spin[imol].item()) + 1}\n")
                tri_values = []
                for i, D in enumerate(['X', 'Y', 'Z']):
                    tri_values.append([
                        f"{D}X{pol[imol][i,0].item():12.6f}  {D}Y{pol[imol][i,1].item():12.6f}  {D}Z{pol[imol][i,2].item():12.6f}"
                        for pol in [real, pred, error]
                    ])
                titles = [f"Real ({p_unit})", f"Predict ({p_unit})", f"Error ({p_unit})"]
                filled_t = [f"{t: <{len(v)}}" for t, v in zip(titles, tri_values[0])]
                wf.write("    ".join(filled_t) + "\n")
                for values in tri_values:
                    wf.write("    ".join(values) + "\n")
                wf.write("\n")
                wf.flush()
        num_mol += len(data.y)
    wf.write(f"Test MAE: {(sum_loss / num_mol / 9) :12.6f} {p_unit}\n")
    wf.close()


@torch.no_grad()
def test_matrix_wavefunc(model, test_loader, device, build_matrix, output_file):
    from xequinet.utils import cal_orbital_and_energies
    from pyscf import gto
    model.eval()
    # total_error_dict = {"total_mole": 0, "total_orb": 0, "mae_e": 0.0, "cosine_similarity": 0.0} 
    sum_mae_e, sum_cosine_similarity = 0.0, 0.0
    num_orbs, num_moles = 0, 0 
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
        overlap = torch.from_numpy(mol.intor("int1e_ovlp"))
        overlap = overlap.to(torch.get_default_dtype()).to(device)
        pred_fock = build_matrix(res_node, res_edge, node_mask, edge_mask, data.at_no, data.fc_edge_index)
        real_fock = build_matrix(data.node_label, data.edge_label, node_mask, edge_mask, data.at_no, data.fc_edge_index)
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
        num_orbs += real_e.shape[0]
        # write to log file 
        with open(output_file, 'a') as wf:
            wf.write(f"Test error for mole {test_batch_idx:6d}: \n")
            wf.write(f"mae_e: {mae_e.item():10.4f}, cosine similarity: {cosine_similarity.item():10.4f}\n")
        
    with open(output_file, 'a') as wf:
        wf.write(f"\nAverage test error: \n")
        wf.write(f"Orbital Energy Absolute Deviation: {sum_mae_e/num_orbs:10.4f}.\n")
        wf.write(f"Wavefunction Cosine Similarity: {sum_cosine_similarity/num_moles:10.4f}.\n")

        
@torch.no_grad()
def test_matrix(model, test_loader, device, output_file):
    model.eval()
    sum_loss_node, sum_loss_edge, sum_loss_total = 0.0, 0.0, 0.0
    num_node, num_edge, num_total = 0, 0, 0
    for data in test_loader:
        data = data.to(device)
        pred_pad_node, pred_pad_edge = model(data)
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


@torch.no_grad()
def test_matrix_per_mole(model, data_loader, device, output_file):
    model.eval() 
    l1loss_diag, l1loss_off_diag, l1loss_total = 0.0, 0.0, 0.0
    num_mole = 0
    for data in data_loader:
        data = data.to(device)
        pred_diag, pred_off_diag = model(data)
        batch_mask_diag, batch_mask_off_diag = data.onsite_mask, data.offsite_mask 
        real_diag, real_off_diag = data.node_label, data.edge_label 
        # calculate mae, average over graph 
        batch_index = data.batch 
        dst_index = data.fc_edge_index[0] 
        edge_batch_index = batch_index[dst_index] 
        ## diagnol  
        diff_diag = pred_diag - real_diag
        mae_diag_per_node = torch.sum(torch.abs(diff_diag) * batch_mask_diag, dim=[1, 2]) 
        num_diag_per_node = torch.sum(batch_mask_diag, dim=[1, 2])
        mae_diag_per_mole = scatter_sum(mae_diag_per_node, batch_index) 
        num_diag_per_mole = scatter_sum(num_diag_per_node, batch_index) 
        ## off-diagnol 
        diff_off_diag = pred_off_diag - real_off_diag  
        mae_off_diag_per_edge = torch.sum(torch.abs(diff_off_diag) * batch_mask_off_diag, dim=[1, 2]) 
        num_off_diag_per_edge = torch.sum(batch_mask_off_diag, dim=[1, 2])  
        mae_off_diag_per_mole = scatter_sum(mae_off_diag_per_edge, edge_batch_index) 
        num_off_diag_per_mole = scatter_sum(num_off_diag_per_edge, edge_batch_index)  
        # accumulation  
        l1loss_diag += torch.sum((mae_diag_per_mole / num_diag_per_mole)).item()
        l1loss_off_diag += torch.sum((mae_off_diag_per_mole / num_off_diag_per_mole)).item()
        l1loss_total += torch.sum((mae_diag_per_mole + mae_off_diag_per_mole) / (num_diag_per_mole + num_off_diag_per_mole)).item()
        num_mole += num_diag_per_mole.size(0)
     
    with open(output_file, 'a') as wf:
        wf.write(f"Test MAE: diagnol {l1loss_diag / num_mole:10.8f}, off-diagnol {l1loss_off_diag / num_mole:10.8f}, total {l1loss_total / num_mole:10.8f}.\n")



def main():
    # parse config
    parser = argparse.ArgumentParser(description="XequiNet test script")
    parser.add_argument(
        "--config", "-C", type=str, default="config.json",
        help="Configuration file (default: config.json).",
    )
    parser.add_argument(
        "--ckpt", "-c", type=str, required=True,
        help="Xequinet checkpoint file. (XXX.pt containing 'model' and 'config')",
    )
    parser.add_argument(
        "--force", "-f", action="store_true",
        help="Whether testing force additionally when the output mode is 'scalar'",
    )
    parser.add_argument(
        "--no-force", "-nf", action="store_true",
        help="Whether not testing force when the output mode is 'grad'",
    )
    parser.add_argument(
        "--diag", default=False, action="store_true",
        help="Whether to diagonalize the Fock matrix for orbital energy and wavefunction.",
    )
    parser.add_argument(
        "--verbose", "-v", type=int, default=0, choices=[0, 1, 2],
        help="Verbose level. (default: 0)",
    )
    parser.add_argument(
        "--batch-size", "-b", type=int, default=32,
        help="Batch size. (default: 32)",
    )
    parser.add_argument(
        "--warning", "-w", action="store_true",
        help="Whether to show warning messages",
    )
    args = parser.parse_args()
    
    # open warning or not
    if not args.warning:
        import warnings
        warnings.filterwarnings("ignore")
    
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load checkpoint and config
    config = NetConfig.model_validate(args.config)
    ckpt = torch.load(args.ckpt, map_location=device)
    config.model_validate(ckpt["config"])
    
    # set default unit
    set_default_unit(config.default_property_unit, config.default_length_unit)

    test_dataset = create_dataset(config, "test")
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=True, drop_last=False,
    )
    
    # adjust some configurations
    if args.force == True and config.output_mode == "scalar":
        config.output_mode = "grad"
    if args.no_force == True and config.output_mode == "grad":
        config.output_mode = "scalar"
    
    # build model
    model = resolve_model(config).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    # test
    output_file = f"{config.run_name}_test.log"
        
    with open(output_file, 'w') as wf:
        wf.write("XequiNet testing\n")
        wf.write(f"Unit: {config.default_property_unit} {config.default_length_unit}\n")

    if config.output_mode == "grad":
        test_grad(model, test_loader, device, output_file, args.verbose)
    elif config.output_mode == "vector" and config.output_dim == 3:
        test_vector(model, test_loader, device, output_file, args.verbose)
    elif config.output_mode == "polar" and config.output_dim == 9:
        test_polar(model, test_loader, device, output_file, args.verbose)
    elif "mat" in config.version:
        if args.diag:
            from xequinet.utils import BuildMatPerMole
            matrix_builder = BuildMatPerMole(config.irreps_out, config.possible_elements, config.target_basisname)
            test_matrix_wavefunc(model, test_loader, device, matrix_builder, output_file)
        else:
            test_matrix(model, test_loader, device, output_file)
    else:
        test_scalar(model, test_loader, device, output_file, config.output_dim, args.verbose)


if __name__ == "__main__":
    main()