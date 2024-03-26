import argparse

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from ..data import create_dataset
from ..nn import resolve_model
from ..utils import (
    NetConfig,
    unit_conversion, set_default_unit, get_default_unit,
    gen_3Dinfo_str,
    MatToolkit,
)


@torch.no_grad()
def test_scalar(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    outfile: str,
    output_dim: int = 1,
    verbose: int = 0,
) -> None:
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
            for imol in range(len(data)):
                datum = data[imol]
                coord = datum.pos * unit_conversion(l_unit, "Angstrom")
                wf.write(f"mol {num_mol + imol + 1}\n")
                if verbose >= 2:  # print atom coordinates
                    wf.write(gen_3Dinfo_str(datum.at_no, coord, title="Coordinates (Angstrom)"))
                    wf.write(f"Charge {int(datum.charge.item())}   Multiplicity {int(datum.spin.item()) + 1}\n")
                wf.write(f"Real:")
                wf.write("".join([f"{r.item():15.9f} " for r in real[imol]]))
                wf.write(f"    Predict:")
                wf.write("".join([f"{p.item():15.9f}" for p in pred[imol]]))
                wf.write(f"    Error:")
                wf.write("".join([f"{l.item():15.9f}" for l in error[imol]]))
                wf.write(f"    ({p_unit})\n\n")
                wf.flush()
        num_mol += len(data)
    avg_loss = sum_loss / num_mol
    wf.write(f"Test MAE:")
    wf.write("".join([f"{l:15.9f}" for l in avg_loss]))
    wf.write(f"  {p_unit}\n")
    wf.close()


def test_grad(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    outfile: str,
    verbose: int = 0,
) -> None:
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
            for imol in range(len(data)):
                idx = (data.batch == imol)
                datum = data[imol]
                coord = datum.pos * unit_conversion(l_unit, "Angstrom")
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
                    wf.write(gen_3Dinfo_str(datum.at_no, info_3ds, titles, precisions))
                    wf.write(f"Charge {int(datum.charge.item())}   Multiplicity {int(datum.spin.item()) + 1}\n")
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
def test_vector(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    outfile: str,
    verbose: int = 0,
) -> None:
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
            for imol in range(len(data)):
                datum = data[imol]
                coord = datum.pos * unit_conversion(l_unit, "Angstrom")
                wf.write(f"mol {num_mol + imol + 1}\n")
                if verbose >= 2:  # print atom coordinates
                    wf.write(gen_3Dinfo_str(datum.at_no, coord, title="Coordinates (Angstrom)"))
                    wf.write(f"Charge {int(datum.charge.item())}   Multiplicity {int(datum.spin.item()) + 1}\n")
                values = [
                    f"X{vec[imol][0].item():12.6f}  Y{vec[imol][1].item():12.6f}  Z{vec[imol][2].item():12.6f}"
                    for vec in [real, pred, error]
                ]
                titles = [f"Real ({p_unit})", f"Predict ({p_unit})", f"Error ({p_unit})"]
                filled_t = [f"{t: <{len(v)}}" for t, v in zip(titles, values)]
                wf.write("    ".join(filled_t) + "\n")
                wf.write("    ".join(values) + "\n\n")
                wf.flush()
        num_mol += len(data)
    wf.write(f"Test MAE: {sum_loss / num_mol / 3 :12.6f} {p_unit}\n")
    wf.close()


@torch.no_grad()
def test_polar(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    outfile: str,
    verbose: int = 0,
) -> None:
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
                datum = data[imol]
                coord = datum.pos * unit_conversion(l_unit, "Angstrom")
                wf.write(f"mol {num_mol + imol + 1}\n")
                if verbose >= 2:  # print atom coordinates
                    wf.write(gen_3Dinfo_str(datum.at_no, coord, title="Coordinates (Angstrom)"))
                    wf.write(f"Charge {int(datum.charge.item())}   Multiplicity {int(datum.spin.item()) + 1}\n")
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
        num_mol += len(data)
    wf.write(f"Test MAE: {(sum_loss / num_mol / 9) :12.6f} {p_unit}\n")
    wf.close()


@torch.no_grad()
def test_tensor(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    outfile: str,
    verbose: int = 0,
) -> None:
    """
    For tensor output, the situation is more complicated.
    So we save the results in a file and do not print them, when verbose >= 1.
    """
    p_unit, _ = get_default_unit()
    sum_loss = 0.0
    tot_numel = 0
    wf = open(outfile, 'a')
    if verbose >= 1:
        X, Y = [], []
    for data in test_loader:
        data = data.to(device)
        pred = model(data)
        real = data.y
        sum_loss += F.l1_loss(pred, real, reduction="sum").item()
        tot_numel += real.numel()
        if verbose >= 1:
            X.append(pred.cpu())
            Y.append(real.cpu())
    if verbose >= 1:
        outpt = f"{outfile.rsplit('.', 1)[0]}.pt"
        X = torch.cat(X, dim=0)
        Y = torch.cat(Y, dim=0)
        torch.save({"X": X, "Y": Y}, outpt)
    wf.write(f"Test MAE: {sum_loss / tot_numel :12.6f} {p_unit}\n")


@torch.no_grad()
def test_matrix(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    outfile: str,
    mat_toolkit: MatToolkit,
    verbose: int = 0,
) -> None:
    """
    For matrix output, the size of the output may be large.
    So we save the results in a file and do not print them, when verbose >= 1.
    """
    p_unit, _ = get_default_unit()
    loss_node, loss_edge = 0.0, 0.0
    num_node, num_edge = 0, 0
    wf = open(outfile, 'a')
    if verbose >= 1:
        X, Y = [], []
    for data in test_loader:
        data = data.to(device)
        pred_node_padded, pred_edge_padded = model(data)
        node_mask, edge_mask = data.node_mask, data.edge_mask
        pred_node, pred_edge = pred_node_padded[node_mask], pred_edge_padded[edge_mask]
        real_node, real_edge = data.node_label[node_mask], data.edge_label[edge_mask]
        loss_node += F.l1_loss(pred_node, real_node, reduction="sum").item()
        loss_edge += F.l1_loss(pred_edge, real_edge, reduction="sum").item()
        num_node += real_node.size(0)
        num_edge += real_edge.size(0)
        if verbose >= 1:
            # steal the sky and change the sun
            data.node_label = pred_edge_padded
            data.edge_label = pred_edge_padded
            for imol in range(len(data)):
                datum = data[imol]
                node_blocks = datum.node_blocks
                edge_blocks = datum.edge_blocks
                if hasattr(datum, "edge_index_full"):
                    mat_edge_index = datum.edge_index_full
                else:
                    mat_edge_index = datum.edge_index
                pred_matrix = mat_toolkit.assemble_blocks(node_blocks, edge_blocks, mat_edge_index)
                X.append(pred_matrix.cpu())
                real_matrix = datum.target_matrix
                Y.append(real_matrix.cpu())
    if verbose >= 1:
        outpt = f"{outfile.rsplit('.', 1)[0]}.pt"
        torch.save({"X": X, "Y": Y}, outpt)
    wf.write(f"Node  MAE: {loss_node / num_node :12.6f} {p_unit}\n")
    wf.write(f"Edge  MAE: {loss_edge / num_edge :12.6f} {p_unit}\n")
    wf.write(f"Total MAE: {(loss_edge+loss_node) / (num_node+num_edge) :12.6f} {p_unit}\n")            



def run_test(args: argparse.Namespace) -> None:
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load checkpoint and config
    with open(args.config, 'r') as json_file:
        config = NetConfig.model_validate_json(json_file.read())
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
    if args.output is None:
        output_file = f"{config.run_name}_test.log"
        
    with open(output_file, 'w') as wf:
        wf.write("XequiNet Testing\n")
        wf.write(f"Unit: {config.default_property_unit} {config.default_length_unit}\n")

    if "mat" in config.version:
        mat_toolkit = MatToolkit(config.target_basis, config.possible_elements)
        test_matrix(model, test_loader, device, output_file, mat_toolkit, args.verbose)
    elif config.output_mode == "grad":
        test_grad(model, test_loader, device, output_file, args.verbose)
    elif config.output_mode == "vector":
        test_vector(model, test_loader, device, output_file, args.verbose)
    elif config.output_mode == "polar":
        test_polar(model, test_loader, device, output_file, args.verbose)
    elif config.output_mode == "cartesian":
        test_tensor(model, test_loader, device, output_file, args.verbose)
    else:
        test_scalar(model, test_loader, device, output_file, config.output_dim, args.verbose)
