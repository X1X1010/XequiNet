import argparse

import torch
from torch_scatter import scatter
from torch_cluster import radius_graph
from torch_geometric.loader import DataLoader

from xequinet.nn import resolve_model
from xequinet.utils import (
    NetConfig,
    set_default_unit, get_default_unit, unit_conversion,
    get_atomic_energy, gen_3Dinfo_str,
    ModelWrapper,
)
from xequinet.data import XYZDataset
from xequinet.interface import mopac_calculation, xtb_calculation


def predict_scalar(
    model: ModelWrapper,
    cutoff: float,
    max_edges: int,
    dataloader: DataLoader,
    device: torch.device,
    output_file: str,
    atom_sp: torch.Tensor,
    base_method: str = None,
    verbose: int = 0,
):
    p_factor = unit_conversion(get_default_unit()[0], "AU")
    c_factor = unit_conversion(get_default_unit()[1], "Angstrom")
    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            data.edge_index = radius_graph(data.pos, r=cutoff, batch=data.batch, max_num_neighbors=max_edges)
            nn_pred = model(data).double()
            sum_atom_sp = scatter(atom_sp[data.at_no], data.batch, dim=0).unsqueeze(-1)
            for i in range(len(data)):
                at_no = data.at_no[data.batch == i]
                coord = data.pos[data.batch == i]
                if base_method in ["pm7", "pm6"]:
                    delta = mopac_calculation(
                        atomic_numbers=at_no.cpu().numpy(),
                        coordinates=coord.cpu().numpy(),
                        method=base_method,
                        charge=int(data.charge[i].item()),
                        multiplicity=int(data.spin[i].item()) + 1,
                    )
                elif base_method in ["gfn2-xtb", "gfn1-xtb", "ipea1-xtb"]:
                    delta = xtb_calculation(
                        atomic_numbers=at_no.cpu().numpy(),
                        coordinates=coord.cpu().numpy(),
                        method=base_method,
                        charge=int(data.charge[i].item()),
                        uhf=int(data.spin[i].item()),
                    )
                else:
                    delta = 0.0
                total = nn_pred[i] + sum_atom_sp[i] + delta
                with open(output_file, 'a') as wf:
                    if verbose >= 2:  # print coordinates
                        wf.write(gen_3Dinfo_str(at_no, coord * c_factor, title="Coordinates (Angstrom)"))
                        wf.write(f"Charge {int(data.charge[i].item())}   Multiplicity {int(data.spin[i].item()) + 1}\n")
                    if verbose >= 1:  # print detailed information
                        wf.write("NN Contribution         :")
                        for v_nn in nn_pred[i]:
                            wf.write(f"    {v_nn.item() * p_factor:10.7f} a.u.")
                        wf.write("\nSum of Atomic Reference :")
                        for v_at in sum_atom_sp[i]:
                            wf.write(f"    {v_at.item() * p_factor:10.7f} a.u.")
                        if base_method is not None:
                            wf.write("\nDelta Method Base       :")
                            wf.write(f"    {delta.item() * p_factor:10.7f} a.u.")
                        wf.write("\n")
                    wf.write("Total Property          :")
                    for v_tot in total:
                        wf.write(f"    {v_tot.item() * p_factor:10.7f} a.u.")
                    wf.write("\n\n")


def predict_grad(
    model: ModelWrapper,
    cutoff: float,
    max_edges: int,
    dataloader: DataLoader,
    device: torch.device,
    output_file: str,
    atom_sp: torch.Tensor,
    base_method: str = None,
    verbose: int = 0,
):
    e_factor = unit_conversion(get_default_unit()[0], "AU")
    c_factor = unit_conversion(get_default_unit()[1], "Angstrom")
    f_factor = unit_conversion(f"{get_default_unit()[0]}/{get_default_unit()[1]}", "AU")
    for data in dataloader:
        data = data.to(device)
        data.edge_index = radius_graph(data.pos, r=cutoff, batch=data.batch, max_num_neighbors=max_edges)
        data.pos.requires_grad = True
        nn_predE, nn_predF = model(data)
        sum_atom_sp = scatter(atom_sp[data.at_no], data.batch, dim=0)
        for i in range(len(data)):
            at_no = data.at_no[data.batch == i]
            coord = data.pos[data.batch == i]
            if base_method in ["pm7", "pm6"]:
                deltaE, deltaF = mopac_calculation(
                    atomic_numbers=at_no.cpu().numpy(),
                    coordinates=coord.cpu().numpy(),
                    method=base_method,
                    charge=int(data.charge[i].item()),
                    multiplicity=int(data.spin[i].item()) + 1,
                    calc_force=True,
                )
            elif base_method in ["gfn2-xtb", "gfn1-xtb", "ipea1-xtb"]:
                deltaE, deltaF = xtb_calculation(
                    atomic_numbers=at_no.cpu().numpy(),
                    coordinates=coord.cpu().numpy(),
                    method=base_method,
                    charge=int(data.charge[i].item()),
                    uhf=int(data.spin[i].item()),
                    calc_force=True,
                )
            else:
                deltaE = 0.0
                deltaF = torch.zeros_like(coord)
            totalE = nn_predE[i] + sum_atom_sp[i] + deltaE
            totalF = nn_predF[i] + deltaF
            with open(output_file, 'a') as wf:
                if verbose >= 2:  # print coordinates
                    wf.write(gen_3Dinfo_str(at_no, coord * c_factor, title="Coordinates (Angstrom)"))
                    wf.write(f"Charge {int(data.charge[i].item())}   Multiplicity {int(data.spin[i].item()) + 1}\n")
                if verbose >= 1:  # print detailed information
                    wf.write("NN Contribution         :")
                    wf.write(f"    {nn_predE[i].item() * e_factor:10.7f} a.u.\n")
                    wf.write("Sum of Atomic Reference :")
                    wf.write(f"    {sum_atom_sp[i].item() * e_factor:10.7f} a.u.\n")
                    if base_method is not None:
                        wf.write("Delta Method Base       :")
                        wf.write(f"    {deltaE * e_factor:10.7f} a.u.\n")
                wf.write(f"Total Energy            :")
                wf.write(f"    {totalE.item() * e_factor:10.7f} a.u.\n")
                allFs, titles = [], []
                if verbose >= 1 and base_method is not None:
                    allFs.extend([nn_predF * f_factor, deltaF * f_factor])
                    titles.append(["NN Forces (a.u.)", "Delta Forces (a.u.)"])
                allFs.append(totalF * f_factor)
                titles.append("Total Forces (a.u.)")
                wf.write(gen_3Dinfo_str(at_no, allFs, titles, precision=9))
                wf.write("\n")


def predict_vector(
    model: ModelWrapper,
    cutoff: float,
    max_edges: int,
    dataloader: DataLoader,
    device: torch.device,
    output_file: str,
    verbose: int = 0,
):
    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            data.edge_index = radius_graph(data.pos, r=cutoff, batch=data.batch, max_num_neighbors=max_edges)
            pred = model(data)
            pred *= unit_conversion(get_default_unit()[0], "AU")
            with open(output_file, 'a') as wf:
                for i in range(len(data)):
                    at_no = data.at_no[data.batch == i]
                    coord = data.pos[data.batch == i]
                    vector = pred[i]
                    if verbose >= 2:  # print coordinates
                        coord *= unit_conversion(get_default_unit()[1], "Angstrom")
                        wf.write(gen_3Dinfo_str(at_no, coord, title="Coordinates (Angstrom)"))
                        wf.write(f"Charge {int(data.charge[i].item())}   Multiplicity {int(data.spin[i].item()) + 1}\n")
                    wf.write(f"Vector Property (a.u.):\n")
                    wf.write(f"X{vector[0].item():12.6f}  Y{vector[1].item():12.6f}  Z{vector[2].item():12.6f}\n\n")


def predict_polar(
    model: ModelWrapper,
    cutoff: float,
    max_edges: int,
    dataloader: DataLoader,
    device: torch.device,
    output_file: str,
    verbose: int = 0,
):
    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            data.edge_index = radius_graph(data.pos, r=cutoff, batch=data.batch, max_num_neighbors=max_edges)
            pred = model(data)
            pred *= unit_conversion(get_default_unit()[0], "AU")
            with open(output_file, 'a') as wf:
                for i in range(len(data)):
                    at_no = data.at_no[data.batch == i]
                    coord = data.pos[data.batch == i]
                    polar = pred[i]
                    if verbose >= 2:  # print coordinates
                        coord *= unit_conversion(get_default_unit()[1], "Angstrom")
                        wf.write(gen_3Dinfo_str(at_no, coord, title="Coordinates (Angstrom)"))
                        wf.write(f"Charge {int(data.charge[i].item())}   Multiplicity {int(data.spin[i].item()) + 1}\n")
                    wf.write("Polar property (a.u.):\n")
                    wf.write(f"XX{polar[0,0].item():12.6f}  XY{polar[0,1].item():12.6f}  XZ{polar[0,2].item():12.6f}\n")
                    wf.write(f"YX{polar[1,0].item():12.6f}  YY{polar[1,1].item():12.6f}  YZ{polar[1,2].item():12.6f}\n")
                    wf.write(f"ZX{polar[2,0].item():12.6f}  ZY{polar[2,1].item():12.6f}  ZZ{polar[2,2].item():12.6f}\n\n")


def main():
    # parse arguments
    parser = argparse.ArgumentParser(description="Xequinet inference script")
    parser.add_argument(
        "--ckpt", "-c", type=str, required=True,
        help="Xequinet checkpoint file. (XXX.pt containing 'model' and 'config')",
    )
    parser.add_argument(
        "--batch-size", "-bz", type=int, default=64,
        help="Batch size.",
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
        "--base-method", "-bm", type=str, default=None,
        help="Base semiempirical method for delta learning."
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output file name."
    )
    parser.add_argument(
        "--verbose", "-v", type=int, default=0, choices=[0, 1, 2],
        help="Verbose level. (default: 0)"
    )
    parser.add_argument(
        "inp", type=str,
        help="Input xyz file."
    )
    args = parser.parse_args()

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load checkpoint and config
    ckpt = torch.load(args.ckpt, map_location=device)
    config = NetConfig.parse_obj(ckpt["config"])

    # set default unit
    set_default_unit(config.default_property_unit, config.default_length_unit)

    # adjust some configurations
    if args.force and config.output_mode == "scalar":
        config.output_mode = "grad"
    if args.no_force and config.output_mode == "grad":
        config.output_mode = "scalar"

    # build model
    model = resolve_model(config).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    # load input data
    dataset = XYZDataset(xyz_file=args.inp, cutoff=config.cutoff)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0)
    outp = f"{args.inp.split('/')[-1].split('.')[0]}.log" if args.output is None else args.output

    # get atom reference
    atom_sp = get_atomic_energy(config.atom_ref) - get_atomic_energy(config.batom_ref)
    atom_sp = atom_sp.to(device)

    with open(outp, 'w') as wf:
        wf.write("XequiNet prediction\n")
        wf.write(f"Coordinates in Angstrom, Properties in Atomic Unit\n\n")
    if config.output_mode == "scalar":
        predict_scalar(
            model=ModelWrapper(model, config.version),
            cutoff=config.cutoff,
            max_edges=config.max_edges,
            dataloader=dataloader,
            device=device,
            output_file=outp,
            atom_sp=atom_sp,
            base_method=args.base_method,
            verbose=args.verbose,
        )
    elif config.output_mode == "grad":
        predict_grad(
            model=ModelWrapper(model, config.version),
            cutoff=config.cutoff,
            max_edges=config.max_edges,
            dataloader=dataloader,
            device=device,
            output_file=outp,
            atom_sp=atom_sp,
            base_method=args.base_method,
            verbose=args.verbose,
        )
    elif config.output_mode == "vector":
        predict_vector(
            model=ModelWrapper(model, config.version),
            cutoff=config.cutoff,
            max_edges=config.max_edges,
            dataloader=dataloader,
            device=device,
            output_file=outp,
            verbose=args.verbose,
        )
    elif config.output_mode == "polar":
        predict_polar(
            model=ModelWrapper(model, config.version),
            cutoff=config.cutoff,
            max_edges=config.max_edges,
            dataloader=dataloader,
            device=device,
            output_file=outp,
            verbose=args.verbose,
        )
    else:
        raise ValueError(f"Unknown output mode: {config.output_mode}")


if __name__ == "__main__":
    main()