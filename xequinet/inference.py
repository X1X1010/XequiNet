import argparse

import torch
from torch_cluster import radius_graph
from torch_geometric.loader import DataLoader

from xequinet.nn import resolve_model
from xequinet.utils import (
    NetConfig,
    set_default_unit, get_default_unit, unit_conversion,
    get_atomic_energy,
    ModelWrapper,
)
from xequinet.utils.qc import ELEMENTS_LIST
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
    base_method: str = None
):
    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            data.edge_index = radius_graph(data.pos, r=cutoff, batch=data.batch, max_num_neighbors=max_edges)
            pred = model(data).double().index_add(0, data.batch, atom_sp[data.at_no])
            for i in range(len(data)):
                at_no = data.at_no[data.batch == i]
                coord = data.pos[data.batch == i]
                scalar = pred[i]
                if base_method in ["pm7", "pm6"]:
                    scalar += mopac_calculation(
                        atomic_numbers=at_no.cpu().numpy(),
                        coordinates=coord.cpu().numpy(),
                        method=base_method,
                    )
                elif base_method in ["gfn2-xtb", "gfn1-xtb", "ipea1-xtb"]:
                    scalar += xtb_calculation(
                        atomic_numbers=at_no.cpu().numpy(),
                        coordinates=coord.cpu().numpy(),
                        method=base_method,
                    )
                scalar *= unit_conversion(get_default_unit()[0], "AU")
                with open(output_file, 'a') as wf:
                    for a, c in zip(at_no, coord):
                        wf.write(f"{ELEMENTS_LIST[a.item()]} {c[0].item():10.7f} {c[1].item():10.7f} {c[2].item():10.7f}\n")
                    wf.write("target property:")
                    for prop in scalar:
                        wf.write(f"  {prop.item():10.7f}")
                    wf.write("\n")


def predict_grad(
    model: ModelWrapper,
    cutoff: float,
    max_edges: int,
    dataloader: DataLoader,
    device: torch.device,
    output_file: str,
    atom_sp: torch.Tensor,
    base_method: str = None
):
    for data in dataloader:
        data = data.to(device)
        data.edge_index = radius_graph(data.pos, r=cutoff, batch=data.batch, max_num_neighbors=max_edges)
        data.pos.requires_grad = True
        predE, predF = model(data)
        predE = predE.double().index_add(0, data.batch, atom_sp[data.at_no])
        for i in range(len(data)):
            at_no = data.at_no[data.batch == i]
            coord = data.pos[data.batch == i]
            energy = predE[i]
            force = predF[data.batch == i]
            if base_method in ["pm7", "pm6"]:
                baseE, baseF = mopac_calculation(
                    atomic_numbers=at_no.cpu().numpy(),
                    coordinates=coord.cpu().numpy(),
                    method=base_method,
                    calc_force=True,
                )
                energy += baseE; force += baseF
            elif base_method in ["gfn2-xtb", "gfn1-xtb", "ipea1-xtb"]:
                baseE, baseF = xtb_calculation(
                    atomic_numbers=at_no.cpu().numpy(),
                    coordinates=coord.cpu().numpy(),
                    method=base_method,
                    calc_force=True,
                )
                energy += baseE; force += baseF
            energy *= unit_conversion(get_default_unit()[0], "AU")
            force *= unit_conversion(f"{get_default_unit()[0]}/{get_default_unit()[1]}", "AU")
            with open(output_file, 'a') as wf:
                for a, c, f in zip(at_no, coord, force):
                    wf.write(f"{ELEMENTS_LIST[a.item()]} {c[0].item():10.7f} {c[1].item():10.7f} {c[2].item():10.7f}")
                    wf.write(f" {f[0].item():10.7f} {f[1].item():10.7f} {f[2].item():10.7f}\n")
                wf.write(f"Energy: {energy.item():10.7f}\n")


def predict_vector(
    model: ModelWrapper,
    cutoff: float,
    max_edges: int,
    dataloader: DataLoader,
    device: torch.device,
    output_file: str,
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
                    for a, c in zip(at_no, coord):
                        wf.write(f"{ELEMENTS_LIST[a.item()]} {c[0].item():10.7f} {c[1].item():10.7f} {c[2].item():10.7f}\n")
                    wf.write("vector property:")
                    wf.write(f"    X  {vector[0].item():10.7f}  Y  {vector[1].item():10.7f}  Z  {vector[2].item():10.7f}\n")


def predict_polar(
    model: ModelWrapper,
    cutoff: float,
    max_edges: int,
    dataloader: DataLoader,
    device: torch.device,
    output_file: str,
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
                    for a, c in zip(at_no, coord):
                        wf.write(f"{ELEMENTS_LIST[a.item()]} {c[0].item():10.7f} {c[1].item():10.7f} {c[2].item():10.7f}\n")
                    wf.write("polar property:")
                    wf.write(f"    XX  {polar[0,0].item():10.7f}  XY  {polar[0,1].item():10.7f}  XZ  {polar[0,2].item():10.7f}\n")
                    wf.write("               ")
                    wf.write(f"    YX  {polar[1,0].item():10.7f}  YY  {polar[1,1].item():10.7f}  YZ  {polar[1,2].item():10.7f}\n")
                    wf.write("               ")
                    wf.write(f"    ZX  {polar[2,0].item():10.7f}  ZY  {polar[2,1].item():10.7f}  ZZ  {polar[2,2].item():10.7f}\n")


def main():
    # parse arguments
    parser = argparse.ArgumentParser(description="Xequinet inference script")
    parser.add_argument(
        "--ckpt", "-c", type=str, required=True,
        help="Xequinet checkpoint file. (XXX.pt containing 'model' and 'config')",
    )
    parser.add_argument(
        "--batch-size", "-bs", type=int, default=64,
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
        wf.write(f"Coordinates in Angstrom, Properties in Atomic Unit\n")
    if config.output_mode == "scalar":
        predict_scalar(
            model=ModelWrapper(model, config.version),
            cutoff=config.cutoff,
            max_edges=config.max_edges,
            dataloader=dataloader,
            device=device,
            output_file=outp,
            atom_sp=atom_sp,
            base_method=args.base_method
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
            base_method=args.base_method
        )
    elif config.output_mode == "vector":
        predict_vector(
            model=ModelWrapper(model, config.version),
            cutoff=config.cutoff,
            max_edges=config.max_edges,
            dataloader=dataloader,
            device=device,
            output_file=outp,
        )
    elif config.output_mode == "polar":
        predict_polar(
            model=ModelWrapper(model, config.version),
            cutoff=config.cutoff,
            max_edges=config.max_edges,
            dataloader=dataloader,
            device=device,
            output_file=outp,
        )
    else:
        raise ValueError(f"Unknown output mode: {config.output_mode}")


if __name__ == "__main__":
    main()