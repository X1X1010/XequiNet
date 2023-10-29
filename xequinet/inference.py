import argparse

import torch

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
    model, dataset, device, output_file,
    atom_sp, base_method=None,
):
    with torch.no_grad():
        for data in dataset:
            data = data.to(device)
            data.batch = torch.zeros_like(data.at_no, dtype=torch.int64)
            pred = model(data).double()
            atom_energy = atom_sp[data.at_no].sum().to(device)
            pred += atom_energy
            if base_method in ["PM7", "PM6"]:
                pred += mopac_calculation(
                    atomic_numbers=data.at_no.numpy(),
                    coordinates=data.pos.numpy(),
                    method=base_method,
                )
            elif base_method in ["gfn2-xtb", "gfn1-xtb", "ipea1-xtb"]:
                pred += xtb_calculation(
                    atomic_numbers=data.at_no.numpy(),
                    coordinates=data.pos.numpy(),
                    method=base_method,
                )
            pred *= unit_conversion(get_default_unit()[0], "AU")
            with open(output_file, 'a') as wf:
                coord = data.pos
                for a, c in zip(data.at_no, coord):
                    wf.write(f"{ELEMENTS_LIST[a.item()]} {c[0].item():10.7f} {c[1].item():10.7f} {c[2].item():10.7f}\n")
                wf.write("target property:")
                for prop in pred:
                    wf.write(f"  {prop.item():10.7f}")
                wf.write("\n")


def predict_grad(
    model, dataset, device, output_file,
    atom_sp, base_method=None,
):
    for data in dataset:
        data = data.to(device)
        data.batch = torch.zeros_like(data.at_no, dtype=torch.int64, device=device)
        data.pos.requires_grad = True
        predE, predF = model(data)
        predE = predE.double()
        atom_energy = atom_sp[data.at_no].sum().to(device)
        predE += atom_energy
        if base_method in ["pm7", "pm6"]:
            baseE, baseF = mopac_calculation(
                atomic_numbers=data.at_no.numpy(),
                coordinates=data.pos.numpy(),
                method=base_method,
                calc_force=True,
            )
            predE += baseE; predF += baseF
        elif base_method in ["gfn2-xtb", "gfn1-xtb", "ipea1-xtb"]:
            baseE, baseF = xtb_calculation(
                atomic_numbers=data.at_no.numpy(),
                coordinates=data.pos.numpy(),
                method=base_method,
                calc_force=True,
            )
            predE += baseE; predF += baseF
        predE *= unit_conversion(get_default_unit()[0], "AU")
        predF *= unit_conversion(f"{get_default_unit()[0]}/{get_default_unit()[1]}", "AU")
        with open(output_file, 'a') as wf:
            coord = data.pos
            for a, c, f in zip(data.at_no, coord, predF):
                wf.write(f"{ELEMENTS_LIST[a.item()]} {c[0].item():10.7f} {c[1].item():10.7f} {c[2].item():10.7f}")
                wf.write(f" {f[0].item():10.7f} {f[1].item():10.7f} {f[2].item():10.7f}\n")
            wf.write(f"Energy: {predE.item():10.7f}\n")


def main():
    # parse arguments
    parser = argparse.ArgumentParser(description="Xequinet inference script")
    parser.add_argument(
        "--ckpt", "-c", type=str, required=True,
        help="Xequinet checkpoint file. (XXX.pt containing 'model' and 'config')",
    )
    parser.add_argument(
        "--max-edges", "-me", type=int, default=None,
        help="Maximum number of edges in a molecule.",
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
        "--base-method", "-bm", type=str, default=None, choices=["pm7", "gfn2-xtb"],
        help="Base semiempirical method for delta learning."
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
    config.node_mean = 0.0; config.graph_mean = 0.0
    if args.force and config.output_mode == "scalar":
        config.output_mode = "grad"
    if args.no_force and config.output_mode == "grad":
        config.output_mode = "scalar"
    if args.max_edges is not None:
        config.max_edges = args.max_edges

    # build model
    model = resolve_model(config).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    # load input data
    dataset = XYZDataset(xyz_file=args.inp, cutoff=config.cutoff, max_edges=config.max_edges)
    outp = f"{args.inp.split('/')[-1].split('.')[0]}.log"

    # get atom reference
    atom_sp = get_atomic_energy(config.atom_ref) - get_atomic_energy(config.batom_ref)

    with open(outp, 'w') as wf:
        wf.write("XequiNet prediction\n")
        wf.write(f"Coordinates in Angstrom, Properties in Atomic Unit\n")
    if config.output_mode == "grad":
        predict_grad(ModelWrapper(model, config.model), dataset, device, outp, atom_sp, args.base_method)
    else:
        predict_scalar(ModelWrapper(model, config.model), dataset, device, outp, atom_sp, args.base_method)


if __name__ == "__main__":
    main()