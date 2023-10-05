import argparse

import torch

from xequinet.nn import xPaiNN
from xequinet.utils import (
    NetConfig,
    set_default_unit, get_default_unit, unit_conversion,
    get_atomic_energy
)
from xequinet.utils.qc import ELEMENTS_LIST
from xequinet.data import XYZDataset
from xequinet.interface import mopac_calculation, xtb_calculation


def calc_atom_ref(at_no, atom_ref, batom_ref, device):
    atom_energy = get_atomic_energy(atom_ref).to(device)[at_no].sum()
    if batom_ref is not None:
        atom_energy -= get_atomic_energy(batom_ref)[at_no].sum()
    return atom_energy.item()


def predict_scalar(
    model, dataset, device, output_file,
    atom_ref=None, batom_ref=None, base_method=None,
):
    model.eval()
    with torch.no_grad():
        for data in dataset:
            data = data.to(device)
            batch = torch.zeros_like(data.at_no, dtype=torch.int64)
            pred = model(data.at_no, data.pos, data.edge_index, batch)
            atom_energy = calc_atom_ref(data.at_no, atom_ref, batom_ref, device)
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
    atom_ref=None, batom_ref=None, base_method=None,
):
    model.eval()
    for data in dataset:
        data = data.to(device)
        batch = torch.zeros_like(data.at_no, dtype=torch.int64, device=device)
        data.pos.requires_grad = True
        predE, predF = model(data.at_no, data.pos, data.edge_index, batch)
        atom_energy = calc_atom_ref(data.at_no, atom_ref, batom_ref, device)
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
        "--force", "-f", action="store_true",
        help="Whether testing force additionally when the output mode is 'scalar'",
    )
    parser.add_argument(
        "--base-method", "-bm", type=str, default=None, choices=["pm7", "gfn2-xtb"],
        help="Base semiempirical method for delta learning."
    )
    parser.add_argument(
        "inp", type=str,
        help="Input xyz file.")
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

    # build model
    model = xPaiNN(config).to(device)
    model.load_state_dict(ckpt["model"])

    # load input data
    dataset = XYZDataset(xyz_file=args.inp, cutoff=config.cutoff)
    outp = f"{args.inp.split('/')[-1].split('.')[0]}.log"

    with open(outp, 'w') as wf:
        wf.write("XequiNet prediction\n")
        wf.write(f"Coordinates in Angstrom, Properties in Atomic Unit\n")
    if config.output_mode == "grad":
        predict_grad(model, dataset, device, outp, config.atom_ref, config.batom_ref, args.base_method)
    else:
        predict_scalar(model, dataset, device, outp, config.atom_ref, config.batom_ref, args.base_method)

if __name__ == "__main__":
    main()