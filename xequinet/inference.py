import os
import argparse

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from xpainn.nn import xPaiNN
from xpainn.utils import NetConfig, set_default_unit, get_atomic_energy
from xpainn.utils.qc import ELEMENTS_LIST
from xpainn.data import XYZDataset
from xpainn.interface import mopac_calculation, xtb_calculation


def calc_atom_ref(at_no, atom_ref, batom_ref):
    atom_energy = get_atomic_energy(atom_ref)[at_no].sum()
    if batom_ref is not None:
        atom_energy -= get_atomic_energy(batom_ref)[at_no].sum()
    return atom_energy.item()


def predict_scalar(
    model, dataset, device, output_file,
    atom_ref=None, batom_ref=None, base_method=None
):
    model.eval()
    with torch.no_grad():
        for data in dataset:
            data = data.to(device)
            batch = torch.zeros_like(data.at_no, dtype=torch.int64)
            pred = model(data.at_no, data.pos, data.edge_index, batch)
            atom_ref = calc_atom_ref(data.at_no, atom_ref, batom_ref)
            pred += atom_ref
            if base_method in ["PM7", "PM6"]:
                pred += mopac_calculation(
                    atomic_numbers=data.at_no.numpy(),
                    coordinates=data.pos.numpy(),
                    method=base_method,
                )
            elif base_method in ["GFN2-xTB", "GFN1-xTB", "IPEA1-xTB"]:
                pred += xtb_calculation(
                    atomic_numbers=data.at_no.numpy(),
                    coordinates=data.pos.numpy(),
                    method=base_method,
                )
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
        batch = torch.zeros_like(data.at_no, dtype=torch.int64)
        predE, predF = model(data.at_no, data.pos, data.edge_index, batch)
        atom_ref = calc_atom_ref(data.at_no, atom_ref, batom_ref)
        predE += atom_ref
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
        with open(output_file, 'a') as wf:
            coord = data.pos
            for a, c, f in zip(data.at_no, coord, predF):
                wf.write(f"{ELEMENTS_LIST[a.item()]} {c[0].item():10.7f} {c[1].item():10.7f} {c[2].item():10.7f}")
                wf.write(f" {f[0].item():10.7f} {f[1].item():10.7f} {f[2].item():10.7f}\n")
            wf.write(f"Energy: {predE.item():10.7f}")


def main():
    # parse arguments
    parser = argparse.ArgumentParser(description="XPainn test script")
    parser.add_argument("--ckpt", "-c", type=str, required=True, help="Checkpoint file")
    parser.add_argument("--output-dir", "-o", type=str, default=".")
    parser.add_argument("--force", "-f", action="store_true", help="Calculate force")
    parser.add_argument("--len-unit", "-lu", type=str, default="Angstrom", help="Unit of coordinates")
    parser.add_argument("--prop-unit", "-pu", type=str, default="eV", help="Unit of properties")
    parser.add_argument("--base", "-b",
                        type=str, default=None,
                        choices=["pm7", "gfn2-xtb"],
                        help="Base semiempirical method")
    parser.add_argument("inp", type=str, help="Input file")
    args = parser.parse_args()

    # load checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device)

    # set default unit
    set_default_unit(args.prop_unit, args.len_unit)

    # load config
    config = NetConfig.parse_obj(ckpt["config"])
    if args.force:
        config.output_mode = "grad"

    # build model
    model = xPaiNN(config).to(device)
    model.load_state_dict(ckpt["model"])

    # load input data
    dataset = XYZDataset(args.inp, config.cutoff)
    outp = f"{args.inp.split('/')[-1].split('.')[0]}.out"
    outp = os.path.join(args.output_dir, outp)

    with open(outp, 'w') as wf:
        wf.write("xPaiNN prediction\n")
        wf.write(f"Coordinates in {args.len_unit}, Properties in {args.prop_unit}\n")
    if args.force:
        predict_grad(model, dataset, device, outp, args.base, config.atom_ref, config.batom_ref)
    else:
        predict_scalar(model, dataset, device, outp, args.base, config.atom_ref, config.batom_ref)

if __name__ == "__main__":
    main()