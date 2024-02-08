import argparse

import torch
import torch_cluster

from xequinet.interface import resolve_md_model, resolve_jit_model
from xequinet.utils import NetConfig, set_default_unit


def main():
    # parse arguments
    parser = argparse.ArgumentParser(description="Just in time script for XequiNet")
    parser.add_argument(
        "--ckpt", "-c", type=str, required=True,
        help="Xequinet checkpoint file. (XXX.pt containing 'model' and 'config')",
    )
    parser.add_argument(
        "--md", action="store_true",
        help="Whether the model is used for molecular dynamics",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output file name. (default: XXX.jit)",
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
    ckpt = torch.load(args.ckpt, map_location=device)
    config = NetConfig.model_validate(ckpt["config"])
    
    # set default unit
    set_default_unit(config.default_property_unit, config.default_length_unit)

    # build model
    if args.md:
        # result contains "energy", atomic "energies", "forces", "virial" and atomic "virials"
        # the input unit of "coordinates" and "shifts" calculate from lattice is Angstrom
        # the output unit of "energy" and "virial" is eV, and the output unit of "forces" is eV/Angstrom
        model = resolve_md_model(config).to(device)
    else:
        # result contains "energy" and nuclear "gradient"
        # the input unit of "coordinates" and "shifts" calculate from lattice is Bohr
        # the output unit of a.u.
        model = resolve_jit_model(config).to(device)

    model.load_state_dict(ckpt["model"], strict=False)
    model_script = torch.jit.script(model)
    output_file = f"{args.ckpt.split('/')[-1].split('.')[0]}.jit" if args.output is None else args.output
    model_script.save(output_file)


if __name__ == "__main__":
    main()