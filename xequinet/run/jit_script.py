import argparse

import torch
import torch_cluster

from xequinet.interface import LmpPaiNN, resolve_jit_model
from xequinet.utils import NetConfig, set_default_unit


def main():
    # parse arguments
    parser = argparse.ArgumentParser(description="Just in time script for XequiNet")
    parser.add_argument(
        "--ckpt", "-c", type=str, required=True,
        help="Xequinet checkpoint file. (XXX.pt containing 'model' and 'config')",
    )
    parser.add_argument(
        "--lammps", "-l", action="store_true",
        help="Whether the model is used in LAMMPS",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output file name. (default: XXX.jit)",
    )
    args = parser.parse_args()

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # load checkpoint and config
    ckpt = torch.load(args.ckpt, map_location=device)
    config = NetConfig.parse_obj(ckpt["config"])
    
    # set default unit
    set_default_unit(config.default_property_unit, config.default_length_unit)

    # build model
    if args.lammps:
        model = LmpPaiNN(config).to(device)
    else:
        model = resolve_jit_model(config).to(device)

    model.load_state_dict(ckpt["model"], strict=False)
    model_script = torch.jit.script(model)
    output_file = f"{args.ckpt.split('/')[-1].split('.')[0]}.jit" if args.output is None else args.output
    model_script.save(output_file)


if __name__ == "__main__":
    main()