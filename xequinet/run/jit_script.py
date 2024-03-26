import argparse

import torch
import torch_cluster

from ..interface import resolve_md_model, resolve_jit_model
from ..utils import NetConfig, set_default_unit


def compile_model(args: argparse.Namespace) -> None:
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # load checkpoint and config
    ckpt = torch.load(args.ckpt, map_location=device)
    config = NetConfig.model_validate(ckpt["config"])
    
    # set default unit
    set_default_unit(config.default_property_unit, config.default_length_unit)

    # build model
    if args.for_md:
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
