import argparse
from typing import cast

import torch
from omegaconf import OmegaConf

from xequinet.interface import resolve_jit_model
from xequinet.utils import ModelConfig, qc, set_default_units


def compile_model(args: argparse.Namespace) -> None:
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load checkpoint and config
    ckpt = torch.load(args.ckpt, map_location=device)
    config = OmegaConf.merge(
        OmegaConf.structured(ModelConfig),
        OmegaConf.load(ckpt["config"]),
    )
    # this will do nothing, only for type annotation
    config = cast(ModelConfig, config)

    # set default unit
    set_default_units(config.default_units)

    # build model
    model = resolve_jit_model(config.model_name, **config.model_kwargs)
    model.eval().to(device)

    model.load_state_dict(ckpt["model"])
    model_script = torch.jit.script(model)

    extra_file = {
        "cutoff_radius": model.cutoff_radius,
    }
    extra_file.update(qc.ELEMENTS_DICT)
    _extra_file = {k: str(v).encode("ascii") for k, v in extra_file.items()}

    output_file = (
        f"{args.ckpt.split('/')[-1].split('.')[0]}.jit"
        if args.output is None
        else args.output
    )
    model_script.save(output_file, _extra_files=_extra_file)
