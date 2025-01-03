import argparse
from typing import cast

import torch
from omegaconf import OmegaConf

from xequinet import keys
from xequinet.interface import resolve_jit_model
from xequinet.utils import ModelConfig, qc, set_default_units


def check_fusion_strategy(value: str) -> bool:
    """
    DYNAMICS,3;STATICS,2;...
    """
    for strategy in value.split(";"):
        splited_strategy = strategy.split(",")
        if len(splited_strategy) != 2:
            return False
        s, n = splited_strategy
        if s not in ["DYNAMICS", "STATICS"]:
            return False
        if not n.isdigit():
            return False
    return True


def compile_model(args: argparse.Namespace) -> None:
    # set device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # check if the strategy is valid
    if not check_fusion_strategy(args.fusion_strategy):
        raise ValueError(f"Invalid fusion strategy: {args.fusion_strategy}")

    # load checkpoint and config
    ckpt = torch.load(args.ckpt, map_location=device)
    config = OmegaConf.merge(
        OmegaConf.structured(ModelConfig),
        ckpt["config"],
    )
    # this will do nothing, only for type annotation
    config = cast(ModelConfig, config)

    # set default unit
    set_default_units(config.default_units)

    # build model
    model = resolve_jit_model(
        model_name=config.model_name,
        mode=args.mode,
        net_charge=args.net_charge,
        **config.model_kwargs,
    )
    model.eval().to(device)

    # load checkpoint
    model.load_state_dict(ckpt["model"])
    model_script = torch.jit.script(model)

    # save model
    n_species = qc.ELEMENTS_DICT["Rn"] + 1  # currently support up to Rn
    extra_files = {
        keys.CUTOFF_RADIUS: model.cutoff_radius,
        keys.JIT_FUSION_STRATEGY: args.fusion_strategy,
        keys.N_SPECIES: n_species,
        keys.PERIODIC_TABLE: " ".join(qc.ELEMENTS_LIST[:n_species]),
    }
    _extra_files = {k: str(v).encode("ascii") for k, v in extra_files.items()}

    if args.net_charge is None:
        chg_mark = ""
    elif args.net_charge > 0:
        chg_mark = f"c+{args.net_charge}"
    else:
        chg_mark = f"c{args.net_charge}"
    output_file = (
        f"{args.ckpt.split('/')[-1].split('.')[0]}-{args.mode}-{chg_mark}.jit"
        if args.output is None
        else args.output
    )
    model_script.save(output_file, _extra_files=_extra_files)
