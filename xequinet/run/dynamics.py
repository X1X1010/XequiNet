import json
import argparse

import numpy as np

import torch

from ase import units
from ase.io import read as ase_read
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution, Stationary, ZeroRotation
)

from xequinet.interface import XeqCalculator


default_settings = {
    "init_xyz": "init.extxyz",
    "charge": 0,
    "multiplicity": 1,
    "ckpt_file": "model.jit",
    "cutoff": 5.0,

    "ensemble": "NVT",         # choices=["NVE", "NVT", "NPT"]
    "dynamics": "Nose-Hoover", # choices=["Langevin", "Andersen", "Nose-Hoover", "Berendsen", "Parrinello-Rahman"]
    "timestep": 1.0,           # (fs)
    "temperature": 298.15,     # (K)
    "friction": 0.002,         # for Langevin
    "andersen_prob": 0.01,     # for Andersen
    "taut": 100.0,             # for Nose-Hoover and Berendsen (fs)
    "taup": 500.0,             # for Berendsen (fs)
    # "externalstress": 1.01325, # for Parrinello-Rahman (bar)
    "pressure": 1.01325,       # for Berendsen or Parrinello-Rahman (bar)
    "pfactor": 1.0,            # for Parrinello-Rahman (GPa)
    "compressibility": 5e-7,   # for Berendsen (1/bar)
    "mask": None,              # for Parrinello-Rahman
    "fixcm": True,
    "trajectory": "traj.xyz",
    "logfile": "md.log",
    "loginterval": 1,
    "append_trajectory": False,

    "seed": None,
}


def resolve_ensemble(ensemble, dynamics, atoms, **kwargs):
    if ensemble == "NVE":
        from ase.md.verlet import VelocityVerlet
        return VelocityVerlet(
            atoms, timestep=kwargs["timestep"] * units.fs,
            trajectory=kwargs["trajectory"], logfile=kwargs["logfile"],
            loginterval=kwargs["loginterval"], append_trajectory=kwargs["append_trajectory"],
        )
    elif ensemble == "NVT":
        if dynamics == "Langevin":
            from ase.md.langevin import Langevin
            return Langevin(
                atoms, timestep=kwargs["timestep"] * units.fs, temperature_K=kwargs["temperature"],
                friction=kwargs["friction"] / units.fs, fixcm=kwargs["fixcm"],
                trajectory=kwargs["trajectory"], logfile=kwargs["logfile"],
                loginterval=kwargs["loginterval"], append_trajectory=kwargs["append_trajectory"],
            )
        if dynamics == "Andersen":
            from ase.md.andersen import Andersen
            return Andersen(
                atoms, timestep=kwargs["timestep"] * units.fs, temperature_K=kwargs["temperature"],
                andersen_prob=kwargs["andersen_prob"], fixcm=kwargs["fixcm"],
                trajectory=kwargs["trajectory"], logfile=kwargs["logfile"],
                loginterval=kwargs["loginterval"], append_trajectory=kwargs["append_trajectory"],
            )
        elif dynamics == "Nose-Hoover":      # Nose-Hoover NVT is NPT with no pressure
            from ase.md.npt import NPT
            return NPT(
                atoms, timestep=kwargs["timestep"] * units.fs, temperature_K=kwargs["temperature"],
                ttime=kwargs["taut"] * units.fs,
                trajectory=kwargs["trajectory"], logfile=kwargs["logfile"],
                loginterval=kwargs["loginterval"], append_trajectory=kwargs["append_trajectory"],
            )  
        elif dynamics == "Berendsen":
            from ase.md.nvtberendsen import NVTBerendsen
            return NVTBerendsen(
                atoms, timestep=kwargs["timestep"] * units.fs, temperature_K=kwargs["temperature"],
                taut=kwargs["taut"] * units.fs, fixcm=kwargs["fixcm"],
                trajectory=kwargs["trajectory"], logfile=kwargs["logfile"],
                loginterval=kwargs["loginterval"], append_trajectory=kwargs["append_trajectory"],
            )
        else:
            raise NotImplementedError(f"Unknown dynamics: {dynamics}")
    elif ensemble == "NPT":
        if dynamics == "Parrinello-Rahman":  # with Nose-Hoover thermostat
            from ase.md.npt import NPT
            return NPT(
                atoms, timestep=kwargs["timestep"] * units.fs, temperature_K=kwargs["temperature"],
                ttime=kwargs["taut"] * units.fs, externalstress=kwargs["pressure"] * units.bar,
                pfactor=kwargs["pfactor"] * units.GPa * (units.fs**2), mask=kwargs["mask"],
                trajectory=kwargs["trajectory"], logfile=kwargs["logfile"],
                loginterval=kwargs["loginterval"], append_trajectory=kwargs["append_trajectory"],
            )
        elif dynamics == "Berendsen":        # with Berendsen thermostat
            from ase.md.nptberendsen import NPTBerendsen
            return NPTBerendsen(
                atoms, timestep=kwargs["timestep"] * units.fs, temperature_K=kwargs["temperature"],
                taut=kwargs["taut"] * units.fs, taup=kwargs["taup"] * units.fs, pressure=kwargs["pressure"] * units.bar,
                compressibility_au=kwargs["compressibility"] / units.bar, fixcm=kwargs["fixcm"],
                trajectory=kwargs["trajectory"], logfile=kwargs["logfile"],
                loginterval=kwargs["loginterval"], append_trajectory=kwargs["append_trajectory"],
            )
        else:
            raise NotImplementedError(f"Unknown dynamics: {dynamics}")
    else:
        raise NotImplementedError(f"Unknown ensemble: {ensemble}")


def main():
    # parse arguments
    parser = argparse.ArgumentParser(description="XequiNet molecular dynamics script")
    parser.add_argument(
        "settings", type=str, default=None,
        help="Setting file for the molecular dynamics. (settings.json)",
    )
    args = parser.parse_args()

    # dump settings
    settings = default_settings.copy()
    if args.settings is not None:
        with open(args.settings, 'r') as f:
            settings.update(json.load(f))

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # set random seed
    seed = settings["seed"]
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    # load atoms
    init_xyz = settings["init_xyz"]
    atoms = ase_read(init_xyz, index=0)

    # set calculator
    calc = XeqCalculator(
        ckpt_file=settings["ckpt_file"],
        device=device,
        cutoff=settings["cutoff"],
        charge=settings["charge"],
        spin=settings["multiplicity"] - 1,
    )
    atoms.set_calculator(calc)

    # set starting tempeature
    MaxwellBoltzmannDistribution(atoms, temperature_K=settings["temperature"])
    # TODO: reconsideration
    ZeroRotation(atoms)
    Stationary(atoms)

    # set ensemble
    ensemble = settings["ensemble"]; dynamics = settings["dynamics"]
    dyn = resolve_ensemble(ensemble, dynamics, atoms, **settings)

    # run dynamics
    dyn.run()


if __name__ == "__main__":
    main()