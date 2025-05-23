import argparse
import json
from typing import Dict, Tuple

import ase.io
import numpy as np
import torch
from pyscf import gto
from pyscf.geomopt import as_pyscf_method, geometric_solver
from pyscf.hessian import thermo

from xequinet import keys
from xequinet.data import (
    NeighborTransform,
    Transform,
    datapoint_from_ase,
    datapoint_from_pyscf,
    datapoint_to_pyscf,
    datapoint_to_xtb,
)
from xequinet.nn import resolve_model
from xequinet.utils import get_default_units, set_default_units, unit_conversion


@torch.no_grad()
def xequi_method(
    mole: gto.Mole,
    transform: Transform,
    model: torch.jit.ScriptModule,
    device: torch.device,
    base_method: str = None,
) -> Tuple[float, np.ndarray]:
    """
    Xequinet method for energy and gradient calculation.
    `energy` in Hartree and `gradient` in au
    """
    default_units = get_default_units()
    data = datapoint_from_pyscf(mole).to(device)
    data = transform(data)
    with torch.enable_grad():
        result: Dict[str, torch.Tensor] = model(
            data.to_dict(),
            compute_forces=True,
            compute_virial=False,
        )
    energy = result[keys.TOTAL_ENERGY].item()
    nuc_grad = -result[keys.FORCES].detach().cpu().numpy()
    # unit conversion
    energy *= unit_conversion(default_units[keys.TOTAL_ENERGY], "Hartree")
    nuc_grad *= unit_conversion(default_units[keys.FORCES], "au")
    if base_method is not None:
        xtb_calc = datapoint_to_xtb(data, method=base_method)
        xtb_res = xtb_calc.singlepoint()
        energy += xtb_res.get("energy")
        nuc_grad += xtb_res.get("gradient")
    return energy, nuc_grad


def calc_analytical_hessian(
    mole: gto.Mole,
    transform: Transform,
    model: torch.jit.ScriptModule,
    device: torch.device,
) -> tuple:
    """
    Calculate the hessian with analytical second derivative.
    `energy` in Hartree and `Hessian` in au
    """
    default_units = get_default_units()
    data = datapoint_from_pyscf(mole).to(device)
    data = transform(data)
    # In order to get the second derivative,
    # we need to set `retain_graph=True` and `create_graph=True`,
    # so we do this hack to set the model to training mode.
    model.train()
    data[keys.POSITIONS].requires_grad_()
    result: Dict[str, torch.Tensor] = model(
        data.to_dict(),
        compute_forces=True,
        compute_virial=False,
    )
    energy = result[keys.TOTAL_ENERGY].item()
    nuc_grad = -result[keys.FORCES]
    hessian = torch.zeros(
        (mole.natm, mole.natm, 3, 3),
        device=device,
    )
    for i in range(mole.natm):
        for j in range(3):
            hessian[i, :, j, :] = torch.autograd.grad(
                nuc_grad[i, j], data[keys.POSITIONS], retain_graph=True
            )[0]
    hessian = hessian.cpu().numpy()
    # unit conversion
    energy *= unit_conversion(default_units[keys.TOTAL_ENERGY], "Hartree")
    hessian *= unit_conversion(
        f"{default_units[keys.TOTAL_ENERGY]}/{default_units[keys.POSITIONS]}^2", "au"
    )
    return energy, hessian


def calc_numerical_hessian(
    mole: gto.Mole,
    transform: Transform,
    model: torch.nn.Module,
    device: torch.device,
    base_method: str = None,
) -> Tuple[float, np.ndarray]:
    """Calculate the hessian with numerical second derivative."""
    energy, _ = xequi_method(mole, transform, model, device, base_method)
    hessian = np.zeros((mole.natm, mole.natm, 3, 3))
    atomic_numbers = mole.atom_charges()
    pos = mole.atom_coords(unit="Bohr")
    h = 1e-5
    for i in range(mole.natm):
        for j in range(3):
            pos[i, j] += h
            d_mol = gto.M(
                atom=[(a, c) for a, c in zip(atomic_numbers, pos)],
                unit="Bohr",
                charge=mole.charge,
            )
            _, gfwd = xequi_method(d_mol, transform, model, device, base_method)
            pos[i, j] -= 2 * h
            d_mol = gto.M(
                atom=[(a, c) for a, c in zip(atomic_numbers, pos)],
                unit="Bohr",
                charge=mole.charge,
            )
            _, gbak = xequi_method(d_mol, transform, model, device, base_method)
            pos[i, j] += h
            hessian[i, :, j, :] = (gfwd - gbak) / (2 * h)
    return energy, hessian


def to_shermo(
    shm_file: str, mol: gto.Mole, energy: float, wavenums: np.ndarray
) -> None:
    with open(shm_file, "w") as f:
        f.write(f"*E\n    {energy:10.6f}\n")
        f.write("*wavenum\n")
        # chage imaginary frequency to negative frequency
        if np.iscomplexobj(wavenums):
            wavenums = wavenums.real - abs(wavenums.imag)
        for wavenum in wavenums:
            f.write(f"    {wavenum:8.4f}\n")
        f.write("*atoms\n")
        elements = mol.elements
        masses = mol.atom_mass_list(isotope_avg=True)
        coords = mol.atom_coords(unit="Angstrom")
        for e, m, c in zip(elements, masses, coords):
            f.write(f"{e: <2} {m:10.6f} {c[0]:10.6f} {c[1]:10.6f} {c[2]:10.6f}\n")
        f.write("*elevel\n    0.000000   1\n")


def run_opt(args: argparse.Namespace) -> None:
    if args.freq:
        # create a new freq log file
        freq_log = f"{args.input.split('.')[0]}_freq.log"
        with open(freq_log, "w") as f:
            f.write(f"XequiNet Frequency Calculation\n\n")

    # set device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # load checkpoint and config
    ckpt = torch.load(args.ckpt, map_location=device)
    model_config = ckpt["config"]
    # set default unit
    set_default_units(model_config["default_units"])
    # build model
    model = resolve_model(
        model_config["model_name"],
        **model_config["model_kwargs"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    transform = NeighborTransform(model.cutoff_radius)

    # set params for optimization
    if args.opt_params:
        with open(args.opt_params, "r") as f:
            opt_params = json.load(f)
    else:
        opt_params = {}

    atoms_list = ase.io.read(args.input, index=":", format=args.format)
    # loop over molecules
    for i, atoms in enumerate(atoms_list):
        data = datapoint_from_ase(atoms)
        mole = datapoint_to_pyscf(data)
        # create a fake method as pyscf method, which returns energy and gradient
        fake_method = as_pyscf_method(
            mole, lambda m: xequi_method(m, transform, model, device, args.delta)
        )
        if args.no_opt:  # given a optimized geometry, so copy the molecule
            conv = True
            new_mole = mole.copy()
        else:  # optimize geometry and return a new molecule
            conv, new_mole = geometric_solver.kernel(
                fake_method,
                constraints=args.constraints,
                maxsteps=args.max_steps,
                **opt_params,
            )

        if args.freq:
            suffix = "" if len(atoms_list) == 1 else f"{i}"
            # open freq output file
            new_mole.stdout = open(freq_log, "a")
            if args.delta is not None:
                # calculate hessian with numerical second derivative
                energy, hessian = calc_numerical_hessian(
                    new_mole, transform, model, device, args.delta
                )
            else:
                # calculate hessian with analytical second derivative
                energy, hessian = calc_analytical_hessian(
                    new_mole, transform, model, device
                )
            # do thermo calculation
            harmonic_res = thermo.harmonic_analysis(new_mole, hessian)
            if args.verbose:
                # dump normal modes in new_mol.stdout, i.e. freq output file
                thermo.dump_normal_mode(new_mole, harmonic_res)
            setattr(fake_method, "e_tot", energy)
            thermo_res = thermo.thermo(fake_method, harmonic_res["freq_au"], args.temp)
            # dump thermo results in new_mol.stdout, i.e. freq output file
            thermo.dump_thermo(new_mole, thermo_res)
            new_mole.stdout.write("\n\n")
            new_mole.stdout.close()
            if args.shermo:
                shm_file = args.input.split(".")[0] + f"_freq{suffix}.shm"
                to_shermo(shm_file, new_mole, energy, harmonic_res["freq_wavenumber"])
            if args.save_hessian:
                hessian_file = args.input.split(".")[0] + f"_h{suffix}.txt"
                np.savetxt(
                    hessian_file,
                    hessian.transpose(0, 2, 1, 3).reshape(
                        new_mole.natm * 3, new_mole.natm * 3
                    ),
                )
            new_mole.stdout.close()

        # write optimized geometry
        if not args.no_opt:
            opt_out = args.input.split(".")[0] + "_opt.xyz"
            with open(opt_out, "a") as f:
                f.write(f"{new_mole.natm}\n")
                if conv:
                    energy, _ = xequi_method(
                        new_mole, transform, model, device, args.delta
                    )
                    f.write(f"Energy: {energy:10.10f} Hartree\n")
                else:
                    f.write(f"Warning!! Optimization not converged!!\n")
                for a, c in zip(
                    new_mole.elements, new_mole.atom_coords(unit="Angstrom")
                ):
                    f.write(f"{a: <2}  {c[0]:10.6f}  {c[1]:10.6f}  {c[2]:10.6f}\n")
