import argparse
from copy import deepcopy

import numpy as np
import torch
import torch_cluster

from pyscf import gto
from pyscf.geomopt import geometric_solver, as_pyscf_method

from xequinet.utils.qc import ELEMENTS_LIST
from xequinet.interface import xtb_calculation


def read_xyz(xyz_file: str) -> tuple:
    """Read a continuous xyz file and return a list of molecule tokens."""
    mol_tokens = []
    charges, spins = [], []
    with open(xyz_file, 'r') as f:
        while True:
            line = f.readline().strip()
            if not line: break
            n_atoms = int(line)
            comment = f.readline()
            try:
                charge, multiplicity = list(map(int, comment.strip().split()))
            except:
                charge, multiplicity = 0, 1
            charges.append(charge)
            spins.append(multiplicity - 1)
            mol_token = []
            for _ in range(n_atoms):
                line = f.readline().strip()
                mol_token.append(line)
            mol_tokens.append("\n".join(mol_token))
    return mol_tokens, charges, spins


def xeq_method(mol: gto.Mole, model: torch.nn.Module, device: torch.device, base_method: str = None) -> tuple:
    """Xequinet method for energy and gradient calculation."""
    at_no = torch.tensor(mol.atom_charges(), dtype=torch.long, device=device)
    coord = torch.tensor(mol.atom_coords(), dtype=torch.get_default_dtype(), device=device)
    energy, nuc_grad = model(at_no, coord, mol.charge, mol.spin)
    energy = energy.item()
    nuc_grad = nuc_grad.detach().cpu().numpy()
    if base_method is not None:
        d_ene, d_grad = xtb_calculation(
            mol.atom_charges(),
            mol.atom_coords(),
            mol.charge,
            mol.spin,
            method=base_method,
            calc_grad=True,
        )
        energy += d_ene
        nuc_grad += d_grad
    return energy, nuc_grad


def calc_analytical_hessian(mol: gto.Mole, model: torch.nn.Module, device: torch.device) -> tuple:
    """Calculate the hessian with analytical second derivative."""
    at_no = torch.tensor(mol.atom_charges(), dtype=torch.long, device=device)
    coord = torch.tensor(mol.atom_coords(), dtype=torch.get_default_dtype(), device=device)
    coord.requires_grad = True
    energy, nuc_grad = model(at_no, coord, mol.charge, mol.spin)
    hessian = torch.zeros((mol.natm, mol.natm, 3, 3), dtype=torch.get_default_dtype(), device=device)
    for i in range(mol.natm):
        for j in range(3):
            hessian[i, :, j, :] = torch.autograd.grad(nuc_grad[i, j], coord, retain_graph=True)[0]
    hessian = hessian.cpu().numpy()
    return energy.item(), hessian


def calc_numerical_hessian(mol: gto.Mole, model: torch.nn.Module, device: torch.device, base_method: str = None) -> tuple:
    """Calculate the hessian with numerical second derivative."""
    energy, _ = xeq_method(mol, model, device, base_method)
    hessian = np.zeros((mol.natm, mol.natm, 3, 3))
    at_no = mol.atom_charges()
    coord = mol.atom_coords()
    h = 1e-5
    for i in range(mol.natm):
        for j in range(3):
            coord[i, j] += h
            d_mol = gto.M(atom=[(a, c) for a, c in zip(at_no, coord)], unit="Bohr", charge=mol.charge, spin=mol.spin)
            _, gfwd = xeq_method(d_mol, model, device, base_method)
            coord[i, j] -= 2 * h
            d_mol = gto.M(atom=[(a, c) for a, c in zip(at_no, coord)], unit="Bohr", charge=mol.charge, spin=mol.spin)
            _, gbak = xeq_method(d_mol, model, device, base_method)
            coord[i, j] += h
            hessian[i, :, j, :] = (gfwd - gbak) / (2 * h)
    return energy, hessian


def main():
    # parse arguments
    parser = argparse.ArgumentParser(description="Xequinet geometry optimization script")
    parser.add_argument(
        "--ckpt", "-c", type=str, required=True,
        help="Path to the checkpoint file. (XXX.jit)",
    )
    parser.add_argument(
        "--base-method", "-bm", type=str, default=None,
        help="Base method for energy and force calculation.",
    )
    parser.add_argument(
        "--max-steps", "-ms", type=int, default=100,
        help="Maximum number of optimization steps.",
    )
    parser.add_argument(
        "--constraints", "-con", type=str, default=None,
        help="Constraints file for optimization.",
    )
    parser.add_argument(
        "--freq", action="store_true",
        help="Calculate vibrational frequencies.",
    )
    parser.add_argument(
        "--numerical", action="store_true",
        help="Calculate hessian with numerical second derivative.",
    )
    parser.add_argument(
        "--no-opt", action="store_true",
        help="Do not perform optimization.",
    )
    parser.add_argument(
        "--temp", "-T", type=float, default=298.15,
        help="Temperature for vibrational frequencies.",
    )
    parser.add_argument(
        "--verbose", "-v", type=int, default=0,
        help="Verbosity level.",
    )
    parser.add_argument(
        "inp", type=str,
        help="Input xyz file."
    )
    args = parser.parse_args()

    if args.freq:
        # create a new freq log file
        freq_log = args.inp.split(".")[0] + "_freq.log"
        with open(freq_log, 'w') as f:
           f.write(f"Xequinet Frequency Calculation\n\n")
        from pyscf.hessian import thermo
    
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load jit model
    model = torch.jit.load(args.ckpt, map_location=device)
    model.eval()

    mol_tokens, charges, spins = read_xyz(args.inp)
    # loop over molecules
    for mol_token, charge, spin in zip(mol_tokens, charges, spins):
        # load molecule
        mol = gto.M(
            atom=mol_token,
            charge=charge,
            spin=spin,        # multiplicity = spin + 1
        )
        # create a fake method as pyscf method, which returns energy and gradient
        fake_method = as_pyscf_method(mol, lambda mol: xeq_method(mol, model, device, args.base_method))
        if args.no_opt:  # given a optimized geometry, so copy the molecule
            conv = True
            new_mol = deepcopy(mol)
        else:            # optimize geometry and return a new molecule
            conv, new_mol = geometric_solver.kernel(fake_method, constraints=args.constraints, maxsteps=args.max_steps)

        if args.freq:
            # open freq output file
            mol.stdout = open(freq_log, 'a')
            if args.numerical or args.base_method is not None:
                # calculate hessian with numerical second derivative
                energy, hessian = calc_numerical_hessian(mol, model, device, args.base_method)
            else:
                # calculate hessian with analytical second derivative
                energy, hessian = calc_analytical_hessian(mol, model, device)
            # do thermo calculation
            results = thermo.harmonic_analysis(mol, hessian)
            if args.verbose >= 1:
                thermo.dump_normal_mode(mol, results)   # dump normal modes in mol.stdout, i.e. freq output file
            setattr(fake_method, "e_tot", energy)
            results = thermo.thermo(fake_method, results['freq_au'], args.temp)
            thermo.dump_thermo(mol, results)   # dump thermo results in mol.stdout, i.e. freq output file
            mol.stdout.write("\n\n")
            mol.stdout.close()

        # write optimized geometry
        if not args.no_opt:
            opt_out = args.inp.split(".")[0] + "_opt.xyz"
            with open(opt_out, 'a') as f:
                f.write(f"{new_mol.natm}\n")
                if conv:
                    energy, _ = xeq_method(new_mol, model, device, args.base_method)
                    f.write(f"Energy: {energy:10.10f}\n")
                else:
                    f.write(f"Warning!! Optimization not converged!!\n")
                for at_no, coord in zip(new_mol.atom_charges(), new_mol.atom_coords(unit="Angstrom")):
                    f.write(f"{ELEMENTS_LIST[at_no]} {coord[0]:10.6f} {coord[1]:10.6f} {coord[2]:10.6f}\n")
    