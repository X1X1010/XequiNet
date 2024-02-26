import warnings
import argparse
from copy import deepcopy

import numpy as np
import torch
import torch_cluster  # must import torch_cluster because it is used in the jit model

from ase.io import read as ase_read
from pyscf import gto
from pyscf.geomopt import geometric_solver, as_pyscf_method
try:
    import tblite.interface as xtb
except:
    warnings.warn("xtb is not installed, xtb calculation will not be performed.")



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


def xeq_method(mol: gto.Mole, model: torch.jit.ScriptModule, device: torch.device, base_method: str = None) -> tuple:
    """Xequinet method for energy and gradient calculation."""
    at_no = torch.tensor(mol.atom_charges(), dtype=torch.long, device=device)
    coord = torch.tensor(mol.atom_coords(), dtype=torch.get_default_dtype(), device=device)
    model_res: dict = model(at_no=at_no, coord=coord, charge=mol.charge, spin=mol.spin)
    energy = model_res.get("energy").item()
    nuc_grad = model_res.get("gradient").detach().cpu().numpy()
    if base_method is not None:
        m_dict = {"gfn2-xtb": "GFN2-xTB", "gfn1-xtb": "GFN1-xTB", "ipea1-xtb": "IPEA1-xTB"}
        calc = xtb.Calculator(
            method=m_dict[base_method.lower()],
            numbers=mol.atom_charges(),
            positions=mol.atom_coords(),
            charge=mol.charge,
            uhf=mol.spin,
        )
        xtb_res = calc.singlepoint()
        energy += xtb_res.get("energy")
        nuc_grad += xtb_res.get("gradient")
    return energy, nuc_grad


def calc_analytical_hessian(mol: gto.Mole, model: torch.jit.ScriptModule, device: torch.device) -> tuple:
    """Calculate the hessian with analytical second derivative."""
    at_no = torch.tensor(mol.atom_charges(), dtype=torch.long, device=device)
    coord = torch.tensor(mol.atom_coords(), dtype=torch.get_default_dtype(), device=device)
    coord.requires_grad = True
    model_res: dict = model(at_no=at_no, coord=coord, charge=mol.charge, spin=mol.spin)
    energy = model_res.get("energy")
    nuc_grad = model_res.get("gradient")
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


def to_shermo(shm_file: str, mol: gto.Mole, energy: float, wavenums: np.ndarray):
    with open(shm_file, 'w') as f:
        f.write(f"*E\n    {energy:10.6f}\n")
        f.write("*wavenum\n")
        # chage imaginary frequency to negative frequency
        if np.iscomplexobj(wavenums):
            wavenums = wavenums.real - abs(wavenums.imag)
        for wavenum in wavenums:
            f.write(f"    {wavenum:8.4f}\n")
        f.write("*atoms\n")
        elements = mol.elements
        masses = mol.atom_mass_list()
        coords = mol.atom_coords(unit="Angstrom")
        for e, m, c in zip(elements, masses, coords):
            f.write(f"{e: <2} {m:10.6f} {c[0]:10.6f} {c[1]:10.6f} {c[2]:10.6f}\n")
        f.write("*elevel\n    0.000000   1\n")


def main():
    # parse arguments
    parser = argparse.ArgumentParser(description="XequiNet geometry optimization script")
    parser.add_argument(
        "--ckpt", "-c", type=str, required=True,
        help="Path to the checkpoint file. (XXX.jit)",
    )
    parser.add_argument(
        "--delta", "-d", type=str, default=None,
        help="Base method for energy and force calculation.",
    )
    parser.add_argument(
        "--max-steps", type=int, default=100,
        help="Maximum number of optimization steps.",
    )
    parser.add_argument(
        "--cons", type=str, default=None,
        help="Constraints file for optimization.",
    )
    parser.add_argument(
        "--freq", action="store_true",
        help="Calculate vibrational frequencies.",
    )
    parser.add_argument(
        "--numer", action="store_true",
        help="Calculate hessian with numerical second derivative.",
    )
    parser.add_argument(
        "--shm", action="store_true",
        help="Whether to write shermo input file.",
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
        "--warning", "-w", action="store_true",
        help="Whether to show warning messages",
    )
    parser.add_argument(
        "inp", type=str,
        help="Input xyz file."
    )
    args = parser.parse_args()

    # open warning or not
    if not args.warning:
        warnings.filterwarnings("ignore")

    if args.freq:
        # create a new freq log file
        freq_log = args.inp.split(".")[0] + "_freq.log"
        with open(freq_log, 'w') as f:
           f.write(f"XequiNet Frequency Calculation\n\n")
        from pyscf.hessian import thermo
    
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load jit model
    model = torch.jit.load(args.ckpt, map_location=device)
    model.eval()

    atoms_list = ase_read(args.inp, index=':')
    # loop over molecules
    for atoms in atoms_list:
        # convert `ase.Atoms` to `pyscf.gto.Mole`
        chem_symb = atoms.get_chemical_symbols()
        positions = atoms.get_positions()
        charge = atoms.info.get("charge", 0)
        if "multiplicity" in atoms.info:
            spin = atoms.info["multiplicity"] - 1
        else:
            spin = atoms.info.get("spin", 0)
        # load molecule
        mol = gto.M(
            atom=[(a, c) for a, c in zip(chem_symb, positions)],  # [(atom, ndarray[x, y, z]), ...]
            charge=charge,
            spin=spin,  # multiplicity = spin + 1
        )
        # create a fake method as pyscf method, which returns energy and gradient
        fake_method = as_pyscf_method(mol, lambda mol: xeq_method(mol, model, device, args.delta))
        if args.no_opt:  # given a optimized geometry, so copy the molecule
            conv = True
            new_mol = deepcopy(mol)
        else:            # optimize geometry and return a new molecule
            conv, new_mol = geometric_solver.kernel(fake_method, constraints=args.cons, maxsteps=args.max_steps)

        if args.freq:
            # open freq output file
            new_mol.stdout = open(freq_log, 'a')
            if args.numer or args.delta is not None:
                # calculate hessian with numerical second derivative
                energy, hessian = calc_numerical_hessian(new_mol, model, device, args.delta)
            else:
                # calculate hessian with analytical second derivative
                energy, hessian = calc_analytical_hessian(new_mol, model, device)
            # do thermo calculation
            harmonic_res = thermo.harmonic_analysis(new_mol, hessian)
            if args.verbose >= 1:
                thermo.dump_normal_mode(new_mol, harmonic_res)   # dump normal modes in new_mol.stdout, i.e. freq output file
            setattr(fake_method, "e_tot", energy)
            thermo_res = thermo.thermo(fake_method, harmonic_res['freq_au'], args.temp)
            thermo.dump_thermo(new_mol, thermo_res)            # dump thermo results in new_mol.stdout, i.e. freq output file
            new_mol.stdout.write("\n\n")
            new_mol.stdout.close()
            if args.shm:
                shm_file = args.inp.split(".")[0] + "_freq.shm"
                to_shermo(shm_file, new_mol, energy, harmonic_res['freq_wavenumber'])

        # write optimized geometry
        if not args.no_opt:
            opt_out = args.inp.split(".")[0] + "_opt.xyz"
            with open(opt_out, 'a') as f:
                f.write(f"{new_mol.natm}\n")
                if conv:
                    energy, _ = xeq_method(new_mol, model, device, args.delta)
                    f.write(f"Energy: {energy:10.10f}\n")
                else:
                    f.write(f"Warning!! Optimization not converged!!\n")
                for e, c in zip(new_mol.elements, new_mol.atom_coords(unit="Angstrom")):
                    f.write(f"{e: <2} {c[0]:10.6f} {c[1]:10.6f} {c[2]:10.6f}\n")
    