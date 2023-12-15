import argparse
import torch
import torch_cluster
import torch_scatter

from pyscf import gto
from pyscf.geomopt import geometric_solver, as_pyscf_method

from xequinet.utils.qc import ELEMENTS_LIST
import tblite.interface as xtb


def read_xyz(xyz_file: str) -> str:
    with open(xyz_file, 'r') as f:
        mol_tokens = []
        while True:
            line = f.readline().strip()
            if not line: break
            n_atoms = int(line)
            comment = f.readline()
            mol_token = []
            for _ in range(n_atoms):
                line = f.readline().strip()
                mol_token.append(line)
            mol_tokens.append("\n".join(mol_token))
    return mol_tokens


def xeq_method(mol: gto.Mole, model: torch.nn.Module, device: torch.device, base_method: str = None) -> tuple:
    """Xequinet method for energy and gradient calculation."""
    at_no = torch.tensor(mol.atom_charges(), dtype=torch.long, device=device)
    coord = torch.tensor(mol.atom_coords(unit="Angstrom"), dtype=torch.get_default_dtype(), device=device)
    coord.requires_grad = True
    batch = torch.zeros_like(at_no, dtype=torch.long, device=device)
    energy, force = model(at_no, coord, batch)
    energy = energy.item()
    force = force.detach().cpu().numpy()
    if base_method is not None:
        m_dict = {"gfn2-xtb": "GFN2-xTB", "gfn1-xtb": "GFN1-xTB", "ipea1-xtb": "IPEA1-xTB"}
        calc = xtb.Calculator(
            method=m_dict[base_method.lower()],
            numbers=mol.atom_charges(),
            positions=mol.atom_coords(unit="Bohr"),
            charge=mol.charge,
            uhf=mol.spin,
        )
        res = calc.singlepoint()
        energy += res.get("energy")
        force -= res.get("gradient")

    return energy, -force
    

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
        "inp", type=str,
        help="Input xyz file."
    )
    args = parser.parse_args()

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load jit model
    model = torch.jit.load(args.ckpt, map_location=device)
    model.eval()

    mol_tokens = read_xyz(args.inp)
    for mol_token in mol_tokens:
        # load molecule
        mol = gto.M(
            atom=mol_token,
            unit="Angstrom",
        )
        fake_method = as_pyscf_method(mol, lambda mol: xeq_method(mol, model, device, args.base_method))
        conv, new_mol = geometric_solver.kernel(fake_method, maxsteps=args.max_steps, constraints=args.constraints)
        
        # write optimized geometry
        wfile = args.inp.split(".")[0] + "_opt.xyz"
        with open(wfile, 'a') as f:
            f.write(f"{mol.natm}\n")
            if conv:
                energy, _ = xeq_method(new_mol, model, device, args.base_method)
                f.write(f"Energy: {energy:10.10f}\n")
            else:
                f.write(f"Warning!! Optimization not converged!!\n")
            for at_no, coord in zip(new_mol.atom_charges(), new_mol.atom_coords(unit="Angstrom")):
                f.write(f"{ELEMENTS_LIST[at_no]} {coord[0]:10.6f} {coord[1]:10.6f} {coord[2]:10.6f}\n")
    