import argparse
import torch
import torch_cluster
import torch_scatter

from pyscf import gto, md
from pyscf.geomopt import as_pyscf_method

from xequinet.utils import unit_conversion


def read_xyz(xyz_file: str) -> str:
    with open(xyz_file, 'r') as f:
        line = f.readline().strip()
        n_atoms = int(line)
        comment = f.readline()
        mol_token = []
        for _ in range(n_atoms):
            line = f.readline().strip()
            mol_token.append(line)
    return mol_token


def xeq_method(mol: gto.Mole, model: torch.nn.Module, device: torch.device):
    """Xequinet method for energy and gradient calculation."""
    at_no = torch.tensor(mol.atom_charges(), dtype=torch.long, device=device)
    coord = torch.tensor(mol.atom_coords(unit="Angstrom"), dtype=torch.get_default_dtype(), device=device)
    coord.requires_grad = True
    batch = torch.zeros_like(at_no, dtype=torch.long, device=device)
    energy, force = model(at_no, coord, batch)
    energy = energy.item()
    force = force.detach().cpu().numpy()
    return energy, -force


def main():
    # parse arguments
    parser = argparse.ArgumentParser(description="Xequinet molecular dynamics script")
    parser.add_argument(
        "--ckpt", "-c", type=str, required=True,
        help="Path to the checkpoint file. (XXX.jit)",
    )
    parser.add_argument(
        "--dt", type=float, required=True,
        help="Time step (fs).",
    )
    parser.add_argument(
        "--steps", "-s", type=int, default=None,
        help="Total steps.",
    )
    parser.add_argument(
        "--temp", "-T", type=float, default=300.0,
        help="Temperature (K).",
    )
    parser.add_argument(
        "--taut", type=float, required=True,
        help="Time constant (fs).",
    )
    parser.add_argument(
        "--trj-out", "-to", type=str, default=None,
        help="Trajectory output file name.",
    )
    parser.add_argument(
        "--data-out", "-do", type=str, default=None,
        help="Data output file name."
    )
    parser.add_argument(
        "inp", type=str,
        help="Input xyz file."
    )
    args = parser.parse_args()

    # set output
    dout = args.inp.split(".")[0] + "_dat.log" if args.data_out is None else args.data_out
    tout = args.inp.split(".")[0] + "_trj.xyz" if args.traj_out is None else args.trj_out

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load jit model
    model = torch.jit.load(args.ckpt, map_location=device)
    model.eval()

    mol_token = read_xyz(args.inp)
    mol = gto.M(atom=mol_token, unit="Angstrom")
    fake_method = as_pyscf_method(mol, lambda mol: xeq_method(mol, model, device))

    init_veloc = md.distributions.MaxwellBoltzmannVelocity(mol, T=args.temp)

    myintegrator = md.integrators.NVTBerendson(
        fake_method,
        dt=args.dt * unit_conversion("fs", "AU"),
        steps=args.steps,
        T=args.T,
        taut = args.taut * unit_conversion("fs", "AU"),
        veloc=init_veloc,
        data_output=dout,
        trajectory_output=tout,
    )

    myintegrator.run()

    # close output file
    myintegrator.data_output.close()
    myintegrator.trajectory_output.close()


if __name__ == "__main__":
    main()