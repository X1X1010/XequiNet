from typing import Optional, Union, Tuple, Dict
from math import pi
import warnings
import re
from pathlib import Path

from pyscf import gto
import numpy as np
import torch

from xequinet.utils import keys


def gen_units_dict():
    """
    Function that creates a dictionary containing all units previously hard
    """
    # constants from the NIST CODATA 2018, unit in SI
    _c = 299792458.            # speed of light, m/s     Exact
    _mu0 = 4.0e-7 * pi         # permeability of vacuum  Exact
    _Grav = 6.67430e-11        # gravitational constant  +/- 0.000_15e-11
    _hplanck = 6.62607015e-34  # Planck constant         Exact
    _e = 1.602176634e-19       # elementary charge       Exact
    _me = 9.1093837015e-31     # electron mass           +/- 0.000_000_0028e-31
    _mp = 1.67262192369e-27    # proton mass             +/- 0.000_000_000_51e-27
    _NA = 6.02214076e23        # Avogadro number         Exact
    _kB = 1.380649e-23         # Boltzmann constant      Exact
    _amu = 1.66053906660e-27   # atomic mass unit, kg    +/- 0.000_000_000_50e-27

    # derived from the CODATA values
    _eps0 = 1 / _mu0 / _c**2     # permittivity of vacuum
    _hbar = _hplanck / (2 * pi)  # Planck constant / 2pi, J s

    u = {}
    # Atomic Units
    u["AU"] = u["a.u."] = 1.0
    # amount of substance
    u["mol"] = _NA
    # charge
    u['e'] = 1.0
    u["Coloumb"] = u['C'] = 1 / _e
    # length
    u["Bohr"] = u["a0"] = 1.0
    u["meter"] = u["m"] = (_me * _e**2) / (4 * pi * _eps0 * _hbar**2)
    u["Angstrom"] = u["Ang"] = u['m'] * 1e-10
    u["cm"] = u["m"] * 1e-2
    u["nm"] = u["Angstrom"] * 10
    # energy
    u["Hartree"] = u["Ha"] = u["Eh"] = 1.0
    u["Joule"] = u['J'] = (4 * pi * _eps0 * _hbar)**2 / (_me * _e**4)
    u["kJoule"] = u["kJ"] = u['J'] * 1000
    u["eV"] = u['J'] * _e
    u["meV"] = u["eV"] / 1000
    u["cal"] = u['J'] * 4.184
    u["kcal"] = u["cal"] * 1000
    # dipole
    u["Debye"] = u['D'] = 1e21 * (4 * pi * _eps0 * _hbar**2 * _c) / (_me * _e)
    # time
    u["second"] = u['s'] = (_me * _e**4) / (4 * pi * _eps0)**2 / _hbar**3
    u["fs"] = u["s"] * 1e-15
    u["ps"] = u["fs"] * 1000
    # pressure
    u["Pascal"] = u["Pa"] = u['J'] / u['m']**3
    u["GPa"] = u["Pa"] * 1e9
    u["bar"] = u["Pa"] * 1e5
    u["kbar"] = u["bar"] * 1e3
    return u

units = gen_units_dict()
globals().update(units)


DEFAULT_UNITS_MAP = {
    keys.POSITIONS: "Angstrom",
}


def check_unit(unit: str) -> bool:
    """Check if the unit is valid and safe."""
    split_unit = re.split(r"[+ | \- | * | / | ^ | ( | )]", unit)
    for u in split_unit:
        if u == '':
            continue
        elif u in units:
            continue
        elif u.isdigit():
            continue
        else:
            return False
    return True


def eval_unit(unit: str) -> float:
    """Evaluate the unit."""
    if not check_unit(unit):
        raise ValueError(f"Invalid unit {unit}")
    unit = unit.replace('^', '**')

    return eval(unit)


def unit_conversion(unit_in: Optional[str], unit_out: Optional[str]) -> float:
    if unit_in is None or unit_out is None:
        return 1.
    if unit_in == unit_out:
        return 1.
    value_in = eval_unit(unit_in)
    value_out = eval_unit(unit_out)

    return value_in / value_out


def set_default_units(unit_dict: Dict[str, str]) -> None:
    for prop, unit in unit_dict.items():
        if prop in keys.GRAD_PROPERTIES:
            raise ValueError(
                "Please do not set units for gradient properties directly, Set the units for the corresponding properties instead."
            )
        if prop in keys.BASE_PROPERTIES:
            if not check_unit(unit):
                raise ValueError(f"Invalid unit {unit} for property {prop}")
        if not check_unit(unit):
            raise ValueError(f"Invalid unit {unit} for property {prop}")
    DEFAULT_UNITS_MAP.update(unit_dict)
    # adjust some settings for gradient properties
    if keys.TOTAL_ENERGY in DEFAULT_UNITS_MAP:
        energy_unit = DEFAULT_UNITS_MAP[keys.TOTAL_ENERGY]
        pos_unit = DEFAULT_UNITS_MAP[keys.POSITIONS]
        DEFAULT_UNITS_MAP[keys.FORCES] = f"{energy_unit}/{pos_unit}"
        DEFAULT_UNITS_MAP[keys.VIRIAL] = f"{energy_unit}/{pos_unit}^3"
    for base_prop, prop in keys.BASE_PROPERTIES.items():
        if prop in DEFAULT_UNITS_MAP:
            DEFAULT_UNITS_MAP[base_prop] = DEFAULT_UNITS_MAP[prop]


def get_default_units() -> Dict[str, str]:
    return DEFAULT_UNITS_MAP


THIS_FOLDER = Path(__file__).parent
BASIS_FOLDER = THIS_FOLDER / "basis"
PRE_FOLDER = THIS_FOLDER / "pre_computed"

# periodic table of elements
# atomic numbers to element symbols
ELEMENTS_LIST = ['d',
  'H' ,                                                                                                 'He',
  'Li', 'Be',                                                             'B' , 'C' , 'N' , 'O' , 'F' , 'Ne',
  'Na', 'Mg',                                                             'Al', 'Si', 'P' , 'S' , 'Cl', 'Ar',
  'K' , 'Ca', 'Sc', 'Ti', 'V' , 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
  'Rb', 'Sr', 'Y' , 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I' , 'Xe',
  'Cs', 'Ba',
              'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
                    'Hf', 'Ta', 'W' , 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn',
]
# element symbols to atomic numbers
ELEMENTS_DICT = {e: i for i, e in enumerate(ELEMENTS_LIST)}
# ground state multiplicities for each element
ATOM_MULT = [-1,
  2,                                                 1,
  2, 1,                               2, 3, 4, 3, 2, 1,
  2, 1,                               2, 3, 4, 3, 2, 1,
  2, 1, 2, 3, 4, 7, 6, 5, 4, 3, 2, 1, 2, 3, 4, 3, 2, 1,
  2, 1, 2, 3, 6, 7, 6, 5, 4, 1, 2, 1, 2, 3, 4, 3, 2, 1,
  2, 1,
        2, 3, 4, 5, 6, 7, 8, 9, 6, 5, 4, 3, 2, 1, 2,
           3, 4, 5, 6, 5, 4, 3, 2, 1, 2, 3, 4, 3, 2, 1,
]
# atomic masses
ATOM_MASS = torch.Tensor([0.0,
    1.008,                                                                                                                 4.003,
    6.941, 9.012,                                                                       10.81, 12.01, 14.01, 16.00, 19.00, 20.18,
    22.99, 24.31,                                                                       26.98, 28.09, 30.97, 32.06, 35.45, 39.95,
    39.10, 40.08, 44.96, 47.87, 50.94, 52.00, 54.94, 55.85, 58.93, 58.69, 63.55, 65.38, 69.72, 72.63, 74.92, 78.96, 79.90, 83.80,
    85.47, 87.62, 88.91, 91.22, 92.91, 95.96, 98.,   101.1, 102.9, 106.4, 107.9, 112.4, 114.8, 118.7, 121.8, 127.6, 126.9, 131.3,
    132.9, 137.3,
                  138.9, 140.1, 140.9, 144.2, 145.,  150.4, 152.0, 157.3, 158.9, 162.5, 164.9, 167.3, 168.9, 173.1, 175.0,
                         178.5, 180.9, 183.8, 186.2, 190.2, 192.2, 195.1, 197.0, 200.6, 204.4, 207.2, 209.,  210.,  210.,  222.,
])


def gen_int2c1e(embed_basis: str = "gfn2-xtb", aux_basis: str = "aux56") -> None:
    """
    Projection of atomic orbitals onto auxiliary basis.
    """
    int2c1e_dict = {}
    if (BASIS_FOLDER / f"{embed_basis}.dat").exists():
        basis = str(BASIS_FOLDER / f"{embed_basis}.dat")
    else:
        basis = embed_basis
    orbaux = str(BASIS_FOLDER / f"{aux_basis}.dat")
    savefile = PRE_FOLDER / f"{embed_basis}_{aux_basis}.pt"
    aux = gto.M(atom="X 0 0 0", basis={'X': orbaux})
    nao_aux = aux.nao
    ao_loc_nr = aux.ao_loc_nr()[:-1]

    for atom, mult in zip(ELEMENTS_LIST[1:], ATOM_MULT[1:]):
        mol = gto.M(
            atom=f"X 0 0 0; {atom} 0 0 0",
            basis={'X': orbaux, atom: basis},
            spin=mult - 1,
        )
        ovlp = mol.intor("int1e_ovlp")
        projection = ovlp[:nao_aux, nao_aux:]
        # embedding = np.sum(np.abs(projection), axis=-1))
        embedding = np.sum(projection, axis=-1)
        int2c1e_dict[atom] = torch.from_numpy(embedding[ao_loc_nr])
    torch.save(int2c1e_dict, savefile)


def gen_atom_sp(atom_ref: str) -> None:
    """
    Calculate the shift of the atomic energies for each element.
    """
    atom_sp_dict = {}
    if "xtb" in atom_ref:
        from tblite.interface import Calculator
        m_dict = {"gfn2-xtb": "GFN2-xTB", "gfn1-xtb": "GFN1-xTB", "ipea1-xtb": "IPEA1-xTB"}
        for at_no, mult in enumerate(ATOM_MULT, start=1):
            calc = Calculator(
                method=m_dict[atom_ref.lower()],
                positions=np.array([[0.0, 0.0, 0.0]]),
                numbers=np.array([at_no]),
                uhf=mult - 1,
            )
            print(f"Calculating single point of atom {atom}")
            energy = calc.singlepoint().get("energy")
            atom_sp_dict[ELEMENTS_LIST[at_no]] = energy
    else:
        from pyscf import dft, scf, cc
        if "xyg" in atom_ref or "xdh" in atom_ref:
            from pyscf import dh
        method, basis = atom_ref.split("/")
        if "def2" in basis:
            ecp = basis
        elif "cc-pv" in basis:
            ecp = basis + "-pp"
        else:
            ecp = None
        for atom, mult in zip(ELEMENTS_LIST[1:], ATOM_MULT[1:]):
            try:
                mol = gto.Mole(atom=f"{atom} 0 0 0", basis=basis, spin=mult - 1)
                if ecp is not None:
                    mol.ecp = ecp
                mol.build()
            except:
                warnings.warn(f"Unsupported atom {atom} for basis {basis}")
                continue
            print(f"Calculating single point of atom {atom}")
            if "xyg" in method or "xdh" in method:
                mf = dh.DH(mol, xc=method).set(max_cycle=150).build_scf().run()
                if not mf.converged:
                    continue
                energy = mf.e_tot
            elif "cc" in method:
                mf = scf.HF(mol).set(max_cycle=150).run()
                mycc = cc.CCSD(mf).set(max_cycle=150).run()
                if not mycc.converged:
                    continue
                if method == "ccsd":
                    energy = mycc.e_tot
                elif method == "ccsd(t)":
                    energy = mycc.e_tot + mycc.ccsd_t()
                else:
                    raise ValueError(f"Unsupported method: {method}")
            else:
                mf = dft.KS(mol, xc=method).set(max_cycle=150).build().run()
                if not mf.converged:
                    continue
                energy = mf.e_tot
            atom_sp_dict[atom] = energy
    torch.save(atom_sp_dict, PRE_FOLDER / f"{method}_{basis}_sp.pt")


def get_embedding_tensor(embed_basis: str = "gfn2-xtb", aux_basis: str = "aux28") -> torch.Tensor:
    """
    Get embedding of atoms in a basis.

    Args:
        `embed_basis`: name of the basis
        `aux_basis`: name of the auxiliary basis
    Returns:
        a tensor of shape ``(n_atoms, n_aux)``
    """
    if not (PRE_FOLDER / f"{embed_basis}_{aux_basis}.pt").exists():
        gen_int2c1e(embed_basis, aux_basis)
    embed_dict = torch.load(PRE_FOLDER / f"{embed_basis}_{aux_basis}.pt")
    embed_tenor = torch.stack([embed_dict[atom] for atom in ELEMENTS_LIST[1:]])
    embed_tenor = torch.cat([torch.zeros(1, embed_tenor.shape[-1]), embed_tenor])
    return embed_tenor.to(torch.get_default_dtype())


def get_atomic_energy(atom_ref: Union[str, dict] = None) -> torch.Tensor:
    """
    Get the shift of the atomic energies for each element. Unit: Hartree

    Args:
        `atom_ref`: type of element shifts
    Returns:
        a tensor of shape ``(n_atoms,)``
    """
    if atom_ref is None:
        return torch.zeros(len(ELEMENTS_LIST), dtype=torch.float64)
    elif isinstance(atom_ref, dict):
        atomic_energy = torch.zeros(len(ELEMENTS_LIST), dtype=torch.float64)
        at_no = list(atom_ref.keys())
        if isinstance(at_no[0], str):
            at_no = [ELEMENTS_DICT[atom] for atom in at_no]
        atomic_energy[at_no] = torch.DoubleTensor(list(atom_ref.values()))
    else:
        atom_ref = atom_ref.lower()
        sp_file_name = f"{atom_ref.replace('/', '_')}_sp.pt"
        if not (PRE_FOLDER / sp_file_name).exists():
            gen_atom_sp(atom_ref)
        atom_sp_dict = torch.load(PRE_FOLDER / sp_file_name)
        atomic_energy = torch.zeros(len(ELEMENTS_LIST), dtype=torch.float64)
        periodic_table = """
H                                                  He
Li Be                               B  C  N  O  F  Ne
Na Mg                               Al Si P  S  Cl Ar
K  Ca Sc Ti V  Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr
Rb Sr Y  Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I  Xe
Cs Ba    Hf Ta W  Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn

      La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu
"""
        for atom, energy in atom_sp_dict.items():
            atomic_energy[ELEMENTS_DICT[atom]] = energy
            periodic_table = periodic_table.replace(f"{atom: <2}", "  ")
        if periodic_table.strip():  # if not all the elements are included
            warning_msg = f"The file {sp_file_name} does not contain single point energy for following atoms:\n"
            warning_msg += f"{periodic_table.strip()}\n"
            warning_msg += "If you need these atoms, please regenerate the file or add them manually."
            warnings.warn(warning_msg)

    return atomic_energy


if __name__ == "__main__":
    gen_int2c1e(embed_basis="gfn2-xtb", aux_basis="aux56")
