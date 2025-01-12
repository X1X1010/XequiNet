import ase
import tblite.interface as xtb
import torch
from pyscf import gto, pbc

from xequinet import keys
from xequinet.utils import get_default_units, qc, unit_conversion

from .datapoint import XequiData

# Only pass the basic structure. The properties will not be passed.


def datapoint_from_ase(atoms: ase.Atoms, dtype: torch.dtype = None) -> XequiData:
    """Convert an ASE Atoms object to a XequiData object."""

    dtype = dtype if dtype is not None else torch.get_default_dtype()
    pos_unit = get_default_units()[keys.POSITIONS]
    pos_factor = unit_conversion("Angstrom", pos_unit)

    atomic_numbers = atoms.get_atomic_numbers()
    pos = atoms.get_positions(wrap=True) * pos_factor
    pbc = atoms.get_pbc()
    cell = atoms.get_cell().array * pos_factor if pbc.any() else None
    # if one want to set charge, please set `charge=...` in the comment line
    charge = int(atoms.info.get("charge", 0))
    # if one want to set spin, please set `spin=...` in the comment line
    if "spin" in atoms.info:
        spin = int(atoms.info["spin"])
    # or one can set mutiplicity instead of spin
    elif "multiplicity" in atoms.info:
        spin = int(atoms.info["multiplicity"]) - 1
    else:
        spin = 0
    return XequiData(
        atomic_numbers=torch.from_numpy(atomic_numbers).to(torch.int),
        pos=torch.from_numpy(pos).to(dtype),
        pbc=torch.from_numpy(pbc).view(1, 3).to(torch.bool),
        cell=torch.from_numpy(cell).view(1, 3, 3).to(dtype)
        if cell is not None
        else None,
        charge=torch.tensor([charge], dtype=torch.int),
        spin=torch.tensor([spin], dtype=torch.int),
    )


def datapoint_from_pyscf(mole: gto.Mole, dtype: torch.dtype = None) -> XequiData:
    """Convert a PySCF Mole object to a XequiData object."""

    dtype = dtype if dtype is not None else torch.get_default_dtype()
    pos_unit = get_default_units()[keys.POSITIONS]
    pos_factor = unit_conversion("Bohr", pos_unit)

    atomic_numbers = mole.atom_charges()
    pos = mole.atom_coords() * pos_factor
    charge = mole.charge
    spin = mole.spin
    return XequiData(
        atomic_numbers=torch.from_numpy(atomic_numbers).to(torch.int),
        pos=torch.from_numpy(pos).to(dtype),
        charge=torch.tensor([charge], dtype=torch.int),
        spin=torch.tensor([spin], dtype=torch.int),
    )


def datapoint_to_ase(datapoint: XequiData) -> ase.Atoms:
    """Convert a XequiData object to an ASE Atoms object."""

    pos_unit = get_default_units()[keys.POSITIONS]
    pos_factor = unit_conversion(pos_unit, "Angstrom")

    atomic_numbers = datapoint.atomic_numbers.numpy()
    pos = datapoint.pos.numpy() * pos_factor
    if hasattr(datapoint, keys.CELL) and datapoint.cell is not None:
        cell = datapoint.cell.view(3, 3).numpy()
    else:
        cell = None
    if hasattr(datapoint, keys.PBC) and datapoint.pbc is not None:
        pbc = datapoint.pbc.view(3).numpy()
    else:
        pbc = None
    return ase.Atoms(
        numbers=atomic_numbers,
        positions=pos,
        pbc=pbc,
        cell=cell,
    )


def datapoint_to_pyscf(datapoint: XequiData) -> gto.MoleBase:
    """Convert a XequiData object to a PySCF Mole object."""

    pos_unit = get_default_units()[keys.POSITIONS]
    pos_factor = unit_conversion(pos_unit, "Angstrom")

    atomic_numbers = datapoint.atomic_numbers.numpy()
    pos = datapoint.pos.numpy() * pos_factor
    if hasattr(datapoint, keys.TOTAL_CHARGE) and datapoint.charge is not None:
        charge = datapoint.charge.item()
    else:
        charge = 0
    if hasattr(datapoint, keys.TOTAL_SPIN) and datapoint.spin is not None:
        spin = datapoint.spin.item()
    else:
        spin = 0
    if (
        hasattr(datapoint, keys.PBC)
        and datapoint.pbc is not None
        and datapoint.pbc.any()
    ):
        cell = datapoint.cell.view(3, 3).numpy()
        mole = pbc.gto.Cell(
            atom=[(a, c) for a, c in zip(atomic_numbers, pos)],
            charge=charge,
            spin=spin,
            a=cell,
        )
    else:
        mole = gto.M(
            atom=[(qc.ELEMENTS_LIST[a], c) for a, c in zip(atomic_numbers, pos)],
            charge=charge,
            spin=spin,
        )
    return mole


def datapoint_to_xtb(datapoint: XequiData, method: str = "GFN1-xTB"):
    """Convert a XequiData object to an xTB Calculator object."""

    pos_unit = get_default_units()[keys.POSITIONS]
    pos_factor = unit_conversion(pos_unit, "Bohr")

    atomic_numbers = datapoint.atomic_numbers.cpu().numpy()
    pos = datapoint.pos.cpu().numpy() * pos_factor
    charge = datapoint.charge.item()
    spin = datapoint.spin.item()
    if hasattr(datapoint, keys.CELL) and datapoint.cell is not None:
        cell = datapoint.cell.view(3, 3).cpu().numpy() * pos_factor
    else:
        cell = None
    if hasattr(datapoint, keys.PBC) and datapoint.pbc is not None:
        pbc = datapoint.pbc.view(3).cpu().numpy()
    else:
        pbc = None
    return xtb.Calculator(
        method=keys.xTB_METHODS[method.lower()],
        numbers=atomic_numbers,
        positions=pos,
        charge=charge,
        uhf=spin,
        lattice=cell,
        periodic=pbc,
    )
