from typing import Iterable
import os, sys
import numpy as np
import tblite.interface as xtb
from .mopac import MOPAC
from ..utils import unit_conversion, get_default_unit


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def xtb_calculation(
    atomic_numbers: Iterable[int],
    coordinates: Iterable[float],
    charge: int = 0,
    multiplicity: int = 1,
    method: str = "gfn2-xtb",
    calc_force: bool = False,
):
    """
    Args:
        `atomic_numbers`: Iterable of atomic numbers.
        `coordinates`: Iterable of coordinates.
        `charge`: Charge of the molecule.
        `multiplicity`: Multiplicity of the molecule.
        `method`: Method of the calculation.
        `calc_force`: Whether calculate the force.
    Returns:
        `energy`: Energy of the molecule.
        `force` (optional): Force of the molecule.
    """
    assert len(atomic_numbers) == len(coordinates)
    prop_unit, len_unit = get_default_unit()
    at_no = np.array(atomic_numbers)
    coord = np.array(coordinates) * unit_conversion(len_unit, "Bohr")
    uhf = multiplicity - 1
    m_dict = {"gfn2-xtb": "GFN2-xTB", "gfn1-xtb": "GFN1-xTB", "ipea1-xtb": "IPEA1-xTB"}
    calc = xtb.Calculator(
        method=m_dict[method.lower()],
        numbers=at_no,
        positions=coord,
        charge=charge,
        uhf=uhf,
    )
    with HiddenPrints():
        res = calc.singlepoint()
    energy = res.get("energy") * unit_conversion("Hartree", prop_unit)
    if calc_force:
        force = -res.get("gradient") * unit_conversion("AU", f"{prop_unit}/{len_unit}")
        return energy, force   
    else:
        return energy


def mopac_calculation(
    atomic_numbers: Iterable[int],
    coordinates: Iterable[float],
    charge: int = 0,
    multiplicity: int = 1,
    method: str = "PM7",
    calc_force: bool = False,
):
    """
    Args:
        `atomic_numbers`: Iterable of atomic numbers.
        `coordinates`: Iterable of coordinates.
        `charge`: Charge of the molecule.
        `multiplicity`: Multiplicity of the molecule.
        `method`: Method of the calculation.
        `calc_force`: Whether calculate the force.
    Returns:
        `energy`: Energy of the molecule.
        `force` (optional): Force of the molecule.
    """
    assert len(atomic_numbers) == len(coordinates)
    prop_unit, len_unit = get_default_unit()
    at_no = np.array(atomic_numbers)
    coord = np.array(coordinates) * unit_conversion(len_unit, "Angstrom")
    mopac = MOPAC(
        atoms=at_no,
        coordinates=coord,
        charge=charge,
        multiplicity=multiplicity,
        method=method,
        calc_force=calc_force,
    )
    mopac.calculate(clean=True)
    energy = mopac.get_final_heat_of_formation(unit=prop_unit)
    if calc_force:
        force = mopac.get_force(unit=f"{prop_unit}/{len_unit}")
        return energy, force
    else:
        return energy    

