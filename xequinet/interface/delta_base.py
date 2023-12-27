from typing import Iterable
import os, sys
import numpy as np
import tblite.interface as xtb
from .mopac import MOPAC


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
    uhf: int = 0,
    method: str = "gfn2-xtb",
    calc_grad: bool = False,
):
    """
    Args:
        `atomic_numbers`: Iterable of atomic numbers.
        `coordinates`: Iterable of coordinates.
        `charge`: Charge of the molecule.
        `multiplicity`: Multiplicity of the molecule.
        `method`: Method of the calculation.
        `calc_grad`: Whether calculate the gradient.
    Returns:
        `energy`: Energy of the molecule.
        `gradient` (optional): gradient of the nuclears.
    """
    assert len(atomic_numbers) == len(coordinates)
    at_no = np.array(atomic_numbers)
    coord = np.array(coordinates)
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
    energy = res.get("energy")
    if calc_grad:
        gradient = res.get("gradient")
        return energy, gradient
    else:
        return energy


def mopac_calculation(
    atomic_numbers: Iterable[int],
    coordinates: Iterable[float],
    charge: int = 0,
    multiplicity: int = 1,
    method: str = "PM7",
    calc_grad: bool = False,
):
    """
    Args:
        `atomic_numbers`: Iterable of atomic numbers.
        `coordinates`: Iterable of coordinates.
        `charge`: Charge of the molecule.
        `multiplicity`: Multiplicity of the molecule.
        `method`: Method of the calculation.
        `calc_grad`: Whether calculate the gradient.
    Returns:
        `energy`: Energy of the molecule.
        `gradient` (optional): gradient of the nuclears.
    """
    assert len(atomic_numbers) == len(coordinates)
    at_no = np.array(atomic_numbers)
    coord = np.array(coordinates)
    mopac = MOPAC(
        atoms=at_no,
        coordinates=coord,
        charge=charge,
        multiplicity=multiplicity,
        method=method,
        calc_grad=calc_grad,
    )
    mopac.calculate(clean=True)
    energy = mopac.get_final_heat_of_formation()
    if calc_grad:
        gradient = mopac.gradient()
        return energy, gradient
    else:
        return energy    

