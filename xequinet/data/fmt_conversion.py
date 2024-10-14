import ase
from pyscf import gto

from .datapoint import XequiData


def datapoint_from_ase(atoms: ase.Atoms) -> XequiData:
    pass

def datapoint_from_pyscf(mole: gto.Mole) -> XequiData:
    pass

def datapoint_to_ase(datapoint: XequiData) -> ase.Atoms:
    pass

def datapoint_to_pyscf(datapoint: XequiData) -> gto.Mole:
    pass