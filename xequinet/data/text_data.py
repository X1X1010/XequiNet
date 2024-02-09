from typing import Optional, Callable

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from ..utils import get_default_unit, unit_conversion

from ase.io import read as ase_read


"""
                  Special XYZ file
number of atoms   >>>  5
informatrion      >>>  charge=0 multiplicity=1
atom1 x y z       >>>  C   0.00000000    0.00000000    0.00000000
atom2 x y z       >>>  H   0.00000000    0.00000000    1.06999996
...               >>>  H  -0.00000000   -1.00880563   -0.35666665
                  >>>  H  -0.87365130    0.50440282   -0.35666665
                  >>>  H   0.87365130    0.50440282   -0.35666665
number of atoms   >>>  3
informatrion      >>>  charge=0 spin=0
atom1 x y z       >>>  O   0.00000000   -0.00000000   -0.11081188
atom2 x y z       >>>  H   0.00000000   -0.78397589    0.44324751
...               >>>  H  -0.00000000    0.78397589    0.44324751


                  Extended XYZ file
number of atoms   >>>  192
informatrion      >>>  pbc="T T T" Lattice="23.46511000 0.00000000 0.00000000   -0.00000100 23.46511000 0.00000000    -0.00000100 -0.00000100 23.46511000" Properties=species:S:1:pos:R:3
atom1 x y z       >>>  O  11.72590000   14.59020000   25.33440000
atom2 x y z       >>>  H  12.69400000   16.13880000   24.72010000
atom3 x y z       >>>  H   9.70021000   15.03790000   25.76530000
atom4 x y z       >>>  O  10.68010000    3.41217000    4.43292000
...
"""


class TextDataset(Dataset):
    """
    Dataset load from file like .xyz, .pdb, .sdf, etc.
    """
    def __init__(
        self,
        file: str,
        transform: Optional[Callable] = None,
    ):
        """
        Args:
            `file`: Path of file.
            `transform`: Transform applied to the data.
        """
        super().__init__()
        self._file = file
        self._transform = transform
        self.data_list = []
        _, self.len_unit = get_default_unit()
        self.process()


    def process(self):
        atoms_list = ase_read(self._file, index=":")
        for atoms in atoms_list:
            at_no = torch.from_numpy(atoms.get_atomic_numbers()).to(torch.long)
            coord = torch.from_numpy(atoms.get_positions()).to(torch.get_default_dtype())
            coord *= unit_conversion("Angstrom", self.len_unit)
            info = atoms.info
            charge = torch.Tensor([info.get("charge", 0.0)]).to(torch.get_default_dtype())
            if "multiplicity" in info:
                spin = torch.Tensor([info["multiplicity"] - 1]).to(torch.get_default_dtype())
            else:
                spin = torch.Tensor([info.get("spin", 0.0)]).to(torch.get_default_dtype())
            data = Data(at_no=at_no, pos=coord, charge=charge, spin=spin)
            pbc = atoms.get_pbc()
            if pbc.any():
                data.pbc = torch.from_numpy(pbc).to(torch.bool)
                data.lattice = torch.from_numpy(atoms.get_cell()).to(torch.get_default_dtype())
            self.data_list.append(data)


    def __len__(self):
        return len(self.data_list)


    def __getitem__(self, idx):
        data = self.data_list[idx]
        if self._transform is not None:
            data = self._transform(data)
        return data
