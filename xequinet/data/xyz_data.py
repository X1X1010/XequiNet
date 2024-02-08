from typing import Optional, Callable

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from ..utils import get_default_unit, unit_conversion
from ..utils.qc import ELEMENTS_DICT


"""
                        Special XYZ file
number of atoms         >>>  5
charge & multiplicity   >>>  0 1
atom1 x y z             >>>  C   0.00000000    0.00000000    0.00000000
atom2 x y z             >>>  H   0.00000000    0.00000000    1.06999996
...                     >>>  H  -0.00000000   -1.00880563   -0.35666665
                        >>>  H  -0.87365130    0.50440282   -0.35666665
                        >>>  H   0.87365130    0.50440282   -0.35666665
number of atoms         >>>  3
charge & multiplicity   >>>  0 1
atom1 x y z             >>>  O   0.00000000   -0.00000000   -0.11081188
atom2 x y z             >>>  H   0.00000000   -0.78397589    0.44324751
...                     >>>  H  -0.00000000    0.78397589    0.44324751
"""


class XYZDataset(Dataset):
    """
    Dataset load from .xyz file
    """
    def __init__(
        self,
        xyz_file: str,
        cutoff: float = 5.0,
        transform: Optional[Callable] = None,
    ):
        """
        Args:
            `xyz_file`: Path of .xyz file.
            `embed_basis`: Basis set used for embedding.
            `cutoff`: Cutoff radius for neighbor list.
            `transform`: Transform applied to the data.
        """
        super().__init__()
        self._file = xyz_file
        self._transform = transform
        self.data_list = []
        _, self.len_unit = get_default_unit()
        self._cutoff = cutoff * unit_conversion("Angstrom", self.len_unit)
        self.process()


    def process(self):
        with open(self._file, "r") as f:
            while True:
                line = f.readline().strip()
                if not line: break
                n_atoms = int(line)
                line = f.readline().strip()
                try:
                    charge, multi = list(map(float, line.split()))
                except:
                    charge, multi = 0.0, 1.0
                at_no, coord = [], []
                for _ in range(n_atoms):
                    line = f.readline().strip().split()
                    if line[0].isdigit():
                        at_no.append(int(line[0]))
                    else:
                        at_no.append(ELEMENTS_DICT[line[0]])
                    coord.append(list(map(float, line[1:])))
                at_no = torch.LongTensor(at_no)
                coord = torch.Tensor(coord).to(torch.get_default_dtype())
                coord *= unit_conversion("Angstrom", self.len_unit)
                charge = torch.Tensor([charge]).to(torch.get_default_dtype())
                spin = torch.Tensor([multi - 1]).to(torch.get_default_dtype())
                data = Data(at_no=at_no, pos=coord, charge=charge, spin=spin)
                self.data_list.append(data)
           

    def __len__(self):
        return len(self.data_list)


    def __getitem__(self, idx):
        data = self.data_list[idx]
        if self._transform is not None:
            data = self._transform(data)
        return data
