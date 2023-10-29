from typing import Union, Optional, Iterable, Callable
import io
import os

import h5py
import torch
import ase
import ase.neighborlist
from torch.utils.data import Dataset
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.data import Dataset as DiskDataset

from ..utils import (
    unit_conversion, get_default_unit
)


def process_pbch5(f_h5: h5py.File, mode: str, cutoff: float, prop_dict: dict):
    len_unit = get_default_unit()[1]
    # loop over samples
    for pbc_name in f_h5[mode].keys():
        pbc_grp = f_h5[mode][pbc_name]
        at_no = torch.LongTensor(pbc_grp['atomic_numbers'][()])
        try:
            coords = torch.Tensor(pbc_grp["coordinates_A"][()]).to(torch.get_default_dtype())
            lattice = torch.Tensor(pbc_grp["lattice_A"][()]).to(torch.get_default_dtype())
            coords *= unit_conversion("Angstrom", len_unit)
            lattice *= unit_conversion("Angstrom", len_unit)
        except:
            coords = torch.Tensor(pbc_grp["coordinates_bohr"][()]).to(torch.get_default_dtype())
            lattice = torch.Tensor(pbc_grp["lattice_bohr"][()]).to(torch.get_default_dtype())
            coords *= unit_conversion("Bohr", len_unit)
            lattice *= unit_conversion("Bohr", len_unit)
        # filter out atoms
        if "atom_filter" in pbc_grp.keys():
            at_filter = torch.BoolTensor(pbc_grp["atom_filter"][()])
        else:
            at_filter = torch.BoolTensor([True for _ in range(at_no.shape[0])])
        # set periodic boundary condition
        if "pbc" in pbc_grp.keys():
            pbc = pbc_grp["pbc"][()]
        else:
            pbc = False
        # loop over configurations
        for icfm, coord in enumerate(coords):
            atoms = ase.Atoms(symbols=at_no, positions=coord, cell=lattice, pbc=pbc)
            idx_i, idx_j, shifts = ase.neighborlist.neighbor_list("ijS", a=atoms, cutoff=cutoff)
            shifts *= lattice
            edge_index = torch.tensor([idx_i, idx_j], dtype=torch.long)
            data = Data(at_no=at_no, pos=coord, edge_index=edge_index, shifts=shifts, at_filter=at_filter)
            for p_attr, p_name in prop_dict.items():
                p_val = torch.Tensor(pbc_grp[p_name][()][icfm])
                if p_attr == "y" or p_attr == "base_y":
                    p_val = p_val.unsqueeze(0)
                setattr(data, p_attr, p_val)
            yield data


class PBCDataset(Dataset):
    def __init__(
        self,
        root: str,
        data_files: Union[str, Iterable[str]],
        data_name: Optional[str] = None,
        mode: str = "train",
        cutoff: float = 5.0,
        max_size: Optional[int] = None,
        mem_process: bool = True,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        **prop_dict,
    ):
        super().__init__()
        assert mode in ["train", "valid", "test"]
        assert {"y", "force", "base_y", "base_force"}.issuperset(prop_dict.keys())
        if isinstance(data_files, str):
            self._raw_paths = [os.path.join(root, "raw", data_files)]
        elif isinstance(data_files, Iterable):
            self._raw_paths = [os.path.join(root, "raw", f) for f in data_files]
        else:
            raise TypeError("data_files must be a string or iterable of strings")
        self._mode = mode
        self._cutoff = cutoff
        self._max_size = max_size if max_size is not None else 1e9
        self._mem_process = mem_process
        self.transform = transform
        self.pre_transform = pre_transform
        self._prop_dict = prop_dict
        self.data_list = []
        _, self.len_unit = get_default_unit()
        self.process()
    
    def process(self):
        ct = 0  # count number of samples
        for raw_path in self._raw_paths:
            # read by memory io-buffer or by disk directly
            if self._mem_process:
                f_disk = open(raw_path, 'rb')
                io_mem = io.BytesIO(f_disk.read())
                f_h5 = h5py.File(io_mem, 'r')
            else:
                f_h5 = h5py.File(raw_path, 'r')
            # skip if current hdf5 file does not contain the `self._mode`
            if self._mode not in f_h5.keys():
                if self._mem_process:
                    f_disk.close(); io_mem.close()
                f_h5.close()
                continue
            date_iter = process_pbch5(f_h5, self._mode, self._cutoff, self._prop_dict)
            for data in date_iter:
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                self.data_list.append(data)
                ct += 1
                if ct >= self._max_size:
                    break
            # close the file
            if self._mem_process:
                f_disk.close(); io_mem.close()
            f_h5.close()
            if ct >= self._max_size: break

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        data = self.data_list[idx]
        if self.transform is not None:
            data = self.transform(data)
        return data



if __name__ == "__main__":
    si_lattice = torch.tensor([
        [1.2, 0., 0.],
        [0., 1.2, 0.],
        [0., 0., 1.2]
    ])
    si_coords = torch.tensor([
        [0. , 0. , 0. ],
        [0. , 0. , 0.6],
    ])
    si_types = ['Cu', 'Cu']

    si = ase.Atoms(symbols=si_types, positions=si_coords, cell=si_lattice, pbc=[False, False, False])
    i, j, d, D, S = ase.neighborlist.neighbor_list("ijdDS", a=si, cutoff=1.21, max_nbins=6)
    print(i, j, d, D, S)
