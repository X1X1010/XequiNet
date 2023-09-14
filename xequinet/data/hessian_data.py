from typing import Union, Optional, Iterable, Callable
import io
import os

import h5py
import torch
from torch_geometric.data import Data
from torch_cluster import radius_graph

from .hdf5_data import H5Dataset, H5MemDataset, H5DiskDataset
from ..utils import (
    unit_conversion,
    get_atomic_energy, get_default_unit, get_centroid,
)


class HessianDataset(H5Dataset):
    """
    Classical torch Dataset for Hessian.
    """
    def __init__(
        self,
        root: str,
        data_files: Union[str, Iterable[str]],
        mode: str = "train",
        cutoff: float = 5.0,
        max_size: Optional[int] = None,
        mem_process: bool = True,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        **prop_dict,
    ):
        super().__init__(
            root=root,
            data_files=data_files,
            mode=mode,
            cutoff=cutoff,
            max_size=max_size,
            mem_process=mem_process,
            transform=transform,
            pre_transform=pre_transform,
            **prop_dict,
        )
    
    def process(self):
        ct = 0  # count of data
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
            # loop over samples
            for mol_name in f_h5[self._mode].keys():
                mol_grp = f_h5[self._mode][mol_name]
                at_no = torch.LongTensor(mol_grp["atomic_numbers"][()])
                try:
                    coords = torch.Tensor(mol_grp["coordinates_A"][()]).to(torch.get_default_dtype())
                    coords *= unit_conversion("angstrom", self.len_unit)
                except:
                    coords = torch.Tensor(mol_grp["coordinates_bohr"][()]).to(torch.get_default_dtype())
                    coords *= unit_conversion("bohr", self.len_unit)
                all_range = torch.arange(0, at_no.size(0), dtype=torch.long)
                fc_edge_index = torch.stack([
                    all_range.repeat_interleave(at_no.size(0)),
                    all_range.repeat(at_no.size(0))
                ])
                for icfm, coord in enumerate(coords):
                    edge_index = radius_graph(coord, self._cutoff)
                    data = Data(at_no=at_no, pos=coord, edge_index=edge_index, fc_edge_index=fc_edge_index)
                    for p_attr, p_name in self._prop_dict.items():
                        p_val = torch.tensor(mol_grp[p_name][()][icfm])
                        p_val = p_val.view(-1, 3, 3)
                        setattr(data, p_attr, p_val)
                    if self.pre_transform is not None:
                        data = self.pre_transform(data)
                    self.data_list.append(data)
                    ct += 1
                    # break if max size is reached
                    if ct >= self._max_size: break
                if ct >= self._max_size: break
            # close the file
            if self._mem_process:
                f_disk.close(); io_mem.close()
            f_h5.close()
            if ct >= self._max_size: break


class HessianMemDataset(H5MemDataset):
    """
    Dataset for Hessian matrix in-memory processing.
    """
    def __init__(
        self,
        root: str,
        data_files: Union[str, Iterable[str]],
        mode: str = 'train',
        cutoff: float = 5.0,
        max_size: Optional[int] = None,
        mem_process: bool = True,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        **prop_dict,
    ):
        super().__init__(
            root=root,
            data_files=data_files,
            mode=mode,
            cutoff=cutoff,
            max_size=max_size,
            mem_process=mem_process,
            transform=transform,
            pre_transform=pre_transform,
            **prop_dict,
        )

    def process(self):
        data_list = []
        ct = 0  # count of data
        for raw_path in self.raw_paths:
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
            # loop over samples
            for mol_name in f_h5[self._mode].keys():
                mol_grp = f_h5[self._mode][mol_name]
                at_no = torch.LongTensor(mol_grp["atomic_numbers"][()])
                try:
                    coords = torch.Tensor(mol_grp["coordinates_A"][()]).to(torch.get_default_dtype())
                    coords *= unit_conversion("Angstrom", self.len_unit)
                except:
                    coords = torch.Tensor(mol_grp["coordinates_bohr"][()]).to(torch.get_default_dtype())
                    coords *= unit_conversion("Bohr", self.len_unit)
                all_range = torch.arange(0, at_no.size(0), dtype=torch.long)
                fc_edge_index = torch.stack([
                    all_range.repeat_interleave(at_no.size(0)),
                    all_range.repeat(at_no.size(0))
                ])
                for icfm, coord in enumerate(coords):
                    edge_index = radius_graph(coord, self._cutoff)
                    data = Data(at_no=at_no, pos=coord, edge_index=edge_index, fc_edge_index=fc_edge_index)
                    for p_attr, p_name in self._prop_dict.items():
                        p_val = torch.tensor(mol_grp[p_name][()][icfm])
                        p_val = p_val.view(-1, 3, 3)
                        setattr(data, p_attr, p_val)
                    data_list.append(data)
                    ct += 1
                    # break if max_size is reached
                    if ct >= self._max_size: break
                if ct >= self._max_size: break
            # close the file
            if self._mem_process:
                f_disk.close(); io_mem.close()
            f_h5.close()
            if ct >= self._max_size: break
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        # save the processed data
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class HessianDiskDataset(H5DiskDataset):
    """
    Dataset for Hessian matrix disk processing.
    """
    def __init__(
        self,
        root: str,
        data_files: Union[str, Iterable[str]],
        mode: str = "train",
        cutoff: float = 5.0,
        max_size: Optional[int] = None,
        mem_process: bool = True,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        **prop_dict,
    ):
        super().__init__(
            root=root,
            data_files=data_files,
            mode=mode,
            cutoff=cutoff,
            max_size=max_size,
            mem_process=mem_process,
            transform=transform,
            pre_transform=pre_transform,
            **prop_dict,
        )

    def process(self):
        idx = 0
        data_dir = os.path.join(self.processed_dir, self._processed_folder)
        for raw_path in self.raw_paths:
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
            # loop over samples
            for mol_name in f_h5[self._mode].keys():
                mol_grp = f_h5[self._mode][mol_name]
                at_no = torch.LongTensor(mol_grp["atomic_numbers"][()])
                try:
                    coords = torch.Tensor(mol_grp["coordinates_A"][()]).to(torch.get_default_dtype())
                    coords *= unit_conversion("angstrom", self.len_unit)
                except:
                    coords = torch.Tensor(mol_grp["coordinates_bohr"][()]).to(torch.get_default_dtype())
                    coords *= unit_conversion("bohr", self.len_unit)
                all_range = torch.arange(0, at_no.size(0), dtype=torch.long)
                fc_edge_index = torch.stack([
                    all_range.repeat_interleave(at_no.size(0)),
                    all_range.repeat(at_no.size(0))
                ])
                for icfm, coord in enumerate(coords):
                    edge_index = radius_graph(coord, self._cutoff)
                    data = Data(at_no=at_no, pos=coord, edge_index=edge_index, fc_edge_index=fc_edge_index)
                    for p_attr, p_name in self._prop_dict.items():
                        p_val = torch.tensor(mol_grp[p_name][()][icfm])
                        p_val = p_val.view(-1, 3, 3)
                        setattr(data, p_attr, p_val)
                    if self.pre_transform is not None:
                        data = self.pre_transform(data)
                    # save the data like `0012/00121234.pt`
                    os.makedirs(os.path.join(data_dir, f"{idx // 10000:04d}"), exist_ok=True)
                    torch.save(data, os.path.join(data_dir, f"{idx // 10000:04d}", f"{idx:08d}.pt"))
                    idx += 1
                    # break if max_size is reached
                    if idx >= self._max_size: break
                if idx >= self._max_size: break
            # close hdf5 file
            if self._mem_process:
                f_disk.close(); io_mem.close()
            f_h5.close()
            if idx >= self._max_size: break
        self._num_data = idx
