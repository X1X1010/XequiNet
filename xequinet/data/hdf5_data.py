from typing import Union, Optional, Iterable, Callable
import io
import os

import h5py
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.data import Dataset as DiskDataset
from torch_cluster import radius_graph

from ..utils import (
    unit_conversion, get_default_unit, get_centroid,
)


def process_h5(f_h5: h5py.File, mode: str, cutoff: float, max_edges: int, prop_dict: dict):
    len_unit = get_default_unit()[1]
    # loop over samples
    for mol_name in f_h5[mode].keys():
        mol_grp = f_h5[mode][mol_name]
        at_no = torch.LongTensor(mol_grp["atomic_numbers"][()])
        try:
            coords = torch.Tensor(mol_grp["coordinates_A"][()]).to(torch.get_default_dtype())
            coords *= unit_conversion("Angstrom", len_unit)
        except:
            coords = torch.Tensor(mol_grp["coordinates_bohr"][()]).to(torch.get_default_dtype())
            coords *= unit_conversion("Bohr", len_unit)
        for icfm, coord in enumerate(coords):
            edge_index = radius_graph(coord, r=cutoff, max_num_neighbors=max_edges)
            data = Data(at_no=at_no, pos=coord, edge_index=edge_index)
            for p_attr, p_name in prop_dict.items():
                p_val = torch.tensor(mol_grp[p_name][()][icfm])
                if p_val.dim() != 2:
                    p_val = p_val.view(1, -1)
                setattr(data, p_attr, p_val)
            yield data


class H5Dataset(Dataset):
    """
    Classical torch Dataset for XequiNet.
    """
    def __init__(
        self,
        root: str,
        data_files: Union[str, Iterable[str]],
        data_name: Optional[str] = None,
        mode: str = "train",
        cutoff: float = 5.0,
        max_size: Optional[int] = None,
        max_edges: Optional[int] = None,
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
        self._max_edges = max_edges if max_edges is not None else 100
        self._max_size = max_size if max_size is not None else 1e9
        self._mem_process = mem_process
        self.transform = transform
        self.pre_transform = pre_transform
        self._prop_dict = prop_dict
        self.data_list = []
        _, self.len_unit = get_default_unit()
        self.process()

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
            data_iter = process_h5(f_h5, self._mode, self._cutoff, self._max_edges, self._prop_dict)
            for data in data_iter:
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                self.data_list.append(data)
                ct += 1
                # break if max size is reached
                if ct >= self._max_size:
                    break
            # close the file
            if self._mem_process:
                f_disk.close(); io_mem.close()
            f_h5.close()
            if ct >= self._max_size: break

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data = self.data_list[index]
        if self.transform is not None:
            data = self.transform(data)
        return data


class H5MemDataset(InMemoryDataset):
    """
    Dataset for XequiNet in-memory processing.
    """
    def __init__(
        self,
        root: str,
        data_files: Union[str, Iterable[str]],
        data_name: Optional[str] = None,
        mode: str = "train",
        cutoff: float = 5.0,
        max_size: Optional[int] = None,
        max_edges: Optional[int] = None,
        mem_process: bool = True,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        **prop_dict,
    ):
        assert mode in ["train", "valid", "test"]
        assert {"y", "force", "base_y", "base_force"}.issuperset(prop_dict.keys())
        if isinstance(data_files, str):
            self._raw_files = [data_files]
        elif isinstance(data_files, Iterable):
            self._raw_files = data_files
        else:
            raise TypeError("data_files must be a string or iterable of strings")
        suffix = f"{mode}.pt" if max_size is None else f"{mode}_{max_size}.pt"
        if data_name is None:
            # the processed file are named after the first raw file
            self._processed_file = f"{self._raw_files[0].split('.')[0]}_{suffix}"
        else:
            self._processed_file = f"{data_name}_{suffix}"
        self._mode = mode
        self._cutoff = cutoff
        self._max_edges = max_edges if max_edges is not None else 100
        self._max_size = max_size if max_size is not None else 1e9
        self._mem_process = mem_process
        self._prop_dict = prop_dict
        _, self.len_unit = get_default_unit()
        super().__init__(root, transform=transform, pre_transform=pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
    
    @property
    def raw_file_names(self) -> Iterable[str]:
        return self._raw_files

    @property
    def processed_file_names(self) -> str:
        return self._processed_file

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
            data_iter = process_h5(f_h5, self._mode, self._cutoff, self._max_edges, self._prop_dict)
            for data in data_iter:
                data_list.append(data)
                ct += 1
                # break if max_size is reached
                if ct >= self._max_size:
                    break
            # close the file
            if self._mem_process:
                f_disk.close(); io_mem.close()
            f_h5.close()
            if ct >= self._max_size:
                break
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        # save the processed data
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class H5DiskDataset(DiskDataset):
    """
    Dataset for XequiNet disk processing.
    """
    def __init__(
        self,
        root: str,
        data_files: Union[str, Iterable[str]],
        data_name: Optional[str] = None,
        mode: str = "train",
        cutoff: float = 5.0,
        max_size: Optional[int] = None,
        max_edges: Optional[int] = None,
        mem_process: bool = True,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        **prop_dict,
    ):
        assert mode in ["train", "valid", "test"]
        assert {"y", "force", "base_y", "base_force"}.issuperset(prop_dict.keys())
        if isinstance(data_files, str):
            self._raw_files = [data_files]
        elif isinstance(data_files, Iterable):
            self._raw_files = data_files
        else:
            raise TypeError("data_files must be a string or iterable of strings")
        suffix = f"{mode}" if max_size is None else f"{mode}_{max_size}"
        if data_name is None:
            self._processed_folder = f"{self._raw_files[0].split('.')[0]}_{suffix}"
        else:
            self._processed_folder = f"{data_name}_{suffix}"
        self._mode = mode
        self._cutoff = cutoff
        self._max_edges = max_edges if max_edges is not None else 100
        self._max_size = max_size if max_size is not None else 1e9
        self._mem_process = mem_process
        self._prop_dict = prop_dict
        self._num_data = None
        _, self.len_unit = get_default_unit()
        super().__init__(root, transform=transform, pre_transform=pre_transform)
    
    @property
    def raw_file_names(self) -> Iterable[str]:
        return self._raw_files

    @property
    def processed_file_names(self) -> str:
        return self._processed_folder

    def process(self):
        data_dir = os.path.join(self.processed_dir, self._processed_folder)
        idx = 0  # count of data
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
            data_iter = process_h5(f_h5, self._mode, self._cutoff, self._max_edges, self._prop_dict)
            for data in data_iter:
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                # save the data like `0012/00121234.pt`
                os.makedirs(os.path.join(data_dir, f"{idx // 10000:04d}"), exist_ok=True)
                torch.save(data, os.path.join(data_dir, f"{idx // 10000:04d}", f"{idx:08d}.pt"))
                idx += 1
                # break if max_size is reached
                if idx >= self._max_size:
                    break
            # close hdf5 file
            if self._mem_process:
                f_disk.close(); io_mem.close()
            f_h5.close()
            if idx >= self._max_size:
                break
        self._num_data = idx
    
    def len(self):
        if self._num_data is None:
            data_dir = os.path.join(self.processed_dir, self._processed_folder)
            max_dir = os.path.join(
                data_dir,
                max([d for d in os.listdir(data_dir) if d.isdigit()])
            )
            data_file = max([f for f in os.listdir(max_dir) if f.endswith(".pt")])
            self._num_data = int(data_file.split(".")[0]) + 1
        return self._num_data
        
    def get(self, idx):
        data = torch.load(os.path.join(
            self.processed_dir,
            self._processed_folder,
            f"{idx // 10000:04d}",
            f"{idx:08d}.pt"
        ))
        return data


def data_unit_transform(
        data: Data,
        y_unit: Optional[str] = None,
        by_unit: Optional[str] = None,
        force_unit: Optional[str] = None,
        bforce_unit: Optional[str] = None,
    ) -> Data:
    """
    Create a deep copy of the data and transform the units of the copy.
    """
    new_data = data.clone()
    prop_unit, len_unit = get_default_unit()
    if hasattr(new_data, "y"):
        new_data.y *= unit_conversion(y_unit, prop_unit)
        if hasattr(new_data, "base_y"):
            new_data.base_y *= unit_conversion(by_unit, prop_unit)

    if hasattr(new_data, "force"):
        new_data.force *= unit_conversion(force_unit, f"{prop_unit}/{len_unit}")
        if hasattr(new_data, "base_force"):
            new_data.base_force *= unit_conversion(bforce_unit, f"{prop_unit}/{len_unit}")

    return new_data


def atom_ref_transform(
    data: Data,
    atom_sp: Optional[torch.Tensor] = None,
    batom_sp: Optional[torch.Tensor] = None,
):
    """
    Create a deep copy of the data and subtract the atomic energy.
    """
    new_data = data.clone()

    if hasattr(new_data, "y"):
        ref_sum = atom_sp[new_data.at_no].sum() if atom_sp is not None else 0.0
        new_data.y -= ref_sum
        new_data.y = new_data.y.to(torch.get_default_dtype())
        if hasattr(new_data, "base_y"):
            bref_sum = batom_sp[new_data.at_no].sum() if batom_sp is not None else 0.0
            new_data.base_y -= bref_sum
            new_data.base_y = new_data.base_y.to(torch.get_default_dtype())
    # change the dtype of force by the way
    if hasattr(new_data, "force"):
        new_data.force = new_data.force.to(torch.get_default_dtype())
        if hasattr(new_data, "base_force"):
            new_data.base_force = new_data.base_force.to(torch.get_default_dtype())

    return new_data


def centroid_transform(
    data: Data,
):
    """
    Create a deep copy of the data and subtract the centroid.
    """
    new_data = data.clone()
    centroid = get_centroid(new_data.at_no, new_data.pos)
    new_data.pos -= centroid
    return new_data
