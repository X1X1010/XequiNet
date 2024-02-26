from typing import Optional, Iterable, Callable
import io
import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.data import Dataset as DiskDataset
from torch_cluster import radius_graph

from ..utils import (
    unit_conversion, get_default_unit, get_atomic_energy,
    distributed_zero_first,
    radius_graph_pbc,
    NetConfig,
)


def set_init_attr(
    dataset: Dataset,
    config: NetConfig,
    mode: str = "train",
    transform: Optional[Callable] = None,
    pre_transform: Optional[Callable] = None,
) -> None:
    """
    Set the initial attributes of the dataset.
    """
    assert mode in ["train", "valid", "test"]
    dataset._mode = mode
    dataset._pbc = True if "pbc" in config.version else False
    dataset._cutoff = config.cutoff
    dataset._max_edges = config.max_edges
    dataset._mem_process = config.mem_process
    dataset.transform = transform
    dataset.pre_transform = pre_transform

    dataset._prop_dict = {'y': config.label_name}
    if config.blabel_name is not None:
        dataset._prop_dict['base_y'] = config.blabel_name
    if config.force_name is not None:
        dataset._prop_dict['force'] = config.force_name
        if config.bforce_name is not None:
            dataset._prop_dict['base_force'] = config.bforce_name
   
    if dataset._pbc:
        dataset._process_h5 = process_pbch5
    else:
        dataset._process_h5 = process_h5


def process_h5(f_h5: h5py.File, mode: str, cutoff: float, prop_dict: str, max_edges: int = 100) -> Iterable[Data]:
    """
    process the hdf5 file of molecular data
    """
    len_unit = get_default_unit()[1]
    # loop over samples
    for mol_name in f_h5[mode].keys():
        mol_grp = f_h5[mode][mol_name]
        # `torch.LongTensor` is required for some pytorch/pyg functions, but I do not remember which one
        at_no = torch.LongTensor(mol_grp["atomic_numbers"][()])
        if "coordinates_A" in mol_grp.keys():
            coords = torch.from_numpy(mol_grp["coordinates_A"][()]).to(torch.get_default_dtype())
            coords *= unit_conversion("Angstrom", len_unit)
        elif "coordinates_bohr" in mol_grp.keys():
            coords = torch.from_numpy(mol_grp["coordinates_bohr"][()]).to(torch.get_default_dtype())
            coords *= unit_conversion("Bohr", len_unit)
        else:
            raise ValueError("Coordinates not found in the hdf5 file.")
        charge = float(mol_grp["charge"][()]) if "charge" in mol_grp.keys() else 0.0
        spin = float(mol_grp["multiplicity"][()] - 1) if "multiplicity" in mol_grp.keys() else 0.0
        charge = torch.Tensor([charge]).to(torch.get_default_dtype())
        spin = torch.Tensor([spin]).to(torch.get_default_dtype())
        for icfm, coord in enumerate(coords):
            edge_index = radius_graph(coord, r=cutoff, max_num_neighbors=max_edges)
            data = Data(at_no=at_no, pos=coord, edge_index=edge_index, charge=charge, spin=spin)
            for p_attr, p_name in prop_dict.items():
                if p_name not in mol_grp.keys():
                    continue
                p_val = torch.tensor(mol_grp[p_name][()][icfm])
                if p_val.dim() == 0:
                    p_val = p_val.unsqueeze(0)
                p_val = p_val.unsqueeze(0)
                setattr(data, p_attr, p_val)
            yield data


def process_pbch5(f_h5: h5py.File, mode: str, cutoff: float, prop_dict: dict, max_edges: int = 100) -> Iterable[Data]:
    """
    process the hdf5 file of system with periodic boundary condition
    """
    len_unit = get_default_unit()[1]
    # loop over samples
    for pbc_name in f_h5[mode].keys():
        pbc_grp = f_h5[mode][pbc_name]
        at_no = torch.LongTensor(pbc_grp['atomic_numbers'][()])
        # set periodic boundary condition
        pbc = np.zeros(3, dtype=bool)
        if "pbc" in pbc_grp.keys():
            pbc = ~pbc * pbc_grp["pbc"][()]
        pbc_condition = pbc if isinstance(pbc, bool) else any(pbc)
        if pbc_condition:
            if "lattice_A" in pbc_grp.keys():
                lattice = pbc_grp["lattice_A"][()]
                lattice *= unit_conversion("Angstrom", len_unit)
            elif "lattice_bohr" in pbc_grp.keys():
                lattice = pbc_grp["lattice_bohr"][()]
                lattice *= unit_conversion("Bohr", len_unit)
            else:
                raise ValueError("Lattice not found in the hdf5 file.")
        else:
            lattice = np.zeros((3, 3))
        if "coordinates_A" in pbc_grp.keys():
            coords = pbc_grp["coordinates_A"][()]
            coords *= unit_conversion("Angstrom", len_unit)
        elif "coordinates_bohr" in pbc_grp.keys():
            coords = pbc_grp["coordinates_bohr"][()]
            coords *= unit_conversion("Bohr", len_unit)
        elif "coordinates_frac" in pbc_grp.keys():
            coords = pbc_grp["coordinates_frac"][()]
            coords = np.einsum("nij, jk -> nik", coords, lattice)
        else:
            raise ValueError("Coordinates not found in the hdf5 file.")
        charge = float(pbc_grp["charge"][()]) if "charge" in pbc_grp.keys() else 0.0
        spin = float(pbc_grp["multiplicity"][()] - 1) if "multiplicity" in pbc_grp.keys() else 0.0
        charge = torch.Tensor([charge]).to(torch.get_default_dtype())
        spin = torch.Tensor([spin]).to(torch.get_default_dtype())
        # loop over configurations
        for icfm, coord in enumerate(coords):
            edge_index, shifts = radius_graph_pbc(
                positions=coord, pbc=pbc, cell=lattice, cutoff=cutoff,
                max_num_neighbors=max_edges,
            )
            edge_index = torch.from_numpy(edge_index).long()
            coord = torch.from_numpy(coord).to(torch.get_default_dtype())
            shifts = torch.from_numpy(shifts).to(torch.get_default_dtype())
            data = Data(at_no=at_no, pos=coord, edge_index=edge_index, shifts=shifts, charge=charge, spin=spin)
            for p_attr, p_name in prop_dict.items():
                p_val = torch.tensor(pbc_grp[p_name][()][icfm])
                if p_val.dim() == 0:
                    p_val = p_val.unsqueeze(0)
                p_val = p_val.unsqueeze(0)
                setattr(data, p_attr, p_val)
            yield data


class H5Dataset(Dataset):
    """
    Classical torch Dataset for XequiNet.
    """
    def __init__(
        self,
        config: NetConfig,
        mode: str = "train",
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        set_init_attr(self, config, mode=mode, transform=transform, pre_transform=pre_transform)
        
        root = config.data_root
        data_files = config.data_files
        if isinstance(data_files, str):
            self._raw_paths = [os.path.join(root, "raw", data_files)]
        elif isinstance(data_files, Iterable):
            self._raw_paths = [os.path.join(root, "raw", f) for f in data_files]
        else:
            raise TypeError("data_files must be a string or iterable of strings")

        self.data_list = []
        self.process()

    def process(self) -> None:
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
            data_iter = self._process_h5(
                f_h5=f_h5, mode=self._mode, cutoff=self._cutoff,
                prop_dict=self._prop_dict,
                max_edges=self._max_edges,
            )
            for data in data_iter:
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                self.data_list.append(data)
            # close the file
            if self._mem_process:
                f_disk.close(); io_mem.close()
            f_h5.close()

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, index) -> Data:
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
        config: NetConfig,
        mode: str = "train",
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ) -> None:
        set_init_attr(self, config, mode=mode, transform=transform, pre_transform=pre_transform)

        root = config.data_root
        data_files = config.data_files
        if isinstance(data_files, str):
            self._raw_files = [data_files]
        elif isinstance(data_files, Iterable):
            self._raw_files = data_files
        else:
            raise TypeError("data_files must be a string or iterable of strings")
        data_name = config.processed_name
        self._data_name: str = f"{self._raw_files[0].split('.')[0]}" if data_name is None else data_name
        self._processed_file = f"{self._data_name}_{self._mode}.pt"        
        
        super().__init__(root, transform=self.transform, pre_transform=self.pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
    
    @property
    def raw_file_names(self) -> Iterable[str]:
        return self._raw_files

    @property
    def processed_file_names(self) -> str:
        return self._processed_file

    def process(self) -> None:
        data_list = []
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
            data_iter = self._process_h5(
                f_h5=f_h5, mode=self._mode, cutoff=self._cutoff,
                max_edges=self._max_edges,
                prop_dict=self._prop_dict,
                virtual_dim=self._virtual_dim
            )
            for data in data_iter:
                data_list.append(data)
            # close the file
            if self._mem_process:
                f_disk.close(); io_mem.close()
            f_h5.close()
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
        config: NetConfig,
        mode: str = "train",
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ) -> None:
        set_init_attr(self, config, mode=mode, transform=transform, pre_transform=pre_transform)

        root = config.data_root
        data_files = config.data_files
        if isinstance(data_files, str):
            self._raw_files = [data_files]
        elif isinstance(data_files, Iterable):
            self._raw_files = data_files
        else:
            raise TypeError("data_files must be a string or iterable of strings")
        
        data_name = config.processed_name
        self._data_name: str = f"{self._raw_files[0].split('.')[0]}" if data_name is None else data_name
        self._processed_folder = f"{self._data_name}_{self._mode}"

        self._num_data = None
        super().__init__(root, transform=self.transform, pre_transform=self.pre_transform)
    
    @property
    def raw_file_names(self) -> Iterable[str]:
        return self._raw_files

    @property
    def processed_file_names(self) -> str:
        return self._processed_folder

    def process(self) -> None:
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
            data_iter = self._process_h5(
                f_h5=f_h5, mode=self._mode, cutoff=self._cutoff,
                max_edges=self._max_edges,
                prop_dict=self._prop_dict,
                virtual_dim=self._virtual_dim
            )
            for data in data_iter:
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                # save the data like `0012/00121234.pt`
                os.makedirs(os.path.join(data_dir, f"{idx // 10000:04d}"), exist_ok=True)
                torch.save(data, os.path.join(data_dir, f"{idx // 10000:04d}", f"{idx:08d}.pt"))
                idx += 1
            # close hdf5 file
            if self._mem_process:
                f_disk.close(); io_mem.close()
            f_h5.close()
        self._num_data = idx
    
    def len(self) -> int:
        if self._num_data is None:
            data_dir = os.path.join(self.processed_dir, self._processed_folder)
            max_dir = os.path.join(
                data_dir,
                max([d for d in os.listdir(data_dir) if d.isdigit()])
            )
            data_file = max([f for f in os.listdir(max_dir) if f.endswith(".pt")])
            self._num_data = int(data_file.split(".")[0]) + 1
        return self._num_data
        
    def get(self, idx) -> Data:
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
    atom_sp: torch.Tensor,
    batom_sp: torch.Tensor,
) -> Data:
    """
    Create a deep copy of the data and subtract the atomic energy.
    """
    new_data = data.clone()

    if hasattr(new_data, "y"):
        at_no = new_data.at_no
        if atom_sp is not None:
            new_data.y -= atom_sp[at_no].sum()
        new_data.y = new_data.y.to(torch.get_default_dtype())
        if hasattr(new_data, "base_y"):
            if batom_sp is not None:
                new_data.base_y -= batom_sp[at_no].sum()
            new_data.base_y = new_data.base_y.to(torch.get_default_dtype())
    # change the dtype of force by the way
    if hasattr(new_data, "force"):
        new_data.force = new_data.force.to(torch.get_default_dtype())
        if hasattr(new_data, "base_force"):
            new_data.base_force = new_data.base_force.to(torch.get_default_dtype())

    return new_data


def create_dataset(config: NetConfig, mode: str = "train", local_rank: int = None) -> Dataset:
    with distributed_zero_first(local_rank):
        # set transform function
        pre_transform = lambda data: data_unit_transform(
            data=data, y_unit=config.label_unit, by_unit=config.blabel_unit,
            force_unit=config.force_unit, bforce_unit=config.bforce_unit,
        )
        atom_sp = get_atomic_energy(config.atom_ref)
        batom_sp = get_atomic_energy(config.batom_ref)
        transform = lambda data: atom_ref_transform(
            data=data,
            atom_sp=atom_sp,
            batom_sp=batom_sp,
        )
        if config.dataset_type == "normal":
            dataset = H5Dataset(config, mode=mode, pre_transform=pre_transform, transform=transform)
        elif config.dataset_type == "memory":
            dataset = H5MemDataset(config, mode=mode, pre_transform=pre_transform, transform=transform)
        elif config.dataset_type == "disk":
            dataset = H5DiskDataset(config, mode=mode, pre_transform=pre_transform, transform=transform)
        else:
            raise ValueError(f"Unknown dataset type: {config.dataset_type}")
        
    return dataset
