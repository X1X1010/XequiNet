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
    MatToolkit,
)



class BaseH5Dataset:
    def _init_base(
        self,
        config: NetConfig,
        mode: str = "train",
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ) -> None:
        """
        The base class for the dataset from hdf5 file of XequiNet.
        """
        assert mode in ["train", "valid", "test"]
        self._mode = mode
        self._mem_process = config.mem_process
        # if pbc in `version`, we load the data with pbc
        self._pbc = True if "pbc" in config.version else False
        self._ele = True if "ele" in config.version else False
        self._mat = True if "mat" in config.version else False
        self._cutoff = config.cutoff         # cutoff radius
        self._max_edges = config.max_edges   # maximum edge numbers for each single system
        self.transform = transform          # transform function, e.g., subtract atomic energy
        self.pre_transform = pre_transform  # pre-transform function, e.g., unit conversion
        # if reduce_op is not None, we need to generize a virtual dimension
        self._graph_prop = False if config.reduce_op is None else True

        # set the property dictionary
        self._prop_dict = {'y': config.label_name}
        if config.blabel_name is not None:
            self._prop_dict["base_y"] = config.blabel_name
        if config.force_name is not None:
            self._prop_dict["force"] = config.force_name
            if config.bforce_name is not None:
                self._prop_dict["base_force"] = config.bforce_name
        
        # set attributes for matrix data
        if "mat" in config.version:
            self._prop_dict["possible_elements"] = config.possible_elements
            self._prop_dict["target_basis"] = config.target_basis
            self._prop_dict["require_full_edges"] = config.require_full_edges

        self._prop_unit, self._len_unit = get_default_unit()
        if self._mat:
            self._process_h5 = self.process_matrixh5
        elif self._pbc:
            self._process_h5 = self.process_pbch5
        else:
            self._process_h5 = self.process_h5


    def process_h5(self, f_h5: h5py.File) -> Iterable[Data]:
        """process the hdf5 file of molecular data"""
        # loop over samples
        for mol_name in f_h5[self._mode].keys():
            mol_grp = f_h5[self._mode][mol_name]
            # `torch.LongTensor` is required for some pytorch/pyg functions, but I do not remember which one
            at_no = torch.LongTensor(mol_grp["atomic_numbers"][()])
            # prepare the base `Data`
            mol_data = Data(at_no=at_no)
            # read coordinates
            if "coordinates_A" in mol_grp.keys():
                coords = torch.from_numpy(mol_grp["coordinates_A"][()]).to(torch.get_default_dtype())
                coords *= unit_conversion("Angstrom", self._len_unit)
            elif "coordinates_Bohr" in mol_grp.keys():
                coords = torch.from_numpy(mol_grp["coordinates_Bohr"][()]).to(torch.get_default_dtype())
                coords *= unit_conversion("Bohr", self._len_unit)
            else:
                raise ValueError("Coordinates not found in the hdf5 file.")
            # read charge and spin if needed
            if self._ele:
                charge = float(mol_grp["charge"][()]) if "charge" in mol_grp.keys() else 0.0
                spin = float(mol_grp["multiplicity"][()] - 1) if "multiplicity" in mol_grp.keys() else 0.0
                charge = torch.Tensor([charge]).to(torch.get_default_dtype())
                spin = torch.Tensor([spin]).to(torch.get_default_dtype())
                mol_data.charge = charge; mol_data.spin = spin
            # loop over conformations
            for icfm, coord in enumerate(coords):
                edge_index = radius_graph(coord, r=self._cutoff, max_num_neighbors=self._max_edges)
                data = mol_data.clone()
                data.pos = coord; data.edge_index = edge_index
                # loop over labels
                for p_attr, p_name in self._prop_dict.items():
                    p_val = torch.tensor(mol_grp[p_name][()][icfm])
                    if p_val.dim() == 0:  # add a dimension for 0-d tensor
                        p_val = p_val.unsqueeze(0)
                    # if `y` or `base_y` is a graph property, add a virtual dimension
                    if p_attr in ["y", "base_y"] and self._graph_prop:
                        p_val = p_val.unsqueeze(0)
                    setattr(data, p_attr, p_val)
                yield data


    def process_pbch5(self, f_h5: h5py.File) -> Iterable[Data]:
        """process the hdf5 file of system with periodic boundary condition"""
        # loop over samples
        for pbc_name in f_h5[self._mode].keys():
            pbc_grp = f_h5[self._mode][pbc_name]
            at_no = torch.LongTensor(pbc_grp['atomic_numbers'][()])
            # set periodic boundary condition
            pbc = np.zeros(3, dtype=bool)
            if "pbc" in pbc_grp.keys():
                pbc = ~pbc * pbc_grp["pbc"][()]
            pbc_condition = pbc if isinstance(pbc, bool) else any(pbc)
            # read lattice
            if pbc_condition:
                if "lattice_A" in pbc_grp.keys():
                    lattice = pbc_grp["lattice_A"][()]
                    lattice *= unit_conversion("Angstrom", self._len_unit)
                elif "lattice_Bohr" in pbc_grp.keys():
                    lattice = pbc_grp["lattice_Bohr"][()]
                    lattice *= unit_conversion("Bohr", self._len_unit)
                else:
                    raise ValueError("Lattice not found in the hdf5 file.")
            else:
                lattice = np.zeros((3, 3))
            # prepare the base `Data`
            pbc_data = Data(at_no=at_no, lattice=torch.from_numpy(lattice).to(torch.get_default_dtype()))
            # read coordinates
            if "coordinates_A" in pbc_grp.keys():
                coords = pbc_grp["coordinates_A"][()]
                coords *= unit_conversion("Angstrom", self._len_unit)
            elif "coordinates_Bohr" in pbc_grp.keys():
                coords = pbc_grp["coordinates_Bohr"][()]
                coords *= unit_conversion("Bohr", self._len_unit)
            elif "coordinates_frac" in pbc_grp.keys():
                coords = pbc_grp["coordinates_frac"][()]
                coords = np.einsum("nij, jk -> nik", coords, lattice)
            else:
                raise ValueError("Coordinates not found in the hdf5 file.")
            # read charge and spin if needed
            if self._ele:
                charge = float(pbc_grp["charge"][()]) if "charge" in pbc_grp.keys() else 0.0
                spin = float(pbc_grp["multiplicity"][()] - 1) if "multiplicity" in pbc_grp.keys() else 0.0
                charge = torch.Tensor([charge]).to(torch.get_default_dtype())
                spin = torch.Tensor([spin]).to(torch.get_default_dtype())
                pbc_data.charge = charge; pbc_data.spin = spin
            # loop over conformations
            for icfm, coord in enumerate(coords):
                edge_index, shifts = radius_graph_pbc(
                    positions=coord, pbc=pbc, cell=lattice, cutoff=self._cutoff,
                    max_num_neighbors=self._max_edges,
                )
                # convert `np.ndarray` to `torch.Tensor`
                edge_index = torch.from_numpy(edge_index).long()
                coord = torch.from_numpy(coord).to(torch.get_default_dtype())
                shifts = torch.from_numpy(shifts).to(torch.get_default_dtype())
                data = pbc_data.clone()
                data.pos = coord; data.edge_index = edge_index; data.shifts = shifts
                # loop over labels
                for p_attr, p_name in self._prop_dict.items():
                    p_val = torch.tensor(pbc_grp[p_name][()][icfm])
                    if p_val.dim() == 0:  # add a dimension for 0-d tensor
                        p_val = p_val.unsqueeze(0)
                    # if `y` or `base_y` is a graph property, add a virtual dimension
                    if p_attr in ["y", "base_y"] and self._graph_prop:
                        p_val = p_val.unsqueeze(0)
                    setattr(data, p_attr, p_val)
                yield data


    def process_matrixh5(self, f_h5: h5py.File) -> Iterable[Data]:
        """
        process the hdf5 file of matrix data, e.g., the Fock matrix
        """
        target_basis = self._prop_dict.pop("target_basis")
        elements = self._prop_dict.pop("possible_elements")
        require_full_edges = self._prop_dict.pop("require_full_edges")
        mat_toolkit = MatToolkit(target_basis, elements)
        # loop over samples
        for mol_name in f_h5[self._mode].keys():
            mol_grp = f_h5[self._mode][mol_name]
            # `torch.LongTensor` is required for some pytorch/pyg functions, but I do not remember which one
            at_no = torch.LongTensor(mol_grp["atomic_numbers"][()])
            # prepare the base `Data`
            mol_data = Data(at_no=at_no)
            # read coordinates
            if "coordinates_A" in mol_grp.keys():
                coords = torch.from_numpy(mol_grp["coordinates_A"][()]).to(torch.get_default_dtype())
                coords *= unit_conversion("Angstrom", self._len_unit)
            elif "coordinates_Bohr" in mol_grp.keys():
                coords = torch.from_numpy(mol_grp["coordinates_Bohr"][()]).to(torch.get_default_dtype())
                coords *= unit_conversion("Bohr", self._len_unit)
            else:
                raise ValueError("Coordinates not found in the hdf5 file.")
            if self._ele:
                charge = float(mol_grp["charge"][()]) if "charge" in mol_grp.keys() else 0.0
                spin = float(mol_grp["multiplicity"][()] - 1) if "multiplicity" in mol_grp.keys() else 0.0
                charge = torch.Tensor([charge]).to(torch.get_default_dtype())
                spin = torch.Tensor([spin]).to(torch.get_default_dtype())
                mol_data.charge = charge; mol_data.spin = spin
            for icfm, coord in enumerate(coords):
                edge_index = radius_graph(coord, r=self._cutoff, max_num_neighbors=self._max_edges)
                data = mol_data.clone()
                data.pos = coord; data.edge_index = edge_index
                matrix = torch.from_numpy(
                    mol_grp[self._prop_dict['y']][()][icfm]
                ).to(torch.get_default_dtype())
                if require_full_edges:
                    mat_edge_index = mat_toolkit.get_edge_index_full(at_no)
                    data.edge_index_full = mat_edge_index
                else:
                    mat_edge_index = edge_index
                if self._mode == "test":
                    data.target_matrix = matrix
                node_blocks, edge_blocks = mat_toolkit.get_padded_blocks(at_no, matrix, mat_edge_index)
                node_mask, edge_mask = mat_toolkit.get_mask(at_no, mat_edge_index)
                data.node_label = node_blocks; data.edge_label = edge_blocks
                data.node_mask = node_mask; data.edge_mask = edge_mask
                yield data



class H5Dataset(BaseH5Dataset, Dataset):
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
        super()._init_base(config, mode=mode, transform=transform, pre_transform=pre_transform)
        super().__init__()
        
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
            data_iter = self._process_h5(f_h5=f_h5)
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



class H5MemDataset(BaseH5Dataset, InMemoryDataset):
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
        super()._init_base(config, mode=mode, transform=transform, pre_transform=pre_transform)

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
            data_iter = self._process_h5(f_h5=f_h5)
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



class H5DiskDataset(BaseH5Dataset, DiskDataset):
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
        super()._init_base(config, mode=mode, transform=transform, pre_transform=pre_transform)

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
            data_iter = self._process_h5(f_h5=f_h5)
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
    if new_data.y is not None:
        new_data.y *= unit_conversion(y_unit, prop_unit)
        if hasattr(new_data, "base_y"):
            new_data.base_y *= unit_conversion(by_unit, prop_unit)
    # for force
    if hasattr(new_data, "force"):
        new_data.force *= unit_conversion(force_unit, f"{prop_unit}/{len_unit}")
        if hasattr(new_data, "base_force"):
            new_data.base_force *= unit_conversion(bforce_unit, f"{prop_unit}/{len_unit}")
    # for matrix
    if hasattr(new_data, "node_label"):
        new_data.node_label *= unit_conversion(y_unit, prop_unit)
    if hasattr(new_data, "edge_label"):
        new_data.edge_label *= unit_conversion(y_unit, prop_unit)

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

    if new_data.y is not None:
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
