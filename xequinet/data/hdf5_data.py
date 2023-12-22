from typing import Union, Optional, Iterable, Callable
import io
import os

import h5py
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.data import Dataset as DiskDataset

from ..utils import (
    unit_conversion, get_default_unit, get_centroid, get_atomic_energy,
    distributed_zero_first,
    NetConfig,
)
from ..utils.qc import ELEMENTS_LIST


def process_h5(f_h5: h5py.File, mode: str, cutoff: float, max_edges: int, prop_dict: dict):
    from torch_cluster import radius_graph
    len_unit = get_default_unit()[1]
    # loop over samples
    for mol_name in f_h5[mode].keys():
        mol_grp = f_h5[mode][mol_name]
        at_no = torch.LongTensor(mol_grp["atomic_numbers"][()])
        if "coordinates_A" in mol_grp.keys():
            coords = torch.Tensor(mol_grp["coordinates_A"][()]).to(torch.get_default_dtype())
            coords *= unit_conversion("Angstrom", len_unit)
        elif "coordinates_bohr" in mol_grp.keys():
            coords = torch.Tensor(mol_grp["coordinates_bohr"][()]).to(torch.get_default_dtype())
            coords *= unit_conversion("Bohr", len_unit)
        else:
            raise ValueError("Coordinates not found in the hdf5 file.")
        for icfm, coord in enumerate(coords):
            edge_index = radius_graph(coord, r=cutoff, max_num_neighbors=max_edges)
            data = Data(at_no=at_no, pos=coord, edge_index=edge_index)
            for p_attr, p_name in prop_dict.items():
                p_val = torch.tensor(mol_grp[p_name][()][icfm])
                if p_attr == "y" or p_attr == "base_y":
                    if p_val.dim() == 0:
                        p_val = p_val.unsqueeze(0)
                    p_val = p_val.unsqueeze(0)
                setattr(data, p_attr, p_val)
            yield data


def process_pbch5(f_h5: h5py.File, mode: str, cutoff: float, prop_dict: dict):
    import ase
    import ase.neighborlist
    len_unit = get_default_unit()[1]
    # loop over samples
    for pbc_name in f_h5[mode].keys():
        pbc_grp = f_h5[mode][pbc_name]
        at_no = torch.LongTensor(pbc_grp['atomic_numbers'][()])
        # set periodic boundary condition
        if "pbc" in pbc_grp.keys():
            pbc = pbc_grp["pbc"][()]
        else:
            pbc = False
        pbc_condition = pbc if isinstance(pbc, bool) else any(pbc)
        if pbc_condition:
            if "lattice_A" in pbc_grp.keys():
                lattice = torch.Tensor(pbc_grp["lattice_A"][()]).to(torch.get_default_dtype())
                lattice *= unit_conversion("Angstrom", len_unit)
            elif "lattice_bohr" in pbc_grp.keys():
                lattice = torch.Tensor(pbc_grp["lattice_bohr"][()]).to(torch.get_default_dtype())
                lattice *= unit_conversion("Bohr", len_unit)
            else:
                raise ValueError("Lattice not found in the hdf5 file.")
        else:
            lattice = None
        if "coordinates_A" in pbc_grp.keys():
            coords = torch.Tensor(pbc_grp["coordinates_A"][()]).to(torch.get_default_dtype())
            coords *= unit_conversion("Angstrom", len_unit)
        elif "coordinates_bohr" in pbc_grp.keys():
            coords = torch.Tensor(pbc_grp["coordinates_bohr"][()]).to(torch.get_default_dtype())
            coords *= unit_conversion("Bohr", len_unit)
        elif "coordinates_frac" in pbc_grp.keys():
            coords = torch.Tensor(pbc_grp["coordinates_frac"][()]).to(torch.get_default_dtype())
            coords = torch.einsum("nij, kj -> nik", coords, lattice)
        else:
            raise ValueError("Coordinates not found in the hdf5 file.")
        # filter out atoms
        if "atom_filter" in pbc_grp.keys():
            at_filter = torch.BoolTensor(pbc_grp["atom_filter"][()])
        else:
            at_filter = torch.BoolTensor([True for _ in range(at_no.shape[0])])
        # loop over configurations
        for icfm, coord in enumerate(coords):
            atoms = ase.Atoms(symbols=at_no, positions=coord, cell=lattice, pbc=pbc)
            idx_i, idx_j, shifts = ase.neighborlist.neighbor_list("ijS", a=atoms, cutoff=cutoff)
            shifts = torch.Tensor(shifts).to(torch.get_default_dtype())
            if lattice is not None:
                shifts = torch.einsum("ij, kj -> ik", shifts, lattice)
            edge_index = torch.tensor([idx_i, idx_j], dtype=torch.long)
            data = Data(at_no=at_no, pos=coord, edge_index=edge_index, shifts=shifts, at_filter=at_filter)
            for p_attr, p_name in prop_dict.items():
                p_val = torch.tensor(pbc_grp[p_name][()][icfm])
                if p_attr == "y" or p_attr == "base_y":
                    if p_val.dim() == 0:
                        p_val = p_val.unsqueeze(0)
                    p_val = p_val.unsqueeze(0)
                setattr(data, p_attr, p_val)
            yield data


def process_math5(f_h5:h5py.File, mode:str, cutoff:float, max_edges:int, config_dict):
    from torch_cluster import radius_graph
    from ..utils import Mat2GraphLabel, TwoBodyBlockMask 
    len_unit = get_default_unit()[1]
    mat2graph = Mat2GraphLabel(config_dict["irreps_out"], config_dict["possible_elements"], config_dict["basisname"])
    genmask = TwoBodyBlockMask(config_dict["irreps_out"], config_dict["possible_elements"], config_dict["basisname"])
    full_edge_index:bool = config_dict["full_edge_index"] if "full_edge_index" in config_dict.keys() else False
    # loop over samples
    for mol_name in f_h5[mode].keys():
        mol_grp = f_h5[mode][mol_name]
        at_no = torch.LongTensor(mol_grp["atomic_numbers"][()])
        if "coordinates_A" in mol_grp.keys():
            coords = torch.Tensor(mol_grp["coordinates_A"][()]).to(torch.get_default_dtype())
            coords *= unit_conversion("Angstrom", len_unit)
        elif "coordinates_bohr" in mol_grp.keys():
            coords = torch.Tensor(mol_grp["coordinates_bohr"][()]).to(torch.get_default_dtype())
            coords *= unit_conversion("Bohr", len_unit)
        else:
            raise ValueError("Coordinates not found in the hdf5 file.")
        for icfm, coord in enumerate(coords):
            edge_index = radius_graph(coord, r=cutoff, max_num_neighbors=max_edges)
            data = Data(at_no=at_no, pos=coord, edge_index=edge_index)
            matrice_target = torch.from_numpy(
                mol_grp[config_dict["label_name"]][()][icfm].copy()
            ).to(torch.get_default_dtype())
            atm_symbols = [ELEMENTS_LIST[atm] for atm in at_no]
            if full_edge_index:
                mole_node_label, mole_edge_label = mat2graph(data, matrice_target, atm_symbols, edge_index)
                mole_node_mask, mole_edge_mask = genmask(at_no, edge_index)
            else:
                # fully connected 
                mole_node_label, mole_edge_label = mat2graph(data, matrice_target, atm_symbols)
                mole_node_mask, mole_edge_mask = genmask(at_no, data.fc_edge_index)
            
            mole_node_label = mole_node_label.to(torch.get_default_dtype())
            mole_edge_label = mole_edge_label.to(torch.get_default_dtype())
            setattr(data, "node_label", mole_node_label)
            setattr(data, "edge_label", mole_edge_label)
            setattr(data, "onsite_mask", mole_node_mask)
            setattr(data, "offsite_mask", mole_edge_mask)
            yield data


class H5Dataset(Dataset):
    """
    Classical torch Dataset for XequiNet.
    """
    def __init__(
        self,
        root: str,
        data_files: Union[str, Iterable[str]],
        prop_dict: dict,
        **kwargs,
    ):
        super().__init__()
        self._mode: str = kwargs.get("mode", "train")
        self._pbc: bool = kwargs.get("pbc", False)
        self._mat: bool = kwargs.get("matrice", False)
        self._cutoff: float = kwargs.get("cutoff", 5.0)
        self._max_size: int = kwargs.get("max_size", None)
        self._max_edges: int = kwargs.get("max_edges", 100)
        self._mem_process: bool = kwargs.get("mem_process", True)
        self.transform: Callable = kwargs.get("transform", None)
        self.pre_transform: Callable = kwargs.get("pre_transform", None)

        assert self._mode in ["train", "valid", "test"]
        assert not (self._pbc and self._mat), "pbc and Mat mode can not be set simultaneously by now."
        assert {"y", "force", "base_y", "base_force"}.issuperset(prop_dict.keys())
        if isinstance(data_files, str):
            self._raw_paths = [os.path.join(root, "raw", data_files)]
        elif isinstance(data_files, Iterable):
            self._raw_paths = [os.path.join(root, "raw", f) for f in data_files]
        else:
            raise TypeError("data_files must be a string or iterable of strings")
        
        if self._mat:
            self._mat_config_dict = {
                "label_name": prop_dict["y"],
                "target_irreps": kwargs.get("irreps_out"), 
                "possible_elements": kwargs.get("possible_elements"),
                "basisname": kwargs.get("target_basisname"), 
                "follow_edge_index": kwargs.get("follow_edge_index"),
            }
        
        self._max_size = int(1e9) if self._max_size is None else self._max_size
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
            if self._pbc:
                data_iter = process_pbch5(f_h5, self._mode, self._cutoff, self._prop_dict)
            elif self._mat:
                data_iter = process_math5(f_h5, self._mode, self._cutoff, self._max_edges, self._mat_config_dict)
            else:
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
            if ct >= self._max_size:
                break

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
        prop_dict: dict,
        **kwargs,
    ):
        self._mode: str = kwargs.get("mode", "train")
        self._pbc: bool = kwargs.get("pbc", False)
        self._mat: bool = kwargs.get("matrice", False)
        self._cutoff: float = kwargs.get("cutoff", 5.0)
        self._max_size: int = kwargs.get("max_size", None)
        self._max_edges: int = kwargs.get("max_edges", 100)
        self._mem_process: bool = kwargs.get("mem_process", True)
        self.transform: Callable = kwargs.get("transform", None)
        self.pre_transform: Callable = kwargs.get("pre_transform", None)

        assert self._mode in ["train", "valid", "test"]
        assert not (self._pbc and self._mat), "pbc and Mat mode can not be set simultaneously by now."
        assert {"y", "force", "base_y", "base_force"}.issuperset(prop_dict.keys())
        if isinstance(data_files, str):
            self._raw_files = [data_files]
        elif isinstance(data_files, Iterable):
            self._raw_files = data_files
        else:
            raise TypeError("data_files must be a string or iterable of strings")
        
        if self._mat:
            self._mat_config_dict = {
                "label_name": prop_dict["y"],
                "target_irreps": kwargs.get("irreps_out"), 
                "possible_elements": kwargs.get("possible_elements"),
                "basisname": kwargs.get("target_basisname"), 
                "follow_edge_index": kwargs.get("follow_edge_index"),
            }
        
        suffix = f"{self._mode}.pt" if self._max_size is None else f"{self._mode}_{self._max_size}.pt"
        self._max_size = int(1e9) if self._max_size is None else self._max_size
        self._data_name: str = kwargs.get("data_name", f"{self._raw_files[0].split('.')[0]}")
        self._processed_file = f"{self._data_name}_{suffix}"
        self._prop_dict = prop_dict
        _, self.len_unit = get_default_unit()
        super().__init__(root, transform=self.transform, pre_transform=self.pre_transform)
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
            if self._pbc:
                data_iter = process_pbch5(f_h5, self._mode, self._cutoff, self._prop_dict)
            elif self._mat:
                data_iter = process_math5(f_h5, self._mode, self._cutoff, self._max_edges, self._mat_config_dict)
            else:
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
        prop_dict: dict,
        **kwargs,
    ):
        self._mode: str = kwargs.get("mode", "train")
        self._pbc: bool = kwargs.get("pbc", False)
        self._mat: bool = kwargs.get("matrice", False)
        self._cutoff: float = kwargs.get("cutoff", 5.0)
        self._max_size: int = kwargs.get("max_size", None)
        self._max_edges: int = kwargs.get("max_edges", 100)
        self._mem_process: bool = kwargs.get("mem_process", True)
        self.transform: Callable = kwargs.get("transform", None)
        self.pre_transform: Callable = kwargs.get("pre_transform", None)
        
        assert self._mode in ["train", "valid", "test"]
        assert not (self._pbc and self._mat), "pbc and Mat mode can not be set simultaneously by now."
        assert {"y", "force", "base_y", "base_force"}.issuperset(prop_dict.keys())
        if isinstance(data_files, str):
            self._raw_files = [data_files]
        elif isinstance(data_files, Iterable):
            self._raw_files = data_files
        else:
            raise TypeError("data_files must be a string or iterable of strings")
        
        if self._mat:
            self._mat_config_dict = {
                "label_name": prop_dict["y"],
                "target_irreps": kwargs.get("irreps_out"), 
                "possible_elements": kwargs.get("possible_elements"),
                "basisname": kwargs.get("target_basisname"), 
                "follow_edge_index": kwargs.get("follow_edge_index"),
            }
        
        suffix = f"{self._mode}" if self._max_size is None else f"{self._mode}_{self._max_size}"
        self._max_size = int(1e9) if self._max_size is None else self._max_size
        self._data_name: str = kwargs.get("data_name", f"{self._raw_files[0].split('.')[0]}")
        self._processed_folder = f"{self._data_name}_{suffix}"
        self._prop_dict = prop_dict
        self._num_data = None
        _, self.len_unit = get_default_unit()
        super().__init__(root, transform=self.transform, pre_transform=self.pre_transform)
    
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
            if self._pbc:
                data_iter = process_pbch5(f_h5, self._mode, self._cutoff, self._prop_dict)
            elif self._mat:
                data_iter = process_math5(f_h5, self._mode, self._cutoff, self._max_edges, self._mat_config_dict)
            else:
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
    atom_sp: torch.Tensor,
    batom_sp: torch.Tensor,
):
    """
    Create a deep copy of the data and subtract the atomic energy.
    """
    new_data = data.clone()

    if hasattr(new_data, "y"):
        at_no = new_data.at_no
        if hasattr(new_data, "at_filter"):
            at_no = at_no[new_data.at_filter]
        new_data.y -= atom_sp[at_no].sum()
        new_data.y = new_data.y.to(torch.get_default_dtype())
        if hasattr(new_data, "base_y"):
            new_data.base_y -= batom_sp[at_no].sum()
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


def create_dataset(config: NetConfig, mode: str = "train", local_rank: int = None):
    with distributed_zero_first(local_rank):
        # set property read from raw data
        prop_dict = {}
        if config.label_name is not None:
            prop_dict["y"] = config.label_name
        if config.blabel_name is not None:
            prop_dict["base_y"] = config.blabel_name
        if config.force_name is not None:
            prop_dict["force"] = config.force_name
        if config.bforce_name is not None:
            prop_dict["base_force"] = config.bforce_name

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
        kwargs = {
            "mode": mode, "max_size": config.max_mol if mode == "train" else config.vmax_mol,
            "root": config.data_root, "data_files": config.data_files, "data_name": config.processed_name,
            "prop_dict": prop_dict, "pbc": config.pbc, "cutoff": config.cutoff,
            "max_edges": config.max_edges, "mem_process": config.mem_process,
            "transform": transform, "pre_transform": pre_transform,
        }
        if config.dataset_type == "normal":
            dataset = H5Dataset(**kwargs)
        elif config.dataset_type == "memory":
            dataset = H5MemDataset(**kwargs)
        elif config.dataset_type == "disk":
            dataset = H5DiskDataset(**kwargs)
        else:
            raise ValueError(f"Unknown dataset type: {config.dataset_type}")
        
    return dataset


def mat_data_unit_transform(data:Data, label_unit:Optional[str] = None) -> Data:
    """
    Create a deep copy of the data and transform the units of the copy.
    """
    new_data = data.clone()
    prop_unit, _ = get_default_unit()
    if hasattr(new_data, "node_label"):
        new_data.node_label *= unit_conversion(label_unit, prop_unit)
    if hasattr(new_data, "edge_label"):
        new_data.edge_label *= unit_conversion(label_unit, prop_unit)
    return new_data


def create_mat_dataset(config:NetConfig, mode:str="train", local_rank:int=None):
    with distributed_zero_first(local_rank):
        # set property read from raw data
        prop_dict = {}
        if config.label_name is not None:
            prop_dict["y"] = config.label_name
        # set transform function
        pre_transform = lambda data: mat_data_unit_transform(data, config.label_unit)
        transform = None 
        kwargs = {
            "mode": mode, "max_size": config.max_mol if mode == "train" else config.vmax_mol,
            "root": config.data_root, "data_files": config.data_files, "data_name": config.processed_name,
            "prop_dict": prop_dict, "pbc": config.pbc, "matrice": True,
            "target_irreps": config.irreps_out, "possible_elements": config.possible_elements,
            "basisname": config.target_basisname, "full_edge_index": config.full_edge_index,
            "cutoff": config.cutoff, "max_edges": config.max_edges, "mem_process": config.mem_process,
            "transform": transform, "pre_transform": pre_transform,
        }
        if config.dataset_type == "normal":
            dataset = H5Dataset(**kwargs)
        elif config.dataset_type == "memory":
            dataset = H5MemDataset(**kwargs)
        elif config.dataset_type == "disk":
            dataset = H5DiskDataset(**kwargs)
        else:
            raise ValueError(f"Unknown dataset type: {config.dataset_type}")
        
    return dataset
        