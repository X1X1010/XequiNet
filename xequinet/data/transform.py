import abc
import functools
from typing import Dict, Iterable, Union

import torch
from torch_cluster import radius_graph

from xequinet import keys
from xequinet.utils import qc

from .datapoint import XequiData
from .radius_graph import radius_graph_pbc


class Transform(abc.ABC):
    @abc.abstractmethod
    def __call__(self, data: XequiData) -> XequiData:
        raise NotImplementedError


class NeighborTransform(Transform):
    def __init__(self, cutoff: float) -> None:
        self.cutoff = cutoff

    def __call__(self, data: XequiData) -> XequiData:
        device = data.pos.device
        num_graphs = data.num_graphs if hasattr(data, keys.NUM_GRAPHS) else 1
        if num_graphs > 1:
            assert hasattr(data, keys.BATCH)
            n_nodes_per_graph = data.ptr[1:] - data.ptr[:-1]  # [num_graphs]
            batch = data.batch
        else:
            n_nodes_per_graph = torch.tensor([data.pos.shape[0]], device=device)  # [1]
            batch = None

        has_pbc = hasattr(data, keys.PBC) and data.pbc.any()
        has_cell = hasattr(data, keys.CELL)

        if has_pbc and has_cell:

            if data.edge_index is not None and data.cell_offsets is not None:
                return data
            edge_index, cell_offsets = radius_graph_pbc(
                pos=data.pos,
                n_nodes_per_graph=n_nodes_per_graph,
                cell=data.cell,
                pbc=data.pbc,
                cutoff=self.cutoff,
            )
            data.edge_index = edge_index
            data.cell_offsets = cell_offsets

        elif not has_pbc and not has_cell:

            if data.edge_index is not None:
                return data
            max_possible_neighbors = torch.sum(n_nodes_per_graph**2).item()
            edge_index = radius_graph(
                x=data.pos,
                r=self.cutoff,
                batch=batch,
                max_num_neighbors=max_possible_neighbors,
                batch_size=num_graphs,
            )
            data.edge_index = edge_index
        else:
            raise ValueError("PBC and cell must be both defined or both undefined.")

        return data


class DataTypeTransform(Transform):
    def __init__(
        self,
        dtype: Union[str, torch.dtype],
    ) -> None:
        if isinstance(dtype, str):
            name_to_dtype = {
                "float16": torch.float16,
                "float32": torch.float32,
                "float64": torch.float64,
            }
            assert dtype in name_to_dtype, f"Invalid dtype {dtype}"
            self.dtype = name_to_dtype[dtype]
        else:
            self.dtype = dtype

    def _is_float_dtype(self, t: torch.Tensor) -> bool:
        float_types = {torch.float16, torch.float32, torch.float64}
        return t.dtype in float_types

    def __call__(self, data: XequiData) -> XequiData:
        for k in data.keys():
            if self._is_float_dtype(data[k]):
                data[k] = data[k].to(self.dtype)

        return data


class UnitTransform(Transform):
    def __init__(self, data_units: Dict[str, str]) -> None:
        self.default_units = qc.get_default_units()
        for k, v in data_units.items():
            assert k in self.default_units, f"Invalid property {k}"
            assert qc.check_unit(v), f"Invalid unit {v} for property {k}"
        self.data_units = data_units

    def __call__(self, data: XequiData) -> XequiData:
        new_data = data.clone()
        for prop, unit in self.data_units.items():
            if prop not in new_data:
                continue
            new_data[prop] *= qc.unit_conversion(unit, self.default_units[prop])
        return new_data


class DeltaTransform(Transform):
    def __init__(self, base_targets: Union[str, Iterable[str]]) -> None:
        self.base_targets = (
            base_targets if isinstance(base_targets, Iterable) else [base_targets]
        )
        self.targets = [keys.BASE_PROPERTIES[t] for t in self.base_targets]

    def __call__(self, data: XequiData) -> XequiData:
        new_data = data.clone()
        for t, bt in zip(self.targets, self.base_targets):
            assert t in new_data, f"Invalid target {t}"
            assert bt in new_data, f"Invalid base target {bt}"
            new_data[t] -= new_data[bt]
            del new_data[bt]
        return new_data


class SVDFrameTransform(Transform):
    """
    Currently only works for vector and atomic vector properties. (e.g. forces, dipoles, etc.)
    For tensorial properties, it may be a little more complicated, and we will add if we need it.
    """

    def __init__(
        self,
        vector_targets: Union[str, Iterable[str]],
        atomic_vector_targets: Union[str, Iterable[str]],
    ) -> None:
        self.vector_targets = (
            vector_targets if isinstance(vector_targets, Iterable) else [vector_targets]
        )
        self.atomic_vector_targets = (
            atomic_vector_targets
            if isinstance(atomic_vector_targets, Iterable)
            else [atomic_vector_targets]
        )

    def __call__(self, data: XequiData) -> XequiData:
        num_graphs = data.num_graphs if hasattr(data, keys.NUM_GRAPHS) else 1
        batch = (
            data.batch
            if hasattr(data, keys.BATCH)
            else torch.zeros(data.pos.shape[0], device=data.pos.device)
        )
        new_pos = []
        new_vec = {k: [] for k in self.vector_targets}
        new_atom_vec = {k: [] for k in self.atomic_vector_targets}
        for i in range(num_graphs):
            pos_batch = data.pos[batch == i]
            # centering
            pos_batch -= pos_batch.mean(dim=0)
            # rotate the structure into its SVD frame
            # only can do this for > 2 atoms
            if pos_batch.shape[0] > 2:
                u, s, v = torch.svd(pos_batch)
                rotated_pos_batch = pos_batch @ v
            else:
                rotated_pos_batch = pos_batch
                v = torch.eye(3, device=pos_batch.device)
            new_pos.append(rotated_pos_batch)
            for k in self.vector_targets:
                new_vec[k].append(data[k][i] @ v)
            for k in self.atomic_vector_targets:
                new_atom_vec[k].append(data[k][batch == i] @ v)
        data.pos = torch.cat(new_pos, dim=0)
        for k, v in (new_vec | new_atom_vec).items():
            data[k] = torch.cat(v, dim=0)
        return data


class SequentialTransform(Transform):
    def __init__(self, transforms: Iterable[Transform]) -> None:
        self.transforms = transforms

    def __call__(self, data: XequiData) -> XequiData:
        return functools.reduce(lambda d, t: t(d), self.transforms, data)
