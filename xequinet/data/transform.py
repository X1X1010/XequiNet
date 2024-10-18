import abc
import functools
from typing import Dict, Iterable, List, Union

import torch
from torch_cluster import radius_graph

from xequinet.utils import keys, qc

from .datapoint import XequiData
from .radius_graph import radius_graph_pbc

INT_MAX = 2**31 - 1


class Transform(abc.ABC):
    @abc.abstractmethod
    def __call__(self, data: XequiData) -> XequiData:
        raise NotImplementedError


class NeighborTransform(Transform):
    def __init__(self, cutoff: float) -> None:
        self.cutoff = cutoff

    def __call__(self, data: XequiData) -> XequiData:
        device = data.pos.device
        num_graphs = data.num_graphs if hasattr(data, "num_graphs") else 1
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
            edge_index = radius_graph(
                x=data.pos,
                r=self.cutoff,
                batch=batch,
                max_num_neighbors=INT_MAX,
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
    def __init__(self, base_targets: Union[str, List[str]]) -> None:
        self.base_targets = (
            base_targets if isinstance(base_targets, list) else [base_targets]
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


class SequentialTransform(Transform):
    def __init__(self, transforms: Iterable[Transform]) -> None:
        self.transforms = transforms

    def __call__(self, data: XequiData) -> XequiData:
        return functools.reduce(lambda d, t: t(d), self.transforms, data)
