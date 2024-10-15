from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn

from xequinet.utils import keys

from .electronic import HierarchicalCutoff
from .output import resolve_output
from .xpainn import XEmbedding, XPainnMessage, XPainnUpdate


def compute_edge_data(
    data: Dict[str, torch.Tensor],
    compute_forces: bool = True,
    compute_virial: bool = False,
) -> Dict[str, torch.Tensor]:
    """Preprocess edge data"""
    pos = data[keys.POSITIONS]
    edge_index = data[keys.EDGE_INDEX]

    single_graph = False
    if keys.BATCH not in data:
        data[keys.BATCH] = torch.zeros(
            pos.shape[0], dtype=torch.long, device=pos.device
        )
        data[keys.BATCH_PTR] = torch.tensor(
            [0, pos.shape[0]], dtype=torch.long, device=pos.device
        )
        single_graph = True
    elif data[keys.BATCH].max() == 0:
        single_graph = True

    batch = data[keys.BATCH]
    n_graphs = data[keys.BATCH_PTR].numel() - 1

    has_cell = keys.CELL in data
    if has_cell:
        cell = data[keys.CELL]
    else:
        cell = torch.empty((0, 3, 3), device=pos.device)

    if compute_forces:
        pos.requires_grad_()

    strain = torch.zeros(
        (n_graphs, 3, 3),
        dtype=pos.dtype,
        device=pos.device,
    )

    if compute_virial:
        strain.requires_grad_()
        symm_strain = 0.5 * (strain + strain.transpose(1, 2))
        # position
        expanded_strain = torch.index_select(symm_strain, 0, batch)
        pos = pos + torch.bmm(pos.unsqueeze(1), expanded_strain).squeeze(1)
        # cell
        if has_cell:
            cell = cell + torch.bmm(cell, symm_strain)

    # vectors are pointing from center to neighbor
    center_idx, neighbor_idx = (
        edge_index[keys.CENTER_IDX],
        edge_index[keys.NEIGHBOR_IDX],
    )
    vectors = torch.index_select(pos, 0, center_idx) - torch.index_select(
        pos, 0, neighbor_idx
    )

    # offsets for periodic boundary conditions
    if has_cell:
        cell_offsets = data[keys.CELL_OFFSETS]
        if single_graph:
            shifts = torch.einsum("ni, ij -> nj", cell_offsets, cell.squeeze(0))
            vectors = vectors - shifts
        else:
            batch_neighbor = torch.index_select(batch, 0, neighbor_idx)
            cell_batch = torch.index_select(cell, 0, batch_neighbor)
            shifts = torch.einsum("ni, nij -> nj", cell_offsets, cell_batch)
            vectors = vectors - shifts

    # compute distances
    dist = torch.linalg.norm(vectors, dim=-1)

    data.update(
        {
            keys.EDGE_LENGTH: dist,
            keys.EDGE_VECTOR: vectors,
            keys.STRAIN: strain,
        }
    )
    return data


class BaseModel(nn.Module):
    
    cutoff_radius: float

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        compute_forces: bool = True,
        compute_virial: bool = False,
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError


class XPaiNN(BaseModel):
    """
    eXtended PaiNN.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__()
        node_dim: int = kwargs.get("node_dim", 128)
        node_irreps: str = kwargs.get("node_irreps", "128x0e + 64x1o + 32x2e")
        embed_basis: str = kwargs.get("embed_basis", "gfn2-xtb")
        aux_basis: str = kwargs.get("aux_basis", "aux56")
        num_basis: int = kwargs.get("num_basis", 20)
        rbf_kernel: str = kwargs.get("rbf_kernel", "bessel")
        cutoff: float = kwargs.get("cutoff", 5.0)
        cutoff_fn: str = kwargs.get("cutoff_fn", "cosine")
        action_blocks: int = kwargs.get("action_blocks", 3)
        activation: str = kwargs.get("activation", "silu")
        norm_type: str = kwargs.get("norm_type", "nonorm")

        self.body_blocks = nn.ModuleList()

        # charge
        coulomb_interaction: bool = kwargs.get("coulomb_interaction", False)
        coulomb_cutoff: Optional[float] = None
        if coulomb_interaction:
            coulomb_cutoff: Optional[float] = kwargs.get("coulomb_cutoff", 10.0)
            assert cutoff <= coulomb_cutoff
            hierarchical_cutoff = HierarchicalCutoff(
                long_cutoff=coulomb_cutoff,
                cutoff=cutoff,
            )
            self.body_blocks.append(hierarchical_cutoff)

        embed = XEmbedding(
            node_dim=node_dim,
            node_irreps=node_irreps,
            embed_basis=embed_basis,
            aux_basis=aux_basis,
            num_basis=num_basis,
            rbf_kernel=rbf_kernel,
            cutoff=cutoff,
            cutoff_fn=cutoff_fn,
        )
        self.body_blocks.append(embed)
        for _ in range(action_blocks):
            message = XPainnMessage(
                node_dim=node_dim,
                node_irreps=node_irreps,
                num_basis=num_basis,
                activation=activation,
                norm_type=norm_type,
            )
            update = XPainnUpdate(
                node_dim=node_dim,
                node_irreps=node_irreps,
                activation=activation,
                norm_type=norm_type,
            )
            self.body_blocks.extend([message, update])

        output_modes: Union[str, List[str]] = kwargs.get("output_modes", ["scalar"])
        if not isinstance(output_modes, list):
            output_modes = [output_modes]
        self.output_blocks = nn.ModuleList()
        for mode in output_modes:
            self.output_blocks.append(resolve_output(mode, **kwargs))

        self.cutoff_radius = coulomb_cutoff if coulomb_interaction else cutoff

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        compute_forces: bool = True,
        compute_virial: bool = False,
    ) -> Dict[str, torch.Tensor]:

        data = compute_edge_data(
            data=data,
            compute_forces=compute_forces,
            compute_virial=compute_virial,
        )
        for body_block in self.body_blocks:
            data = body_block(data)
        result = {}
        for output_block in self.output_blocks:
            data, output = output_block(data, compute_forces, compute_virial)
            result.update(output)

        return result


def resolve_model(model_name: str, **kwargs) -> BaseModel:
    models_factory = {
        "xpainn": XPaiNN,
    }
    if model_name.lower() not in models_factory:
        raise NotImplementedError(f"Unsupported model {model_name}")
    return models_factory[model_name.lower()](**kwargs)
