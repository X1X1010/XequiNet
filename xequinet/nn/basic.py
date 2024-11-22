from typing import Dict, List, Optional, Tuple

import torch

from xequinet import keys


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


def compute_forces_only(
    energy: torch.Tensor,
    pos: torch.Tensor,
    training: bool = True,
) -> torch.Tensor:
    """Compute forces from energy and positions"""
    grad_outputs: Optional[List[Optional[torch.Tensor]]] = [torch.ones_like(energy)]
    pos_grad = torch.autograd.grad(
        outputs=[energy],
        inputs=[pos],
        grad_outputs=grad_outputs,
        retain_graph=training,
        create_graph=training,
    )[0]
    if pos_grad is None:
        pos_grad = torch.zeros_like(pos)
    return -1.0 * pos_grad


def compute_virial_only(
    energy: torch.Tensor,
    strain: torch.Tensor,
    training: bool = True,
) -> torch.Tensor:
    """Compute virial from energy and strain"""
    grad_outputs: Optional[List[Optional[torch.Tensor]]] = [torch.ones_like(energy)]
    strain_grad = torch.autograd.grad(
        outputs=[energy],
        inputs=[strain],
        grad_outputs=grad_outputs,
        retain_graph=training,
        create_graph=training,
    )[0]
    if strain_grad is None:
        strain_grad = torch.zeros_like(strain)
    return -1.0 * strain_grad


def compute_forces_and_virial(
    energy: torch.Tensor,
    pos: torch.Tensor,
    strain: torch.Tensor,
    training: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    grad_outputs: Optional[List[Optional[torch.Tensor]]] = [torch.ones_like(energy)]
    pos_grad, strain_grad = torch.autograd.grad(
        outputs=[energy],
        inputs=[pos, strain],
        grad_outputs=grad_outputs,
        retain_graph=training,
        create_graph=training,
    )
    if pos_grad is None:
        pos_grad = torch.zeros_like(pos)
    if strain_grad is None:
        strain_grad = torch.zeros_like(strain)
    return -1.0 * pos_grad, -1.0 * strain_grad


def compute_properties(
    data: Dict[str, torch.Tensor],
    compute_forces: bool = True,
    compute_virial: bool = False,
    training: bool = True,
    extra_properties: Optional[List[str]] = None,
) -> Dict[str, torch.Tensor]:
    """Compute properties from data"""
    results = {}
    if compute_forces and compute_virial:
        forces, virial = compute_forces_and_virial(
            energy=data[keys.TOTAL_ENERGY],
            pos=data[keys.POSITIONS],
            strain=data[keys.STRAIN],
            training=training,
        )
        results[keys.FORCES] = forces
        results[keys.VIRIAL] = virial
    elif compute_forces:
        forces = compute_forces_only(
            energy=data[keys.TOTAL_ENERGY],
            pos=data[keys.POSITIONS],
            training=training,
        )
        results[keys.FORCES] = forces
    elif compute_virial:
        virial = compute_virial_only(
            energy=data[keys.TOTAL_ENERGY],
            strain=data[keys.STRAIN],
            training=training,
        )
        results[keys.VIRIAL] = virial

    if extra_properties is not None:
        results.update({k: data[k] for k in extra_properties})

    return results
