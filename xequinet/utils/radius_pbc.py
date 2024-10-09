from typing import Tuple, Union
import numpy as np
import torch
from ase.neighborlist import primitive_neighbor_list


def radius_graph_pbc(
    positions: np.ndarray,
    pbc: Union[bool, np.ndarray],
    cell: np.ndarray,
    cutoff: float,
    loop: bool = False,
    max_num_neighbors: int = 32,
    flow: str = "source_to_target",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the radius graph with a periodic boundary condition.
    """
    assert flow in ["source_to_target", "target_to_source"]
    # make sure `pbc` is a 3-element array
    pbc = np.ones(3, dtype=bool) * pbc
    # here we define `offsets` as how many unit cells does the `idx_j` atom shift from its original position
    # and `shifts` as the actual shift vector
    idx_c, idx_n, distances, offsets = primitive_neighbor_list(
        "ijdS",
        pbc=pbc,
        cell=cell,
        positions=positions,
        cutoff=cutoff,
        self_interaction=True,  # we want edges from atom to itself in different periodic images
    )
    # from https://github.com/mir-group/nequip/blob/main/nequip/data/AtomicData.py#L770
    if not loop:  # remove the real self-interaction
        bad_edge = idx_c == idx_n
        bad_edge &= np.all(offsets == 0, axis=1)
        keep_edge = ~bad_edge
        idx_c = idx_c[keep_edge]
        idx_n = idx_n[keep_edge]
        offsets = offsets[keep_edge]
        distances = distances[keep_edge]
    # limit max number of neighbors
    nonmax_idx = []
    num_atoms = positions.shape[0]
    for i in range(num_atoms):
        idx_i = (idx_c == i).nonzero()[0]
        # sort neighbors by distance, remove edges large than `max_num_neighbors`
        idx_sorted = np.argsort(distances[idx_i])[:max_num_neighbors]
        nonmax_idx.append(idx_i[idx_sorted])
    nonmax_idx = np.concatenate(nonmax_idx)
    idx_c = idx_c[nonmax_idx]
    idx_n = idx_n[nonmax_idx]
    offsets = offsets[nonmax_idx]
    # calculate `shifts` from `offsets`
    shifts = np.dot(offsets, cell)
    if flow == "source_to_target":
        edge_index = np.stack([idx_c, idx_n], axis=0)
    else:
        edge_index = np.stack([idx_n, idx_c], axis=0)

    return edge_index, shifts


def radius_batch_pbc(
    positions: torch.Tensor,
    pbc: torch.Tensor,
    lattice: torch.Tensor,
    cutoff: float,
    loop: bool = False,
    max_num_neighbors: int = 32,
    flow: str = "source_to_target",
    batch: torch.Tensor = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate the radius graph with PBC for a batch one by one.
    Fake batch process for `radius_graph_pbc`.
    """
    assert flow in ["source_to_target", "target_to_source"]
    if batch is None:
        batch = torch.zeros(positions.shape[0], dtype=torch.long)
    num_points = batch.max().item() + 1
    edge_index_list = []
    shifts_list = []
    for i in range(num_points):
        _mask = batch == i
        _pos = positions[_mask].cpu().numpy()
        _pbc = pbc[i].cpu().numpy()
        _cell = lattice[i].cpu().numpy()
        _edge_index, _shifts = radius_graph_pbc(
            _pos, _pbc, _cell, cutoff, loop, max_num_neighbors, flow
        )
        edge_index_list.append(torch.from_numpy(_edge_index))
        shifts_list.append(torch.from_numpy(_shifts))
    edge_index = torch.cat(edge_index_list, dim=1).to(torch.long).to(positions.device)
    shifts = (
        torch.cat(shifts_list, dim=0).to(torch.get_default_dtype()).to(positions.device)
    )
    return edge_index, shifts
