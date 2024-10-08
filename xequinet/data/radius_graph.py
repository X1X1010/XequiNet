from typing import Tuple, List
import torch


def wrap_positions(
    pos: torch.Tensor,  # [n_atoms, 3]
    cell: torch.Tensor,  # [n_graphs, 3, 3]
    n_nodes_per_graph: torch.Tensor,  # [n_graphs]
    pbc: list[bool],  # [3]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Wrap positions to unit cell with PBC."""

    if not any(pbc):
        return pos, torch.zeros_like(pos)

    # by default not changing positions
    shift_T = torch.zeros_like(pos).T

    cell = cell.repeat_interleave(n_nodes_per_graph, dim=0)
    cell_inv = torch.linalg.inv(cell)

    fractional = torch.bmm(pos.unsqueeze(1), cell_inv).squeeze(1)
    fractional_T = fractional.T

    for i, periodic in enumerate(pbc):
        if periodic:
            shift_T[i] = torch.floor(fractional_T[i])
            fractional_T[i] -= shift_T[i]
    pos_wrap = torch.bmm(fractional_T.T.unsqueeze(1), cell).squeeze(1)

    return pos_wrap, shift_T.T


@torch.no_grad()
def radius_graph_pbc(
    pos: torch.Tensor,
    n_nodes_per_graph: torch.Tensor,
    pbc: torch.Tensor,
    cell: torch.Tensor,
    cutoff: float,
):
    """
    Calculate the radius graph with PBC for a batch of graphs.
    """
    device = pos.device
    dtype = pos.dtype
    batch_size = n_nodes_per_graph.shape[0]

    assert pbc.dim() == 2 and pbc.shape[1] == 3, "Invalid pbc shape"
    assert torch.all(pbc[0] == pbc), "PBC must be the same for all graphs"
    pbc_ = pbc[0].detach().cpu().numpy().tolist()

    # Calculate required number of unit cells in each direction.
    # Smallest distance between planes separated by a1 is
    # 1 / ||(a2 x a3) / V ||_2, since a2 x a3 is the area fo the plane.
    # Note that the unit cell volume V = a1 * (a2 x a3) and that
    # (a2 x a3) / V is also the reciprocal primitive vector
    # (crystallographer's definition)

    cross_a2a3 = torch.cross(cell[:, 1], cell[:, 2], dim=-1)
    cell_vol = torch.sum(cell[:, 0] * cross_a2a3, dim=-1, keepdim=True)

    if pbc_[0]:
        inv_min_dist_a1 = torch.norm(cross_a2a3 / cell_vol, p=2, dim=-1)
        rep_a1 = torch.ceil(cutoff * inv_min_dist_a1)
    else:
        rep_a1 = cell.new_zeros(1)

    if pbc_[1]:
        cross_a3a1 = torch.cross(cell[:, 2], cell[:, 0], dim=-1)
        inv_min_dist_a2 = torch.norm(cross_a3a1 / cell_vol, p=2, dim=-1)
        rep_a2 = torch.ceil(cutoff * inv_min_dist_a2)
    else:
        rep_a2 = cell.new_zeros(1)
    
    if pbc_[2]:
        cross_a1a2 = torch.cross(cell[:, 0], cell[:, 1], dim=-1)
        inv_min_dist_a3 = torch.norm(cross_a1a2 / cell_vol, p=2, dim=-1)
        rep_a3 = torch.ceil(cutoff * inv_min_dist_a3)
    else:
        rep_a3 = cell.new_zeros(1)

    # Take the max over all images for uniformity. This is essentially padding.
    # Note that this can significantly increase the number of computed distances
    # if the required repetitions are very different between images
    # (which they usually are). Changing this to sparse (scatter) operations
    # might be worth the effort if this function becomes a bottleneck.
    max_rep = [rep_a1.max().item(), rep_a2.max().item(), rep_a3.max().item()]

    # Tensor of units cells
    cells_per_dim = [torch.arange(-rep, rep + 1, device=device, dtype=dtype) for rep in max_rep]
    # [ncells=8*m1*m2*m3, 3], covering all possible combinations of cells in 3 directions
    # potential memory hog here
    cell_offsets = torch.cartesian_prod(*cells_per_dim)
    n_cells = cell_offsets.shape[0]
    # [n_graphs, n_cells, 3]
    unit_cell_batch = cell_offsets.view(1, n_cells, 3).expand(batch_size, -1, -1).contiguous()

    # Compute the x, y, z positional offsets for each cell in each image
    # [n_graphs, n_cells, 3]
    pbc_offsets = torch.bmm(unit_cell_batch, cell)
    # repeat into: [n_atoms, n_cells, 3]
    pbc_offsets_per_atom = pbc_offsets.repeat_interleave(n_nodes_per_graph, dim=0)

    # Wrap positions to unit cell
    pos_orig_wrapped, shift_to_unwrapped = wrap_positions(pos, cell, n_nodes_per_graph, pbc_)
    pos_orig_wrapped = pos_orig_wrapped.view(-1, 1, 3)
    # broadcasted to [n_atoms, n_cells, 3]
    pos_pbc_shift = pos_orig_wrapped + pbc_offsets_per_atom

    @torch.no_grad()
    def dist_thresh(
        A: torch.Tensor, B: torch.Tensor, cutoff: float,
    ) -> torch.Tensor:
        D = torch.cdist(A, B)
        idx = torch.nonzero(torch.logical_and(D < cutoff, D > 0.01), as_tuple=False)
        return idx
        
    @torch.no_grad()
    def blockwise_dist_thresh(
        A: torch.Tensor, B: torch.Tensor, cutoff: float, block_size: int
    ) -> List[torch.Tensor]:
        """Iter over blocks of A and B to compute distances."""
        n, m = A.shape[0], B.shape[0]
        n_blocks = (n + block_size - 1) // block_size
        m_blocks = (m + block_size - 1) // block_size

        ret_idx = []
        for i in range(n_blocks):
            for j in range(m_blocks):
                A_block = A[i * block_size : (i + 1) * block_size]
                B_block = B[j * block_size : (j + 1) * block_size]

                idx = dist_thresh(A_block, B_block, cutoff)
                ret_idx.append(idx)
        return ret_idx
    
    @torch.no_grad()
    def compute_dist_one_graph(i: int, j: int) -> List[torch.Tensor]:
        # [Vi, 3]
        A = pos_orig_wrapped[i:j].reshape(-1, 3).contiguous()
        # [Vi * n_cells, 3]
        B = pos_pbc_shift[i:j].reshape(-1, 3).contiguous()

        # ix, iy: [n_edges]
        return blockwise_dist_thresh(A, B, cutoff, 65536) # 32GB memory limit
    
    # index offset between images
    graph_end = torch.cumsum(n_nodes_per_graph, dim=0)  # [n_graphs]
    graph_begin = graph_end - n_nodes_per_graph

    # [sender, receiver]
    compute_dist = [
        compute_dist_one_graph(i, j) for i, j in zip(graph_begin, graph_end)
    ]

    def _compute_nr_edges(edges: List[torch.Tensor]) -> int:
        # edges: [fragment of the edge list]
        return sum(map(len, edges))

    n_neighbors_image = torch.tensor(list(map(_compute_nr_edges, compute_dist)), device=device)
    # flatten index to get 0-based indices
    # [ [ix0, iy0], [ix1, iy1], ...]: [n_edges, 2]
    index0 = torch.cat(sum(compute_dist, []))
    # iy is in range (0, ... sum(Vi * n_cells)) but we need (0, ... sum(Vi))
    ix = index0[:, 0]
    iy = torch.div(index0[:, 1], n_cells, rounding_mode="floor")
    # get displacement offsets: [sum(Vi)] -> [n_edges] -> [1, n_edges]
    graph_offset = torch.repeat_interleave(graph_begin, n_neighbors_image).view(1, -1)
    edge_index = torch.stack([ix, iy]) + graph_offset
    # pos: [n_atoms, n_cells, 3], and reshape to [n_atoms * n_cells, 3] therefore,
    # the indeces are arranged like:
    # | [G0] a0c0 a0c1 ... a1c0 a1c1 ... | [G1] a0c0 a0c1 ... a1c0 a1c1 ... |
    # hence, we can easily obtain cell index with iy % n_cells
    cell_offsets_index = index0[:, 1] % n_cells
    cell_offsets = cell_offsets[cell_offsets_index]
    cell_offsets += (
        shift_to_unwrapped[edge_index[0]] - shift_to_unwrapped[edge_index[1]]
    )

    return edge_index, cell_offsets