from pathlib import Path
from typing import Dict, List, Tuple, Union

import torch
from e3nn import o3
from pyscf import gto

from .qc import ELEMENTS_DICT, ELEMENTS_LIST

M_IDX_COMMON = {
    0: [0],
    1: [1, 2, 0],                   # (x, y, z) -> (y, z, x)
    2: [0, 1, 2, 3, 4],             # (xy, yz, z2, xz, x2y2)
    3: [0, 1, 2, 3, 4, 5, 6],       # (-3, -2, -1, 0, 1, 2, 3)
    4: [0, 1, 2, 3, 4, 5, 6, 7, 8], # (-4, -3, -2, -1, 0, 1, 2, 3, 4)
}

M_IDX_ORCA = {
    0: [0],
    1: [1, 2, 0],                   # (z, y, x)
    2: [2, 3, 1, 4, 0],             # (z^2, xz, yz, x2y2, xy)
    3: [3, 4, 2, 5, 1, 6, 0],       # (0, 1, -1, 2, -2, 3, -3)
    4: [4, 5, 3, 6, 2, 7, 1, 8, 0]  # (0, 1, -1, 2, -2, 3, -3, 4, -4)
}

M_IDX_ORIG = {
    0: [0],
    1: [0, 1, 2],                   # (y, z, x)
    2: [0, 1, 2, 3, 4],             # (xy, yz, z2, xz, x2y2)
    3: [0, 1, 2, 3, 4, 5, 6],       # (-3, -2, -1, 0, 1, 2, 3)
    4: [0, 1, 2, 3, 4, 5, 6, 7, 8], # (-4, -3, -2, -1, 0, 1, 2, 3, 4)
}


THIS_FOLDER = Path(__file__).parent
BASIS_FOLDER = THIS_FOLDER / "basis"


def get_l_from_basis(basis: str, element: str) -> List[int]:
    """get the list of angular momentum of the element from the basis set."""
    if basis == "hessian":
        return [1]
    if (BASIS_FOLDER / f"{basis}.dat").exists():
        basis_file = BASIS_FOLDER / f"{basis}.dat"
        basis = gto.basis.load(basis_file, element)
    else:
        basis = gto.basis.load(basis, element)
    return [b[0] for b in basis]


def resolve_m_idx_type(map_type:str="pyscf") -> Dict[int, List[int]]:
    map_type = map_type.lower()
    if map_type in ["orca"]:
        return M_IDX_ORCA 
    elif map_type in ["pyscf", "gaussian"]:
        return M_IDX_COMMON
    else:
        return M_IDX_ORIG


class MatToolkit:
    """
    Toolkit for converting matrices to e3nn format.
    """

    def __init__(
        self,
        target_basis: str,
        elements: List[Union[str, int]],
        map_type: str = "pyscf",
    ) -> None:
        """
        Args:
            `target_basis`: target basis set
            `elements`: list of existing elements in the dataset
            `map_type`: map type for resolve m idx ordering of input
        """
        self.target_basis = target_basis
        self.elements = [
            e if isinstance(e, str) else ELEMENTS_LIST[e] for e in elements
        ]  # List[str]
        self.atomic_numbers = [
            e if isinstance(e, int) else ELEMENTS_DICT[e] for e in elements
        ]  # List[int]
        self.num_orb_per_angular, self.max_orb_per_angular = self._resolve_basis()
        self.atom2mask = self._gen_mask()

        basis_irreps = []
        for l, mul in enumerate(self.max_orb_per_angular):
            basis_irreps.append([mul.item(), (l, (-1) ** l)])
        self.basis_irreps = o3.Irreps(basis_irreps).simplify()

        self.m_idx_map = resolve_m_idx_type(map_type)
        self.m_idx = self._gen_m_idx()

    def _resolve_basis(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Resolve the basis set.
        Get the number of orbitals for each angular momentum, and the maximum number of orbitals among selected elements.
        """
        # we don't know the maximum number of orbitals for each angular momentum, so we set it to 10 first
        num_orb_per_angular = torch.zeros((len(self.elements), 10), dtype=torch.long)
        for i, ele in enumerate(self.elements):
            l_list = get_l_from_basis(self.target_basis, ele)
            l_idx, l_num = torch.unique(
                torch.tensor(l_list), return_counts=True
            )  # [l1, l2, ...], [num_l1, num_l2, ...]
            num_orb_per_angular[i, l_idx] = l_num
        max_orb_per_angular = num_orb_per_angular.max(
            dim=0
        ).values  # [max_l1, max_l2, ...]
        max_l = torch.nonzero(max_orb_per_angular).max().item() + 1
        max_orb_per_angular = max_orb_per_angular[:max_l]
        num_orb_per_angular = num_orb_per_angular[:, :max_l]
        return num_orb_per_angular, max_orb_per_angular

    def _gen_mask(self) -> Dict[int, torch.Tensor]:
        """
        Generate the mask for each element.
        """
        atom2mask = {}
        for i, at in enumerate(self.atomic_numbers):
            mask = []
            for l, (num_orb, max_orb) in enumerate(
                zip(self.num_orb_per_angular[i], self.max_orb_per_angular)
            ):
                # mask for each angular momentum
                # e.g. max: 2p. current: 1p. mask: [T, T, T, F, F, F]
                l_mask = torch.zeros(max_orb * (2 * l + 1), dtype=bool)
                l_mask[: num_orb * (2 * l + 1)] = True
                mask.append(l_mask)
            atom2mask[at] = torch.cat(mask)
        return atom2mask

    def _gen_m_idx(self) -> torch.Tensor:
        """
        Generate the m index.
        e.g. For 1s 1p 1d, the m index is [0, 2, 3, 1, 4, 5, 6, 7, 8],
             where (x, y, z) of p orbital is changed to (y, z, x).
        """
        m_idx = []
        offset = 0
        for mul, (l, p) in self.basis_irreps:
            offsets = torch.arange(offset, offset + mul * (2 * l + 1), 2 * l + 1).view(
                -1, 1
            )
            idx = torch.tensor([self.m_idx_map[l]]).repeat(mul, 1)
            m_idx.append((idx + offsets).flatten())
            offset += mul * (2 * l + 1)
        return torch.cat(m_idx)

    def get_basis_irreps(self) -> o3.Irreps:
        """
        Get the basis irreps.
        """
        return self.basis_irreps

    def padding_matrix(self, at_no: torch.Tensor, mat: torch.Tensor) -> torch.Tensor:
        """
        Pad the matrix with zeros.
        """
        num_atoms = at_no.shape[0]
        padded_mat = torch.zeros(
            (num_atoms * self.basis_irreps.dim, num_atoms * self.basis_irreps.dim),
            dtype=mat.dtype,
            device=mat.device,
        )
        mask1d = torch.cat([self.atom2mask[at.item()] for at in at_no])
        mask2d = torch.outer(mask1d, mask1d)
        padded_mat[mask2d] = mat.flatten()
        return padded_mat

    def unpadding_matrix(self, at_no: torch.Tensor, mat: torch.Tensor) -> torch.Tensor:
        """
        Unpad the matrix.
        """
        mask1d = torch.cat([self.atom2mask[at.item()] for at in at_no])
        mask2d = torch.outer(mask1d, mask1d)
        length_unpad = mask1d.sum().item()
        return mat[mask2d].reshape(length_unpad, length_unpad)

    def get_edge_index_full(self, at_no: torch.Tensor) -> torch.Tensor:
        """
        Get the full edge index for the system without self-loop.
        """
        num_atoms = at_no.shape[0]
        edge_index_full = torch.zeros(
            (2, num_atoms * (num_atoms - 1)), dtype=torch.long, device=at_no.device
        )
        for i in range(num_atoms):
            edge_index_full[0, i * (num_atoms - 1) : (i + 1) * (num_atoms - 1)] = i
            edge_index_full[
                1, i * (num_atoms - 1) : (i + 1) * (num_atoms - 1)
            ] = torch.cat(
                [
                    torch.arange(0, i, device=at_no.device),
                    torch.arange(i + 1, num_atoms, device=at_no.device),
                ]
            )
        return edge_index_full

    def get_padded_blocks(
        self, at_no: torch.Tensor, mat: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Get the padded blocks for the matrix.
        """
        device = mat.device
        num_atoms = at_no.shape[0]
        padded_mat = self.padding_matrix(
            at_no, mat
        )  # [num_atoms * num_orb, num_atoms * num_orb]
        padded_mat = padded_mat.view(
            num_atoms, self.basis_irreps.dim, num_atoms, self.basis_irreps.dim
        )
        padded_mat = padded_mat.permute(0, 2, 1, 3)
        # for p orbital, change the order to (y, z, x)
        padded_mat = padded_mat[:, :, self.m_idx, :][:, :, :, self.m_idx]
        node_blocks = padded_mat[
            torch.arange(num_atoms, device=device),
            torch.arange(num_atoms, device=device),
        ]
        edge_blocks = padded_mat[edge_index[0], edge_index[1]]
        return node_blocks, edge_blocks

    def get_mask(self, at_no: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Get the mask for the atom.
        """
        atom_mask = torch.stack(
            [self.atom2mask[at.item()] for at in at_no]
        )  # [num_atoms, num_orb]
        node_mask = atom_mask.unsqueeze(2) * atom_mask.unsqueeze(
            1
        )  # [num_atoms, num_orb, num_orb]
        edge_mask = atom_mask[edge_index[0]].unsqueeze(2) * atom_mask[
            edge_index[1]
        ].unsqueeze(
            1
        )  # [num_edges, num_orb, num_orb]
        return node_mask, edge_mask

    def assemble_blocks(
        self,
        at_no: torch.Tensor,
        node_blocks: torch.Tensor,
        edge_blocks: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Assemble the blocks to the full matrix.
        """
        device = node_blocks.device
        num_atoms = node_blocks.shape[0]
        num_orb = node_blocks.shape[1]
        padded_mat = torch.zeros(
            (num_atoms, num_atoms, num_orb, num_orb),
            dtype=node_blocks.dtype,
            device=device,
        )
        padded_mat[
            torch.arange(num_atoms, device=device),
            torch.arange(num_atoms, device=device),
        ] = node_blocks
        padded_mat[edge_index[0], edge_index[1]] = edge_blocks
        # for p orbital, change the order to (x, y, z)
        padded_mat_ = padded_mat.clone()
        padded_mat[:, :, self.m_idx, :] = padded_mat_
        padded_mat_ = padded_mat.clone()
        padded_mat[:, :, :, self.m_idx] = padded_mat_
        padded_mat = padded_mat.permute(0, 2, 1, 3).reshape(
            num_atoms * num_orb, num_atoms * num_orb
        )
        mat = self.unpadding_matrix(at_no, padded_mat)
        return mat


if __name__ == "__main__":
    mole = gto.M(atom="H 0 0 0; O 0 0 2.1", basis="sto-3g", charge=-1)
    at_no = torch.from_numpy(mole.atom_charges())
    ovlp = mole.intor("int1e_ovlp")
    ovlp = torch.tensor(ovlp)
    mat_toolkit = MatToolkit("sto-3g", ["H", "O"])
    edge_index_full = mat_toolkit.get_edge_index_full(at_no)
    node_blocks, edge_blocks = mat_toolkit.get_padded_blocks(
        at_no, ovlp, edge_index_full
    )
    node_mask, edge_mask = mat_toolkit.get_mask(at_no, edge_index_full)
    print(node_blocks, "\n", edge_blocks)
    print(node_mask, "\n", edge_mask)
    full_mat = mat_toolkit.assemble_blocks(node_blocks, edge_blocks, edge_index_full)
    print(ovlp)
    print(full_mat)
    print((full_mat == ovlp).all())
