from typing import Optional

import torch
from torch_geometric.data import Data


class XData(Data):
    # for type annotation
    atomic_numbers: torch.Tensor
    pos: torch.Tensor
    edge_index: torch.Tensor
    batch: torch.Tensor

    def __init__(
        self,
        atomic_numbers: torch.Tensor,  # [N_atoms]
        pos: torch.Tensor,  # [N_atoms, 3]
        pbc: Optional[torch.Tensor] = None,  # [1, 3]
        cell: Optional[torch.Tensor] = None,  # [1, 3, 3]
        edge_index: Optional[torch.Tensor] = None,  # [2, N_edges]
        cell_offsets: Optional[torch.Tensor] = None,  # [N_edges, 3]
        charge: Optional[torch.Tensor] = None,  # [1]
        energy: Optional[torch.Tensor] = None,  # [1]
        forces: Optional[torch.Tensor] = None,  # [N_atoms, 3]
        virial: Optional[torch.Tensor] = None,  # [1, 3, 3]
        atomic_charges: Optional[torch.Tensor] = None,  # [N_atoms]
        **kwargs,
    ) -> None:
        super().__init__(edge_index=edge_index, pos=pos, **kwargs)
        # edge index
        if self.edge_index is not None:
            assert (
                isinstance(self.edge_index, torch.Tensor)
                and self.edge_index.dim() == 2
                and self.edge_index.shape[0] == 2
                and self.edge_index.dtype == torch.long
            )
        # positions
        assert (
            isinstance(self.pos, torch.Tensor)
            and self.pos.dim() == 2
            and self.pos.shape[1] == 3
        )
        n_atoms = self.pos.shape[0]
        dtype = self.pos.dtype

        # atomic_numbers
        assert atomic_numbers.shape == (n_atoms,) and atomic_numbers.dtype == torch.int
        self.atomic_numbers = atomic_numbers

        # pbc and cell
        if pbc is not None or pbc is not None:
            assert pbc.shape == (1, 3) and pbc.dtype == torch.bool
            assert cell.shape == (1, 3, 3) and cell.dtype == dtype
            self.pbc = pbc
            self.cell = cell
        # cell offsets
        if cell_offsets is not None:
            assert self.edge_index is not None
            assert (
                cell_offsets.shape == (self.edge_index.shape[1], 3)
                and cell_offsets.dtype == dtype
            )
            self.cell_offsets = cell_offsets
        # charge
        if charge is not None:
            assert charge.shape == (1,) and charge.dtype == torch.int
            self.charge = charge

        # energy
        if energy is not None:
            assert energy.shape == (1,) and energy.dtype == dtype
            self.energy = energy
        # forces
        if forces is not None:
            assert forces.shape == (n_atoms, 3) and forces.dtype == dtype
            self.forces = forces
        # virial
        if virial is not None:
            assert virial.shape == (1, 3, 3) and virial.dtype == dtype
            self.virial = virial
        # charges
        if atomic_charges is not None:
            assert atomic_charges.shape == (n_atoms,) and atomic_charges.dtype == dtype
            assert torch.allclose(atomic_charges.sum(), charge)
            self.atomic_charges = atomic_charges
