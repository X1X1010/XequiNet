from typing import Optional

import torch
from torch_geometric.data import Data


class XequiData(Data):

    # for type annotation
    atomic_numbers: torch.Tensor
    pos: torch.Tensor
    edge_index: torch.Tensor
    batch: torch.Tensor

    def __init__(
        self,
        atomic_numbers: Optional[torch.Tensor] = None,  # [N_atoms]
        pos: Optional[torch.Tensor] = None,  # [N_atoms, 3]
        pbc: Optional[torch.Tensor] = None,  # [1, 3]
        cell: Optional[torch.Tensor] = None,  # [1, 3, 3]
        edge_index: Optional[torch.Tensor] = None,  # [2, N_edges]
        cell_offsets: Optional[torch.Tensor] = None,  # [N_edges, 3]
        charge: Optional[torch.Tensor] = None,  # [1]
        energy: Optional[torch.Tensor] = None,  # [1]
        forces: Optional[torch.Tensor] = None,  # [N_atoms, 3]
        base_energy: Optional[torch.Tensor] = None,  # [1]
        base_forces: Optional[torch.Tensor] = None,  # [N_atoms, 3]
        virial: Optional[torch.Tensor] = None,  # [1, 3, 3]
        atomic_charges: Optional[torch.Tensor] = None,  # [N_atoms]
        base_charges: Optional[torch.Tensor] = None,  # [1]
        dipole: Optional[torch.Tensor] = None,  # [1, 3]
        base_dipole: Optional[torch.Tensor] = None,  # [1, 3]
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
        # positions and atomic numbers
        n_atoms: Optional[int] = None
        dtype: Optional[torch.dtype] = None
        if self.pos is not None or atomic_numbers is not None:
            assert self.pos.dim() == 2 and self.pos.shape[1] == 3
            n_atoms = self.pos.shape[0]
            dtype = self.pos.dtype

            assert (
                atomic_numbers.shape == (n_atoms,) and atomic_numbers.dtype == torch.int
            )
            self.atomic_numbers = atomic_numbers

        # pbc
        if pbc is not None:
            assert pbc.shape == (1, 3) and pbc.dtype == torch.bool
            self.pbc = pbc
        else:
            self.pbc = torch.zeros(1, 3, dtype=torch.bool)
        # cell
        if cell is not None and self.pbc.any():
            assert cell.shape == (1, 3, 3) and cell.dtype == dtype
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
        if base_energy is not None:
            assert base_energy.shape == (1,) and base_energy.dtype == dtype
            self.base_energy = base_energy
        # forces
        if forces is not None:
            assert forces.shape == (n_atoms, 3) and forces.dtype == dtype
            self.forces = forces
        if base_forces is not None:
            assert base_forces.shape == (n_atoms, 3) and base_forces.dtype == dtype
            self.base_forces = base_forces
        # virial
        if virial is not None:
            assert virial.shape == (1, 3, 3) and virial.dtype == dtype
            self.virial = virial
        # charges
        if atomic_charges is not None:
            assert atomic_charges.shape == (n_atoms,) and atomic_charges.dtype == dtype
            assert torch.allclose(atomic_charges.sum().round().to(torch.int), charge)
            self.atomic_charges = atomic_charges
        if base_charges is not None:
            assert base_charges.shape == (1,) and base_charges.dtype == dtype
            assert torch.allclose(base_charges.sum().round().to(torch.int), charge)
            self.base_charges = base_charges
        if dipole is not None:
            assert dipole.shape == (1, 3) and dipole.dtype == dtype
            self.dipole = dipole
        if base_dipole is not None:
            assert base_dipole.shape == (1, 3) and base_dipole.dtype == dtype
            self.base_dipole = base_dipole
