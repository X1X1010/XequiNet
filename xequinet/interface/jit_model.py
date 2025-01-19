from typing import Dict, Literal, Optional

import torch

from xequinet import keys
from xequinet.data.radius_graph import single_radius_graph
from xequinet.nn.basic import compute_edge_data, compute_properties
from xequinet.nn.model import BaseModel, XPaiNN
from xequinet.utils import get_default_units, unit_conversion


class XPaiNNLMP(XPaiNN):
    """
    XPaiNN script model for force field. This model does not consider batch.
    """

    def __init__(
        self,
        unit_style: str = "metal",
        net_charge: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        lammps_units = keys.LAMMPS_UNIT_STYLE[unit_style]
        default_units = get_default_units()
        self.pos_unit_factor = unit_conversion(
            lammps_units[keys.POSITIONS], default_units[keys.POSITIONS]
        )  # LAMMPS -> NNP
        self.energy_unit_factor = unit_conversion(
            default_units[keys.TOTAL_ENERGY], lammps_units[keys.TOTAL_ENERGY]
        )  # NNP -> LAMMPS
        self.forces_unit_factor = unit_conversion(
            f"{default_units[keys.FORCES]}",
            f"{lammps_units[keys.FORCES]}",
        )  # NNP -> LAMMPS
        self.net_charge = net_charge
        self.cutoff_radius /= (
            self.pos_unit_factor
        )  # note: NNP -> LAMMPS, so use /= instead of *=

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        compute_forces: bool = True,
        compute_virial: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            `data`: Dictionary containing the following keys,
               - `atomic_numbers`: Atomic numbers.
               - `positions`: Atomic positions.
               - `pbc` (Optional): Periodic boundary conditions.
               - `cell` (Optional): Cell parameters.
            `compute_forces`: Whether to compute forces.
            `compute_virial`: Whether to compute virial.
        Returns:
            A dictionary containing the following keys:
            - `energy`: Total energy.
            - `atomic_energies`: Atomic energies.
            - `forces` (Optional): Atomic forces.
            - `virial` (Optional): Virial tensor.
        """
        data[keys.POSITIONS] *= self.pos_unit_factor
        if self.net_charge is not None:
            data[keys.TOTAL_CHARGE] = torch.tensor(
                [self.net_charge], device=data[keys.POSITIONS].device
            )
        # we cannot use `super().forward` because it is nor supported by TorchScript
        # so we manually call the forward method of the parent class
        data: Dict[str, torch.Tensor] = compute_edge_data(
            data=data,
            compute_forces=compute_forces,
            compute_virial=compute_virial,
        )
        for mod in self.mods.values():
            data = mod(data)
        result = compute_properties(
            data=data,
            compute_forces=compute_forces,
            compute_virial=compute_virial,
            training=self.training,
            extra_properties=self.extra_properties,
        )
        result[keys.TOTAL_ENERGY] *= self.energy_unit_factor
        if compute_forces:
            result[keys.FORCES] *= self.forces_unit_factor
        if compute_virial:
            result[keys.VIRIAL] *= self.energy_unit_factor
        return result


class XPaiNNDipole(XPaiNN):
    """
    XPaiNN script model for dipole moment. This model does not consider batch.
    """

    def __init__(
        self,
        unit_style: str = "metal",
        net_charge: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        default_units = get_default_units()
        lammps_units = keys.LAMMPS_UNIT_STYLE[unit_style]
        self.pos_unit_factor = unit_conversion(
            lammps_units[keys.POSITIONS], default_units[keys.POSITIONS]
        )  # LAMMPS -> NNP
        self.dipole_unit_factor = unit_conversion(
            default_units[keys.DIPOLE],
            f"{lammps_units[keys.TOTAL_CHARGE]}*{lammps_units[keys.POSITIONS]}",
        )  # NNP -> LAMMPS
        self.net_charge = net_charge
        self.cutoff_radius /= (
            self.pos_unit_factor
        )  # note: NNP -> LAMMPS, so use /= instead of *=

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            `data`: Dictionary containing the following keys,
               - `atomic_numbers`: Atomic numbers.
               - `positions`: Atomic positions.
               - `pbc` (Optional): Periodic boundary conditions.
               - `cell` (Optional): Cell parameters.
        Returns:
            A dictionary containing the following keys:
            - `dipole`: Dipole moment.
        """
        data[keys.POSITIONS] *= self.pos_unit_factor
        if self.net_charge is not None:
            data[keys.TOTAL_CHARGE] = torch.tensor(
                [self.net_charge], device=data[keys.POSITIONS].device
            )
        data = compute_edge_data(
            data=data,
            compute_forces=False,
            compute_virial=False,
        )
        for mod in self.mods.values():
            data = mod(data)
        dipole = data[keys.DIPOLE]
        dipole *= self.dipole_unit_factor

        return {keys.DIPOLE: dipole}


class XPaiNNGMX(XPaiNN):
    """
    XPaiNN script model for force field. This model does not consider batch.
    """

    def __init__(self, net_charge: Optional[int] = None, **kwargs) -> None:
        super().__init__(**kwargs)
        default_units = get_default_units()
        self.pos_unit_factor = unit_conversion("nm", default_units[keys.POSITIONS])
        self.energy_unit_factor = unit_conversion(
            default_units[keys.TOTAL_ENERGY], "kJ/mol"
        )  # NNP -> GROMACS
        self.forces_unit_factor = unit_conversion(
            f"{default_units[keys.FORCES]}", "kJ/(mol*nm)"
        )  # NNP -> GROMACS
        self.net_charge = net_charge

    def forward(
        self,
        positions: torch.Tensor,
        atomic_numbers: torch.Tensor,
        box: Optional[torch.Tensor] = None,
        pbc: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            `positions`: Atomic positions.
            `atomic_numbers`: Atomic numbers.
            `box` (Optional): Box vectors.
            `pbc` (Optional): Periodic boundary conditions.
        Returns:
            Energy.
        """
        positions = positions * self.pos_unit_factor
        if box is None:
            cell = torch.eye(3, dtype=positions.dtype, device=positions.device)
        else:
            cell = box * self.pos_unit_factor
        if pbc is None:
            pbc = torch.zeros(3, dtype=torch.bool, device=positions.device)

        with torch.no_grad():
            edge_index, cell_offsets = single_radius_graph(
                pos=positions,
                cell=cell,
                pbc=pbc,
                cutoff=self.cutoff_radius,
            )
        data = {
            keys.POSITIONS: positions,
            keys.ATOMIC_NUMBERS: atomic_numbers,
            keys.CELL: cell.unsqueeze(0),
            keys.PBC: pbc.unsqueeze(0),
            keys.EDGE_INDEX: edge_index,
            keys.CELL_OFFSETS: cell_offsets,
        }
        if self.net_charge is not None:
            data[keys.TOTAL_CHARGE] = torch.tensor(
                [self.net_charge], device=positions.device
            )
        data = compute_edge_data(
            data=data,
            compute_forces=True,
            compute_virial=False,
        )
        for mod in self.mods.values():
            data = mod(data)

        return data[keys.TOTAL_ENERGY] * self.energy_unit_factor


def resolve_jit_model(
    mode: Literal["lmp", "gmx", "dipole"] = "lmp",
    unit_style: str = "metal",
    net_charge: Optional[int] = None,
    **kwargs,
) -> BaseModel:
    models_factory = {
        "lmp": XPaiNNLMP,
        "dipole": XPaiNNDipole,
        "gmx": XPaiNNGMX,
    }
    if mode not in models_factory:
        raise NotImplementedError(f"Unsupported mode {mode}")

    return models_factory[mode](
        unit_style=unit_style,
        net_charge=net_charge,
        **kwargs,
    )
