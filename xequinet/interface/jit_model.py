from typing import Dict

import torch

from xequinet import keys
from xequinet.nn.basic import compute_edge_data, compute_properties
from xequinet.nn.model import BaseModel, XPaiNN, XPaiNNEwald
from xequinet.utils import get_default_units, unit_conversion


def get_ff_unit_factor() -> float:
    """from default units to LAMMPS metal units"""
    default_units = get_default_units()
    energy_unit = default_units[keys.TOTAL_ENERGY]
    pos_unit = default_units[keys.POSITIONS]
    pf = unit_conversion("Angstrom", pos_unit)
    ef = unit_conversion(energy_unit, "eV")
    ff = unit_conversion(f"{energy_unit}/{pos_unit}", "eV/Angstrom")
    vf = unit_conversion(f"{energy_unit}/{pos_unit}^3", "eV/Angstrom^3")
    return pf, ef, ff, vf


class XPaiNNFF(XPaiNN):
    """
    XPaiNN model for JIT script. This model does not consider batch.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        (
            self.pos_unit_factor,
            self.energy_unit_factor,
            self.forces_unit_factor,
            self.virial_unit_factor,
        ) = get_ff_unit_factor()

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

        # we cannot use `super().forward` because it is nor supported by TorchScript
        # so we manually call the forward method of the parent class
        data = compute_edge_data(
            data=data,
            compute_forces=compute_forces,
            compute_virial=compute_virial,
        )
        for module in self.module_list:
            data = module(data)
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
            result[keys.VIRIAL] *= self.virial_unit_factor
        return result


@torch.jit.script
def pos_svd_frame(pos: torch.Tensor) -> torch.Tensor:
    """
    Args:
        `pos`: Atomic positions.
    Returns:
        Atomic positions in the SVD frame.
    """
    pos = pos - pos.mean(dim=0, keepdim=True)
    if pos.shape[0] <= 2:
        return pos
    u, s, v = torch.linalg.svd(pos, full_matrices=True)
    return pos @ v


class XPaiNNEwaldFF(XPaiNNEwald):
    """
    XPaiNNEwald model for JIT script. This model does not consider batch.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        (
            self.pos_unit_factor,
            self.energy_unit_factor,
            self.forces_unit_factor,
            self.virial_unit_factor,
        ) = get_ff_unit_factor()
        self.use_pbc = kwargs.get("use_pbc", False)

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

        if not self.use_pbc:
            assert compute_virial is False, "Virial is not supported without PBC"
            # use SVD frame for non-periodic systems
            data[keys.POSITIONS] = pos_svd_frame(data[keys.POSITIONS])

        # we cannot use `super().forward` because it is nor supported by TorchScript
        # so we manually call the forward method of the parent class
        data = compute_edge_data(
            data=data,
            compute_forces=compute_forces,
            compute_virial=compute_virial,
        )
        for module in self.module_list:
            data = module(data)
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
            result[keys.VIRIAL] *= self.virial_unit_factor
        return result


def resolve_jit_model(model_name: str, **kwargs) -> BaseModel:
    models_factory = {"xpainn": XPaiNNFF}
    if model_name.lower() not in models_factory:
        raise NotImplementedError(f"Unsupported model {model_name}")
    return models_factory[model_name.lower()](**kwargs)
