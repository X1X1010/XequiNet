from typing import Dict, List, Union

import torch

from xequinet import keys
from xequinet.nn.model import BaseModel, XPaiNN, compute_edge_data
from xequinet.utils import get_default_units, unit_conversion


class XPaiNNFF(XPaiNN):
    """
    XPaiNN model for JIT script. This model does not consider batch.
    """

    def __init__(self, **kwargs) -> None:
        output_modes: Union[str, List[str]] = kwargs["output_modes"]
        assert output_modes[-1] == "forcefield"
        super().__init__(**kwargs)
        default_units = get_default_units()
        energy_unit = default_units[keys.TOTAL_ENERGY]
        pos_unit = default_units[keys.POSITIONS]

        self.pos_unit_factor = unit_conversion("Angstrom", pos_unit)
        self.energy_unit_factor = unit_conversion(energy_unit, "eV")
        self.forces_unit_factor = unit_conversion(
            f"{energy_unit}/{pos_unit}", "eV/Angstrom"
        )
        self.virial_unit_factor = unit_conversion(
            f"{energy_unit}/{pos_unit}^3", "eV/Angstrom^3"
        )

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
        for body_block in self.body_blocks:
            data = body_block(data)
        result = {}
        for output_block in self.output_blocks:
            data, output = output_block(data, compute_forces, compute_virial)
            result.update(output)

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
