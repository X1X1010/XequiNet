from typing import Dict, List, Optional

import torch
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress

from xequinet import keys
from xequinet.data import (
    DataTypeTransform,
    NeighborTransform,
    SequentialTransform,
    Transform,
    datapoint_from_ase,
)
from xequinet.nn import resolve_model
from xequinet.utils import get_default_units, set_default_units, unit_conversion


class XequiCalculator(Calculator):
    """
    ASE calculator for XequiNet models.
    """

    implemented_properties = ["energy", "energies", "forces", "stress"]
    default_parameters = {
        "ckpt_file": "model.pt",
        "dtype": "float32",
        "device": None,
    }
    atoms: Atoms

    def __init__(self, **kwargs) -> None:
        self.dtype = torch.float32
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[torch.ScriptModule] = None
        self.transform: Optional[Transform] = None
        Calculator.__init__(self, **kwargs)

    def set(self, **kwargs) -> None:
        changed_parameters = Calculator.set(self, **kwargs)
        if changed_parameters:
            self.reset()
        if "dtype" in changed_parameters:
            dtype_map = {
                "float16": torch.float16,
                "float32": torch.float32,
                "float64": torch.float64,
            }
            self.dtype = dtype_map[self.parameters.dtype]
            torch.set_default_dtype(self.dtype)
        if "device" in changed_parameters:
            self.device = self.parameters.device
        if "ckpt_file" in changed_parameters or self.model is None:
            device = torch.device(self.device)
            ckpt = torch.load(self.parameters.ckpt_file, map_location=device)
            model_config = ckpt["config"]
            set_default_units(model_config["default_units"])
            self.model = (
                resolve_model(
                    model_config["model_name"],
                    **model_config["model_kwargs"],
                )
                .to(device=device)
                .eval()
            )
            self.transform = SequentialTransform(
                [
                    DataTypeTransform(self.dtype),
                    NeighborTransform(self.model.cutoff_radius),
                ]
            )

    def calculate(
        self,
        atoms: Atoms = None,
        properties: Optional[List[str]] = None,
        system_changes: List[str] = all_changes,
    ) -> None:
        if properties is None:
            properties = self.implemented_properties

        Calculator.calculate(self, atoms, properties, system_changes)

        atoms.wrap()
        data = datapoint_from_ase(self.atoms).to(self.device)
        data = self.transform(data).to_dict()
        compute_forces = "forces" in properties
        compute_virial = "stress" in properties
        result: Dict[str, torch.Tensor] = self.model(
            data,
            compute_forces,
            compute_virial,
        )
        default_units = get_default_units()
        self.results["energy"] = result[keys.TOTAL_ENERGY].item() * unit_conversion(
            default_units[keys.TOTAL_ENERGY], "eV"
        )
        self.results["energies"] = result[
            keys.ATOMIC_ENERGIES
        ].detach().cpu().numpy() * unit_conversion(
            default_units[keys.TOTAL_ENERGY], "eV"
        )
        if compute_forces:
            self.results["forces"] = result[
                keys.FORCES
            ].detach().cpu().numpy() * unit_conversion(
                default_units[keys.FORCES], "eV/Angstrom"
            )
        if compute_virial:
            assert self.atoms.cell.rank == 3
            virial = result[keys.VIRIAL].detach().cpu().numpy() * unit_conversion(
                default_units[keys.TOTAL_ENERGY], "eV"
            )
            self.results["stress"] = (
                full_3x3_to_voigt_6_stress(virial) / self.atoms.get_volume()
            )
