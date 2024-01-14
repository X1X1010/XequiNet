from typing import Dict
import numpy as np
import torch

from ase.neighborlist import neighbor_list
from ase.stress import full_3x3_to_voigt_6_stress
from ase.calculators.calculator import Calculator, all_changes



class XeqCalculator(Calculator):
    """
    ASE calculator for XequiNet models.

    Args:
    """
    implemented_properties = ["energy", "energies", "forces"]
    implemented_properties += ["stress", "stresses"]
    default_parameters = {
        "ckpt_file": "model.jit",
        "device": torch.device("cpu"),
        "cutoff": 5.0,
        "charge": 0,
        "spin": 0,
    }

    def __init__(self, **kwargs):
        self.model = None
        Calculator.__init__(**kwargs)
    
    def set(self, **kwargs):
        changed_parameters = Calculator.set(self, **kwargs)
        if changed_parameters:
            self.reset()
        if "ckpt_file" in changed_parameters or self.model is None:
            self.model = torch.jit.load(self.ckpt_file, map_location=self.device)


    def calculate(
        self,
        atoms=None,
        properties=None,
        system_changes=all_changes,
    ):
        if properties is None:
            properties = self.implemented_properties

        Calculator.calculate(self, atoms, properties, system_changes)

        positions = self.atoms.positions
        cell = self.atoms.cell
        idx_i, idx_j, offsets = neighbor_list("ijS", self.atoms, self.cutoff)
        shifts = torch.from_numpy(np.einsum("ij, kj -> ki", offsets, cell)).to(self.device)
        edge_index = torch.tensor([idx_i, idx_j], dtype=torch.long, device=self.device)
        at_no = torch.tensor(self.atoms.numbers, dtype=torch.long, device=self.device)
        coord = torch.from_numpy(positions).to(self.device)
        model_res: Dict[str, torch.Tensor] = self.model(
            at_no=at_no, coord=coord,
            edge_index=edge_index, shifts=shifts,
            charge=self.charge, spin=self.spin
        )
        self.results["energy"] = model_res.get("energy").item()
        self.results["energies"] = model_res.get("energies").cpu().numpy()
        self.results["forces"] = model_res.get("forces").cpu().numpy()
        virials = model_res.get("virials").cpu().numpy()
        virial = model_res.get("virial").cpu().numpy()
        self.results["stress"] = full_3x3_to_voigt_6_stress(virial) / self.atoms.get_volume()
        self.results["stresses"] = full_3x3_to_voigt_6_stress(virials) / self.atoms.get_volume()
