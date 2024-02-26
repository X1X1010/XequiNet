from typing import Dict
import torch

from ase.stress import full_3x3_to_voigt_6_stress
from ase.calculators.calculator import Calculator, all_changes

from ..utils import radius_graph_pbc


class XeqCalculator(Calculator):
    """
    ASE calculator for XequiNet models.
    """
    implemented_properties = ["energy", "energies", "forces"]
    implemented_properties += ["stress", "stresses"]
    default_parameters = {
        "ckpt_file": "model.jit",
        "cutoff": 5.0,
        "max_edges": 100,
        "charge": 0,
        "spin": 0,
    }

    def __init__(self, **kwargs) -> None:
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        Calculator.__init__(self, **kwargs)
    
    def set(self, **kwargs) -> None:
        changed_parameters = Calculator.set(self, **kwargs)
        if changed_parameters:
            self.reset()
        if "ckpt_file" in changed_parameters or self.model is None:
            self.model = torch.jit.load(self.parameters.ckpt_file, map_location=self.device)


    def calculate(
        self,
        atoms=None,
        properties=None,
        system_changes=all_changes,
    ) -> None:
        if properties is None:
            properties = self.implemented_properties

        Calculator.calculate(self, atoms, properties, system_changes)
        positions = self.atoms.positions
        edge_index, shifts = radius_graph_pbc(
            positions=positions,
            pbc=self.atoms.pbc,
            cell=self.atoms.cell,
            cutoff=self.parameters.cutoff,
            max_num_neighbors=self.parameters.max_edges,
        )
        edge_index = torch.from_numpy(edge_index).to(torch.long).to(self.device)
        shifts = torch.from_numpy(shifts).to(torch.get_default_dtype()).to(self.device)
        at_no = torch.from_numpy(self.atoms.numbers).to(torch.long).to(self.device)
        coord = torch.from_numpy(positions).to(torch.get_default_dtype()).to(self.device)
        model_res: Dict[str, torch.Tensor] = self.model(
            at_no=at_no, coord=coord,
            edge_index=edge_index, shifts=shifts,
            charge=self.parameters.charge,
            spin=self.parameters.spin,
        )
        self.results["energy"] = model_res.get("energy").item()
        self.results["energies"] = model_res.get("energies").detach().cpu().numpy()
        self.results["forces"] = model_res.get("forces").detach().cpu().numpy()
        # no lattice, no stress
        if self.atoms.cell.rank == 3:
            virials = model_res.get("virials").detach().cpu().numpy()
            virial = model_res.get("virial").detach().cpu().numpy()
            self.results["stress"] = full_3x3_to_voigt_6_stress(virial) / self.atoms.get_volume()
            self.results["stresses"] = full_3x3_to_voigt_6_stress(virials) / self.atoms.get_volume()
