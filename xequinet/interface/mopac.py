from typing import Iterable, Union
import os
import subprocess
import numpy as np

# from ..utils import unit_conversion
from xpainn.utils import unit_conversion


class MOPAC:
    """
    MOPAC interface for calculating single-point energy and force.
    """
    def __init__(
        self,
        atoms: Iterable[Union[int, str]],
        coordinates: Iterable[float],
        charge: int = 0,
        multiplicity: int = 1,
        method: str = "PM7",
        calc_force: bool = False,
        name: str = "mopac",
        scratch: str = "./",
        keywords: str = "",
    ):
        """
        Args:
            `atoms`: Iterable of atoms.
            `coordinates`: Iterable of coordinates.
            `charge`: Charge of the molecule.
            `multiplicity`: Multiplicity of the molecule.
            `method`: Method of the calculation.
            `calc_force`: Whether calculate the force.
            `name`: Name of the mopac files (.mop, .out, .arc, etc.).
            `scratch`: Scratch directory.
            `keywords`: Other keywords of the mopac calculation.
        """
        self.atoms = atoms
        self.coords = coordinates
        assert len(atoms) == len(coordinates)
        self.charge = charge
        self.multi = multiplicity
        self.method = method
        self.calc_force = calc_force
        self.name = name
        self.keywords = keywords
        self.scratch = scratch
        os.makedirs(self.scratch, exist_ok=True)

        self.final_hof = None  # kcal/mol
        self.force = None      # kcal/mol/angstrom

    def _write_input(self):
        s = f"{self.method}"
        if self.calc_force:
            s += " FORCE NOREOR"
        if self.charge != 0:
            s += f"CHARGE={self.charge} "
        if self.multi != 1:
            multi_dict = {2: "DOUBLET", 3: "TRIPLET", 4: "QUARTET", 5: "QUNITET"}
            s += f"{multi_dict[self.multi]} UHF "
        s += f"{self.keywords}"
        
        s += "\nTitle\n\n"

        for atom, coord in zip(self.atoms, self.coords):
            s += "{:2} {:15.10f} 0 {:15.10f} 0 {:15.10f} 0\n".format(atom, *coord)

        with open(f"{self.scratch}/{self.name}.mop", 'w') as wf:
            wf.write(s)

    def _read_output(self):
        out_file = f"{self.name}.out"
        if not os.path.isfile(f"{self.scratch}/{out_file}"):
            raise FileNotFoundError(f"No such file: '{out_file}'")
        # read heat of formation
        line = subprocess.check_output(["grep", "HEAT OF FORMATION", out_file], cwd=self.scratch)
        line = str(line, "utf-8")
        self.final_hof = float(line.split()[-2])
        # read force
        if self.calc_force:
            lines = subprocess.check_output(
                ["grep", "-A", str(len(self.atoms) + 3), "CARTESIAN COORDINATE DERIVATIVES", out_file],
                cwd=self.scratch
            )
            lines = str(lines, "utf-8").strip().split("\n")[4:]
            force = [l.strip().split()[2:5] for l in lines]
            self.force = np.array(force, dtype=np.float32)

    def calculate(self, clean=False):
        self._write_input()
        subprocess.call(f"mopac {self.name}.mop > /dev/null", shell=True, cwd=self.scratch)
        self._read_output()
        if clean:
            os.remove(os.path.join(self.scratch, f"{self.name}.mop"))
            os.remove(os.path.join(self.scratch, f"{self.name}.out"))
            if os.path.exists(os.path.join(self.scratch, f"{self.name}.arc")):
                os.remove(os.path.join(self.scratch, f"{self.name}.arc"))

    def get_final_heat_of_formation(self, unit="AU", clean=False):
        if self.final_hof is None:
            self.calculate(clean=clean)
        return self.final_hof * unit_conversion("kcal_per_mol", unit)
    
    def get_force(self, unit="AU", clean=False):
        if self.force is None:
            self.calculate(clean=clean)
        return self.force * unit_conversion("kcal_per_mol/Angstrom", unit)
    

if __name__ == "__main__":
    atoms = ['O', 'H', 'H']
    coords = [
        [0.0, 0.0, 0.11930800],
        [0.0, 0.75895300, -0.47723200],
        [0.0, -0.75895300, -0.47723200],
    ]
    mopac = MOPAC(atoms, coords, charge=0, multiplicity=1, method="PM7", calc_force=True, name="mopac", scratch="./")
    mopac.calculate(clean=True)
    print(mopac.get_final_heat_of_formation(unit="hartree"))
    print(mopac.get_force(unit="hartree/bohr"))
