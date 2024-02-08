from typing import Dict
import socket
import struct
import numpy as np
import torch

from ..utils import radius_graph_pbc, unit_conversion

# number of bytes
INT_BYTES = 4
DOUBLE_BYTES = 8

# header length
HDRLEN = 12
def Message(mystr):
    """Returns a header of standard length HDRLEN."""

    # convert to bytestream since we'll be sending this over a socket
    return str.ljust(str.upper(mystr), HDRLEN).encode()


class iPIDriver:
    """
    Interface for the i-PI driver.
    """
    def __init__(
        self,
        ckpt_file: str,
        address: str = "localhost",
        port: int = 31415,
        **kwargs,
    ):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.settimeout(10)
        try:
            self.socket.connect((address, port))
            self.socket.settimeout(None)
        except socket.timeout as e:
            raise RuntimeError("Connection to i-PI failed") from e
        self.ifInit = False
        self.ifForce = False
        self.cell: np.ndarray = None
        self.inverse: np.ndarray = None
        self.coord: np.ndarray = None
        self.energy: float = None
        self.forces: np.ndarray = None
        self.virials: np.ndarray = None
        self.extra = ""
        self.nbead = -1
        self.natoms = -1
        # parameters for the model
        self.pbc = kwargs.get("pbc", False)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.jit.load(ckpt_file, map_location=self.device)
        self.cutoff = kwargs.get("cutoff", 5.0)
        self.max_edges = kwargs.get("max_edges", 100)
        self.charge = kwargs.get("charge", 0)
        self.spin = kwargs.get("multiplicity", 1) - 1


    def calculate(self):
        """Compute the energy, forces, and virials."""
        # convert the units from a.u. to the MD model's unit
        positions = self.coord * unit_conversion("Bohr", "Angstrom")
        lattice = self.cell * unit_conversion("Bohr", "Angstrom")
        edge_index, shifts = radius_graph_pbc(
            positions=positions,
            pbc=self.pbc,
            cell=lattice,
            cutoff=self.cutoff,
            max_num_neighbors=self.max_edges,
        )
        edge_index = torch.from_numpy(edge_index).to(torch.long).to(self.device)
        shifts = torch.from_numpy(shifts).to(torch.get_default_dtype()).to(self.device)
        at_no = torch.from_numpy(self.atoms.numbers).to(torch.long).to(self.device)
        coord = torch.from_numpy(positions).to(torch.get_default_dtype()).to(self.device)
        model_res: Dict[str, torch.Tensor] = self.model(
            at_no=at_no, coord=coord,
            edge_index=edge_index, shifts=shifts,
            charge=self.charge,
            spin=self.spin,
        )
        # convert the units from the MD model's unit to a.u.
        energy = model_res["energy"].item() * unit_conversion("eV", "Hartree")
        forces = model_res["forces"].cpu().numpy() * unit_conversion("eV/Angstrom", "AU")
        virials = model_res.get("virials", None)
        if virials is not None:
            virials = virials.cpu().numpy() * unit_conversion("eV", "Hartree")
        return energy, forces, virials


    def update(self, text: str):
        """Update system message from INIT motion."""
        pass


    def init(self):
        """Deal with message from INIT motion."""
        self.nbead = np.frombuffer(
            self.socket.recv(INT_BYTES * 1), dtype=np.int32,
        )[0]
        offset = np.frombuffer(
            self.socket.recv(INT_BYTES * 1), dtype=np.int32,
        )[0]
        self.update(self.socket.recv(offset).decode())
        self.ifInit = True


    def status(self):
        """Reply STATUS."""
        if self.ifInit and not self.ifForce:
            self.socket.send(Message("ready"))
        elif self.ifForce:
            self.socket.send(Message("havedata"))
        else:
            self.socket.send(Message("needinit"))


    def posdata(self):
        """Read position data."""
        self.cell = np.frombuffer(
            self.socket.recv(DOUBLE_BYTES * 9), dtype=np.float64,
        ).reshape(3, 3)
        self.inverse = np.frombuffer(
            self.socket.recv(DOUBLE_BYTES * 9), dtype=np.float64,
        ).reshape(3, 3)
        self.natom = np.frombuffer(
            self.socket.recv(INT_BYTES * 1), dtype=np.int32,
        )[0]
        self.coord = np.frombuffer(
            self.socket.recv(DOUBLE_BYTES * 3 * self.natom), dtype=np.float64,
        ).reshape(-1, 3)
        energy, forces, virials = self.calculate()
        self.energy = energy; self.forces = forces; self.virials = virials
        self.ifForce = True


    def getforce(self):
        """Reply GETFORCES."""
        self.socket.send(Message("forceready"))
        self.socket.send(struct.pack("d", self.energy))
        self.socket.send(struct.pack("i", self.natom))
        for f in self.forces.ravel():
            self.socket.send(struct.pack("d", f))
        if self.virials is None:
            self.virials = self.forces * self.coord
        for v in self.virials.ravel():
            self.socket.send(struct.pack("d", v))
        extra = self.extra if len(self.extra) > 0 else " "
        self.socket.send(struct.pack("i", len(extra)))
        self.socket.send(extra.encode())
        self.ifForce = False
    

    def run_driver(self):
        """Reply the request from server."""
        while True:
            header = self.socket.recv(HDRLEN)
            if header == Message("STATUS"):
                self.status()
            elif header == Message("INIT"):
                self.init()
            elif header == Message("POSDATA"):
                self.posdata()
            elif header == Message("GETFORCE"):
                self.getforce()
            elif header == Message("EXIT"):
                print("Received exit message from i-PI. Bye bye!")
                return