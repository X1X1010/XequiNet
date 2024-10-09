from typing import Dict, Tuple
import socket
import numpy as np
import torch
from ase.io import read as ase_read
from ipi._driver.driver import (
    HDRLEN,
    Message,
    recv_data,
    send_data,
)

from ..utils import radius_graph_pbc, unit_conversion


class iPIDriver:
    """
    Interface for the i-PI driver.
    """

    def __init__(
        self,
        ckpt_file: str,
        init_file: str,
        address: str = "localhost",
        port: int = 31415,
        verbose: int = 0,
        **kwargs,
    ) -> None:
        # set verbosity
        self.verbose = verbose
        # Opens a socket to i-PI
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.settimeout(10)
        try:
            self.socket.connect((address, port))
            self.socket.settimeout(None)
        except socket.timeout:
            raise TimeoutError("Timeout when connecting to i-PI server.")
        # set flag
        self.f_init = False
        self.f_data = False
        # read the initial structure file
        at_no = self.get_atomic_numbers(init_file)
        self.natoms = at_no.shape[0]
        self.at_no = torch.LongTensor(at_no)
        # initialize structure array
        self.cell = np.zeros((3, 3), float)
        self.icell = np.zeros((3, 3), float)
        self.coord = np.zeros((self.natoms, 3), float)
        # initialize return arrays
        self.energy = 0.0
        self.forces = np.zeros((self.natoms, 3), float)
        self.virial = np.zeros((3, 3), float)

        # parameters for the model
        self.pbc = kwargs.get("pbc", False)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.jit.load(ckpt_file, map_location=self.device)
        self.cutoff = kwargs.get("cutoff", 5.0)
        self.max_edges = kwargs.get("max_edges", 100)
        self.charge = kwargs.get("charge", 0)
        if "multiplicity" in kwargs:
            self.spin = kwargs["multiplicity"] - 1
        else:
            self.spin = kwargs.get("spin", 0)

    def get_atomic_numbers(self, file: str) -> np.ndarray:
        atoms = ase_read(file)
        return atoms.get_atomic_numbers()

    def calculate(self) -> Tuple[float, np.ndarray, np.ndarray]:
        """Compute the energy, forces, and virial."""
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
        at_no = self.at_no.to(self.device)
        coord = (
            torch.from_numpy(positions).to(torch.get_default_dtype()).to(self.device)
        )
        model_res: Dict[str, torch.Tensor] = self.model(
            at_no=at_no,
            coord=coord,
            edge_index=edge_index,
            shifts=shifts,
            charge=self.charge,
            spin=self.spin,
        )
        # convert the units from the MD model's unit to a.u.
        energy = model_res["energy"].item() * unit_conversion("eV", "Hartree")
        forces = model_res["forces"].detach().cpu().numpy() * unit_conversion(
            "eV/Angstrom", "AU"
        )
        virial = model_res.get("virial", None)
        if virial is not None:
            virial = virial.detach().cpu().numpy() * unit_conversion("eV", "Hartree")
        return energy, forces, virial

    def init(self) -> None:
        """Deal with message from `INIT` motion."""
        rid = recv_data(self.socket, np.int32())
        initlen = recv_data(self.socket, np.int32())
        initstr = recv_data(self.socket, np.chararray(initlen))
        if self.verbose > 0:
            print(rid, initstr)
        self.f_init = True

    def status(self) -> None:
        """Reply `STATUS`."""
        if not self.f_init:
            self.socket.sendall(Message("NEEDINIT"))
        elif self.f_data:
            self.socket.sendall(Message("HAVEDATA"))
        else:
            self.socket.sendall(Message("READY"))

    def posdata(self) -> None:
        """Receive `POSDATA` and calculate."""
        # receives structural information
        self.cell = recv_data(self.socket, self.cell)
        self.icell = recv_data(
            self.socket, self.icell
        )  # inverse of the cell. mostly useless legacy stuff
        natom = recv_data(self.socket, np.int32())
        assert self.natoms == natom, "Number of atoms does not match!"
        self.coord = recv_data(self.socket, self.coord)
        # calculate the energy, forces, and virial
        energy, forces, virial = self.calculate()
        self.energy = energy
        self.forces = forces
        self.virial = virial
        # set the flag
        self.f_data = True

    def getforce(self) -> None:
        """Reply `GETFORCE`."""
        self.socket.sendall(Message("FORCEREADY"))
        send_data(self.socket, np.float64(self.energy))
        send_data(self.socket, np.int32(self.natoms))
        send_data(self.socket, self.forces.astype(np.float64))
        send_data(self.socket, self.virial.astype(np.float64))
        extra = "mamba out"  # what can I say?
        send_data(self.socket, np.int32(len(extra)))
        self.socket.sendall(extra.encode("utf-8"))
        self.f_data = False

    def exit(self) -> None:
        """Exit the driver."""
        self.socket.close()
        raise SystemExit("Received exit message from i-PI. Bye bye!")

    def parse(self) -> None:
        """Reply the request from server."""
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
            self.exit()
