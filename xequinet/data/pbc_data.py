import torch
import ase
import ase.neighborlist
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.data import Dataset as DiskDataset



if __name__ == "__main__":
    si_lattice = torch.tensor([
        [1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.]
    ])
    si_coords = torch.tensor([
        [0.      , 0.      , 0.      ]
    ])
    si_types = ['Cu']

    si = ase.Atoms(symbols=si_types, positions=si_coords, cell=si_lattice, pbc=True)
    i, j, d, D, S = ase.neighborlist.neighbor_list("ijdDS", a=si, cutoff=1.1)
    print(i)
    print(j)
    print(d)
    print(D)
    print(S)

