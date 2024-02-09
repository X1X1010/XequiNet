## Prepare dataset
Dataset for Xequinet is storing with HDF5 in the following structure:
```
MyDataset.hdf5
├── train
│   ├── group1 (molecule)
|   |   ├── atomic_numbers
|   |   ├── coordinates_A / coordinates_bohr
|   |   ├── property1
|   |   ├── property2
|   |   └── ...
│   ├── group2 (pbc)
|   |   ├── atomic_numbers
|   |   ├── pbc
|   |   ├── lattice_A / lattice_bohr
|   |   ├── coordinates_A / coordinates_bohr / coordinates_frac
|   |   ├── property1
|   |   ├── property2
|   |   └── ...
|   |
|   ├── ...
|   |
|   └── groupX
|       └── ...
|
├── valid
|   └── ...
|
└── test
      └── ...
```
The dataset is divided into `train`, `valid` and `test`, and each containing their respective molecule or pbc groups. During the training process, `test` group is not necessarily required.

Each group contain following datasets, while the names of the groups are not important. `N` is the number of atoms in the system and `M` is the number of conformations.

**The following key names are fixed**
- `"atomic_numbers"`: `(N,)`, `uint8`. Atomic number of each atom.
- `"coordinates_A" | "coordinates_bohr" | "coordinates_frac"`: `(M, N, 3)`, `float | double`. Cartesian coordinates for every conformation. `A` is for Ångstrom, `bohr` is for Bohr and `frac` means fractional coordinates of crystals.
- `"pbc"`: `(3,) | ()`, `bool`. Periodic boundary conditions. Whether there is a periodicity (in the three directions).
- `"lattice_A" | "lattice_bohr"`: `(3, 3)`, `float | double`. Lattice vectors. `A` is for Ångstrom and `bohr` is for Bohr.

Note that `"coordinates_frac"`, `"pbc"` and `"lattice_A" | "lattice_bohr"` are only used when training periodic systems.

**The property key names can be costumed**<br>
e.g.
- `"E_wb97x_def2tzvp_Ha"`: `(M,)`, `double`. Electron energies calculated under ωB97X/def2-TZVP in Hartree.
- `"F_gfn2-xtb_AU"`: `(M, N, 3)`, `double`. Nuclear forces calculated under GFN2-xTB in a.u.

Due to the additional files may generated during data processing by `torch_geometric`, you may set your dataset directory like this as well:
```
mydata
├── raw
│   └── mydata.hdf5
└── processed
    └── ...
```

### Training
Firstly, set the training configuration file `config.json` in working directory as follows. Details can be viewed in the file `xequinet/utils/config.py`.
```
{
    "run_name": "qm9"

    "node_dim": 128,
    "edge_irreps": "128x0e + 64x1o + 32x2e",
    "num_basis": 20,
    "action_blocks": 3,
    "output_mode": "scalar",
    ...
}
```
Then simply run the following command according to your GPUs and port.
```
torchrun --nproc_per_node=${n_gpu} --master_port=${port} --no-python xeqtrain --config config.json
```
`--config` can be abbreviated to `-C`. If you don't specify it, it will read `config.json` by default.

During training, `loss.log` , `run_name_k.pt` and `run_name_last.pt` will be automatically generated, which records loss information and net parameters respectively.

### Test
Similarily, prepare the dataset and configuration file `config.json`. Run
```
xeqtest --config config.json --ckpt run_name_0.pt
```
You can obtain the detailed arguments by
```
xeqtest -h
```
Test result will be recorded in `run_name_test.log`.