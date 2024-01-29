## XequiNet
XequiNet is an equivariant graph neural network for predicting properties of chemical molecules or periodical systems.

## Requirements
**The following versions of these packages are only recommended for use.**

python 3.9.17<br>
cuda 11.7<br>
pytorch 2.0.1<br>
pyg 2.3.1<br>
pytorch-cluster 1.6.1<br>
pytorch-scatter 2.1.1<br>
e3nn 0.5.1<br>
pytorch-warmup 0.1.1<br>
pydantic 1.10.8 (recommand this version)<br>
pyscf 2.3.0<br>

**The following packages are only required when using delta learning with GFN2-xTB as base.**

tblite 0.3.0<br>
tblite-python 0.3.0<br>

**The ASE package is used for periodic boundary conditions and molecular dynamics.**
ase 3.22.1<br>

## Setups
### From Source
Once the requirements are installed, running
```
pip install -e .
```

## Usage
### Prepare dataset
Our dataset is storing with HDF5 in the following structure:
```
MyDataset.hdf5
├── train
│   ├── molecule1
|   |   ├── atomic_numbers
|   |   ├── coordinates_A / coordinates_bohr
|   |   ├── properties1_{unit}
|   |   ├── properties2_{unit}
|   |   ├── properties3_{unit}
|   |   └── ...
|   ├── molecule2
|   |   └── ...
|   |
|   ├── ...
|   |
|   └── moleculeX
|       └── ...
├── valid
|   └── ...
└── test
      └── ...
```
The dataset is divided into `train`, `valid` and `test`, and each containing their respective molecule subgroups. During the training process, `test` group is not necessarily required.

Each molecule subgroup contain following datasets. `N` is the number of atoms in the molecule and `M` is the number of conformations.

- `atomic_numbers`: `shape=(N,), dtype=np.uint8`. Atomic number of each atom.

- `coordinates_A` / `coordinates_bohr`: `shape=(M, N, 3), dtype=np.float64/np.float32`. Cartesian coordinates for every conformation. `A` for Angstrom and `bohr` for Bohr.

- `properties_{unit}`: Properties of the molecule in "unit".

    e.g.

    `E_wb97x_def2tzvp_Ha`: `shape=(M,), dtype=np.float64`. Electron energies for every conformation calculated at ωB97X/def2-TZVP in Hartree.

    `Hf_gfn2-xtb_kcal`: `shape=(M,), dtype=np.float64`. Heat of formations for every conformation calculated at GFN2-xTB in kcal/mol. (`kcal` refers to kcal/mol)

Due to the additional files may generated during data processing, your dataset directory should be set like this:
```
mydata
├── raw
│   └── mydata.hdf5
└── processed
    └──
```


### Training
Firstly, set the training configuration file `config.json` in working directory as follows. Details can be viewed in the file `xequinet/utils/config.py`.
```
{
    "run_name": "run_name"
    "node_dim": 128,
    "edge_irreps": "128x0e + 64x1e + 32x2e",
    "num_basis": 20,
    "action_blocks": 3,
    "output_mode": "scalar",
    "output_dim": 1,
    ...
}
```
Then simply run the following command according to your GPUs and port.
```
torchrun --nproc_per_node=${n_gpu} --master_port=${port} --no-python xeqtrain --config config.json
```
During training, `loss.log` , `run_name_k.pt` and `run_name_last.pt` will be automatically generated, which records loss information and net parameters respectively.

### Test
Similarily, prepare the dataset and configuration file `config.json`. Run
```
xeqtest --config config.json --ckpt run_name_0.pt
```
Test result will be recorded in `run_name_test.log`.

### Inference
Write the molecules you need to predict into the a standard xyz file `inf_mol.xyz` like:
```
5
methane
C   0.00000000    0.00000000    0.00000000
H   0.00000000    0.00000000    1.06999996
H  -0.00000000   -1.00880563   -0.35666665
H  -0.87365130    0.50440282   -0.35666665
H   0.87365130    0.50440282   -0.35666665
3
water
O   0.00000000   -0.00000000   -0.11081188
H   0.00000000   -0.78397589    0.44324751
H  -0.00000000    0.78397589    0.44324751
...
```
Run
```
xeqinfer --ckpt run_name_k.pt inf_mol.xyz
```
the prediction result will be writen in `inf_mol.log`.

### JIT script
Finally you can jit compile the model for cross platform tasks by running
```
xeqjit --ckpt run_name_k.pt
```
You will get a jit compiled model `run_name_k.jit`.