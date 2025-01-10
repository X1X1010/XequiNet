# Dataset
## Datapoint class

`N` for number of atoms.

Following is the input of the model.
| Name | Type | Shape | Description |
| - | - | - | - |
| `atomic_numbers` | `int` | `[N,]` | Atomic number in periodic table |
| `pos` | `float`, `double` | `[N,3]` | Cartesian coordinate of atoms |
| `cell` | `float`, `double` | `[1,3,3]` | Lattice vector (optional) |
| `charge` | `float`, `double` | `[1,]` | Net charge (optional) |
| `spin` | `float`, `double` | `[1,]` | Total spin (2S or M-1) (optional) |

Following is the label for training.
| Name | Type | Shape | Description |
| - | - | - | - |
| `energy` | `float`, `double` | `[1,]` | Total energy of the system. Can be subtracted by atomic energies for better performance. |
| `forces` | `float`, `double` | `[N,3]` | Negative values of derivative of energy with respect to nuclear coordinates |
| `base_energy` | `float`, `double` | `[1,]` | Total Energy of low level method for delta-learning |
| `base_forces` | `float`, `double` | `[N,3]` | Forces of low level method for delta-learning |
| `virial` | `float`, `double` | `[1,3,3]` | Negative values of derivative of energy with respect to cell strain (stress * volume, be careful with sign) |
| `atomic_charges` | `float`, `double` | `[N,]` | Charges on atoms. Can be charge population |
| `dipole` | `float`, `double` | `[1,3]` | Dipole moment vector |
| `polar` | `float`, `double` | `[1,3,3]` | Poloarizability tensor |
| `new_label` | | | Create new labels as you wish |

## Prepare dataset
A complete dataset for Xequinet is consists of three parts: `data.lmdb`, `info.json`, `<split>.json`

### LMDB file:`data.lmdb`
The reason for choosing lmdb is its support for concurrent reads. Each key in `data.lmdb` is corresponded to an `XequiData`. Here is an example to generate an LMDB data file.

```python
import lmdb
import pickle
from xequinet.data import XequiData

lmdb_data = lmdb.open(
    path="data.lmdb",
    map_size=2**40,  # Here is 1 TB. You can set other size suitable for your dataset
    subdir=False,
    sync=False,
    writemap=False,
    meminit=False,
    map_async=True,
    create=True,
    readonly=False,
    lock=True,
)

for i in range(<dataset_len>):
    # all the components are torch.Tensor
    # components of float points must be consistent (dtype should be the same)
    datapoint = XequiData(
        atomic_numbers=...  # [n_atoms,] int
        pos=...  # [n_atoms, 3] float/double
        pbc=...  # [1, 3]  bool (optional)
        cell=...  # [1, 3, 3]  float/double (optional)
        energy=...  # [1,]  float/double
        forces=...  # [n_atoms, 3]  float/double
        virial=...  # [1, 3, 3]  float/double
        xxx=...  # any other property is OK
    )

    with lmdb_data.begin(write=True) as txn:
        key = i.to_bytes(8, byteorder="little")
        txn.put(key, pickle.dumps(datapoint))
    ...
```
Theoretically it's possible to write concurrently, but I haven't tried it yet.

By the way, it is highly recommended to record some information about the data in `csv` or other format, so that it will be conventient for dataset splitting or data filtering. e.g.
| | Formula | Subset | Energy | Charge | ... |
| - | - | - | - | - | - |
| 0 | CH4 | monomer | -3.054 | 0 | |
| 1 | H2O | monomer | -3.032 | 0 | |
| 2 | H3O+ | ion | -3.033 | 1 | |
|... | | | | | |

### Info file: `info.json`
This file is mainly used for saving the information of dataset, especially the **UNIT** of each item. e.g.

```json
{
  "units": {
    "energy": "eV",
    "pos": "Ang",
    "forces": "eV/Ang"
  },
}
```
You can also record additional information for everyone's use. e.g.
```json
{
  "units": {
    "energy": "eV",
    "pos": "Ang",
    "forces": "eV/Ang"
  },
  "method": "XYGJ-OS/cc-pVQZ",
  "atomic_energies": {
    "H": -0.5,
    "C": -37.8,
    "O": -75.1
  }
}
```
Be careful with the json format.

### Split file: `<split>.json`
This file can be named whatever you want. It contained the indices of the train, validation and test set. e.g.
```json
{
  "trian": [0, 1, 2, ...],
  "valid": [3, 4, 5, ...],
  "test": [6, 7, 8, ...]
}
```
You can use this to divide the dataset, or just use a small subset of it to train.
