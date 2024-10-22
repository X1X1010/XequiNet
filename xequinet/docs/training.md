## Prepare dataset
A complete dataset for Xequinet is consists of three parts: `data.lmdb`, `info.json`, `<split>.json`

### `data.lmdb`
Each key in `data.lmdb` is corresponded to an `XequiData`. Here is an example to generate an LMDB data file.
```python
import lmdb
import pickle
from xequinet.data import XequiData

lmdb_data = lmdb.open(
    path="data.lmdb",
    map_size=2**40,  # Here is 1 TB. You can set any thing suitable for your dataset
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

    with data.begin(write=True) as txn:
        key = i.to_byte(8, byteorder="little")
        txn.put(key, pickle.dumps(datapoint))
    ...
```

### `info.json`
This is used to store some necessary infomation. especially the unit and name of the properties in the `data.lmdb`. Here is an example:
```json
{
  "method": "<functional>/<basis>",
  "units": {
    "energy": "Hartree",
      "pos": "Angstron",
      "forces": "a.u."
      ...
  },
  "atomic_energies": {
    "H": -0.5,
    "C": ...
  }
}
```
The most important part is "units", because the unit conversions in the model are based on this json file. Detailed infomation for valid unit type and format can be check in `xequinet/utils/qc`. In general, combinations of commonly used units are available.

### `<split>.json`
This file can be named whatever you want. It contained the indices of the train, validation and test set. For example:
```json
{
  "trian": [0, 1, 2, ...],
  "valid": [3, 4, 5, ...],
  "test": [6, 7, 8, ...]
}
```
You can use this to divide the dataset, or just use a small subset of it to train.

## Training
Firstly, set the training configuration file `config.yaml` in working directory as follows. Details can be viewed in the file `xequinet/utils/config.py`.
```yaml
model:
  model_name: xpainn
  model_kwargs:
    node_dim: 128
    node_irreps: 128x0e+64x1o+32x2e
    ...
  default_units:
    energy: eV
    pos: Angstrom
data:
  db_path: /xxxx/dataset/
  cutoff: 5.0
  split: <split>
  targets: [energy, forces]
  ...
trainer:
  run_name: xxx
  lossfn: smoothl1
  ...
```
Then simply run the following command according to your GPUs and port.
```
torchrun --nproc_per_node=<n_gpu> --master_port=<port> --no-python xeq train --config config.yaml
```
`--config` can be abbreviated to `-C`. If you don't specify it, it will read `config.yaml` by default.

During training, `loss.log` , `run_name_k.pt` and `run_name_last.pt` will be automatically generated, which records loss information and net parameters respectively.

### Test
Similarily, prepare the dataset and configuration file `config.yaml`. Run
```
xeq test --config config.json --ckpt run_name_0.pt
```
You can obtain the detailed arguments by
```
xeq -h
```
