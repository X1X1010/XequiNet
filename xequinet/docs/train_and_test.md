# Training and testing

## Data preparing
See [dataset doc](./dataset.md) for data format. Your data directory should contain at least these three files:
```
<path_to_data>
  ├─ data.lmdb
  ├─ info.json
  └─ <split>.json
```
A large number of concurrent reads will affect the speed. If you have multiple training tasks using the same dataset, it is highly recommended to make a separate copy of the lmdb file for each task to the disk of the compute node, e.g. `/scratch/...`, and remove them after training.

## Training
Set the training configuration file `config.yaml` in working directory as follows. Details can be viewed in the [config doc](./config.md).
```yaml
model:
  model_name: xpainn
  model_kwargs:
    node_dim: 128
    node_irreps: 128x0e+64x1o+32x2e
    ...
  default_units:
    energy: eV
    pos: Ang
data:
  db_path: /scratch/.../dataset/
  cutoff: 5.0
  split: random42
  targets: [energy, forces]
  ...
trainer:
  run_name: xxx
  lossfn: smoothl1
  ...
```
Then simply run the following command according to your GPUs and port.
```
torchrun --nproc-per-node=<n_gpu> --master-port=<port> --no-python xeq train --config config.yaml
```
- `torchrun`: A python console script for pytorch to conduct DDP training, see [torchrun](https://pytorch.org/docs/stable/elastic/run.html).
- `--nproc-per-node`: Number of GPUs
- `--master-port`: Port number of the master node, used for communication with other nodes. Just avoid using the same.
- `--no-python`: This is because `xeq` is not a python file, but an entry point.
- `xeq train`: Entry point to training.
- `--config` / `-C`: Training config file name. (default: `config.yaml`)

During training, `loss.log` , `run_name_k.pt` and `run_name_last.pt` will be automatically generated, which records loss information and model parameters respectively.

## Testing
Similarily, prepare the dataset and configuration file `config.yaml`. Run testing with this command. Then testing will be preformed on the `test` part in `<split>.json`
```
xeq test --config config.yaml --ckpt run_name_0.pt
```
Here is the detailed command line arguments.

`--config` / `-C`: Testing config file name. (default: `config.yaml`)

`--ckpt` / `-c`: Checkpoint file name.

`--device`: `cuda` or `cpu`. (default: automatically detect if GPU is available.)

`--ouput` / `-o`: Output file name. (default: `<run_name>.log`)

`--verbose` / `-v`: Whether to print detailed information and save the result in `.pt` file.

