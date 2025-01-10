## XequiNet
XequiNet is an equivariant graph neural network for predicting properties of chemical molecules or periodical systems.

## Install environments
**The following versions of these packages are only recommended for use.**
```
conda env create -f environment.yaml -n <env_name>
```


## Setups
### From Source
Once the requirements are installed, running
```
conda activate <env_name>
pip install -e .
```

## Usage
See the markdown files in [docs](./xequinet/docs) for details.

`docs/training.md`: Training and testing with dataset in `lmdb` format. No support for CPU training at this time.

`docs/inference.md`: Prediction with a trained model or **JIT** compiled model.

`docs/geometry.md`: Geometry optimization and molecular dynamics with a **JIT** model `xxx.jit`.

`docs/md.md`: Molecular dynamics with ASE.