## XequiNet
XequiNet is an equivariant graph neural network for predicting properties of chemical molecules or periodical systems.

## Requirements
**The following versions of these packages are only recommended for use.**

python 3.9<br>
pytorch 2.0<br>
pyg (follow pytorch)<br>
pytorch-cluster (follow pytorch)<br>
pytorch-scatter (follow pytorch)<br>
e3nn 0.5<br>
pytorch-warmup 0.1<br>
pydantic 2.6<br>
ase 3.22<br>
pyscf 2.4

## Extra requirements

**The geomeTRIC package is used for geometry optimization.**

geometric 1.0

**The TBLite packages are used for delta learning with GFN2-xTB as base.**

tblite 0.3<br>
tblite-python 0.3

**The i-PI package is used for path integral molecular dynamics**

ipi 2.6

## Setups
### From Source
Once the requirements are installed, running
```
pip install -e .
```

## Usage
See the markdown files in `docs` for details.

`docs/training.md`: Training and testing with dataset in `hdf5` format. No support for CPU training at this time.

`docs/inference.md`: Prediction with a trained model `xxx.pt`.

`docs/geometry.md`: Geometry optimization and molecular dynamics with a **JIT** model `xxx.jit`.