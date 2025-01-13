[中文](README_CN.md)/EN

# XequiNet
XequiNet is a package of equivariant graph neural network for predicting properties of chemical molecules or periodical systems.

## Install environments
```
git clone https://github.com/X1X1010/XequiNet.git
cd XequiNet
conda env create -f environment.yaml -n <env_name>
```
If the automatic installation fails, or if you want to install packages of other versions, you can install it manually, mainly for the following packages.

- [PyTorch](https://pytorch.org/): Greatness speaks for itself.
- [PyG](https://pytorch-geometric.readthedocs.io/en/latest/index.html): For constructing graph neural networks.
- [torch-cluster](https://pypi.org/project/torch-cluster/): For constructing `edge_index` with cutoff radius.
- [torch-scatter](https://pypi.org/project/torch-scatter/): For index selecting operations in GNN structure.
- [e3nn](https://e3nn.org/): For constructing equivariant modules.
- [pytorch-warmup](https://tony-y.github.io/pytorch_warmup/master/index.html): For convenient learning rate warmup.
- [OmegaConf](https://omegaconf.readthedocs.io/en/2.3_branch/): A nice package for parsing configuration files
- LMDB: [Conda](https://anaconda.org/conda-forge/python-lmdb) / [PyPI](https://pypi.org/project/lmdb/). Dataset format. Be careful with installation by conda.
- [ASE](https://wiki.fysik.dtu.dk/ase/#): A nice package for atomistic simulation. We mainly use for read input files with various format.
- [PySCF](https://pyscf.org/index.html): For handling some quantum chemical computation.
- [geomeTRIC](https://geometric.readthedocs.io/en/latest/): Geometry optimization engine used by PySCF.
- [TBlite](https://tblite.readthedocs.io/en/latest/): Python interface of xTB. Note that there are two packages, `tblite` and `tblite-python`. Both are needed.

## Extra packages
- [cuda-toolkit](https://anaconda.org/nvidia/cuda-toolkit): Nvidia CUDA Toolkit. Please be super careful with the **cuda version**.
- [i-Pi](https://ipi-code.org/): For PIMD simulation.

## Setups
Once the requirements are installed, running
```
conda activate <env_name>
pip install -e .
```

## Usage
See [docs](./xequinet/docs) for details.


## Citation
```bibtex
@article{doi:10.1021/acs.jctc.4c01151,
    author = {Chen, Yicheng and Yan, Wenjie and Wang, Zhanfeng and Wu, Jianming and Xu, Xin},
    title = {Constructing Accurate and Efficient General-Purpose Atomistic Machine Learning Model with Transferable Accuracy for Quantum Chemistry},
    journal = {J. Chem. Theory Comput.},
    volume = {20},
    number = {21},
    pages = {9500-9511},
    year = {2024},
    doi = {10.1021/acs.jctc.4c01151},
}
```