[EN](README.md)/中文

# XequiNet
XequiNet 是一个等变图神经网络框架，用于预测化学分子或周期系统的性质。

## 安装环境
```
git clone https://github.com/X1X1010/XequiNet.git
cd XequiNet
conda env create -f environment.yaml -n <env_name>
```
如果自动安装失败，或者你希望安装其他版本的库，你可以选择手动安装，主要包括以下几个重要的库。

- [PyTorch](https://pytorch.org/)：伟大，无需多言。
- [PyG](https://pytorch-geometric.readthedocs.io/en/latest/index.html)：用于图网络的构建。
- [torch-cluster](https://pypi.org/project/torch-cluster/)：用于构建基于截断半径的邻居表。
- [torch-scatter](https://pypi.org/project/torch-scatter/)：用于GNN的节点索引操作。
- [e3nn](https://e3nn.org/)：用于构建等变模块。
- [pytorch-warmup](https://tony-y.github.io/pytorch_warmup/master/index.html)：用于简易地实现学习率热身。
- [OmegaConf](https://omegaconf.readthedocs.io/en/2.3_branch/)：一个好用的解析配置文件的库。
- LMDB: [Conda](https://anaconda.org/conda-forge/python-lmdb) / [PyPI](https://pypi.org/project/lmdb/). 数据集储存格式。注意Conda安装的方法。
- [ASE](https://wiki.fysik.dtu.dk/ase/#)：非常好用的原子模拟库，这里主要用来读各种格式的输入文件。
- [PySCF](https://pyscf.org/index.html)：用于一些量子化学计算。
- [geomeTRIC](https://geometric.readthedocs.io/en/latest/)：PySCF使用的结构优化引擎。
- [TBlite](https://tblite.readthedocs.io/en/latest/): xTB的Python接口。注意有两个包要装，`tblite` 和 `tblite-python`。
- [tabulate](https://pypi.org/project/tabulate/): 用来美化输出格式的。

## 其他库
- [cuda-toolkit](https://anaconda.org/nvidia/cuda-toolkit): 英伟达 CUDA Toolkit。千万注意**cuda版本**。
- [i-Pi](https://ipi-code.org/): 可以用于路径积分动力学（太贵了我跑不动）。

## 安装
在配置完环境之后，执行
```
conda activate <env_name>
pip install -e .
```

## 使用方法
详见[docs](./xequinet/docs)。

## 引用
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