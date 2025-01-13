# Deployment

## JIT script
To use model for MD simulation, we need to JIT compile the model.
```
xeq jit -c example.pt
```

- `--ckpt` / `-c`: Checkpoint file name `<ckpt>.pt`.
- `--device`: `cuda` or `cpu`. (default: automatically detect if GPU is available.)
- `--fusion-strategy`: Type and number of specializations that can occur during fusion. Format like `TYPE1,DEPTH1;TYPE2,DEPTH2;...`, see [torch.jit.set_fusion_strategy](https://pytorch.org/docs/stable/generated/torch.jit.set_fusion_strategy.html). (default `DYNAMIC,3`)
- `--mode`: `lmp` for LAMMPS `pair xequinet`; `dipole` for LAMMPS compute `dipole/xequinet`, see [xequinet-lammps](https://github.com/X1X1010/xequinet-lammps). `gmx` for GROMACS NNP, see [NNP/MM](https://manual.gromacs.org/2025.0-beta/reference-manual/special/nnpot.html).
- `--unit-style`: LAMMPS unit style, see [units](https://docs.lammps.org/units.html).
- `--net-charge`: Net charge for your system of sub-system in NNP/MM simulation.


## TODO: How to use deployed model in LAMMPS