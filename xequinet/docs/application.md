# Application
## Input file
Any input format that can be recognized by ASE can be used. For example, the `xyz` or `extxyz` format.

`example.xyz`:
```
5
charge=0  multiplicity=1
C   0.00000000    0.00000000    0.00000000
H   0.00000000    0.00000000    1.06999996
H  -0.00000000   -1.00880563   -0.35666665
H  -0.87365130    0.50440282   -0.35666665
H   0.87365130    0.50440282   -0.35666665
3
charge=0  spin=0
O   0.00000000   -0.00000000   -0.11081188
H   0.00000000   -0.78397589    0.44324751
H  -0.00000000    0.78397589    0.44324751
...
```
`example.extxyz`:
```
192
pbc="T T T" lattice="23.46511000 0.00000000 0.00000000   -0.00000100 23.46511000 0.00000000    -0.00000100 -0.00000100 23.46511000" Properties=species:S:1:pos:R:3
O  11.72590000   14.59020000   25.33440000
H  12.69400000   16.13880000   24.72010000
H   9.70021000   15.03790000   25.76530000
O  10.68010000    3.41217000    4.43292000
...
```
Specially, one can use `charge=?` and `spin=?` to denote the properties of the molecule in the comment line. And one can also use `pbc="? ? ?"` and `lattice="ax ay az  bx by bz  cx cy cz"` to denote the periodic boundary conditions.

## Inference
Since PyTorch takes time to initialize and load model, try to use consecutive xyz files to save multiple structrues in one file if there are a large number of inference tasks. And also try to increase `batch_size` to speed up.
```
xeq infer -c <ckpt>.pt -in <mol>.xyz
```
The prediction result will be writen in `<mol>.log`. Following is the detailed command line arguments.

- `--ckpt` / `-c`: Checkpoint file name `<ckpt>.pt`.
- `--device`: `cuda` or `cpu` (default: automatically detect if GPU is available).
- `--input` / `-in`: Input file name.
- `--delta` / `-d`: Base semi-empirical method for delta-learning model. Choice: `gfn2-xtb`, `gfn1-xtb`. 
- `--forces`: Whether to compute forces.
- `--stress`: Whether to compute stress.
- `--batch-size`: Batch size, literally.
- `--ouput` / `-o`: Output file name. By default, change the suffix of input file to `.log` as output file.
- `--verbose` / `-v`: Whether to save the result in `.pt` file.

## Geometry optimization and frequency analysis
The geometry optimization is implemented with the interface to [geomTRIC](https://geometric.readthedocs.io/en/latest/index.html) provided by [PySCF](https://pyscf.org/), and the vibrational frequency analysis is implemented with PySCF.
```
xeq opt -c <ckpt>.pt -in <mol>.xyz
```

- `--ckpt` / `-c`: Checkpoint file name `<ckpt>.pt`.
- `--device`: `cuda` or `cpu`. (default: automatically detect if GPU is available.)
- `--input` / `-in`: Input file name.
- `--opt-params`: `.json` file to set optimization parameters, see [geomeTRIC](https://geometric.readthedocs.io/en/latest/options.html#optimization-parameters).
- `--constraints`: File to set constraints for constrained optimization, see [constrained optimization](https://geometric.readthedocs.io/en/latest/constraints.html).
- `--max-steps`: Maximum iteration steps for optimization.
- `--delta` / `-d`: Base semi-empirical method for delta-learning model. Choice: `gfn2-xtb`, `gfn1-xtb`. 
- `--freq`: Whether to do frequency analysis.
- `--temp` / `-T`: Temperature for vibrational analysis (default 298.15 K).
- `--no-opt`: Doing frequency analysis without optimization. Be careful with imaginary frequencies.
- `--shermo`: Whether to save shermo input file for quasi-harmonic analysis, see [Shermo](http://sobereva.com/soft/shermo/).
- `--save-hessian`: Whether to save hessian matrix.
- `--verbose` / `-v`: Whether to print detailed frequencies information.

## MD simulation with ASE
If you have some simple MD task, you can use the MD modules implemented in ASE.
```
xeq md --config <md_config>.yaml
```
- `--config` / `-c`: MD config file.

e.g.
```yaml
input_file: water.xyz
model_file: model.pt

init_temperature: 300.  # Kelvin

ensembles:
  - name: VelocityVerlet
    timestep: 1  # fs
    steps: 1000
    loginterval: 100
  - name: NVTBerendsen
    timestep: 1  # fs
    taut: 100    # fs
    temperature: 300.  # Kelvin
    loginterval: 100
    steps: 1000000
  
logfile: md.log
trajectory: md.traj

xyz_traj: traj.xyz

columns: ["symbols", "positions"]
```

Here is the detailed MD config option:
| Name | Type | Default | Description |
| - | - | - | - |
| `emsenbles` | `List[Any]` | - | Emsemble options, see [ASE MD](https://wiki.fysik.dtu.dk/ase/ase/md.html). (`logfile` and `trajectory` are excluded and will be set globally). |
| `input_file` | `str` | `input.xyz` | Input structure file known by ASE. |
| `model_file` | `str` | `model.pt` | Model checkpoint file. |
| `delta_method` | `[str]` | `null` | Delta learning base method. `GFN2-xTB` or `GFN1-xTB` |
| init_temperature | `float` | `300.0` | Initial temperature to generate velocity under Maxwell-Boltzmann distribution. |
| `logfile` | `str` | `md.log` | File name for logging. |
| `append_logfile` | `bool` | `False` | Whether to append write in `logfile`. |
| `trajectory` | `[str]` | `null` | Trajectory file name for saving ASE binary trajectory. |
| `append_trajectory` | `str` | `False` | Whether to append write in trajectory file. |
| `xyz_traj` | `[str]` | `null` | xyz file name to which trajectory file is converted. |
| `columns` | `List[str]` | `[symbol, position]` | Content in xyz file. Options: [`positions`, `numbers`, `charges`, `symbols`] |