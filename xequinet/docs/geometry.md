## JIT script
We use JIT model for geometry optimization and molecular dynamics and the two type of jit models are slightly different.

One can JIT compile a model for geometry optimization by running
```shell
xeqjit -c example.pt
```
Then one can get `example.jit`, which can be used like this:
```python
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.jit.load("example.jit", map_location=device)
...
result = model(
    at_no = at_no,  # atomic_numbers: torch.LongTensor[natom,]
    coord = coord,  # coordinates in Bohr: torch.Tensor[natom, 3]
    charge = 0,     # total charge of a molecule: int
    spin = 0,       # 2S, equals multiplicity - 1: int
)                   # reslts: Dict[str, torch.Tensor]
energy = result["energy"]      # electronic energy in Hartree: float
gradient = result["gradient"]  # nuclear gradient in Hartree/Bohr: torch.Tensor[natom, 3]
```
The other type is for molecular dynamics which can be obtained by running
```shell
xeqjit -c example.jit --md
```
Similarly, one can get `example.jit`, which is a little bit more complex to use.
```python
...
model = torch.jit.load("example.jit", map_location=device)
result = model(
    at_no = at_no,   # atomic_numbers: torch.LongTensor[natom,]
    coord = coord,   # coordinates in Angstrom: torch.Tensor[natom, 3]
    edge_index = edge_index,  # interconnected pair indexes: torch.LongTensor[2, nedge]
    shifts = shifts, # corrections of relative positions due to pbc (quite hard to explain): torch.Tensor[nedge, 3]
    charge = 0,      # total charge of a molecule: int
    spin = 0,        # 2S, equals multiplicity - 1: int
)                    # reslts: Dict[str, torch.Tensor]
energy = result["energy"]     # total energy in eV: float
energies = result["energies"] # atomic energies in eV: torch.Tensor[natom,]
forces = result["forces"]     # forces felt by nucleus in eV/Angstrom: torch.Tensor[natom, 3]
virial = result["virial"]     # total virial in eV: torch.Tensor[3, 3]
virials = result["virials"]   # atomic virials in eV: torch.Tensor[natom, 3, 3]
```
## Geometry Optimization
Any input format that can be recognized by ASE can be used. For example, to optimize molecule written in `ABC.xyz`, one can run
```shell
xeqopt -c example.jit ABC.xyz
```
Then the optimized struct will be written in `ABC_opt.xyz`. We can further do vibrational frequency analysis by
```shell
xeqopt -c example.jit --freq ABC.xyz
```
Additional file `ABC_freq.log` will be written to show the thermodynamic result.
```shell
xeqopt -h
```
for detailed arguments

## Molecular Dynamics
For classical MD, one can prepare a `md_set.json` like this first.
```json
{
    "init_xyz": "ABC.xyz",
    "ckpt_file": "example.jit",
    "cutoff": 5.0,

    "ensemble": "NVE",
    "timestep": 1.0,
    "steps": 50,
    "temperature": 298.15,
    "loginterval": 5,
    "traj_xyz": "traj.xyz"
    ...
}
```
Check out `xequinet/run/dynamics.py` for detailed parameters.

Then start running MD by
```shell
xeqmd md_set.json
```
For PIMD, we use **i-PI** through socket. For **i-PI**, see https://ipi-code.org/

Add the following settings to **i-PI** input file `pimd_set.xml`
```xml
<simulation>
    ...
    <ffsocket mode='inet' pbc='False'>
        <address>localhost</address>
        <port>31415</port>
        <parameters>
            {
                "ckpt_file": "example.jit",
                "cutoff": 5.0,
                "max_edges": 100,
                "charge": 0,
                "multiplicity": 1
            }
        </parameters>
    </ffsocket>
    ...
</simulation>
```
Then start PIMD by running
```shell
i-pi pimd_set.xml & xeqipi pimd_set.xml
```
