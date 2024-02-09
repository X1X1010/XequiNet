## Inference
**Input file**

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
Specially, one can use `charge=?` and `multiplicity=?` to denote the properties of the molecule in the comment line. And one can also use `pbc="? ? ?"` and `lattice="ax ay az  bx by bz  cx cy cz"` to denote the periodic boundary conditions.

**Run inference**
```
xeqinfer -c run_name_k.pt inf_mol.xyz
```
The prediction result will be writen in `example.log`. Use
```
xeqinfer -h
```
for detailed arguments.