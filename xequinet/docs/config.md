# Specific Configurations

`[xxx]` in the Type column means optinal parameters.

## Model Config
| Name | Type | Example | Description |
| - | - | - | - |
| `model_name` | `str` | `xpainn`, `xpainn-ewald` | Model architecture to use.<br> `xpainn` refers to XPaiNN and `xpainn-ewald` add Ewald message passing block based on XPaiNN |
| `model_kwargs` | `dict` | see below | Model hyper-parameters. |
| `default_units` | `dict[str, str]` | `energy: eV, pos: Angstrom` | Units you model uses. See [supporting unit](#supporting-units) |

## Trainer Config
| Name | Type | Example | Description |
| - | - | - | - |
| `run_name` | `str` | `xxx` | Name of the experiment. Checkpoint file will be named after this |
| `ckpt_file` | `[str]` | `xxx.pt` | Checkpoint file to load |
| `finetune_modules` | `[list[str]]` | `output` | Keywords of parameters to be finetuned |
| `warmup_scheduler` | `str` | `linear`, `exponential`, `untuned_linear`, `untuned_exponential` | Warmup scheduler type, see [pytorch-warmup](https://pypi.org/project/pytorch-warmup/) |
| `warmup_epochs` | `int` | `10` | Number of epochs for learning rate raise from zero to `max_lr` |
| `max_epochs` | `int` | `300` | Maximum number of training epochs |
| `max_lr` | `float` | `5e-4` | Maximum learning rate |
| `min_lr` | `float` | `0.0` | Minimum learning rate during scheduling |
| `lossfn` | `str` | `smoothl1`, `mae`, `mse` | Loss function type |
| `losses_weight` | `dict[str, float]` | `energy: 1.0, forces: 100.0` | Weights for each part in loss function |
| `grad_clip` | `[float]` | `50` | Gradient clipping value |
| `optimizer` | `str` | `adamW`, `adam` | Optimizer type |
| `optimizer_kwargs` | `dict` | | See [torch.optim](https://pytorch.org/docs/stable/optim.html) |
| `lr_scheduler` | `str` | `cosine_annealing`, `cosine_warmup`, `reduce_on_plateau` | Learning rate scheduler type |
| `lr_scheduler_kwargs` | `dict` | `Tmax: 100` | See [torch.optim](https://pytorch.org/docs/stable/optim.html) |
| `early_stop` | `[int]` | `100` | Stop training if validation loss does not drop for certain epochs |
| `ema_decay` | `[float]` | `0.999` | Weight for exponential moving average |
| `seed` | `[int]` | `42` | Random seed |
| `num_workers` | `int` | `0` | Extra number of threads for PyG to prepare batch data |

## Data Config
| Name | Type | Example | Description |
| - | - | - | - |
| `db_path` | `str` | `/share/home/.../spice-v2/` | Path to the directory containing your data files |
| `cutoff` | `float` | `5.0` | Cutoff radius, need to be the same in model kwargs |
| `split` | `str` | `random42` | Name of the split file without appendix. i.e. `random42` if your file is `random42.json` |
| `targets` | `list[str]` | `[energy, forces, virial]` | Target label you want to train in the dataset |
| `base_targets` | `[list[str]]` | `[base_energy, base_forces, base_virial]` | Base target label for delta-learning |
| `default_dtype` | `str` | `float32`, `float64` | Float point precision |
| `node_scale` | `bool` / `float` | `True`, `1.14` | The standard deviation value per atom of the label, which will be multiplied to the output. If `True`, calculate the value on-the-fly. If `float`, use this value |
| `node_shift` | `bool` / `float` | `True`, `5.14` | The average value per atom of the label, which will be added to the output. If `True`, calculate the value on-the-fly. If `float`, use this value |
| `max_num_samples` | `int` | `100000` | Numbers of samples used to calculate `node_scale` and `node_shift` on-the-fly` |
| `batch_size` | `int` | `1024` | Training batch size |
| `valid_batch_size` | `int` | `1024` | Validation batch size |


## Model Hyper-parameters
### XPaiNN

| Name | Type | Example | Description |
| - | - | - | - |
| `node_dim` | `int` | `128` | Length of scalar features |
| `node_irreps` | `str` | `128x0e + 64x1o + 32x2e` | Irreps of equivariant features, see [e3nn/Irreps](https://docs.e3nn.org/en/latest/api/o3/o3_irreps.html) |
| `embed_basis` | `str` | `one-hot`, `gfn2-xtb` | AO used to embed atomic numbers |
| `aux_basis` | `str` | `aux56`, `aux28` | Auxiliary orbital used to get projected from AO |
| `rbf_kernel` | `str` | `bessel`, `gaussian` | Radial basis function type |
| `num_basis` | `int` | `20` | Number of RBF |
| `cutoff` | `float` | `5.0` | Cutoff radius for building graph (unit follow user's setting) |
| `cutoff_fn` | `str` | `cosine`, `polynomial` | Smooth envelope function type |
| `action_blocks` | `int` | `3` | Number of layers |
| `activation` | `str` | `silu`, `relu`, `leakyrelu`, `softplus` | Activation function type |
| `layer_norm` | `bool` | `True` | Whether to use layer normalization |
| `charge_embed` | `bool` | `False` | Whether to include net charge into account. If `True`, your datset need contain `charge` |
| `spin_embed` | `bool` | `False` | Whether to include spin into account. If `True`, your dataset need contain `spin` |
| `output_mode` | `str`, `list[str]` | `energy`, `charges`, `dipole`, `polar` | Output mode. Passing list for multi-task |
| `hidden_dim` | `int` | `64` | Hidden dimension of scalar MLP in output layer |
| `hidden_irreps` | `str` | `64x0e + 32x1o + 16x2e` | Hidden Irreps for equivariant MLP in output layer for vector or tensor properties |
| `conservation` | `bool` | `True` | For `charges` output. Whether to enssure conservation of electric charge |
| `magnitude` | `bool` | `False` | For `dipole` output. Whether to return the magnitude of dipole moment vector |
| `isotropic` | `bool` | `False` | For `polar` output. Whether to return the isotropic of polarizability |

### Extra Hyper-parameters for Ewald Massage Passing
| Name | Type | Example | Description |
| - | - | - | - |
| `use_pbc` | `bool` | `True` | Whether your dataset is periodic, since whether PBC or not affects the architectures for Ewald-MP |
| `projection_dim` | `int` | `8` | Hidden dimension in Ewald-MP |
| `ewald_blocks` | `int` | `1` | Number of Ewald-MP layers |
| `ewald_output_mode` | `str`, `list[str]` | `energy` | Extra output mode after Ewald-MP |
| `num_k_points` | `list[int]` | `[16, 16, 16]` | Numbers of k points of 3 dimension |
| `k_cutoff` | `float` | `0.4` | For `use_pbc = False`. Cutoff radius in reciprocal space (unit follow user's setting) |
| `num_k_basis` | `int` | `20` | For `use_pbc = False`. Number of RBF in reciprocal space |
| `k_offset` | `[float]` | `0.2` | For `use_pbc = False`. Extension raidus for `k_cutoff` |


## Supporting units
| Quantity | Supporting units |
| - | - |
| Atomic unit | `AU`(`au`) |
| Amount of substance | `mol` |
| Charge | `e`, `Coulomb`(`C`) |
| Length | `Bohr`(`a0`), `meter`(`m`), `Angstrom`(`Ang`), `cm`, `nm` |
| Mass | `kg`, `g` |
| Energy | `Hartree`(`Ha`,`Eh`), `Joule`(`J`), `kJoule`(`kJ`), `eV`, `meV`, `cal`, `kcal` |
| Dipole | `Debye`(`D`) |
| Time | `second`(`s`), `fs`, `ps` |
| Pressure | `Pascal`(`Pa`), `GPa`, `bar`, `kbar` |
| Magnetic moment | `Bohr_magneton`(`muB`) |

Further, mixed units, like `kcal/mol` and `eV/Ang`, are also supported.