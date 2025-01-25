# Specific Configurations

`[xxx]` in the **Type** column means optinal parameters.

## Model Config
| Name | Type | Default | Description |
| - | - | - | - |
| `model_name` | `str` | `xpainn` | Model architecture to use.<br> `xpainn` refers to XPaiNN and `xpainn-ewald` add Ewald message passing block based on XPaiNN. |
| `model_kwargs` | `dict` | - | Model hyper-parameters, see [model hyper-parameters](#model-hyper-parameters). |
| `default_units` | `dict[str, str]` | - | Units you model uses, see [supporting unit](#supporting-units). e.g. `{energy: eV, pos: Ang}` |

## Trainer Config
| Name | Type | Default | Description |
| - | - | - | - |
| `run_name` | `str` | `xequinet` | Name of the experiment. Checkpoint file will be named after this. |
| `ckpt_file` | `[str]` | `null` | Checkpoint file to load. e.g. `water_xdh.pt` |
| `finetune_modules` | `[list[str]]` | `null` | Keywords of parameters to be finetuned. e.g. `[output, embed]` |
| `warmup_scheduler` | `str` | `linear` | Warmup scheduler type, see [pytorch-warmup](https://pypi.org/project/pytorch-warmup/). Other choices: `exponential`, `untuned_linear`, `untuned_exponential` |
| `warmup_epochs` | `int` | `10` | Number of epochs for learning rate raise from zero to `max_lr`. |
| `max_epochs` | `int` | `300` | Maximum number of training epochs. |
| `max_lr` | `float` | `5e-4` | Maximum learning rate. |
| `min_lr` | `float` | `0.0` | Minimum learning rate during scheduling. |
| `lossfn` | `str` | `smoothl1` | Loss function type. Other choices: `mae`, `l1`, `mse`, `l2`. |
| `losses_weight` | `dict[str, float]` | - | Weights for each part in loss function. e.g.`{energy: 1.0, forces: 100.0}`. |
| `grad_clip` | `[float]` | `null` | Gradient clipping value. |
| `optimizer` | `str` | `adamW` | Optimizer type. Other choices: `adam` |
| `optimizer_kwargs` | `dict` | `{}` | See [torch.optim](https://pytorch.org/docs/stable/optim.html) |
| `lr_scheduler` | `str` | `cosine_annealing` | Learning rate scheduler type. Other choices: `cosine_warmup`, `reduce_on_plateau`. |
| `lr_scheduler_kwargs` | `dict` | `{}` | See [torch.optim](https://pytorch.org/docs/stable/optim.html). e.g. `{Tmax: 100}`. |
| `early_stoppings` | `[dict[str, Any]]` | `null` | keys for monitoring properties, and values are early stopping settings. e.g. `{energy: {metric: mae, patience: 50, threshold: 0.0, lower_bound: 0.043}, forces: {metric: mae, patience: null, threshold: 0.0, lower_bound: 0.043}}` |
| `early_stropping_mode` | `str` | `and` | Conditions for multiple early stoppings. `and` for stopping training when all conditions are met concurrently. `or` for stopping training when any condition is met. |
| `ema_decay` | `[float]` | `null` | Weight for exponential moving average. Recommended value from 0.9 to 0.999. |
| `seed` | `[int]` | `null` | Random seed for everything. |
| `num_workers` | `int` | `0` | Extra number of threads for PyG to prepare batch data. |
| `save_dir` | `str` | `./` | Path to save loss file and checkpoint files. |
| `best_k` | `int` | `1` | Save best k models on validation loss. |
| `log_file` | `str` | `loss.log` | File for logging loss information. |
| `log_steps` | `int` | `50` | Log MAE every # steps. |
| `log_epochs` | `int` | `1` | Log loss information every # epochs |



## Data Config
| Name | Type | Default | Description |
| - | - | - | - |
| `db_path` | `str` | `./` | Path to the directory containing your data files |
| `cutoff` | `float` | `5.0` | Cutoff radius, need to be the same in model kwargs |
| `split` | `str` | `split` | Name of the split file without suffix, i.e. `random42` if your file is `random42.json` |
| `targets` | `list[str]` | - | Target label you want to train in the dataset. e.g. `[energy, forces, virial]`. |
| `base_targets` | `[list[str]]` | - | Base target label for delta-learning. e.g. `[base_energy, base_forces, base_virial]`. |
| `default_dtype` | `str` | `float32` | Float point precision. Other choice: `float64`. |
| `node_scale` | `bool` / `float` | `False` | The standard deviation value per atom of the label, which will be multiplied to the output. If `True`, calculate the value on-the-fly. If `float`, use this value |
| `node_shift` | `bool` / `float` | `False` | The average value per atom of the label, which will be added to the output. If `True`, calculate the value on-the-fly. If `float`, use this value |
| `max_num_samples` | `int` | `1000000` | Numbers of samples used to calculate `node_scale` and `node_shift` on-the-fly`. |
| `batch_size` | `int` | `64` | Training batch size. |
| `valid_batch_size` | `int` | `64` | Validation batch size. |


## Model Hyper-parameters
### XPaiNN

| Name | Type | Default | Description |
| - | - | - | - |
| `node_dim` | `int` | `128` | Length of scalar features |
| `node_irreps` | `str` | `128x0e + 64x1o + 32x2e` | Irreps of equivariant features, see [e3nn/Irreps](https://docs.e3nn.org/en/latest/api/o3/o3_irreps.html). |
| `embed_basis` | `str` | `gfn2-xtb` | AO used to embed atomic numbers. Other choice: `one-hot`. |
| `aux_basis` | `str` | `aux56` | Auxiliary orbital used to get projected from AO. Other choice: `aux28`. |
| `rbf_kernel` | `str` | `bessel` | Radial basis function type. Other choice: `gaussian`. |
| `num_basis` | `int` | `20` | Number of RBF. |
| `cutoff` | `float` | `5.0` | Cutoff radius for building graph (unit follow user's setting). |
| `cutoff_fn` | `str` | `cosine` | Smooth envelope function type. Other choice: `polynomial`. |
| `action_blocks` | `int` | `3` | Number of layers. |
| `activation` | `str` | `silu` | Activation function type. Other choices: `relu`, `leakyrelu`, `softplus`. |
| `layer_norm` | `bool` | `True` | Whether to use layer normalization. |
| `charge_embed` | `bool` | `False` | Whether to include net charge into account. If `True`, your datset need contain `charge`. |
| `spin_embed` | `bool` | `False` | Whether to include spin into account. If `True`, your dataset need contain `spin`. |
| `output_mode` | `str`, `list[str]` | `[energy]` | Output mode. Passing list for multi-task. Other choices: `charges`, `dipole`, `polar`. |
| `hidden_dim` | `int` | `64` | Hidden dimension of scalar MLP in output layer. |
| `hidden_irreps` | `str` | `64x0e + 32x1o + 16x2e` | Hidden Irreps for equivariant MLP in output layer for vector or tensor properties. |
| `conservation` | `bool` | `True` | For `charges` output. Whether to enssure conservation of electric charge. |
| `magnitude` | `bool` | `False` | For `dipole` output. Whether to return the magnitude of dipole moment vector. |
| `isotropic` | `bool` | `False` | For `polar` output. Whether to return the isotropic of polarizability. |

### Extra Hyper-parameters for Ewald Massage Passing (Not recommended)
| Name | Type | Default | Description |
| - | - | - | - |
| `use_pbc` | `bool` | `True` | Whether your dataset is periodic, since whether PBC or not affects the architectures for Ewald-MP |
| `projection_dim` | `int` | `8` | Hidden dimension in Ewald-MP |
| `ewald_blocks` | `int` | `1` | Number of Ewald-MP layers |
| `ewald_output_mode` | `str`, `list[str]` | `[energy]` | Extra output mode after Ewald-MP |
| `num_k_points` | `list[int]` | `[3, 3, 3]` | Numbers of k points of 3 dimension |
| `k_cutoff` | `float` | `0.4` | For `use_pbc = False`. Cutoff radius in reciprocal space (unit follow user's setting) |
| `delta_k` | `float` | `0.2` | For `use_pbc = False`. Grid length of reciprocal space. |
| `num_k_basis` | `int` | `20` | For `use_pbc = False`. Number of RBF in reciprocal space |
| `k_offset` | `[float]` | `null` | For `use_pbc = False`. Extension raidus for `k_cutoff` |


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