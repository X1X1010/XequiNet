from typing import Union, List
from pydantic import BaseModel, Extra


class NetConfig(BaseModel):
    """
    Network configuration
    """
    class Config:
        arbitrary_types_allowed = True
        extra = Extra.allow
    
    # non-essential
    run_name: str = "my_task"                      # name of the run

    # configurations about the model
    pbc: bool = False                              # whether to use the periodic boundary condition
    embed_basis: str = "gfn2-xtb"                  # embedding basis type
    aux_basis: str = "aux56"                       # auxiliary basis type
    node_dim: int = 128                            # node irreps for the input
    edge_irreps: str = "128x0e + 64x1e + 32x2e"    # edge irreps for the input
    hidden_dim: int = 64                           # hidden dimension for the output
    hidden_irreps: str = "64x0e + 32x1e + 16x2e"   # hidden irreps for the output
    rbf_kernel: str = "bessel"                     # radial basis function type
    num_basis: int = 20                            # number of the radial basis functions
    cutoff: float = 5.0                            # cutoff distance for the neighbor atoms
    cutoff_fn: str = "cosine"                      # cutoff function type
    max_edges: int = 100                           # maximum number of the edges
    action_blocks: int = 3                         # number of the action blocks
    activation: str = "silu"                       # activation function type
    norm_type: str = "nonorm"                      # normalization layer type
    output_mode: str = "scalar"                    # task type (`scalar` is for energy like, `grad` is for force like, etc.)
    output_dim: int = 1                            # output dimension of multi-task (only for `scalar` mode)
    atom_ref: Union[str, dict] = None              # atomic reference (only for `scalar` mode)
    batom_ref: Union[str, dict] = None             # base atomic reference (only for `scalar` mode)
    add_mean: Union[bool, float] = False           # whether to add the mean atomic energy to the output (only for `scalar` mode)
    divided_by_atoms: bool = True                  # whether to be divided by the number of atoms when calculating the mean
    default_length_unit: str = "Angstrom"          # unit of the input coordinates
    default_property_unit: str = "eV"              # unit of the input properties
    default_dtype: str = "float32"                 # default data type

    # configurations about the dataset
    dataset_type: str = "normal"                   # dataset type (`memory` is for the dataset in memory, `disk` is for the dataset on disk)
    data_root: str = None                          # root directory of the dataset
    data_files: Union[List[str], str] = None       # list of the data files
    processed_name: str = None                     # name of the processed dataset
    mem_process: bool = True                       # whether to process the dataset in memory
    label_name: str = None                         # name of the label
    blabel_name: str = None                        # name of the basis label
    force_name: str = None                         # name of the force
    bforce_name: str = None                        # name of the basis force
    label_unit: str = None                         # unit of the input label
    blabel_unit: str = None                        # unit of the input base label
    force_unit: str = None                         # unit of the input force
    bforce_unit: str = None                        # unit of the input base force
    max_mol: int = 1e9                             # maximum number of the training molecules
    vmax_mol: int = 1e9                            # maximum number of the validation molecules
    batch_size: int = 64                           # training batch size
    vbatch_size: int = 64                          # validation batch size

    # configurations about the training
    ckpt_file: str = None                          # checkpoint file to load
    resumption: bool = False                       # whether to resume the training
    finetune: bool = False                         # whether to finetune the model
    warmup_scheduler: str = "linear"               # warmup scheduler type
    warmup_epochs: int = 10                        # number of the warmup epochs
    max_epochs: int = 300                          # maximum number of the training epochs
    max_lr: float = 5e-4                           # maximum learning rate
    min_lr: float = 0.0                            # minimum learning rate
    lossfn: str = "smoothl1"                       # loss function type
    force_weight: float = 100.0                    # weight of the force loss
    grad_clip: float = None                        # gradient clip
    optimizer: str = "adamW"                       # optimizer type
    optim_kwargs: dict = {}                        # kwargs for the optimizer
    lr_scheduler: str = "cosine_annealing"         # learning rate scheduler type
    lr_sche_kwargs: dict = {}                      # kwargs for the learning rate scheduler
    early_stop: int = None                         # number of the epochs to wait before stopping the training
    ema_decay: float = None                        # exponential moving average decay

    # configurations about the logging
    save_dir: str = './'                           # directory to save the model
    best_k: int = 1                                # number of the best models to keep
    log_file: str = "loss.log"                     # name of the logging file
    log_interval: int = 50                         # number of the steps to log the training information

    # others
    seed: int = None                               # random seed
    num_workers: int = 0                           # number of the workers for the data loader

    def model_hyper_params(self):
        hyper_params = self.dict(include={
            "pbc", "embed_basis", "aux_basis", "node_dim", "edge_irreps", "hidden_dim", "hidden_irreps",
            "rbf_kernel", "num_basis", "cutoff", "cutoff_fn", "max_edges", "action_blocks",
            "activation", "norm_type", "output_mode", "output_dim",
            "atom_ref", "batom_ref", "default_property_unit", "default_length_unit",
            "default_dtype",
        })
        return hyper_params


if __name__ == "__main__":
    config = NetConfig()
    print(config.model_hyper_params())
