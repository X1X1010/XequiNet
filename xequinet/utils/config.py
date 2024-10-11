from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Union, List


@dataclass
class ModelConfig:
    """Config for the model"""

    model_name: str = "xpainn"
    model_config: Dict[str, Any] = field(
        default_factory=lambda: dict(
            embed_basis="gfn2-xtb",
            aux_basis="aux56",
            node_dim=128,
            node_irreps="128x0e + 64x1o + 32x2e",
            hidden_dim=64,
            hidden_irreps="64x0e + 32x1o + 16x2e",
            rbf_kernel="bessel",
            num_basis=20,
            cutoff=5.0,
            cutoff_fn="cosine",
            action_blocks=3,
            acitvation="silu",
            norm_type="nonorm",
            output_mode="scalar",
        )
    )


@dataclass
class TrainerConfig:
    """Config for the trainer"""

    run_name: str = "xequinet"
    batch_size: int = 64
    valid_batch_size: int = 64
    ckpt_file: Optional[str] = None
    resume: bool = False
    finetune: bool = False
    warmup_scheduler: str = "linear"
    warmup_epochs: int = 10
    max_epochs: int = 300
    max_lr: float = 5e-4
    min_lr: float = 0.0
    lossfn: str = "smoothl1"
    losses_weight: Dict[str, float] = field(default_factory=lambda: dict(energy=1.0))
    grad_clip: Optional[float] = None
    optimizer: str = "adamW"
    optim_kwargs: Dict[str, Any] = field(default_factory=dict)
    lr_scheduler: str = "cosine_annealing"
    lr_sche_kwargs: Dict[str, Any] = field(default_factory=dict)
    early_stop: Optional[int] = None
    ema_decay: Optional[float] = None
    seed: Optional[int] = None
    num_workers: int = 0

    save_dir: str = "./"
    best_k: int = 1
    log_file: str = "loss.log"
    log_step: int = 50
    log_epoch: int = 1


@dataclass
class DataConfig:
    """Config for the dataset"""

    db_path: str = "./"
    cutoff: float = 5.0
    split: str = "split"
    # Note: the default values for the following fields are not the values of the dataset,
    #       but the default values you want to use in the training.
    targets: Union[str, List[str]] = "energy"
    base_targets: Optional[Union[str, List[str]]] = None
    default_dtype: str = "float32"
    default_units: Dict[str, str] = field(
        default_factory=lambda: dict(energy="eV", pos="Angstrom")
    )
    node_shift: Union[float, bool] = False
    node_scale: Union[float, bool] = False
    max_num_samples: int = 1000000


@dataclass
class XequiConfig:
    """Config for the XequiNet"""

    model: ModelConfig = ModelConfig()
    trainer: TrainerConfig = TrainerConfig()
    data: DataConfig = DataConfig()


if __name__ == "__main__":
    from omegaconf import OmegaConf
    from typing import cast

    schema = OmegaConf.structured(XequiConfig)
    config = cast(XequiConfig, schema)
    print(type(config))
