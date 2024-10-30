from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ModelConfig:
    """Config for the model"""

    model_name: str = "xpainn"
    model_kwargs: Dict[str, Any] = field(default_factory=dict)
    default_units: Dict[str, str] = field(default_factory=dict)


@dataclass
class TrainerConfig:
    """Config for the trainer"""

    run_name: str = "xequinet"
    ckpt_file: Optional[str] = None
    resume: bool = False
    finetune: bool = False
    warmup_scheduler: str = "linear"
    warmup_epochs: int = 10
    max_epochs: int = 300
    max_lr: float = 5e-4
    min_lr: float = 0.0
    lossfn: str = "smoothl1"
    losses_weight: Dict[str, float] = field(default_factory=dict)
    grad_clip: Optional[float] = None
    optimizer: str = "adamW"
    optimizer_kwargs: Dict[str, Any] = field(default_factory=dict)
    lr_scheduler: str = "cosine_annealing"
    lr_scheduler_kwargs: Dict[str, Any] = field(default_factory=dict)
    early_stop: Optional[int] = None
    ema_decay: Optional[float] = None
    seed: Optional[int] = None
    num_workers: int = 0

    save_dir: str = "./"
    best_k: int = 1
    log_file: str = "loss.log"
    log_steps: int = 50
    log_epochs: int = 1


@dataclass
class DataConfig:
    """Config for the dataset"""

    db_path: str = "./"
    cutoff: float = 5.0
    split: str = "split"
    # Note: the default values for the following fields are not the values of the dataset,
    #       but the default values you want to use in the training.
    targets: List[str] = field(default_factory=list)
    base_targets: Optional[List[str]] = None
    default_dtype: str = "float32"
    node_shift: Any = False
    node_scale: Any = False
    max_num_samples: int = 1000000
    batch_size: int = 64
    valid_batch_size: int = 64


@dataclass
class XequiConfig:
    """Config for the XequiNet"""

    model: ModelConfig = ModelConfig()
    trainer: TrainerConfig = TrainerConfig()
    data: DataConfig = DataConfig()


@dataclass
class MDConfig:
    """Config for the Molecular Dynamics with ASE"""

    ensembles: List[Dict[str, Any]] = field(default_factory=list)
    input_file: str = "input.xyz"
    model_file: str = "model.jit"

    init_temperature: float = 300.0  # Kelvin

    logfile: str = "md.log"
    append_logfile: bool = False
    trajectory: Optional[str] = None
    append_trajectory: bool = False
    xyz_traj: Optional[str] = None
    columns: Optional[List[str]] = None

    dtype: str = "float32"
    seed: Optional[int] = None


if __name__ == "__main__":
    from typing import cast

    from omegaconf import OmegaConf

    schema = OmegaConf.structured(XequiConfig)
    config = cast(XequiConfig, schema)
    print(type(config))
