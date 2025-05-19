import argparse
import os
import random
from typing import cast

import numpy as np
import torch
import torch.backends.cudnn
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import DataLoader

from xequinet import keys
from xequinet.data import create_lmdb_dataset
from xequinet.nn import resolve_model
from xequinet.utils import (
    Trainer,
    XequiConfig,
    ZeroLogger,
    calculate_stats,
    distributed_zero_first,
    set_default_units,
)


def run_train(args: argparse.Namespace) -> None:
    # ------------------- set up ------------------- #
    # load config
    if os.path.isfile(args.config):
        config = OmegaConf.merge(
            OmegaConf.structured(XequiConfig),
            OmegaConf.load(args.config),
        )
        # this will do nothing, only for type annotation
        config = cast(XequiConfig, config)
    else:
        Warning(f"Config file {args.config} not found. Default config will be used.")
        config = XequiConfig()

    # parallel process
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # set logger (only log when rank0)
    log = ZeroLogger(
        is_rank0=(local_rank == 0),
        output_dir=config.trainer.save_dir,
        log_file=config.trainer.log_file,
        resume=config.trainer.resume,
    )

    # set random seed
    if config.trainer.seed is not None:
        torch.manual_seed(config.trainer.seed)
        torch.cuda.manual_seed(config.trainer.seed)
        np.random.seed(config.trainer.seed)
        random.seed(config.trainer.seed)
        torch.backends.cudnn.deterministic = True

    # set default dtype
    name_to_dtype = {
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
    }
    torch.set_default_dtype(name_to_dtype[config.data.default_dtype])

    # set default unit
    set_default_units(config.model.default_units)

    # set device
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    torch.cuda.set_device(device)

    # ------------------- load dataset ------------------- #
    # for non-pbc Ewald model, we should use SVD frame to rotate the atomic positions
    if (
        config.model.model_name.lower() == "xpainn-ewald"
        and not config.model.model_kwargs["use_pbc"]
    ):
        svd_frame = True
    else:
        svd_frame = False
    train_dataset, valid_dataset = create_lmdb_dataset(
        db_path=config.data.db_path,
        cutoff=config.data.cutoff,
        split=config.data.split,
        dtype=config.data.default_dtype,
        targets=config.data.targets,
        base_targets=config.data.base_targets,
        mode="train",
        svd_frame=svd_frame,
    )
    # set dataloader
    sampler_seed = config.trainer.seed if config.trainer.seed is not None else 0
    train_sampler = DistributedSampler(
        dataset=train_dataset,
        num_replicas=world_size,
        rank=local_rank,
        seed=sampler_seed,
        shuffle=True,
    )
    valid_sampler = DistributedSampler(
        dataset=valid_dataset,
        num_replicas=world_size,
        rank=local_rank,
        seed=sampler_seed,
        shuffle=False,
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.data.batch_size // world_size,
        sampler=train_sampler,
        num_workers=config.trainer.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=config.data.valid_batch_size // world_size,
        sampler=valid_sampler,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    # calculate the mean and std of the training dataset
    # currently only for total energy
    node_shift, node_scale = 0.0, 1.0
    if config.trainer.ckpt_file is not None:
        pass  # the mean and std are in the parameters
    elif config.data.node_shift is True or config.data.node_scale is True:
        with distributed_zero_first(local_rank):
            mean, std = calculate_stats(
                dataloader=train_loader,
                divided_by_atoms=True,
                target_property=keys.TOTAL_ENERGY,
                base_property=keys.BASE_ENERGY,
                max_num_samples=config.data.max_num_samples,
            )
        log.s.info(f"Mean: {mean:6.4f} {config.model.default_units[keys.TOTAL_ENERGY]}")
        log.s.info(f"Std : {std:6.4f} {config.model.default_units[keys.TOTAL_ENERGY]}")
        if config.data.node_shift is True:
            node_shift = mean
        if config.data.node_scale is True:
            node_scale = std
    if isinstance(config.data.node_shift, float):
        node_shift = config.data.node_shift
    if isinstance(config.data.node_scale, float):
        node_scale = config.data.node_scale

    # -------------------  build model ------------------- #
    # initialize model
    model = resolve_model(
        model_name=config.model.model_name,
        node_shift=node_shift,
        node_scale=node_scale,
        **config.model.model_kwargs,
    )
    log.s.info(model)
    model.to(device)

    # several cases need to find unused params
    if not config.trainer.finetune_modules:
        # case 1: finetuning, where frozen params are not used
        find_unused = True
    elif (
        keys.FORCES in config.data.targets
        and keys.TOTAL_ENERGY not in config.data.targets
    ):
        # case 2: trianing forces without energy
        # bias in the final linear layer is not used (y = xW^T + b; dy/dx = W^T)
        find_unused = True
        log.s.info(
            "Training forces without energy need to set `find_unused_parameters` to True."
        )
    else:
        find_unused = False

    # distributed training
    ddp_model = DDP(
        module=model,
        device_ids=[local_rank],
        output_device=device,
        find_unused_parameters=find_unused,
    )

    # record the number of parameters
    n_params = 0
    for name, param in ddp_model.named_parameters():
        # train all parameters, no finetune
        if config.trainer.finetune_modules is None:
            n_params += param.numel()
            log.s.info(f"{name}: {param.numel()}")
        # train only the specified modules
        elif any(
            finetune_name in name for finetune_name in config.trainer.finetune_modules
        ):
            n_params += param.numel()
            log.s.info(f"{name}: {param.numel()}")
        # freeze the parameters
        else:
            param.requires_grad = False
            log.s.info(f"{name}: {param.numel()} (frozen)")
    log.s.info(f"Total number of parameters to be optimized: {n_params}")

    # -------------------  train model ------------------- #
    trainer = Trainer(
        model=ddp_model,
        config=config,
        device=device,
        train_loader=train_loader,
        valid_loader=valid_loader,
        dist_sampler=train_sampler,
        log=log,
    )
    trainer.start()
