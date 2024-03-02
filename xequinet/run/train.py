import os
import random
import argparse

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import DataLoader

from xequinet.nn import resolve_model
from xequinet.utils import (
    NetConfig, ZeroLogger, set_default_unit,
    calculate_stats, distributed_zero_first,
)
from xequinet.data import create_dataset


def main():
    # ------------------- set up ------------------- #
    # parse config
    parser = argparse.ArgumentParser(description="XequiNet distributed training script")
    parser.add_argument(
        "--config", "-C", type=str, default="config.json",
        help="Configuration file (default: config.json).",
    )
    parser.add_argument(
        "--warning", "-w", action="store_true",
        help="Whether to show warning messages",
    )
    args = parser.parse_args()

    # open warning or not
    if not args.warning:
        import warnings
        warnings.filterwarnings("ignore")
    
    # load config
    if os.path.isfile(args.config):
        with open(args.config, "r") as json_file:
            config = NetConfig.model_validate_json(json_file.read())
    else:
        Warning(f"Config file {args.config} not found. Default config will be used.")
        config = NetConfig()

    # parallel process
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # set logger (only log when rank0)
    is_rank0 = (local_rank == 0)
    log = ZeroLogger(is_rank0=is_rank0, output_dir=config.save_dir, log_file=config.log_file)
    
    # set random seed
    if config.seed is not None:
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed(config.seed)
        np.random.seed(config.seed)
        random.seed(config.seed)
        torch.backends.cudnn.deterministic = True

    # set default dtype
    if config.default_dtype == "float32":
        torch.set_default_dtype(torch.float32)
    elif config.default_dtype == "float64":
        torch.set_default_dtype(torch.float64)
    else:
        raise NotImplementedError(f"Unknown default data type: {config.default_dtype}")

    # set default unit
    set_default_unit(config.default_property_unit, config.default_length_unit)

    # set device
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(device)

    # ------------------- load dataset ------------------- #
    train_dataset = create_dataset(config, "train", local_rank)
    valid_dataset = create_dataset(config, "valid", local_rank)
    
    # set dataloader
    train_sampler = DistributedSampler(train_dataset, world_size, local_rank, shuffle=True)
    valid_sampler = DistributedSampler(valid_dataset, world_size, local_rank, shuffle=False)
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size // world_size, sampler=train_sampler,
        num_workers=config.num_workers, pin_memory=True, drop_last=True,
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=config.vbatch_size // world_size, sampler=valid_sampler,
        num_workers=0, pin_memory=True, drop_last=False,
    )

    # calculate the mean and std of the training dataset
    if config.node_average:
        if config.node_average is True:
            with distributed_zero_first(local_rank):
                mean, std = calculate_stats(train_loader)
                log.s.info(f"Mean: {mean:6.4f} {config.default_property_unit}")
                log.s.info(f"Std : {std:6.4f} {config.default_property_unit}")
                config.node_average = mean
        else:
            config.node_average = float(config.node_average)
    else:
        config.node_average = 0.0

    # -------------------  build model ------------------- #
    # initialize model
    model = resolve_model(config)
    log.s.info(model)
    model.to(device)

    # distributed training
    find_unused = True if config.finetune else False
    ddp_model = DDP(model, device_ids=[local_rank], output_device=device, find_unused_parameters=find_unused)

    # record the number of parameters
    n_params = 0
    for name, param in ddp_model.named_parameters():
        if config.finetune and "embed.node_lin" not in name:
            param.requires_grad = False
            log.s.info(f"{name}: {param.numel()} (frozen)")
        else:
            n_params += param.numel()
            log.s.info(f"{name}: {param.numel()}")
    log.s.info(f"Total number of parameters to be optimized: {n_params}")
    
    # -------------------  train model ------------------- #
    if config.output_mode == "grad":
        from xequinet.utils import GradTrainer as MyTrainer
    else:
        from xequinet.utils import Trainer as MyTrainer
    trainer = MyTrainer(ddp_model, config, device, train_loader, valid_loader, train_sampler, log)
    trainer.start()
    

if __name__ == "__main__":
    main()
