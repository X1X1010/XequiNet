import os
import argparse

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import DataLoader

from xequinet.nn import xPaiNN
from xequinet.utils import (
    NetConfig, ZeroLogger,
    set_default_unit,
    distributed_zero_first, calculate_stats,
    Trainer, GradTrainer,
)
from xequinet.data import(
    H5Dataset, H5MemDataset, H5DiskDataset,
    data_unit_transform, atom_ref_transform,
)


def main():
    # ------------------- set up ------------------- #
    # parse config
    parser = argparse.ArgumentParser(description="Xequinet distributed training script")
    parser.add_argument(
        "--config", type=str, default="config.json",
        help="Configuration file (default: config.json).",
    )
    args = parser.parse_args()

    if os.path.isfile(args.config):
        config = NetConfig.parse_file(args.config)
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
    # select dataset type
    if config.dataset_type == "normal":
        Dataset = H5Dataset      # inherit from torch.utils.data.Dataset
    elif config.dataset_type == "memory":
        Dataset = H5MemDataset   # inherit from torch_geometric.data.InMemoryDataset
    elif config.dataset_type == "disk":
        Dataset = H5DiskDataset  # inherit from torch_geometric.data.Dataset
    else:
        raise ValueError(f"Unknown dataset type: {config.dataset_type}")
    
    # set property read from raw data
    prop_dict = {}
    if config.label_name is not None:
        prop_dict["y"] = config.label_name
    if config.blabel_name is not None:
        prop_dict["base_y"] = config.blabel_name
    if config.force_name is not None:
        prop_dict["force"] = config.force_name
    if config.bforce_name is not None:
        prop_dict["base_force"] = config.bforce_name

    # set transform function
    pre_transform = lambda data: data_unit_transform(
        data, config.label_unit, config.blabel_unit,
        config.force_unit, config.bforce_unit,
    )
    transform = lambda data: atom_ref_transform(data, config.atom_ref, config.batom_ref)
    # set dataset
    with distributed_zero_first(local_rank):
        train_dataset = Dataset(
            config.data_root, config.data_files, config.processed_name,
            "train", config.cutoff, config.max_mol, config.mem_process,
            transform, pre_transform, **prop_dict,
        )
        valid_dataset = Dataset(
            config.data_root, config.data_files, config.processed_name,
            "valid", config.cutoff, config.vmax_mol, config.mem_process,
            transform, pre_transform, **prop_dict,
        )
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

    # calculate element shifts and set to config
    config.node_mean = 0.0
    config.graph_mean = 0.0
    if isinstance(config.add_mean, float):
        if config.divided_by_atoms:
            config.node_mean = config.add_mean
        else:
            config.graph_mean = config.add_mean
    elif config.add_mean == True:
        mean, std = calculate_stats(train_loader, config.divided_by_atoms)
        log.s.info(f"Mean : {mean:.4f} {config.default_property_unit}")
        log.s.info(f"Std. : {std:.4f} {config.default_property_unit}")
        if config.divided_by_atoms:
            config.node_mean = mean
        else:
            config.graph_mean = mean

    # -------------------  build model ------------------- #
    # initialize model
    model = xPaiNN(config)
    log.s.info(model)
    model.to(device)

    # distributed training
    find_unused_parameters = True if config.finetune else False
    ddp_model = DDP(model, device_ids=[local_rank], output_device=device,
                    find_unused_parameters=find_unused_parameters)

    # record the number of parameters and frozen some when finetuning
    n_params = 0
    for name, param in ddp_model.named_parameters():
        if config.finetune and (not "out" in name) and (not "update" in name):
            param.requires_grad = False
            log.s.info(f"{name}: {param.numel()} (frozen)")
        else:
            n_params += param.numel()
            log.s.info(f"{name}: {param.numel()}")
    log.s.info(f"Total number of parameters to be optimized: {n_params}")
    
    # -------------------  train model ------------------- #
    if config.output_mode == "grad":
        NetTrainer = GradTrainer
    else:
        NetTrainer = Trainer
    trainer = NetTrainer(ddp_model, config, device, train_loader, valid_loader, train_sampler, log)
    trainer.start()
    

if __name__ == "__main__":
    main()
