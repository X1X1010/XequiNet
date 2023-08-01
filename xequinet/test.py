import os
import argparse

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from xpainn.data import H5Dataset, H5MemDataset, H5DiskDataset, data_unit_transform, atom_ref_transform
# from xphormer.nn.painn_ori import PaiNN
from xpainn.nn import XPaiNN
from xpainn.utils import NetConfig, unit_conversion, set_default_unit
from xpainn.utils.qc import ELEMENTS_LIST


def test_scalar(model, test_loader, device, outfile):
    model.eval()
    sum_loss, num_mol = 0.0, 0
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            pred = model(data.at_no, data.pos, data.edge_index, data.batch)
            if hasattr(data, "base_y"):
                pred += data.base_y
            real = data.y
            l1loss = F.l1_loss(pred, real, reduce=False)
            sum_loss += l1loss.sum().item()
            num_mol += len(data.y)
            with open(outfile, 'a') as wf:
                for imol in range(len(data.y)):
                    idx = (data.batch == imol)
                    at_no = data.at_no[idx]
                    coord = data.pos[idx]
                    for a, c in zip(at_no, coord):
                        wf.write(f"{ELEMENTS_LIST[a.item()]} {c[0].item():10.7f} {c[1].item():10.7f} {c[2].item():10.7f}\n")
                    wf.write(f"real: {real[imol].tolist()}  pred: {pred[imol].tolist()}  loss: {l1loss[imol].tolist()}\n")
    with open(outfile, 'a') as wf:
        wf.write(f"Test MAE: {sum_loss / num_mol:10.7f}\n")


def test_grad(model, test_loader, device, outfile):
    model.eval()
    sum_lossE, sum_lossF, num_mol, num_atm = 0.0, 0.0, 0, 0
    for data in test_loader:
        data = data.to(device)
        predE, predF = model(data.at_no, data.pos, data.edge_index, data.batch)
        with torch.no_grad():
            if hasattr(data, "base_y"):
                predE += data.base_y
            if hasattr(data, "base_force"):
                predF += data.base_force
            realE, realF = data.y, data.force
            l1lossE = F.l1_loss(predE, realE, reduce=False)
            l1lossF = F.l1_loss(predF, realF, reduce=False)
            sum_lossE += l1lossE.sum().item()
            sum_lossF += l1lossF.sum().item()
            num_mol += data.y.numel()
            num_atm += data.at_no.numel()
        with open(outfile, 'a') as wf:
            for imol in range(len(data.y)):
                idx = (data.batch == imol)
                at_no = data.at_no[idx]
                coord = data.pos[idx] * unit_conversion("bohr", "angstrom")
                for a, c in zip(at_no, coord):
                    wf.write(f"{ELEMENTS_LIST[a.item()]} {c[0].item():10.7f} {c[1].item():10.7f} {c[2].item():10.7f}  ")
                    wf.write(f"real: {realF[imol][0].item():10.7f} {realF[imol][1].item():10.7f} {realF[imol][2].item():10.7f}  ")
                    wf.write(f"pred: {predF[imol][0].item():10.7f} {predF[imol][1].item():10.7f} {predF[imol][2].item():10.7f}  ")
                    wf.write(f"loss: {l1lossF[imol][0].item():10.7f} {l1lossF[imol][1].item():10.7f} {l1lossF[imol][2].item():10.7f}\n")
                wf.write(f"real: {realE[imol].item():10.7f}  pred: {predE[imol].item():10.7f}  loss: {l1lossE[imol].item():10.7f}\n")
    with open(outfile, 'a') as wf:
        wf.write(f"Test MAE: Energy {sum_lossE / num_mol:10.7f}  Force {sum_lossF / (3*num_atm):10.7f}\n")



def main():
    # parse config
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.json", help="configuration file")
    parser.add_argument("--output-file", "-o", default=None, help="output file name")
    args = parser.parse_args()
    config = NetConfig.parse_file(args.config)
    
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    
    # set default unit
    set_default_unit(config.default_property_unit, config.default_length_unit)

    # update configuration with config saved in checkpoint
    ckpt = torch.load(config.ckpt_file, map_location=device)
    config.parse_obj(ckpt["config"])

    # choose dataset type
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

    # set unit transform function
    pre_transform = lambda data: atom_ref_transform(
        data, config.atom_ref, config.batom_ref, config.label_unit, config.blabel_unit,
    )
    transform = lambda data: data_unit_transform(
        data, config.label_unit, config.blabel_unit, config.force_unit, config.bforce_unit,
    )
    # load dataset
    test_dataset = Dataset(
        config.data_root, config.data_files, "test", config.embed_basis,
        config.cutoff, config.vmax_mol, config.mem_process, transform, pre_transform,
        **prop_dict,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.vbatch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=True, drop_last=False,
    )
    
    # build model
    model = XPaiNN(config).to(device)
    model.load_state_dict(ckpt["model"])

    # test
    if args.output_file is None:
        args.output_file = f"{config.run_name}.out"
        
    with open(args.output_file, 'w') as wf:
        wf.write("Xphormer testing\n")

    if config.output_mode == "scalar":
        test_scalar(model, test_loader, device, args.output_file)
    elif config.output_mode == "grad":
        test_grad(model, test_loader, device, args.output_file)
    else:
        raise NotImplementedError(f"Unknown output mode: {config.output_mode}")
    

if __name__ == "__main__":
    main()