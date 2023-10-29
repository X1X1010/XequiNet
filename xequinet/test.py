import argparse

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from xequinet.data import H5Dataset, data_unit_transform, atom_ref_transform
from xequinet.nn import resolve_model
from xequinet.utils import NetConfig, unit_conversion, set_default_unit, get_default_unit, ModelWrapper
from xequinet.utils.qc import ELEMENTS_LIST


def test_scalar(model, test_loader, device, outfile):
    sum_loss, num_mol = 0.0, 0
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            pred = model(data)
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
                    coord = data.pos[idx] * unit_conversion(get_default_unit()[1], "Angstrom")
                    for a, c in zip(at_no, coord):
                        wf.write(f"{ELEMENTS_LIST[a.item()]} {c[0].item():10.7f} {c[1].item():10.7f} {c[2].item():10.7f}\n")
                    wf.write("real: ")
                    wf.write(" ".join([f"{r.item():10.7f}" for r in real[imol]]))
                    wf.write("  pred: ")
                    wf.write(" ".join([f"{p.item():10.7f}" for p in pred[imol]]))
                    wf.write("  loss: ")
                    wf.write(" ".join([f"{l.item():10.7f}" for l in l1loss[imol]]))
                    wf.write("\n")
    with open(outfile, 'a') as wf:
        wf.write(f"Test MAE: {sum_loss / num_mol:10.7f}\n")


def test_grad(model, test_loader, device, outfile):
    sum_lossE, sum_lossF, num_mol, num_atm = 0.0, 0.0, 0, 0
    for data in test_loader:
        data = data.to(device)
        data.pos.requires_grad = True
        predE, predF = model(data)
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
                coord = data.pos[idx] * unit_conversion(get_default_unit()[1], "Angstrom")
                for a, c, pF, rF, l in zip(at_no, coord, predF[idx], realF[idx], l1lossF[idx]):
                    wf.write(f"{ELEMENTS_LIST[a.item()]} {c[0].item():10.7f} {c[1].item():10.7f} {c[2].item():10.7f}  ")
                    wf.write(f"real: {rF[0].item():10.7f} {rF[1].item():10.7f} {rF[2].item():10.7f}  ")
                    wf.write(f"pred: {pF[0].item():10.7f} {pF[1].item():10.7f} {pF[2].item():10.7f}  ")
                    wf.write(f"loss: {l[0].item():10.7f} {l[1].item():10.7f} {l[2].item():10.7f}\n")
                wf.write(f"real: {realE[imol].item():10.7f}  pred: {predE[imol].item():10.7f}  loss: {l1lossE[imol].item():10.7f}\n")
    with open(outfile, 'a') as wf:
        wf.write(f"Test MAE: Energy {sum_lossE / num_mol:10.7f}  Force {sum_lossF / (3*num_atm):10.7f}\n")



def main():
    # parse config
    parser = argparse.ArgumentParser(description="XequiNet test script")
    parser.add_argument(
        "--config", "-C", type=str, default="config.json",
        help="Configuration file (default: config.json).",
    )
    parser.add_argument(
        "--ckpt", "-c", type=str, required=True,
        help="Xequinet checkpoint file. (XXX.pt containing 'model' and 'config')",
    )
    parser.add_argument(
        "--force", "-f", action="store_true",
        help="Whether testing force additionally when the output mode is 'scalar'",
    )
    args = parser.parse_args()

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load checkpoint and config
    config = NetConfig.parse_file(args.config)
    ckpt = torch.load(args.ckpt, map_location=device)
    config.parse_obj(ckpt["config"])
    
    # set default unit
    set_default_unit(config.default_property_unit, config.default_length_unit)

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
    pre_transform = lambda data: data_unit_transform(
        data, config.label_unit, config.blabel_unit, config.force_unit, config.bforce_unit,
    )
    transform = lambda data: atom_ref_transform(data, config.atom_ref, config.batom_ref)
    # load dataset
    test_dataset = H5Dataset(
        config.data_root, config.data_files, config.processed_name,
        "test", config.cutoff, config.vmax_mol,
        config.mem_process, transform, pre_transform,
        **prop_dict,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.vbatch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=True, drop_last=False,
    )
    
    # adjust some configurations
    config.node_mean = 0.0; config.graph_mean = 0.0
    if args.force == True and config.output_mode == "scalar":
        config.output_mode = "grad"
    
    # build model
    model = resolve_model(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # test
    output_file = f"{config.run_name}_test.log"
        
    with open(output_file, 'w') as wf:
        wf.write("XequiNet testing\n")
        wf.write(f"Unit: {config.default_property_unit} {config.default_length_unit}\n")

    if config.output_mode == "grad":
        test_grad(ModelWrapper(model, config.pbc), test_loader, device, output_file)
    else:
        test_scalar(ModelWrapper(model, config.pbc), test_loader, device, output_file)


if __name__ == "__main__":
    main()