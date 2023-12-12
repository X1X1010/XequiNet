import argparse

import torch
from torch_geometric.loader import DataLoader

from xequinet.data import create_dataset
from xequinet.nn import resolve_model
from xequinet.utils import (
    NetConfig,
    unit_conversion, set_default_unit, get_default_unit,
    ModelWrapper,
    gen_3Dinfo_str,
)


def test_scalar(model, test_loader, device, outfile, output_dim=1, verbose=0):
    p_unit, l_unit = get_default_unit()
    sum_loss = torch.zeros(output_dim, device=device)
    num_mol = 0
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            pred = model(data)
            if hasattr(data, "base_y"):
                pred += data.base_y
            real = data.y
            error = real - pred
            sum_loss += error.abs().sum(dim=0)
            if verbose >= 1:
                with open(outfile, 'a') as wf:
                    for imol in range(len(data.y)):
                        at_no = data.at_no[data.batch == imol]
                        coord = data.pos[data.batch == imol] * unit_conversion(l_unit, "Angstrom")
                        wf.write(f"mol {num_mol + imol + 1}\n")
                        if verbose >= 2:  # print atom coordinates
                            wf.write(gen_3Dinfo_str(at_no, coord, title="Coordinates (Angstrom)"))
                        wf.write(f"Real:")
                        wf.write("".join([f"{r.item():15.9f} " for r in real[imol]]))
                        wf.write(f"    Predict:")
                        wf.write("".join([f"{p.item():15.9f}" for p in pred[imol]]))
                        wf.write(f"    Error:")
                        wf.write("".join([f"{l.item():15.9f}" for l in error[imol]]))
                        wf.write(f"    ({p_unit})\n\n")
        num_mol += len(data.y)
    with open(outfile, 'a') as wf:
        avg_loss = sum_loss / num_mol
        wf.write(f"Test MAE:")
        wf.write("".join([f"{l:15.9f}" for l in avg_loss]))
        wf.write(f"  {p_unit}\n")


def test_grad(model, test_loader, device, outfile, verbose=0):
    p_unit, l_unit = get_default_unit()
    sum_lossE, sum_lossF, num_mol, num_atom = 0.0, 0.0, 0, 0
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
            errorE = realE - predE
            errorF = realF - predF
            sum_lossE += errorE.abs().sum()
            sum_lossF += errorF.abs().sum()
        if verbose >= 1:
            with open(outfile, 'a') as wf:
                for imol in range(len(data.y)):
                    idx = (data.batch == imol)
                    at_no = data.at_no[idx]
                    coord = data.pos[idx] * unit_conversion(l_unit, "Angstrom")
                    wf.write(f"mol {num_mol + imol + 1}\n")
                    if verbose >= 2:  # print atom coordinates
                        info_3ds = [coord, predF[idx], realF[idx], errorF[idx]]
                        titles = [
                            "Coordinates (Angstrom)",
                            f"Predicted Forces ({p_unit}/{l_unit})",
                            f"Real Forces ({p_unit}/{l_unit})",
                            f"Error Forces ({p_unit}/{l_unit})"
                        ]
                        precisions = [6, 9, 9, 9]
                        wf.write(gen_3Dinfo_str(at_no, info_3ds, titles, precisions))
                    wf.write(f"Energy | Real: {realE[imol].item():15.9f}    ")
                    wf.write(f"Predict: {predE[imol].item():15.9f}    ")
                    wf.write(f"Error: {errorE[imol].item():15.9f}    {p_unit}\n")
                    wf.write(f"Force  | MAE : {errorF[idx].abs().mean():15.9f}   {p_unit}/{l_unit}\n\n")
        num_mol += data.y.numel()
        num_atom += data.at_no.numel()
    with open(outfile, 'a') as wf:
        wf.write(f"Energy MAE : {sum_lossE / num_mol:15.9f}    {p_unit}\n")
        wf.write(f"Force  MAE : {sum_lossF / (3*num_atom):15.9f}   {p_unit}/{l_unit}\n")


def test_vector(model, test_loader, device, outfile, verbose=0):
    p_unit, l_unit = get_default_unit()
    sum_loss = torch.zeros(3, device=device)
    num_mol = 0
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            pred = model(data)
            real = data.y
            error = real - pred
            sum_loss += error.abs().sum()
            num_mol += len(data.y)
            if verbose >= 1:
                with open(outfile, 'a') as wf:
                    for imol in range(len(data.y)):
                        at_no = data.at_no[data.batch == imol]
                        coord = data.pos[data.batch == imol] * unit_conversion(l_unit, "Angstrom")
                        wf.write(f"mol {num_mol + imol + 1}\n")
                        if verbose >= 2:  # print atom coordinates
                            wf.write(gen_3Dinfo_str(at_no, coord, title="Coordinates (Angstrom)"))
                        values = [
                            f"X{vec[imol][0].item():12.6f}  Y{vec[imol][1].item():12.6f}  Z{vec[imol][2].item():12.6f}"
                            for vec in [real, pred, error]
                        ]
                        titles = [f"Real ({p_unit})", f"Predict ({p_unit})", f"Error ({p_unit})"]
                        filled_t = [f"{t: <{len(v)}}" for t, v in zip(titles, values)]
                        wf.write("    ".join(filled_t) + "\n")
                        wf.write("    ".join(values) + "\n\n")
        num_mol += len(data.y)
    with open(outfile, 'a') as wf:
        wf.write(f"Test MAE: {sum_loss / num_mol / 3 :12.6f} {p_unit}\n")


def test_polar(model, test_loader, device, outfile, verbose=0):
    p_unit, l_unit = get_default_unit()
    sum_loss = torch.zeros((3,3), device=device)
    num_mol = 0
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            pred = model(data)
            real = data.y
            error = real - pred
            sum_loss += error.abs().sum()
            if verbose >= 1:
                with open(outfile, 'a') as wf:
                    for imol in range(len(data.y)):
                        at_no = data.at_no[data.batch == imol]
                        coord = data.pos[data.batch == imol] * unit_conversion(l_unit, "Angstrom")
                        wf.write(f"mol {num_mol + imol + 1}\n")
                        if verbose >= 2:  # print atom coordinates
                            wf.write(gen_3Dinfo_str(at_no, coord, title="Coordinates (Angstrom)"))
                        tri_values = []
                        for i, D in enumerate(['X', 'Y', 'Z']):
                            tri_values.append([
                                f"{D}X{pol[imol][i,0].item():12.6f}  {D}Y{pol[imol][i,1].item():12.6f}  {D}Z{pol[imol][i,2].item():12.6f}"
                                for pol in [real, pred, error]
                            ])
                        titles = [f"Real ({p_unit})", f"Predict ({p_unit})", f"Error ({p_unit})"]
                        filled_t = [f"{t: <{len(v)}}" for t, v in zip(titles, tri_values[0])]
                        wf.write("    ".join(filled_t) + "\n")
                        for values in tri_values:
                            wf.write("    ".join(values) + "\n")
                        wf.write("\n")
        num_mol += len(data.y)
    with open(outfile, 'a') as wf:
        wf.write(f"Test MAE: {sum_loss / num_mol / 9 :12.6f} {p_unit}\n")


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
    parser.add_argument(
        "--no-force", "-nf", action="store_true",
        help="Whether not testing force when the output mode is 'grad'",
    )
    parser.add_argument(
        "--verbose", "-v", type=int, default=0, choices=[0, 1, 2],
        help="Verbose level. (default: 0)",
    )
    parser.add_argument(
        "--batch-size", "-bz", type=int, default=32,
        help="Batch size. (default: 32)",
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

    test_dataset = create_dataset(config, "test")
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=True, drop_last=False,
    )
    
    # adjust some configurations
    if args.force == True and config.output_mode == "scalar":
        config.output_mode = "grad"
    if args.no_force == True and config.output_mode == "grad":
        config.output_mode = "scalar"
    
    # build model
    model = resolve_model(config).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    # test
    output_file = f"{config.run_name}_test.log"
        
    with open(output_file, 'w') as wf:
        wf.write("XequiNet testing\n")
        wf.write(f"Unit: {config.default_property_unit} {config.default_length_unit}\n")

    if config.output_mode == "grad":
        test_grad(ModelWrapper(model, config.version), test_loader, device, output_file, args.verbose)
    elif config.output_mode == "vector" and config.output_dim == 3:
        test_vector(ModelWrapper(model, config.version), test_loader, device, output_file, args.verbose)
    elif config.output_mode == "polar" and config.output_dim == 9:
        test_polar(ModelWrapper(model, config.version), test_loader, device, output_file, args.verbose)
    elif config.output_mode == "scalar":
        test_scalar(ModelWrapper(model, config.version), test_loader, device, output_file, config.output_dim, args.verbose)
    else:
        raise ValueError(f"Unsupported output mode: {config.output_mode}")


if __name__ == "__main__":
    main()