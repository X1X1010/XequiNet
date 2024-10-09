import warnings
import argparse

import torch
from torch_scatter import scatter
from torch_cluster import radius_graph
from torch_geometric.loader import DataLoader

try:
    import tblite.interface as xtb
except:
    warnings.warn("xtb is not installed, xtb calculation will not be performed.")

from ..nn import resolve_model
from ..utils import (
    NetConfig,
    set_default_unit,
    get_default_unit,
    unit_conversion,
    get_atomic_energy,
    gen_3Dinfo_str,
    radius_batch_pbc,
)
from ..data import TextDataset


@torch.no_grad()
def predict_scalar(
    model: torch.nn.Module,
    cutoff: float,
    max_edges: int,
    dataloader: DataLoader,
    device: torch.device,
    output_file: str,
    atom_sp: torch.Tensor,
    base_method: str = None,
    verbose: int = 0,
) -> None:
    wf = open(output_file, "a")
    p_unit, l_unit = get_default_unit()
    for data in dataloader:
        if hasattr(data, "pbc") and data.pbc.any():
            data.edge_index, data.shifts = radius_batch_pbc(
                data.pos, data.pbc, data.lattice, cutoff, max_num_neighbors=max_edges
            )
            data = data.to(device)
        else:
            data = data.to(device)
            data.edge_index = radius_graph(
                data.pos, r=cutoff, batch=data.batch, max_num_neighbors=max_edges
            )
        # change the prediction unit to atomic unit
        nn_preds = model(data).double() * unit_conversion(p_unit, "AU")
        # the atomic reference is already in atomic unit
        atom_refs = scatter(atom_sp[data.at_no], data.batch, dim=0).unsqueeze(-1)
        for i in range(len(data)):
            datum = data[i].cpu()
            nn_pred = nn_preds[i].cpu()
            atom_ref = atom_refs[i].cpu()
            if base_method is not None:
                m_dict = {
                    "gfn2-xtb": "GFN2-xTB",
                    "gfn1-xtb": "GFN1-xTB",
                    "ipea1-xtb": "IPEA1-xTB",
                }
                calc = xtb.Calculator(
                    method=m_dict[base_method.lower()],
                    numbers=datum.at_no.numpy(),
                    positions=datum.pos.numpy() * unit_conversion(l_unit, "Bohr"),
                    charge=int(datum.charge.item()),
                    uhf=int(datum.spin.item()),
                )
                res = calc.singlepoint()
                delta = res.get("energy")  # the delta value is already in atomic unit
            else:
                delta = 0.0
            total = nn_pred + atom_ref + delta
            if verbose >= 2:  # print coordinates
                wf.write(
                    gen_3Dinfo_str(
                        datum.at_no,
                        datum.pos
                        * unit_conversion(l_unit, "Angstrom"),  # change to Angstrom
                        title="Coordinates (Angstrom)",
                    )
                )
                wf.write(
                    f"Charge {int(datum.charge.item())}   Multiplicity {int(datum.spin.item()) + 1}\n"
                )
            if verbose >= 1:  # print detailed information
                wf.write("NN Contribution         :")
                for v_nn in nn_pred:
                    wf.write(f"    {v_nn.item():10.7f} a.u.")
                wf.write("\nSum of Atomic Reference :")
                for v_at in atom_ref:
                    wf.write(f"    {v_at.item():10.7f} a.u.")
                if base_method is not None:
                    wf.write("\nDelta Method Base       :")
                    wf.write(f"    {delta.item():10.7f} a.u.")
                wf.write("\n")
            wf.write("Total Property          :")
            for v_tot in total:
                wf.write(f"    {v_tot.item():10.7f} a.u.")
            wf.write("\n\n")
            wf.flush()
    wf.close()


def predict_grad(
    model: torch.nn.Module,
    cutoff: float,
    max_edges: int,
    dataloader: DataLoader,
    device: torch.device,
    output_file: str,
    atom_sp: torch.Tensor,
    base_method: str = None,
    verbose: int = 0,
) -> None:
    wf = open(output_file, "a")
    p_unit, l_unit = get_default_unit()
    for data in dataloader:
        if hasattr(data, "pbc") and data.pbc.any():
            data.edge_index, data.shifts = radius_batch_pbc(
                data.pos, data.pbc, data.lattice, cutoff, max_num_neighbors=max_edges
            )
            data = data.to(device)
        else:
            data = data.to(device)
            data.edge_index = radius_graph(
                data.pos, r=cutoff, batch=data.batch, max_num_neighbors=max_edges
            )
        data.pos.requires_grad = True
        nn_predEs, nn_predFs = model(data)
        # change the prediction unit to atomic unit
        nn_predEs *= unit_conversion(p_unit, "AU")
        nn_predFs *= unit_conversion(f"{p_unit}/{l_unit}", "AU")
        # the atomic reference is already in atomic unit
        atom_Es = scatter(atom_sp[data.at_no], data.batch, dim=0)
        for i in range(len(data)):
            datum = data[i].cpu()
            nn_predE = nn_predEs[i].cpu()
            nn_predF = nn_predFs[i].cpu()
            atom_E = atom_Es[i].cpu()
            if base_method is not None:
                m_dict = {
                    "gfn2-xtb": "GFN2-xTB",
                    "gfn1-xtb": "GFN1-xTB",
                    "ipea1-xtb": "IPEA1-xTB",
                }
                calc = xtb.Calculator(
                    method=m_dict[base_method.lower()],
                    numbers=datum.at_no.numpy(),
                    positions=datum.pos.numpy() * unit_conversion(l_unit, "Bohr"),
                    charge=int(datum.charge.item()),
                    uhf=int(datum.spin.item()),
                )
                res = calc.singlepoint()
                d_ene = res.get("energy")  # the delta value is already in atomic unit
                d_grad = res.get(
                    "gradient"
                )  # the delta gradient is already in atomic unit
            else:
                d_ene = 0.0
                d_grad = torch.zeros_like(nn_predF)
            totalE = nn_predE + atom_E + d_ene
            totalF = nn_predF - d_grad
            if verbose >= 2:  # print coordinates
                wf.write(
                    gen_3Dinfo_str(
                        datum.at_no,
                        datum.pos
                        * unit_conversion(
                            get_default_unit()[1], "Angstrom"
                        ),  # change to Angstrom
                        title="Coordinates (Angstrom)",
                    )
                )
                wf.write(
                    f"Charge {int(datum.charge.item())}   Multiplicity {int(datum.spin.item()) + 1}\n"
                )
            if verbose >= 1:  # print detailed information
                wf.write("NN Contribution         :")
                wf.write(f"    {nn_predE.item():10.7f} a.u.\n")
                wf.write("Sum of Atomic Reference :")
                wf.write(f"    {atom_E[i].item():10.7f} a.u.\n")
                if base_method is not None:
                    wf.write("Delta Method Base       :")
                    wf.write(f"    {d_ene:10.7f} a.u.\n")
            wf.write(f"Total Energy            :")
            wf.write(f"    {totalE.item():10.7f} a.u.\n")
            allFs, titles = [], []
            if verbose >= 1 and base_method is not None:
                allFs.extend([nn_predF, -d_grad])
                titles.append(["NN Forces (a.u.)", "Delta Forces (a.u.)"])
            allFs.append(totalF)
            titles.append("Total Forces (a.u.)")
            wf.write(gen_3Dinfo_str(datum.at_no, allFs, titles, precision=9))
            wf.write("\n")
            wf.flush()
    wf.close()


@torch.no_grad()
def predict_vector(
    model: torch.nn.Module,
    cutoff: float,
    max_edges: int,
    dataloader: DataLoader,
    device: torch.device,
    output_file: str,
    verbose: int = 0,
) -> None:
    wf = open(output_file, "a")
    p_unit, l_unit = get_default_unit()
    for data in dataloader:
        if hasattr(data, "pbc") and data.pbc.any():
            data.edge_index, data.shifts = radius_batch_pbc(
                data.pos, data.pbc, data.lattice, cutoff, max_num_neighbors=max_edges
            )
            data = data.to(device)
        else:
            data = data.to(device)
            data.edge_index = radius_graph(
                data.pos, r=cutoff, batch=data.batch, max_num_neighbors=max_edges
            )
        pred = model(data)
        pred *= unit_conversion(p_unit, "AU")
        for i in range(len(data)):
            datum = data[i]
            vector = pred[i]
            if verbose >= 2:  # print coordinates
                wf.write(
                    gen_3Dinfo_str(
                        datum.at_no,
                        datum.pos * unit_conversion(l_unit, "Angstrom"),
                        title="Coordinates (Angstrom)",
                    )
                )
                wf.write(
                    f"Charge {int(datum.charge.item())}   Multiplicity {int(datum.spin.item()) + 1}\n"
                )
            wf.write(f"Vector Property (a.u.):\n")
            wf.write(
                f"X{vector[0].item():12.6f}  Y{vector[1].item():12.6f}  Z{vector[2].item():12.6f}\n\n"
            )
            wf.flush()
    wf.close()


@torch.no_grad()
def predict_polar(
    model: torch.nn.Module,
    cutoff: float,
    max_edges: int,
    dataloader: DataLoader,
    device: torch.device,
    output_file: str,
    verbose: int = 0,
) -> None:
    wf = open(output_file, "a")
    p_unit, l_unit = get_default_unit()
    for data in dataloader:
        if hasattr(data, "pbc") and data.pbc.any():
            data.edge_index, data.shifts = radius_batch_pbc(
                data.pos, data.pbc, data.lattice, cutoff, max_num_neighbors=max_edges
            )
            data = data.to(device)
        else:
            data = data.to(device)
            data.edge_index = radius_graph(
                data.pos, r=cutoff, batch=data.batch, max_num_neighbors=max_edges
            )
        pred = model(data)
        pred *= unit_conversion(p_unit, "AU")
        for i in range(len(data)):
            datum = data[i]
            polar = pred[i]
            if verbose >= 2:  # print coordinates
                wf.write(
                    gen_3Dinfo_str(
                        datum.at_no,
                        datum.pos * unit_conversion(l_unit, "Angstrom"),
                        title="Coordinates (Angstrom)",
                    )
                )
                wf.write(
                    f"Charge {int(datum.charge.item())}   Multiplicity {int(datum.spin.item()) + 1}\n"
                )
            wf.write("Polar property (a.u.):\n")
            wf.write(
                f"XX{polar[0,0].item():12.6f}  XY{polar[0,1].item():12.6f}  XZ{polar[0,2].item():12.6f}\n"
            )
            wf.write(
                f"YX{polar[1,0].item():12.6f}  YY{polar[1,1].item():12.6f}  YZ{polar[1,2].item():12.6f}\n"
            )
            wf.write(
                f"ZX{polar[2,0].item():12.6f}  ZY{polar[2,1].item():12.6f}  ZZ{polar[2,2].item():12.6f}\n\n"
            )
            wf.flush()
    wf.close()


def run_infer(args: argparse.Namespace) -> None:
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load checkpoint and config
    ckpt = torch.load(args.ckpt, map_location=device)
    config = NetConfig.model_validate(ckpt["config"])

    # set default unit
    set_default_unit(config.default_property_unit, config.default_length_unit)

    # adjust some configurations
    if args.force:
        config.output_mode = "grad"
    elif config.output_mode == "grad":
        config.output_mode = "scalar"

    # build model
    model = resolve_model(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # load input data
    dataset = TextDataset(file=args.input)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0)
    outp = (
        f"{args.input.split('/')[-1].split('.')[0]}.log"
        if args.output is None
        else args.output
    )

    # get atom reference
    atom_sp = get_atomic_energy(config.atom_ref) - get_atomic_energy(config.batom_ref)
    atom_sp = atom_sp.to(device) * unit_conversion(get_default_unit()[0], "AU")

    with open(outp, "w") as wf:
        wf.write("XequiNet prediction\n")
        wf.write(f"Coordinates in Angstrom, Properties in Atomic Unit\n\n")
    if config.output_mode == "scalar":
        predict_scalar(
            model=model,
            cutoff=config.cutoff,
            max_edges=config.max_edges,
            dataloader=dataloader,
            device=device,
            output_file=outp,
            atom_sp=atom_sp,
            base_method=args.delta,
            verbose=args.verbose,
        )
    elif config.output_mode == "grad":
        predict_grad(
            model=model,
            cutoff=config.cutoff,
            max_edges=config.max_edges,
            dataloader=dataloader,
            device=device,
            output_file=outp,
            atom_sp=atom_sp,
            base_method=args.delta,
            verbose=args.verbose,
        )
    elif config.output_mode == "vector":
        predict_vector(
            model=model,
            cutoff=config.cutoff,
            max_edges=config.max_edges,
            dataloader=dataloader,
            device=device,
            output_file=outp,
            verbose=args.verbose,
        )
    elif config.output_mode == "polar":
        predict_polar(
            model=model,
            cutoff=config.cutoff,
            max_edges=config.max_edges,
            dataloader=dataloader,
            device=device,
            output_file=outp,
            verbose=args.verbose,
        )
    else:
        raise ValueError(f"Unknown output mode: {config.output_mode}")
