import argparse
from typing import Any, Dict, Optional

import ase.io
import torch
from tabulate import tabulate
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

from xequinet import keys
from xequinet.data import (
    NeighborTransform,
    Transform,
    XequiData,
    datapoint_from_ase,
    datapoint_to_xtb,
)
from xequinet.nn import resolve_model
from xequinet.utils import get_default_units, qc, set_default_units, unit_conversion


@torch.no_grad()
def inference(
    model: torch.nn.Module,
    transform: Transform,
    dataloader: DataLoader,
    device: torch.device,
    output_file: str,
    compute_forces: bool = False,
    compute_virial: bool = False,
    base_method: Optional[str] = None,
    verbose: bool = False,
) -> Optional[Dict[str, Any]]:
    # get default units
    default_units = get_default_units()
    # save the results if verbose
    data_list, results_list = [], []
    # loop over the dataloader
    for data in dataloader:
        data: XequiData  # batch data
        data = data.to(device)
        data = transform(data)
        batch = data[keys.BATCH]
        with torch.enable_grad():
            result = model(data.to_dict(), compute_forces, compute_virial)

        if all(prop not in result for prop in keys.STANDARD_PROPERTIES):
            if not verbose:
                raise RuntimeError(
                    "No standard properties found in the model output, so nothing will be printed. \
                    Please use `--verbose` or `-v` to save the results in `.pt` file."
                )
            continue
        # loop over the batch
        for i in range(len(data)):
            datum: XequiData = data[i]  # single data point
            # add semi-empirical results if using delta-learning
            if base_method is not None:
                xtb_calc = datapoint_to_xtb(datum, method=base_method)
                xtb_res = xtb_calc.singlepoint()
                if keys.TOTAL_ENERGY in result:
                    xtb_energy = xtb_res.get("energy") * unit_conversion(
                        "Hartree", default_units[keys.TOTAL_ENERGY]
                    )
                    result[keys.TOTAL_ENERGY][i] += xtb_energy
                if keys.FORCES in result:
                    xtb_forces = -xtb_res.get("gradient") * unit_conversion(
                        "au", default_units[keys.FORCES]
                    )
                    result[keys.FORCES][batch == i] += torch.tensor(
                        xtb_forces, device=device
                    )
                if keys.VIRIAL in result:
                    xtb_virial = xtb_res.get("virial") * unit_conversion(
                        "au", default_units[keys.VIRIAL]
                    )
                    result[keys.VIRIAL][i] += torch.tensor(xtb_virial, device=device)
                if keys.ATOMIC_CHARGES in result:
                    xtb_charges = xtb_res.get("charges") * unit_conversion(
                        "e", default_units[keys.TOTAL_CHARGE]
                    )
                    result[keys.ATOMIC_CHARGES][batch == i] += torch.tensor(
                        xtb_charges, device=device
                    )
                if keys.DIPOLE in result:
                    xtb_dipole = xtb_res.get("dipole") * unit_conversion(
                        "au", default_units[keys.DIPOLE]
                    )
                    result[keys.DIPOLE][i] += torch.tensor(xtb_dipole, device=device)

            # write to output file
            with open(output_file, "a") as f:

                # write the cell information
                if hasattr(datum, keys.PBC) and datum[keys.PBC].any():
                    header = ["Cell", "x", "y", "z"]
                    table = [
                        ["a"] + [datum[keys.CELL][0].tolist()],
                        ["b"] + [datum[keys.CELL][1].tolist()],
                        ["c"] + [datum[keys.CELL][2].tolist()],
                    ]
                    f.write(tabulate(table, headers=header, tablefmt="plain"))
                    f.write("\n")
                # write the positions, forces, and charges
                header = ["Atoms", "x", "y", "z"]
                if keys.FORCES in result:
                    header.extend(["Fx", "Fy", "Fz"])
                if keys.ATOMIC_CHARGES in result:
                    header.append("Charges")
                table = []
                for j, atomic_number in enumerate(datum.atomic_numbers):
                    row = [qc.ELEMENTS_LIST[atomic_number.item()]]
                    row.extend(datum[keys.POSITIONS][j].tolist())
                    if keys.FORCES in result:
                        row.extend(result[keys.FORCES][j].tolist())
                    if keys.ATOMIC_CHARGES in result:
                        row.append(result[keys.ATOMIC_CHARGES][j].item())
                    table.append(row)
                f.write(
                    tabulate(table, headers=header, tablefmt="simple", floatfmt=".6f")
                )
                f.write("\n")

                # write the energy
                if keys.TOTAL_ENERGY in result:
                    f.write(f"Total energy {result[keys.TOTAL_ENERGY][i].item():.6f}\n")

                # write the stress
                if keys.VIRIAL in result:
                    header = ["", "xx", "yy", "zz", "yz", "zx", "xy"]
                    stress = result[keys.VIRIAL][i] / torch.det(datum[keys.CELL]).abs()
                    stress = stress.flatten()[0, 4, 8, 5, 2, 1]
                    table = [["Stress"] + stress.tolist()]
                    f.write(
                        tabulate(
                            table, headers=header, tablefmt="plain", floatfmt=".6f"
                        )
                    )
                    f.write("\n")

                # write the dipole
                if keys.DIPOLE in result:
                    header = ["", "x", "y", "z", "Magnitude"]
                    dipole = result[keys.DIPOLE][i]
                    magnitude = torch.linalg.norm(dipole)
                    table = [["Dipole"] + [dipole.tolist()] + [magnitude.item()]]
                    f.write(
                        tabulate(
                            table, headers=header, tablefmt="plain", floatfmt=".6f"
                        )
                    )
                    f.write("\n")

                # write the polarizability
                if keys.POLARIZABILITY in result:
                    header = ["", "xx", "yy", "zz", "yz", "zx", "xy", "Isotropic"]
                    polar = result[keys.POLARIZABILITY][i]
                    isotropic = (torch.trace(polar) / 3.0).item()
                    polar = polar.flatten()[0, 4, 8, 5, 2, 1]
                    table = [
                        ["Polarizability"] + [p.item() for p in polar] + [isotropic]
                    ]
                    f.write(
                        tabulate(
                            table, headers=header, tablefmt="plain", floatfmt=".6f"
                        )
                    )
                    f.write("\n")
                f.write("\n")
        # save the results if verbose
        if verbose:
            data_list.append(data)
            results_list.append(result)

    # collate the results if verbose
    if verbose:
        # collate data
        batch_data = Batch.from_data_list(data_list)
        # concatenate results
        cated_results = {}
        for prop in results_list[0].keys():
            cated_results[prop] = torch.cat(
                [result[prop] for result in results_list], dim=0
            )
        results = {"data": batch_data, "results": cated_results}
        return results

    return None


def run_infer(args: argparse.Namespace) -> None:
    # set device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # load checkpoint and config
    ckpt = torch.load(args.ckpt, map_location=device)
    model_config = ckpt["config"]

    # set default unit
    set_default_units(model_config["default_units"])

    # build model
    model = resolve_model(
        model_config["model_name"],
        **model_config["model_kwargs"],
    ).to(device)
    model.load_state_dict(ckpt["model"]).eval()
    transform = NeighborTransform(model.cutoff_radius)

    # whether to compute forces and virial
    compute_forces = args.forces
    compute_virial = args.stress

    # load input data
    atoms_list = ase.io.read(args.input, index=":")
    dataset = [datapoint_from_ase(atoms) for atoms in atoms_list]

    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0)
    output_file = (
        f"{args.input.split('/')[-1].split('.')[0]}.log"
        if args.output is None
        else args.output
    )

    with open(output_file, "w") as f:
        f.write(" --- XequiNet Inference\n")
        for prop, unit in get_default_units().items():
            if prop in keys.BASE_PROPERTIES:
                continue
            f.write(f" --- Property: {prop} ---- Unit: {unit}\n")
        f.write("\n")

    results = inference(
        model=model,
        transform=transform,
        dataloader=dataloader,
        device=device,
        output_file=output_file,
        compute_forces=compute_forces,
        compute_virial=compute_virial,
        base_method=args.delta,
        verbose=args.verbose,
    )

    if args.verbose:
        results_file = f"{args.input.split('/')[-1].split('.')[0]}.pt"
        torch.save(results, results_file)
