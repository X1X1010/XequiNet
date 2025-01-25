import argparse
from typing import Any, Dict, Iterable, Optional, Tuple, cast

import torch
from omegaconf import OmegaConf
from tabulate import tabulate
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

from xequinet import keys
from xequinet.data import XequiData, create_lmdb_dataset
from xequinet.nn import resolve_model
from xequinet.utils import (
    DataConfig,
    ModelConfig,
    get_default_units,
    qc,
    set_default_units,
)
from xequinet.utils.loss import ErrorMetric


class AverageMetric:
    def __init__(self, properties: Iterable) -> None:
        self.properties: Dict[str, float] = {prop: 0.0 for prop in properties}
        self.counts: Dict[str, int] = {prop: 0 for prop in properties}
        self.reset()

    def reset(self) -> None:
        for prop in self.properties:
            self.properties[prop] = 0.0
            self.counts[prop] = 0

    def update(self, property: str, value: float, n: int = 1) -> None:
        assert property in self.properties, f"Property {property} not found"
        self.properties[property] += value
        self.counts[property] += n

    def reduce(self) -> Dict[str, float]:
        result = {}
        for prop, val in self.properties.items():
            count = self.counts[prop]
            result[prop] = val / count
        return result


@torch.no_grad()
def test(
    model: torch.nn.Module,
    data_loader: DataLoader,
    err_metric: ErrorMetric,
    meter: AverageMetric,
    device: torch.device,
    compute_forces: bool = False,
    compute_virial: bool = False,
    verbose: bool = False,
) -> Tuple[Dict[str, float], Optional[Dict[str, Any]]]:

    data_list, results_list = [], []
    for data in data_loader:
        data: XequiData
        data = data.to(device)
        with torch.enable_grad():
            result = model(data.to_dict(), compute_forces, compute_virial)
        errors = err_metric(result, data)
        for prop, (l1, l2, n) in errors.items():
            meter.update(prop, l1, l2, n)
        if verbose:
            data_list.append(data)
            results_list.append(result)
    # reduce the meter
    reduced_err = meter.reduce()
    if verbose:
        # collate data
        batch_data = Batch.from_data_list(data_list)
        # concatenate results
        cated_results = {}
        for prop in results_list[0]:
            cated_results[prop] = torch.cat(
                [result[prop] for result in results_list], dim=0
            )
        results = {"data": batch_data, "results": cated_results}
    else:
        results = None

    return reduced_err, results


def write_results(results: Dict[str, Any], output_file: str) -> None:
    data, pred = results["data"], results["results"]
    batch = data[keys.BATCH]
    pbc = hasattr(data, keys.PBC) and data[keys.PBC].any()
    n_total = batch.max().item() + 1
    with open(output_file, "a") as f:
        for i in range(n_total):
            mask = batch == i
            if pbc:
                cell = data[keys.CELL][i]
                header = ["Cell", "x", "y", "z"]
                table = [
                    ["a"] + [a.item() for a in cell[0]],
                    ["b"] + [b.item() for b in cell[1]],
                    ["c"] + [c.item() for c in cell[2]],
                ]
                f.write(
                    tabulate(table, headers=header, tablefmt="plain", floatfmt=".6f")
                )
                f.write("\n")

            # write the positions forces and charges
            header = ["Atom", "x", "y", "z"]
            if keys.FORCES in pred and keys.FORCES in data:
                header.extend(["Pred Fx", "Pred Fy", "Pred Fz"])
                header.extend(["Real Fx", "Real Fy", "Real Fz"])
            if keys.ATOMIC_CHARGES in pred and keys.ATOMIC_CHARGES in data:
                header.extend(["Pred q", "Real q"])
            table = []
            for j, atomic_number in enumerate(data[keys.ATOMIC_NUMBERS][mask]):
                row = [qc.ELEMENTS_LIST[atomic_number.item()]]
                row.extend(data[keys.POSITIONS][mask][j].tolist())
                if keys.FORCES in pred and keys.FORCES in data:
                    row.extend(pred[keys.FORCES][mask][j].tolist())
                    row.extend(data[keys.FORCES][mask][j].tolist())
                if keys.ATOMIC_CHARGES in pred and keys.ATOMIC_CHARGES in data:
                    row.append(pred[keys.ATOMIC_CHARGES][mask][j].item())
                    row.append(data[keys.ATOMIC_CHARGES][mask][j].item())
                table.append(row)
            f.write(tabulate(table, headers=header, tablefmt="plain", floatfmt=".6f"))
            f.write("\n")

            # write the energy
            if keys.TOTAL_ENERGY in pred and keys.TOTAL_ENERGY in data:
                f.write(f"Pred Energy: {pred[keys.TOTAL_ENERGY][i].item()}  ")
                f.write(f"Real Energy: {data[keys.TOTAL_ENERGY][i].item()}\n")

            # write the virial
            if keys.VIRIAL in pred and keys.VIRIAL in data:
                header = ["", "xx", "yy", "zz", "yz", "zx", "xy"]
                table = [["Pred Virial"], ["Real Virial"]]
                table[0].extend(
                    pred[keys.VIRIAL][i].flatten()[0, 4, 8, 5, 1, 2].tolist()
                )
                table[1].extend(
                    data[keys.VIRIAL][i].flatten()[0, 4, 8, 5, 1, 2].tolist()
                )
                f.write(
                    tabulate(table, headers=header, tablefmt="plain", floatfmt=".6f")
                )
                f.write("\n")

            # write the dipole
            if keys.DIPOLE in pred and keys.DIPOLE in data:
                header = ["", "x", "y", "z"]
                table = [["Pred Dipole"], ["Real Dipole"]]
                table[0].extend(pred[keys.DIPOLE][i].tolist())
                table[1].extend(data[keys.DIPOLE][i].tolist())
                f.write(
                    tabulate(table, headers=header, tablefmt="plain", floatfmt=".6f")
                )
                f.write("\n")

            # write the polar
            if keys.POLARIZABILITY in pred and keys.POLARIZABILITY in data:
                header = ["", "xx", "yy", "zz", "yz", "zx", "xy"]
                table = [["Pred Polar"], ["Real Polar"]]
                table[0].extend(
                    pred[keys.POLARIZABILITY][i].flatten()[0, 4, 8, 5, 1, 2].tolist()
                )
                table[1].extend(
                    data[keys.POLARIZABILITY][i].flatten()[0, 4, 8, 5, 1, 2].tolist()
                )
                f.write(
                    tabulate(table, headers=header, tablefmt="plain", floatfmt=".6f")
                )
                f.write("\n")
            f.write("\n")


def run_test(args: argparse.Namespace) -> None:
    # set device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # load checkpoint and config
    yaml_config = OmegaConf.load(args.config)
    data_config = OmegaConf.merge(
        OmegaConf.structured(DataConfig),
        yaml_config["data"],
    )
    ckpt = torch.load(args.ckpt, map_location=device)
    model_config = OmegaConf.merge(
        OmegaConf.structured(ModelConfig),
        yaml_config["model"],
    )
    # these will do nothing, only for type annotation
    data_config = cast(DataConfig, data_config)
    model_config = cast(ModelConfig, model_config)

    # set default unit
    set_default_units(model_config.default_units)

    # set default type
    name_to_dtype = {
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
    }
    torch.set_default_dtype(name_to_dtype[data_config.default_dtype])

    # build model
    model = resolve_model(
        model_name=model_config.model_name,
        **model_config.model_kwargs,
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    # for non-pbc Ewald model, we should use SVD frame to rotate the atomic positions
    if (
        model_config.model_name.lower() == "xpainn-ewald"
        and not model_config.model_kwargs["use_pbc"]
    ):
        svd_frame = True
    else:
        svd_frame = False
    # load test dataset
    test_dataset = create_lmdb_dataset(
        db_path=data_config.db_path,
        cutoff=model.cutoff_radius,
        split=data_config.split,
        targets=data_config.targets,
        base_targets=data_config.base_targets,
        dtype=data_config.default_dtype,
        mode="test",
        svd_frame=svd_frame,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=data_config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    # set meter
    err_metric = ErrorMetric(*data_config.targets)
    meter = AverageMetric(err_metric.properties)

    # testing
    compute_forces = keys.FORCES in data_config.targets
    compute_virial = (
        keys.VIRIAL in data_config.targets or keys.STRESS in data_config.targets
    )

    reduced_l1, results = test(
        model=model,
        data_loader=test_loader,
        err_metric=err_metric,
        meter=meter,
        device=device,
        compute_forces=compute_forces,
        compute_virial=compute_virial,
        verbose=args.verbose,
    )
    if args.output is None:
        output_file = f"{yaml_config['trainer']['run_name']}_test.log"
    else:
        output_file = args.output
    with open(output_file, "w") as f:
        f.write(" --- XequiNet Testing\n")
        for prop, unit in get_default_units().items():
            if prop in keys.BASE_PROPERTIES:
                continue
            f.write(f" --- Property: {prop} --- Unit: {unit}\n")
        f.write(" --- Test Results\n")

    if args.verbose:
        write_results(results, output_file)

    with open(output_file, "a") as f:
        header = [""]
        tabulate_data = ["Test MAE"]
        header.extend(list(map(lambda x: x.capitalize(), reduced_l1.keys())))
        tabulate_data.extend(list(reduced_l1.values()))
        f.write(
            tabulate([tabulate_data], headers=header, tablefmt="plain", floatfmt=".6f")
        )
        f.write("\n")

    if args.verbose:
        results_file = output_file.split(".")[0] + "_results.pt"
        torch.save(results, results_file)
