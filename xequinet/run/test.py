import argparse
from typing import Any, Dict, Iterable, Tuple, cast

import torch
from omegaconf import OmegaConf
from tabulate import tabulate
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

from xequinet.data import XequiData, create_lmdb_dataset
from xequinet.nn import resolve_model
from xequinet.utils import (
    DataConfig,
    ModelConfig,
    get_default_units,
    keys,
    set_default_units,
)
from xequinet.utils.loss import L1Metric


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


def test(
    model: torch.nn.Module,
    data_loader: DataLoader,
    l1_metric: L1Metric,
    meter: AverageMetric,
    device: torch.device,
    compute_forces: bool = False,
    compute_virial: bool = False,
) -> Tuple[Dict[str, float], Dict[str, Any]]:

    data_list, results_list = [], []
    for data in data_loader:
        data: XequiData
        data = data.to(device)
        result = model(data.to_dict(), compute_forces, compute_virial)
        l1_losses = l1_metric(result, data)
        for prop, (l1, n) in l1_losses.items():
            meter.update(prop, l1, n)
        data_list.append(data)
        results_list.append(result)
    # reduce the meter
    reduced_l1 = meter.reduce()
    # collate data
    batch_data = Batch.from_data_list(data_list)
    # concatenate results
    cated_results = {}
    for prop in results_list[0]:
        cated_results[prop] = torch.cat(
            [result[prop] for result in results_list], dim=0
        )

    return reduced_l1, {"data": batch_data, "results": cated_results}


def run_test(args: argparse.Namespace) -> None:
    # set device
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

    # load checkpoint and config
    data_config = OmegaConf.merge(
        OmegaConf.structured(DataConfig),
        OmegaConf.load(args.config),
    )
    ckpt = torch.load(args.ckpt, map_location=device)
    model_config = OmegaConf.merge(
        OmegaConf.structured(ModelConfig),
        ckpt["config"],
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

    # load test dataset
    test_dataset = create_lmdb_dataset(
        db_path=data_config.db_path,
        cutoff=model.cutoff_radius,
        split=data_config.split,
        targets=data_config.targets,
        base_targets=data_config.base_targets,
        dtype=data_config.default_dtype,
        mode="test",
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
    meter = AverageMetric(data_config.targets)
    l1_metric = L1Metric(*data_config.targets)

    # testing
    compute_forces = keys.FORCES in data_config.targets
    compute_virial = (
        keys.VIRIAL in data_config.targets or keys.STRESS in data_config.targets
    )

    reduced_l1, results = test(
        model=model,
        data_loader=test_loader,
        l1_metric=l1_metric,
        meter=meter,
        device=device,
        compute_forces=compute_forces,
        compute_virial=compute_virial,
    )
    if args.output is None:
        output_file = f"{args.config.split('/')[-1].split('.')[0]}_test.log"
    else:
        output_file = args.output
    with open(output_file, "w") as f:
        f.write(" --- XequiNet Testing\n")
        for prop, unit in get_default_units().items():
            if prop in keys.BASE_PROPERTIES:
                continue
            f.write(f" --- Property: {prop} --- Unit: {unit}\n")
        f.write(" --- Test Results\n")
        header = [""]
        tabulate_data = ["Test MAE"]
        header.extend(list(map(lambda x: x.capitalize(), reduced_l1.keys())))
        tabulate_data.extend(list(map(lambda x: f"{x:.7f}", reduced_l1.values())))
        f.write(tabulate([tabulate_data], headers=header, tablefmt="plain"))
        f.write("\n")

    if args.verbose >= 1:
        results_file = output_file.replace(".log", ".pt")
        torch.save(results, results_file)
