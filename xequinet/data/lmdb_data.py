import json
import os.path as osp
import pickle
from typing import Dict, Iterable, List, Literal, Optional, Tuple, TypeVar, Union

import lmdb
import torch
import torch.utils.data as torch_data

from .transform import (
    DataTypeTransform,
    DeltaTransform,
    NeighborTransform,
    SequentialTransform,
    Transform,
    UnitTransform,
)

T = TypeVar("T")


def generate_lmdb_key(index: int) -> bytes:
    return index.to_bytes(length=8, byteorder="little")


class LMDBDataset(torch_data.Dataset[T]):

    data: lmdb.Environment
    entries: int  # Number of datapoints in the LMDB

    def __init__(
        self,
        db_path: str,
        transform: Optional[Transform] = None,
    ) -> None:
        super().__init__()
        self.db_path = db_path

        # There are only readers, so we can set map_size to file size
        file_size = osp.getsize(self.db_path)  # bytes

        self.data = lmdb.open(
            path=self.db_path,
            map_size=file_size,
            subdir=False,
            sync=False,
            writemap=False,
            meminit=False,
            map_async=False,
            create=False,
            readonly=True,
            lock=False,
        )
        self.entries = self.data.stat()["entries"]
        self.is_open = True
        self.transform = transform

    def __getitem__(self, index) -> T:
        assert self.is_open
        if not isinstance(index, int):
            raise IndexError(f"Index must be an integer, not {type(index)}")
        with self.data.begin(write=False) as txn:
            key = generate_lmdb_key(index)
            buf = txn.get(key)
            if not buf:
                raise IndexError(f"Index {index} out of bounds")
            val: T = pickle.loads(buf)
            if self.transform is not None:
                val = self.transform(val)
            return val

    def __len__(self) -> int:
        return self.entries

    def close(self) -> None:
        if not self.is_open:
            return
        self.data.close()
        self.is_open = False

    def __del__(self) -> None:
        self.close()


def create_lmdb_dataset(
    db_path: str,
    cutoff: float,
    split: str,
    targets: Union[str, Iterable[str]] = "energy",
    base_targets: Optional[Union[str, Iterable[str]]] = None,
    dtype: Optional[Union[str, torch.dtype]] = None,
    mode: Literal["train", "test"] = "train",
) -> Union[Tuple[torch_data.Dataset[T], torch_data.Dataset[T]], torch_data.Dataset[T]]:
    """
    Create the dataset from an LMDB database
    If mode is "train", returns a tuple of training and validation datasets
    If mode is "test", returns the test dataset
    """
    # check if the necessary files exist
    lmdb_path = osp.join(db_path, "data.lmdb")
    assert osp.exists(lmdb_path), f"LMDB file not found at {lmdb_path}"
    info_path = osp.join(db_path, "info.json")
    assert osp.exists(info_path), f"Info file not found at {info_path}"
    split_path = osp.join(db_path, f"{split}.json")
    assert osp.exists(split_path), f"Split file not found at {split_path}"

    # set transforms
    transform_list = []

    # data type transform
    if dtype is None:
        dtype = torch.get_default_dtype()
    transform_list.append(DataTypeTransform(dtype=dtype))

    # unit transform
    targets = targets if isinstance(targets, Iterable) else [targets]
    with open(info_path, "r") as f:
        info = json.load(f)
        data_units = {k: info["units"][k] for k in targets}
        if base_targets is not None:
            data_units.update({k: info["units"][k] for k in base_targets})
    transform_list.append(UnitTransform(data_units=data_units))

    # delta transform
    if base_targets is not None:
        base_targets = (
            base_targets if isinstance(base_targets, Iterable) else [base_targets]
        )
        transform_list.append(DeltaTransform(base_targets=base_targets))

    # neighbor transform
    neighbor_transform = NeighborTransform(cutoff=cutoff)
    transform_list.append(neighbor_transform)

    transform = SequentialTransform(transforms=transform_list)

    # load the dataset
    dataset = LMDBDataset(db_path=lmdb_path, transform=transform)

    # load the split and return the dataset(s)
    if mode == "train":
        with open(split_path, "r") as f:
            split_dict: Dict[str, List[int]] = json.load(f)
            train_indices = split_dict["train"]
            valid_indices = split_dict["valid"]
        train_dataset = torch_data.Subset(dataset, train_indices)
        valid_dataset = torch_data.Subset(dataset, valid_indices)
        return train_dataset, valid_dataset
    elif mode == "test":
        with open(split_path, "r") as f:
            split_dict: Dict[str, List[int]] = json.load(f)
            test_indices = split_dict["test"]
        test_dataset = torch_data.Subset(dataset, test_indices)
        return test_dataset
    else:
        raise ValueError(f"Invalid mode {mode}")
