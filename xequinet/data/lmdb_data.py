import os.path as osp
from typing import TypeVar, Optional, Union, Literal, Tuple

import lmdb
import pickle
import json
import torch
import torch.utils.data as torch_data

from .transform import (
    Transform,
    NeighborTransform,
    UnitTransform,
    DataTypeTransform,
    SequentialTransform,
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
    # unit transform
    with open(info_path, "r") as f:
        info = json.load(f)
        data_units = info["data_units"]
    unit_transform = UnitTransform(data_units=data_units)
    # data type transform
    if dtype is None:
        dtype = torch.get_default_dtype()
    dtype_transform = DataTypeTransform(dtype=dtype)
    # neighbor transform
    neighbor_transform = NeighborTransform(cutoff=cutoff)
    transform = SequentialTransform(
        transforms=[unit_transform, dtype_transform, neighbor_transform]
    )

    # load the dataset
    dataset = LMDBDataset(db_path=lmdb_path, transform=transform)

    # load the split and return the dataset(s)
    if mode == "train":
        with open(split_path, "r") as f:
            split = json.load(f)
            train_indices = split["train"]
            valid_indices = split["valid"]
        train_dataset = torch_data.Subset(dataset, train_indices)
        valid_dataset = torch_data.Subset(dataset, valid_indices)
        return train_dataset, valid_dataset
    elif mode == "test":
        with open(split_path, "r") as f:
            split = json.load(f)
            test_indices = split["test"]
        test_dataset = torch_data.Subset(dataset, test_indices)
        return test_dataset
    else:
        raise ValueError(f"Invalid mode {mode}")
