import os.path as osp
from typing import TypeVar

import lmdb
import pickle
import torch.utils.data as torch_data

T = TypeVar("T")


def generate_lmdb_key(index: int) -> bytes:
    return index.to_bytes(length=8, byteorder="little")


class LMDBDataset(torch_data.Dataset[T]):

    data: lmdb.Environment
    entries: int  # Number of datapoints in the LMDB

    def __init__(self, db_path: str) -> None:
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
            return val

    def __len__(self) -> int:
        return self.entries

    def __repr__(self) -> str:
        return f"LMDBDataset({self.db_path})"

    def close(self) -> None:
        if not self.is_open:
            return
        self.data.close()
        self.is_open = False

    def __del__(self) -> None:
        self.close()
