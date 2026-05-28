"""
Read a parquet split into a PyTorch Dataset / DataLoader.

Grids are returned as LongTensors of shape (H*W,) with token values:
    0 = free, 1 = obstacle, 2 = start marker, 3 = goal marker
(The start/goal markers are written over the flattened grid so the model
can see them as input tokens. The TRM embedding has vocab size 4.)

Labels are LongTensors of shape (H*W,) with action ids; masked cells hold
IGNORE_INDEX so cross-entropy skips them.
"""
import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, DataLoader

FREE, OBSTACLE, START_TOKEN, GOAL_TOKEN = 0, 1, 2, 3
INPUT_VOCAB_SIZE = 4  # {free, obstacle, start, goal}

class OccupancyGridDataset(Dataset):
    def __init__(self, parquet_path: str):
        table = pq.read_table(parquet_path)
        self.grids = table["grid_flat"].to_pylist()
        self.actions = table["actions_flat"].to_pylist()
        self.start_idxs = table["start_idx"].to_numpy()
        self.goal_idxs = table["goal_idx"].to_numpy()
        self.size = table["grid_size"][0].as_py()  # All grids have the same size

    def __len__(self):
        return len(self.grids)
    
    def __getitem__(self, idx):
        grid_flat = np.asarray(self.grids[idx], dtype=np.int64)  # (H*W,)
        grid_flat[self.start_idxs[idx]] = START_TOKEN
        grid_flat[self.goal_idxs[idx]] = GOAL_TOKEN
        action_flat = np.asarray(self.actions[idx], dtype=np.int64)  # (H*W,)
        return torch.from_numpy(grid_flat), torch.from_numpy(action_flat)


def make_loader(parquet_path: str, batch_size: int, shuffle: bool = True, num_workers: int = 4) -> DataLoader:
    dataset = OccupancyGridDataset(parquet_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True, drop_last=shuffle)