import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class ForestTrainDataset(Dataset):
    def __init__(self, h5_path, csv_path, augment=False):
        self.h5_path = h5_path
        self.augment = augment
        with h5py.File(h5_path, "r") as f:
            raw_ids = f["ids"][:]
            all_ids = [s.decode() if isinstance(s, bytes) else s for s in raw_ids]
        self.id_to_idx = {id_: i for i, id_ in enumerate(all_ids)}
        df = pd.read_csv(csv_path)
        label_cols = [c for c in df.columns if c.startswith("c")]
        self.sample_ids = df["id"].tolist()
        self.labels = df[label_cols].values.astype(np.float32)
        self.indices = [self.id_to_idx[sid] for sid in self.sample_ids]

    def __len__(self):
        return len(self.sample_ids)

    def _augment(self, x):
        if np.random.rand() > 0.5:
            x = x[:, ::-1, :].copy()
        if np.random.rand() > 0.5:
            x = x[:, :, ::-1].copy()
        k = np.random.randint(0, 4)
        return np.rot90(x, k, axes=(1, 2)).copy()

    def __getitem__(self, idx):
        h5_idx = self.indices[idx]
        with h5py.File(self.h5_path, "r") as f:
            x_hr = f["x_highres"][h5_idx].astype(np.float32) / 255.0
            x_ts = f["x_ts"][h5_idx].astype(np.float32)
        if self.augment:
            x_hr = self._augment(x_hr)
        y = self.labels[idx]
        return torch.from_numpy(x_hr), torch.from_numpy(x_ts), torch.from_numpy(y)


class ForestUnlabeledDataset(Dataset):
    def __init__(self, h5_path, ids=None):
        self.h5_path = h5_path
        with h5py.File(h5_path, "r") as f:
            raw_ids = f["ids"][:]
            all_ids = [s.decode() if isinstance(s, bytes) else s for s in raw_ids]
        self.id_to_idx = {id_: i for i, id_ in enumerate(all_ids)}
        if ids is not None:
            self.sample_ids = ids
            self.indices = [self.id_to_idx[sid] for sid in ids]
        else:
            self.sample_ids = all_ids
            self.indices = list(range(len(all_ids)))

    def __len__(self):
        return len(self.sample_ids)

    def _augment(self, x):
        if np.random.rand() > 0.5:
            x = x[:, ::-1, :].copy()
        if np.random.rand() > 0.5:
            x = x[:, :, ::-1].copy()
        k = np.random.randint(0, 4)
        return np.rot90(x, k, axes=(1, 2)).copy()

    def __getitem__(self, idx):
        h5_idx = self.indices[idx]
        with h5py.File(self.h5_path, "r") as f:
            x_hr = f["x_highres"][h5_idx].astype(np.float32) / 255.0
            x_ts = f["x_ts"][h5_idx].astype(np.float32)
        x_hr1 = self._augment(x_hr.copy())
        x_hr2 = self._augment(x_hr.copy())
        return torch.from_numpy(x_hr1), torch.from_numpy(x_hr2), torch.from_numpy(x_ts)
