import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class ForestDataset(Dataset):
    def __init__(self, h5_path, csv_path=None, augment=False):
        self.h5_path = h5_path
        self.augment = augment
        with h5py.File(h5_path, "r") as f:
            self.ids = list(f.keys())
        if csv_path is not None:
            df = pd.read_csv(csv_path)
            df = df.set_index("id")
            self.labels = df
            self.ids = [i for i in self.ids if i in df.index]
        else:
            self.labels = None

    def __len__(self):
        return len(self.ids)

    def _load(self, idx):
        sample_id = self.ids[idx]
        with h5py.File(self.h5_path, "r") as f:
            g = f[sample_id]
            x_hr = g["x_highres"][()].astype(np.float32)
            x_ts = g["x_ts"][()].astype(np.float32)
        return sample_id, x_hr, x_ts

    def _augment_hr(self, x):
        if np.random.rand() > 0.5:
            x = x[:, ::-1, :].copy()
        if np.random.rand() > 0.5:
            x = x[:, :, ::-1].copy()
        k = np.random.randint(0, 4)
        x = np.rot90(x, k, axes=(1, 2)).copy()
        return x

    def __getitem__(self, idx):
        sample_id, x_hr, x_ts = self._load(idx)

        if x_hr.ndim == 2:
            x_hr = x_hr[np.newaxis]
        if x_ts.ndim == 2:
            x_ts = x_ts

        if self.augment:
            x_hr = self._augment_hr(x_hr)

        x_hr = torch.from_numpy(x_hr)
        x_ts = torch.from_numpy(x_ts)

        if self.labels is not None:
            row = self.labels.loc[sample_id]
            target_cols = [c for c in row.index if c.startswith("c")]
            y = torch.tensor(row[target_cols].values.astype(np.float32))
            return x_hr, x_ts, y
        return sample_id, x_hr, x_ts


class UnlabeledDataset(Dataset):
    def __init__(self, h5_path, ids=None, augment=False):
        self.h5_path = h5_path
        self.augment = augment
        with h5py.File(h5_path, "r") as f:
            all_ids = list(f.keys())
        self.ids = ids if ids is not None else all_ids

    def __len__(self):
        return len(self.ids)

    def _augment_hr(self, x):
        if np.random.rand() > 0.5:
            x = x[:, ::-1, :].copy()
        if np.random.rand() > 0.5:
            x = x[:, :, ::-1].copy()
        k = np.random.randint(0, 4)
        x = np.rot90(x, k, axes=(1, 2)).copy()
        return x

    def __getitem__(self, idx):
        sample_id = self.ids[idx]
        with h5py.File(self.h5_path, "r") as f:
            g = f[sample_id]
            x_hr = g["x_highres"][()].astype(np.float32)
            x_ts = g["x_ts"][()].astype(np.float32)

        if x_hr.ndim == 2:
            x_hr = x_hr[np.newaxis]

        x_hr1 = self._augment_hr(x_hr)
        x_hr2 = self._augment_hr(x_hr)

        return torch.from_numpy(x_hr1), torch.from_numpy(x_hr2), torch.from_numpy(x_ts)
