import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class ForestTrainDataset(Dataset):
    def __init__(self, h5_path, csv_path, augment=False):
        self.augment = augment
        with h5py.File(h5_path, "r") as f:
            raw_ids = [s.decode() if isinstance(s, bytes) else s for s in f["ids"][:]]
            id_to_idx = {id_: i for i, id_ in enumerate(raw_ids)}
            df = pd.read_csv(csv_path)
            label_cols = [c for c in df.columns if c.startswith("c")]
            self.sample_ids = df["id"].tolist()
            self.labels = df[label_cols].values.astype(np.float32)
            indices = np.array([id_to_idx[sid] for sid in self.sample_ids])
            print(f"Loading train data into RAM...")
            self.x_hr = f["x_highres"][indices]
            self.x_ts = f["x_ts"][indices].astype(np.float32)
            print(f"Loaded {len(indices)} train samples.")

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
        x_hr = self.x_hr[idx].astype(np.float32) / 255.0
        x_ts = self.x_ts[idx]
        if self.augment:
            x_hr = self._augment(x_hr)
        return (
            torch.from_numpy(x_hr),
            torch.from_numpy(x_ts),
            torch.from_numpy(self.labels[idx]),
        )


class ForestUnlabeledDataset(Dataset):
    def __init__(self, h5_path, ids=None):
        with h5py.File(h5_path, "r") as f:
            raw_ids = [s.decode() if isinstance(s, bytes) else s for s in f["ids"][:]]
            id_to_idx = {id_: i for i, id_ in enumerate(raw_ids)}
            if ids is not None:
                self.sample_ids = ids
                indices = np.array([id_to_idx[sid] for sid in ids])
            else:
                self.sample_ids = raw_ids
                indices = np.arange(len(raw_ids))
            print(f"Loading unlabeled data into RAM ({len(indices)} samples)...")
            self.x_hr = f["x_highres"][indices]
            self.x_ts = f["x_ts"][indices].astype(np.float32)
            print("Done loading.")

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
        x_hr = self.x_hr[idx].astype(np.float32) / 255.0
        x_ts = self.x_ts[idx]
        x_hr1 = self._augment(x_hr.copy())
        x_hr2 = self._augment(x_hr.copy())
        return (
            torch.from_numpy(x_hr1),
            torch.from_numpy(x_hr2),
            torch.from_numpy(x_ts),
        )
