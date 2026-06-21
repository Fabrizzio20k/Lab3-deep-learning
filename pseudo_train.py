import argparse
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from dataset import ForestUnlabeledDataset, ForestTrainDataset
from model import ForestModel
import h5py


class PseudoLabelDataset(Dataset):
    def __init__(self, h5_path, pseudo_ids, pseudo_labels, augment=True):
        self.augment = augment
        with h5py.File(h5_path, "r") as f:
            raw_ids = [s.decode() if isinstance(s, bytes) else s for s in f["ids"][:]]
            id_to_idx = {id_: i for i, id_ in enumerate(raw_ids)}
            indices = np.array([id_to_idx[sid] for sid in pseudo_ids])
            print(f"Loading {len(indices)} pseudo-labeled samples into RAM...")
            self.x_hr = f["x_highres"][indices]
            self.x_ts = f["x_ts"][indices].astype(np.float32)
            print("Done.")
        self.labels = pseudo_labels.astype(np.float32)

    def __len__(self):
        return len(self.labels)

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


class CombinedDataset(Dataset):
    def __init__(self, ds1, ds2):
        self.ds1 = ds1
        self.ds2 = ds2

    def __len__(self):
        return len(self.ds1) + len(self.ds2)

    def __getitem__(self, idx):
        if idx < len(self.ds1):
            return self.ds1[idx]
        return self.ds2[idx - len(self.ds1)]


def brier_loss(pred, target):
    return ((pred - target) ** 2).sum(dim=-1).mean()


def generate_pseudo_labels(model, h5_path, all_ids, device, batch_size=256):
    ds = ForestUnlabeledDataset(h5_path)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    all_preds = []
    model.eval()
    with torch.no_grad():
        for x_hr, _, x_ts in dl:
            x_hr, x_ts = x_hr.to(device), x_ts.to(device)
            with torch.amp.autocast(device_type="cuda"):
                preds = model(x_hr, x_ts)
            all_preds.append(preds.float().cpu().numpy())
    return np.concatenate(all_preds, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_h5", default="data/train.h5")
    parser.add_argument("--train_csv", default="data/train.csv")
    parser.add_argument("--unlabeled_h5", default="data/unlabeled.h5")
    parser.add_argument("--pretrained", default="pretrained.pt")
    parser.add_argument("--init_model", default=None)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--pseudo_epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--embed", type=int, default=256)
    parser.add_argument("--out", default="model_pseudo.pt")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--confidence", type=float, default=0.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = ForestModel(embed_dim=args.embed).to(device)

    if args.init_model:
        model.load_state_dict(torch.load(args.init_model, map_location=device))
        print(f"Loaded init model from {args.init_model}")
    elif args.pretrained:
        state = torch.load(args.pretrained, map_location=device)
        missing, _ = model.load_state_dict(state, strict=False)
        print(f"Loaded pretrained. Missing={len(missing)}")

    labeled_ds = ForestTrainDataset(args.train_h5, args.train_csv, augment=True)

    scaler = torch.amp.GradScaler()

    def run_training(dataset, num_epochs, lr, out_path):
        dl = DataLoader(dataset, batch_size=args.batch, shuffle=True,
                        num_workers=args.workers, pin_memory=True, drop_last=False)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=num_epochs)
        best = float("inf")
        for epoch in range(num_epochs):
            model.train()
            total = 0.0
            for x_hr, x_ts, y in dl:
                x_hr, x_ts, y = x_hr.to(device), x_ts.to(device), y.to(device)
                with torch.amp.autocast(device_type="cuda"):
                    pred = model(x_hr, x_ts)
                    loss = brier_loss(pred, y)
                opt.zero_grad()
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                total += loss.item()
            sched.step()
            tl = total / len(dl)
            print(f"  Epoch {epoch+1}/{num_epochs} | train={tl:.4f}")
            if tl < best:
                best = tl
                torch.save(model.state_dict(), out_path)
        print(f"  Best train loss: {best:.4f}")

    print("\n=== Phase 1: Train on labeled data ===")
    run_training(labeled_ds, args.epochs, args.lr, args.out)

    print("\n=== Generating pseudo-labels for unlabeled data ===")
    model.load_state_dict(torch.load(args.out, map_location=device))

    with h5py.File(args.unlabeled_h5, "r") as f:
        all_ids = [s.decode() if isinstance(s, bytes) else s for s in f["ids"][:]]

    pseudo_preds = generate_pseudo_labels(model, args.unlabeled_h5, all_ids, device)

    if args.confidence > 0:
        max_vals = pseudo_preds.max(axis=1)
        mask = max_vals >= args.confidence
        selected_ids = [id_ for id_, m in zip(all_ids, mask) if m]
        selected_preds = pseudo_preds[mask]
        print(f"Selected {len(selected_ids)}/{len(all_ids)} samples (confidence >= {args.confidence})")
    else:
        selected_ids = all_ids
        selected_preds = pseudo_preds
        print(f"Using all {len(selected_ids)} unlabeled samples as pseudo-labeled")

    pseudo_ds = PseudoLabelDataset(args.unlabeled_h5, selected_ids, selected_preds, augment=True)
    combined_ds = CombinedDataset(labeled_ds, pseudo_ds)
    print(f"Combined dataset: {len(combined_ds)} samples ({len(labeled_ds)} labeled + {len(pseudo_ds)} pseudo)")

    print("\n=== Phase 2: Train on labeled + pseudo-labeled data ===")
    run_training(combined_ds, args.pseudo_epochs, args.lr * 0.3, args.out)

    print(f"\nFinal model saved to {args.out}")


if __name__ == "__main__":
    main()
