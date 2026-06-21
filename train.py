import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from dataset import ForestDataset
from model import ForestModel
import h5py
import numpy as np


def brier_loss(pred, target):
    return ((pred - target) ** 2).sum(dim=-1).mean()


def peek_shapes(h5_path):
    with h5py.File(h5_path, "r") as f:
        k = list(f.keys())[0]
        hr = f[k]["x_highres"][()]
        ts = f[k]["x_ts"][()]
    return hr, ts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_h5", default="data/train.h5")
    parser.add_argument("--train_csv", default="data/train.csv")
    parser.add_argument("--pretrained", default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embed", type=int, default=256)
    parser.add_argument("--out", default="model_best.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    hr_sample, ts_sample = peek_shapes(args.train_h5)
    if hr_sample.ndim == 2:
        hr_sample = hr_sample[None]
    hr_c = hr_sample.shape[0]
    ts_c = ts_sample.shape[-1] if ts_sample.ndim == 2 else ts_sample.shape[0]
    ts_len = ts_sample.shape[0] if ts_sample.ndim == 2 else ts_sample.shape[1]

    ds = ForestDataset(args.train_h5, args.train_csv, augment=True)
    val_n = max(1, int(0.1 * len(ds)))
    train_n = len(ds) - val_n
    train_ds, val_ds = random_split(ds, [train_n, val_n], generator=torch.Generator().manual_seed(42))
    val_ds.dataset.augment = False

    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=4, pin_memory=True)

    model = ForestModel(hr_c, ts_c, ts_len, embed_dim=args.embed).to(device)

    if args.pretrained:
        state = torch.load(args.pretrained, map_location=device)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"Loaded pretrained. Missing: {len(missing)}, Unexpected: {len(unexpected)}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_val = float("inf")
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for x_hr, x_ts, y in train_dl:
            x_hr, x_ts, y = x_hr.to(device), x_ts.to(device), y.to(device)
            pred = model(x_hr, x_ts)
            loss = brier_loss(pred, y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_hr, x_ts, y in val_dl:
                x_hr, x_ts, y = x_hr.to(device), x_ts.to(device), y.to(device)
                pred = model(x_hr, x_ts)
                val_loss += brier_loss(pred, y).item()

        sched.step()
        tl = train_loss / len(train_dl)
        vl = val_loss / len(val_dl)
        print(f"Epoch {epoch+1}/{args.epochs} | train={tl:.4f} | val={vl:.4f}")

        if vl < best_val:
            best_val = vl
            torch.save(model.state_dict(), args.out)
            print(f"  -> Saved best model (val={vl:.4f})")

    print(f"Best val Brier: {best_val:.4f}")


if __name__ == "__main__":
    main()
