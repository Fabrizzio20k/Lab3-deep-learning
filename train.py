import argparse
import torch
from torch.utils.data import DataLoader, random_split
from dataset import ForestTrainDataset
from model import ForestModel


def brier_loss(pred, target):
    return ((pred - target) ** 2).sum(dim=-1).mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_h5", default="data/train.h5")
    parser.add_argument("--train_csv", default="data/train.csv")
    parser.add_argument("--pretrained", default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embed", type=int, default=256)
    parser.add_argument("--out", default="model_best.pt")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--no-val", action="store_true", default=False)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ds = ForestTrainDataset(args.train_h5, args.train_csv, augment=True)

    if args.no_val:
        train_dl = DataLoader(ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
        val_dl = None
        print(f"Training on all {len(ds)} samples (no val split)")
    else:
        val_n = max(1, int(0.1 * len(ds)))
        train_n = len(ds) - val_n
        train_ds, val_ds = random_split(ds, [train_n, val_n],
                                        generator=torch.Generator().manual_seed(42))
        train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
        val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    model = ForestModel(embed_dim=args.embed).to(device)

    if args.pretrained:
        state = torch.load(args.pretrained, map_location=device)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"Loaded pretrained. Missing={len(missing)}, Unexpected={len(unexpected)}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler()

    best_loss = float("inf")
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for x_hr, x_ts, y in train_dl:
            x_hr, x_ts, y = x_hr.to(device), x_ts.to(device), y.to(device)
            with torch.amp.autocast(device_type="cuda"):
                pred = model(x_hr, x_ts)
                loss = brier_loss(pred, y)
            opt.zero_grad()
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            train_loss += loss.item()

        tl = train_loss / len(train_dl)
        sched.step()

        if val_dl is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x_hr, x_ts, y in val_dl:
                    x_hr, x_ts, y = x_hr.to(device), x_ts.to(device), y.to(device)
                    with torch.amp.autocast(device_type="cuda"):
                        pred = model(x_hr, x_ts)
                    val_loss += brier_loss(pred, y).item()
            vl = val_loss / len(val_dl)
            print(f"Epoch {epoch+1}/{args.epochs} | train={tl:.4f} | val={vl:.4f}")
            if vl < best_loss:
                best_loss = vl
                torch.save(model.state_dict(), args.out)
                print(f"  -> Best model saved (val={vl:.4f})")
        else:
            print(f"Epoch {epoch+1}/{args.epochs} | train={tl:.4f}")
            if tl < best_loss:
                best_loss = tl
                torch.save(model.state_dict(), args.out)

    print(f"Best loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()
