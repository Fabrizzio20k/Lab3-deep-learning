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
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr_backbone", type=float, default=1e-5)
    parser.add_argument("--embed", type=int, default=256)
    parser.add_argument("--backbone", type=str, default="efficientnet_b0")
    parser.add_argument("--out", default="model_best.pt")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--no-val", action="store_true", default=False)
    parser.add_argument("--freeze-epochs", type=int, default=10)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | backbone={args.backbone}")

    ds = ForestTrainDataset(args.train_h5, args.train_csv, augment=True)

    if args.no_val:
        train_dl = DataLoader(ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
        val_dl = None
        print(f"Training on all {len(ds)} samples")
    else:
        val_n = max(1, int(0.1 * len(ds)))
        train_n = len(ds) - val_n
        train_ds, val_ds = random_split(ds, [train_n, val_n],
                                        generator=torch.Generator().manual_seed(42))
        train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
        val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    model = ForestModel(embed_dim=args.embed, backbone=args.backbone).to(device)

    if args.pretrained:
        state = torch.load(args.pretrained, map_location=device)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"Loaded pretrained. Missing={len(missing)}, Unexpected={len(unexpected)}")

    # Freeze backbone initially
    for p in model.hr_enc.backbone.parameters():
        p.requires_grad_(False)
    print(f"Backbone frozen for first {args.freeze_epochs} epochs")

    param_groups = model.get_param_groups(lr_backbone=args.lr_backbone, lr_head=args.lr)
    opt = torch.optim.AdamW(param_groups, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler()

    best_loss = float("inf")
    for epoch in range(args.epochs):
        if epoch == args.freeze_epochs:
            for p in model.hr_enc.backbone.parameters():
                p.requires_grad_(True)
            print(f"Epoch {epoch+1}: backbone unfrozen")

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

        sched.step()
        tl = train_loss / len(train_dl)

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
            monitor = vl
        else:
            print(f"Epoch {epoch+1}/{args.epochs} | train={tl:.4f}")
            monitor = tl

        if monitor < best_loss:
            best_loss = monitor
            torch.save(model.state_dict(), args.out)
            print(f"  -> Saved best (loss={monitor:.4f})")

    print(f"Best loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()
