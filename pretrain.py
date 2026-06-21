import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import ForestUnlabeledDataset
from model import ForestModel


def nt_xent_loss(z1, z2, temperature=0.5):
    N = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    sim = torch.mm(z, z.T) / temperature
    mask = torch.eye(2 * N, device=z.device).bool()
    sim.masked_fill_(mask, float("-inf"))
    labels = torch.cat([torch.arange(N, 2 * N), torch.arange(0, N)]).to(z.device)
    return F.cross_entropy(sim, labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/unlabeled.h5")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--embed", type=int, default=256)
    parser.add_argument("--out", default="pretrained.pt")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ds = ForestUnlabeledDataset(args.data)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                    pin_memory=True, drop_last=True)

    model = ForestModel(embed_dim=args.embed).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler()

    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for x_hr1, x_hr2, _ in dl:
            x_hr1, x_hr2 = x_hr1.to(device), x_hr2.to(device)
            with torch.amp.autocast(device_type="cuda"):
                z1 = model.forward_with_proj(x_hr1)
                z2 = model.forward_with_proj(x_hr2)
                loss = nt_xent_loss(z1, z2)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += loss.item()
        sched.step()
        print(f"Epoch {epoch+1}/{args.epochs} | loss={total/len(dl):.4f} | lr={sched.get_last_lr()[0]:.6f}")

    torch.save(model.state_dict(), args.out)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
