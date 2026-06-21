import argparse
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from dataset import ForestUnlabeledDataset
from model import ForestModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unlabeled_h5", default="data/unlabeled.h5")
    parser.add_argument("--test_ids", default="data/testIDs.csv")
    parser.add_argument("--model", default="model_best.pt")
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--embed", type=int, default=256)
    parser.add_argument("--out", default="submission.csv")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    test_df = pd.read_csv(args.test_ids)
    test_ids = test_df.iloc[:, 0].tolist()

    model = ForestModel(embed_dim=args.embed).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    ds = ForestUnlabeledDataset(args.unlabeled_h5, ids=test_ids)
    dl = DataLoader(
        ds, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=True
    )

    all_preds = []
    with torch.no_grad():
        for x_hr, _, x_ts in dl:
            x_hr, x_ts = x_hr.to(device), x_ts.to(device)
            with torch.amp.autocast(device_type="cuda"):
                preds = model(x_hr, x_ts)
            all_preds.append(preds.float().cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)

    rows = [
        {"Id": sid, "Target": " ".join(f"{v:.6f}" for v in pred)}
        for sid, pred in zip(test_ids, all_preds)
    ]
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"Saved {len(rows)} predictions to {args.out}")


if __name__ == "__main__":
    main()
