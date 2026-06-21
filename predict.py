import argparse
import torch
import pandas as pd
from torch.utils.data import DataLoader
from dataset import UnlabeledDataset
from model import ForestModel
import h5py
import numpy as np


def peek_shapes(h5_path):
    with h5py.File(h5_path, "r") as f:
        k = list(f.keys())[0]
        hr = f[k]["x_highres"][()]
        ts = f[k]["x_ts"][()]
    return hr, ts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unlabeled_h5", default="data/unlabeled.h5")
    parser.add_argument("--test_ids", default="data/testIDs.csv")
    parser.add_argument("--model", default="model_best.pt")
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--embed", type=int, default=256)
    parser.add_argument("--out", default="submission.csv")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    test_df = pd.read_csv(args.test_ids)
    if "id" in test_df.columns:
        test_ids = test_df["id"].tolist()
    else:
        test_ids = test_df.iloc[:, 0].tolist()

    hr_sample, ts_sample = peek_shapes(args.unlabeled_h5)
    if hr_sample.ndim == 2:
        hr_sample = hr_sample[None]
    hr_c = hr_sample.shape[0]
    ts_c = ts_sample.shape[-1] if ts_sample.ndim == 2 else ts_sample.shape[0]
    ts_len = ts_sample.shape[0] if ts_sample.ndim == 2 else ts_sample.shape[1]

    model = ForestModel(hr_c, ts_c, ts_len, embed_dim=args.embed).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    ds = UnlabeledDataset(args.unlabeled_h5, ids=test_ids, augment=False)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=4, pin_memory=True)

    all_ids = []
    all_preds = []

    with torch.no_grad():
        for x_hr1, _, x_ts in dl:
            x_hr1, x_ts = x_hr1.to(device), x_ts.to(device)
            preds = model(x_hr1, x_ts)
            all_preds.append(preds.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)

    rows = []
    for sid, pred in zip(test_ids, all_preds):
        target_str = " ".join(f"{v:.6f}" for v in pred)
        rows.append({"id": sid, "Target": target_str})

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out, index=False)
    print(f"Submission saved to {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
