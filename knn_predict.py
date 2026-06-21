import argparse
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from dataset import ForestTrainDataset, ForestUnlabeledDataset
from model import ForestModel
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import normalize


def extract_features(model, dl, device):
    model.eval()
    all_hr = []
    all_ts = []
    with torch.no_grad():
        for batch in dl:
            if len(batch) == 3 and isinstance(batch[0], torch.Tensor):
                if batch[2].shape[-1] == 15:
                    x_hr, x_ts, _ = batch
                else:
                    x_hr, _, x_ts = batch
            else:
                x_hr, _, x_ts = batch
            x_hr, x_ts = x_hr.to(device), x_ts.to(device)
            with torch.amp.autocast(device_type="cuda"):
                hr_feat = model.hr_enc(x_hr)
                ts_feat = model.ts_enc(x_ts)
            all_hr.append(hr_feat.float().cpu().numpy())
            all_ts.append(ts_feat.float().cpu().numpy())
    return np.concatenate(all_hr), np.concatenate(all_ts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_h5", default="data/train.h5")
    parser.add_argument("--train_csv", default="data/train.csv")
    parser.add_argument("--unlabeled_h5", default="data/unlabeled.h5")
    parser.add_argument("--test_ids", default="data/testIDs.csv")
    parser.add_argument("--model", default="model_pseudo.pt")
    parser.add_argument("--embed", type=int, default=256)
    parser.add_argument("--k", type=int, default=15)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--out", default="submission_knn.csv")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | k={args.k}")

    model = ForestModel(embed_dim=args.embed).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    print("Extracting train features...")
    train_ds = ForestTrainDataset(args.train_h5, args.train_csv, augment=False)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=False,
                          num_workers=args.workers, pin_memory=True)
    train_hr, train_ts = extract_features(model, train_dl, device)
    train_feat = np.concatenate([train_hr, train_ts], axis=1)
    train_labels = train_ds.labels
    print(f"Train features: {train_feat.shape}")

    test_df = pd.read_csv(args.test_ids)
    test_ids = test_df.iloc[:, 0].tolist()

    print("Extracting test features...")
    test_ds = ForestUnlabeledDataset(args.unlabeled_h5, ids=test_ids)
    test_dl = DataLoader(test_ds, batch_size=args.batch, shuffle=False,
                         num_workers=args.workers, pin_memory=True)
    test_hr, test_ts = extract_features(model, test_dl, device)
    test_feat = np.concatenate([test_hr, test_ts], axis=1)
    print(f"Test features: {test_feat.shape}")

    train_feat_n = normalize(train_feat)
    test_feat_n = normalize(test_feat)

    results = {}
    for k in [5, 10, 15, 25, 50]:
        knn = KNeighborsRegressor(n_neighbors=k, metric="cosine", n_jobs=-1)
        knn.fit(train_feat_n, train_labels)
        preds = knn.predict(test_feat_n)
        preds = np.clip(preds, 0, 1)
        preds = preds / preds.sum(axis=1, keepdims=True)
        results[k] = preds
        print(f"k={k} done")

    for k, preds in results.items():
        rows = [{"id": sid, "Target": " ".join(f"{v:.6f}" for v in pred)}
                for sid, pred in zip(test_ids, preds)]
        out_name = args.out.replace(".csv", f"_k{k}.csv")
        pd.DataFrame(rows).to_csv(out_name, index=False)
        print(f"Saved k={k} → {out_name}")

    best_k = args.k
    rows = [{"id": sid, "Target": " ".join(f"{v:.6f}" for v in results[best_k][i])}
            for i, sid in enumerate(test_ids)]
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"Default submission (k={best_k}) saved to {args.out}")


if __name__ == "__main__":
    main()
