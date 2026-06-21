import h5py
import numpy as np

for path in ["data/unlabeled.h5", "data/train.h5"]:
    print(f"\n=== {path} ===")
    with h5py.File(path, "r") as f:
        def show(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  {name}: shape={obj.shape}, dtype={obj.dtype}")
        f.visititems(show)
        print("  Top-level keys:", list(f.keys()))
