"""Model 3 (hybrid): frozen ImageNet MobileNetV2 embeddings (1280-d) -> XGBoost.

Uses the generic ImageNet backbone (not the fine-tuned one) so that the hybrid is an
independent model: deep features + gradient-boosted trees. Train embeddings are the
average of the original and horizontally-flipped image (cheap augmentation).
"""
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from xgboost import XGBClassifier
from config import MODELS, SEED
from common import (ImageDataset, build_mobilenet, load_split, eval_tf, embed, pil_aug,
                    AUG_COPIES, balanced_weights, set_seed)
from torchvision import transforms

CACHE = MODELS / "cnn_embeddings.npz"
AUG_CACHE = MODELS / "cnn_embeddings_aug.npz"
torch.set_num_threads(4)


@torch.no_grad()
def embeddings(model, df, tf=eval_tf):
    dl = DataLoader(ImageDataset(df, tf), 64, num_workers=3)
    out = []
    for x, _ in dl:
        out.append(((embed(model, x) + embed(model, torch.flip(x, [3]))) / 2).numpy())
    return np.concatenate(out)


def augmented_train(df_train):
    """Embeddings of AUG_COPIES augmented copies of every training image."""
    if AUG_CACHE.exists() and len(np.load(AUG_CACHE)["E"]) == AUG_COPIES * len(df_train):
        d = np.load(AUG_CACHE); return d["E"], d["y"]
    set_seed(); m = build_mobilenet().eval(); t = time.time()
    tf = transforms.Compose([pil_aug, eval_tf])
    E = np.concatenate([embeddings(m, df_train, tf) for _ in range(AUG_COPIES)])
    y = np.tile(df_train.y.values, AUG_COPIES)
    np.savez_compressed(AUG_CACHE, E=E, y=y); print(f"augmented embeddings {E.shape} in {time.time()-t:.0f}s")
    return E, y


def get_embeddings():
    df = load_split()
    if CACHE.exists() and len(np.load(CACHE)["E"]) == len(df):
        E = np.load(CACHE)["E"]
    else:
        m = build_mobilenet().eval(); t = time.time()
        E = embeddings(m, df); np.savez_compressed(CACHE, E=E)
        print(f"embeddings {E.shape} in {time.time()-t:.0f}s")
    return {s: (E[(df.split == s).values], df.y[df.split == s].values) for s in ["train", "val", "test"]}


def main():
    d = get_embeddings()
    (Xtr, ytr), (Xva, yva) = d["train"], d["val"]
    Ea, ya = augmented_train(load_split("train"))
    Xtr, ytr = np.concatenate([Xtr, Ea]), np.concatenate([ytr, ya])
    print(f"train: {len(ytr)} samples (original + {AUG_COPIES} augmented copies)")
    xgb = XGBClassifier(n_estimators=1000, learning_rate=0.05, max_depth=4, subsample=0.8,
                        colsample_bytree=0.3, tree_method="hist", n_jobs=-1,
                        early_stopping_rounds=60, eval_metric="mlogloss", random_state=SEED)
    t = time.time(); xgb.fit(Xtr, ytr, sample_weight=balanced_weights(ytr), eval_set=[(Xva, yva)], verbose=False)
    print(f"Hybrid val acc={xgb.score(Xva, yva):.3f} best_iter={xgb.best_iteration} ({time.time()-t:.0f}s)")
    xgb.save_model(MODELS / "xgb_cnn_embeddings.json")


if __name__ == "__main__":
    main()
