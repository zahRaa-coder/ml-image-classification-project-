"""Model 1: Random Forest + XGBoost on handcrafted features (HSV hist + HOG + LBP)."""
import time
import joblib
import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from config import ROOT, MODELS, SEED, SLUGS
import torch
from common import load_split, load_rgb, pil_aug, AUG_COPIES, balanced_weights, set_seed
from features import extract

CACHE = MODELS / "handcrafted_features.npz"
AUG_CACHE = MODELS / "handcrafted_features_aug.npz"


def featurize(paths):
    return np.stack(Parallel(n_jobs=-1)(delayed(lambda p: extract(np.asarray(load_rgb(ROOT / p))))(p)
                                        for p in paths))


def _aug_feat(p, seed):
    torch.manual_seed(seed)
    return extract(np.asarray(pil_aug(load_rgb(ROOT / p))))


def augmented_train(df_train):
    """AUG_COPIES augmented feature vectors per training image (train split only)."""
    if AUG_CACHE.exists() and len(np.load(AUG_CACHE)["X"]) == AUG_COPIES * len(df_train):
        d = np.load(AUG_CACHE); return d["X"], d["y"]
    jobs = [(p, k * 100000 + i) for k in range(AUG_COPIES) for i, p in enumerate(df_train.path)]
    t = time.time()
    X = np.stack(Parallel(n_jobs=-1)(delayed(_aug_feat)(p, s) for p, s in jobs))
    y = np.tile(df_train.y.values, AUG_COPIES)
    np.savez_compressed(AUG_CACHE, X=X, y=y); print(f"augmented features {X.shape} in {time.time()-t:.0f}s")
    return X, y


def get_features():
    df = load_split()
    if CACHE.exists() and len(np.load(CACHE)["X"]) == len(df):
        X = np.load(CACHE)["X"]
    else:
        t = time.time(); X = featurize(df.path); np.savez_compressed(CACHE, X=X)
        print(f"features {X.shape} in {time.time()-t:.0f}s")
    return {s: (X[(df.split == s).values], df.y[df.split == s].values) for s in ["train", "val", "test"]}


def main():
    set_seed(); MODELS.mkdir(exist_ok=True)
    d = get_features()
    (Xtr, ytr), (Xva, yva) = d["train"], d["val"]
    Xa, ya = augmented_train(load_split("train"))
    Xtr, ytr = np.concatenate([Xtr, Xa]), np.concatenate([ytr, ya])
    print(f"train: {len(ytr)} samples (original + {AUG_COPIES} augmented copies)")

    rf = RandomForestClassifier(n_estimators=600, max_features="sqrt", min_samples_leaf=1,
                                class_weight="balanced", n_jobs=-1, random_state=SEED)
    t = time.time(); rf.fit(Xtr, ytr)
    print(f"RF   val acc={rf.score(Xva, yva):.3f}  ({time.time()-t:.0f}s)")
    joblib.dump(rf, MODELS / "rf_handcrafted.joblib", compress=3)

    xgb = XGBClassifier(n_estimators=800, learning_rate=0.05, max_depth=5, subsample=0.8,
                        colsample_bytree=0.4, tree_method="hist", n_jobs=-1,
                        early_stopping_rounds=50, eval_metric="mlogloss", random_state=SEED)
    t = time.time(); xgb.fit(Xtr, ytr, sample_weight=balanced_weights(ytr), eval_set=[(Xva, yva)], verbose=False)
    print(f"XGB  val acc={xgb.score(Xva, yva):.3f}  best_iter={xgb.best_iteration} ({time.time()-t:.0f}s)")
    xgb.save_model(MODELS / "xgb_handcrafted.json")


if __name__ == "__main__":
    main()
