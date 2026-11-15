"""Unknown / out-of-scope image handling ("this does not look like waste").

Two confidence scores are compared:
  * msp      - maximum softmax probability of the classifier (all 4 models)
  * centroid - cosine similarity between the fine-tuned CNN embedding and the
               nearest class centroid computed on the TRAIN split (CNN only)

Out-of-scope images: data/ood/ (openly-licensed photos of people, animals, places, objects ...,
collected by collect_ood.py),
split 50/50 into ood-val (used to choose the score) and ood-test (reported only).
The threshold is set on the in-distribution VAL split so that 95% of real waste
images are accepted (TPR = 95%); everything scoring below it is shown as "unknown".

Outputs: models/ood.json (thresholds used by the website), results/ood_report.csv,
results/ood_scores.png
"""
import json
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from config import ROOT, MODELS, RESULTS, OOD_DIR, SEED
from common import load_split, load_rgb, eval_tf, embed
from inference import load_all

TPR = 0.95


@torch.no_grad()
def cnn_embed(cnn, img):
    return F.normalize(embed(cnn.m, eval_tf(img)[None]), dim=1)[0].numpy()


def main():
    models = load_all()
    cnn = models["cnn"]
    tr, va, te = load_split("train"), load_split("val"), load_split("test")
    ood = sorted(OOD_DIR.glob("*.jpg"))
    random.Random(SEED).shuffle(ood)
    ood_val, ood_test = ood[: len(ood) // 2], ood[len(ood) // 2:]
    imgs = {
        "val": [load_rgb(ROOT / p) for p in va.path],
        "test": [load_rgb(ROOT / p) for p in te.path],
        "ood_val": [load_rgb(p) for p in ood_val],
        "ood_test": [load_rgb(p) for p in ood_test],
    }
    # class centroids of fine-tuned CNN embeddings (train split only)
    Etr = np.stack([cnn_embed(cnn, load_rgb(ROOT / p)) for p in tr.path])
    C = np.stack([Etr[tr.y.values == k].mean(0) for k in range(len(np.unique(tr.y)))])
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    np.save(MODELS / "cnn_centroids.npy", C)

    def scores(key, method, ims):
        if method == "centroid":
            return np.array([(C @ cnn_embed(cnn, im)).max() for im in ims])
        return np.array([models[key].predict_proba(im).max() for im in ims])

    cfg, rows = {}, []
    for key in models:
        methods = ["msp", "centroid"] if key == "cnn" else ["msp"]
        best = None
        for method in methods:
            s = {k: scores(key, method, v) for k, v in imgs.items()}
            auroc_val = roc_auc_score(np.r_[np.ones(len(s["val"])), np.zeros(len(s["ood_val"]))],
                                      np.r_[s["val"], s["ood_val"]])
            thr = float(np.quantile(s["val"], 1 - TPR))
            row = dict(model=models[key].name, method=method, threshold=round(thr, 4),
                       auroc_val=auroc_val,
                       auroc_test=roc_auc_score(np.r_[np.ones(len(s["test"])), np.zeros(len(s["ood_test"]))],
                                                np.r_[s["test"], s["ood_test"]]),
                       waste_accepted_test=float((s["test"] >= thr).mean()),
                       non_waste_rejected_test=float((s["ood_test"] < thr).mean()))
            rows.append(row); print(row, flush=True)
            if best is None or auroc_val > best[0]:
                best = (auroc_val, method, thr, s)
        cfg[key] = {"method": best[1], "threshold": best[2]}
        if key == "cnn":
            s = best[3]
            plt.figure(figsize=(7, 4))
            plt.hist(s["test"], bins=30, alpha=.6, label="waste (test)", density=True)
            plt.hist(s["ood_test"], bins=30, alpha=.6, label="not waste (ood-test)", density=True)
            plt.axvline(best[2], c="k", ls="--", label=f"threshold ({best[1]})")
            plt.xlabel(f"CNN confidence score ({best[1]})"); plt.legend(); plt.title("Unknown-image detection")
            plt.tight_layout(); plt.savefig(RESULTS / "ood_scores.png", dpi=130); plt.close()
    json.dump(cfg, open(MODELS / "ood.json", "w"), indent=1)
    pd.DataFrame(rows).round(4).to_csv(RESULTS / "ood_report.csv", index=False)
    print(json.dumps(cfg, indent=1))


if __name__ == "__main__":
    main()
