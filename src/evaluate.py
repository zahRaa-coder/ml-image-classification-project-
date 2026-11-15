"""Step 7: evaluate all models on the held-out TEST set.

Outputs (results/):
  comparison.csv / comparison.md   - accuracy, top-3, macro P/R/F1, inference ms/img
  confusion_<model>.png            - normalized confusion matrices
  per_class_f1.png, model_comparison.png, cnn_training_curves.png
  classification_report_<model>.txt
"""
import json
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_recall_fscore_support,
                             top_k_accuracy_score, classification_report)
from config import ROOT, RESULTS, SLUGS, DISPLAY, TRASHNET_DIR
from common import load_split, load_rgb
from inference import load_all

TIMING_N = 100  # images used for single-image latency measurement
EXTERNAL_PER_CLASS = 120  # TrashNet images per class for the external test (seeded sample)


def external_test(models, names):
    """Generalisation check on TrashNet (a different source, never used for training)."""
    import random
    rng = random.Random(0); paths, ys = [], []
    for k, s in enumerate(SLUGS):
        fs = sorted((TRASHNET_DIR / s).glob("*.jpg"))
        for f in rng.sample(fs, min(EXTERNAL_PER_CLASS, len(fs))):
            paths.append(f); ys.append(k)
    if not paths:
        return
    imgs, y = [load_rgb(p) for p in paths], np.array(ys)
    rows = []
    for key, m in models.items():
        P = np.stack([m.predict_proba(im) for im in imgs]); pred = P.argmax(1)
        p, r, f, _ = precision_recall_fscore_support(y, pred, average="macro", zero_division=0)
        rows.append(dict(model=m.name, accuracy=accuracy_score(y, pred),
                         top3_accuracy=top_k_accuracy_score(y, P, k=3, labels=range(len(SLUGS))),
                         macro_precision=p, macro_recall=r, macro_f1=f))
        if key == "cnn":
            cm = confusion_matrix(y, pred, labels=range(len(SLUGS)), normalize="true")
            plt.figure(figsize=(8, 6.5))
            sns.heatmap(cm, annot=True, fmt=".2f", cmap="Greens", xticklabels=names, yticklabels=names, cbar=False)
            plt.title("External test (TrashNet) – MobileNetV2"); plt.xlabel("Predicted"); plt.ylabel("True")
            plt.xticks(rotation=45, ha="right"); plt.tight_layout()
            plt.savefig(RESULTS / "confusion_external_cnn.png", dpi=130); plt.close()
    df = pd.DataFrame(rows); df.to_csv(RESULTS / "comparison_external.csv", index=False)
    md = df.copy()
    for c in md.columns[1:]:
        md[c] = (md[c] * 100).map("{:.1f}%".format)
    (RESULTS / "comparison_external.md").write_text(
        f"External test set: {len(y)} TrashNet images ({EXTERNAL_PER_CLASS}/class max) - different source, never seen in training\n\n"
        + md.to_markdown(index=False))
    print("\nEXTERNAL (TrashNet):\n" + md.to_string(index=False))


def main():
    RESULTS.mkdir(exist_ok=True)
    test = load_split("test")
    imgs = [load_rgb(ROOT / p) for p in test.path]
    y = test.y.values
    names = [DISPLAY[s] for s in SLUGS]
    models = load_all()
    rows, f1s = [], {}
    for key, m in models.items():
        P = np.stack([m.predict_proba(im) for im in imgs])
        t = time.perf_counter()
        for im in imgs[:TIMING_N]:
            m.predict_proba(im)
        ms = (time.perf_counter() - t) / min(TIMING_N, len(imgs)) * 1000
        pred = P.argmax(1)
        p, r, f, _ = precision_recall_fscore_support(y, pred, average="macro", zero_division=0)
        rows.append(dict(model=m.name, accuracy=accuracy_score(y, pred),
                         top3_accuracy=top_k_accuracy_score(y, P, k=3, labels=range(len(SLUGS))),
                         macro_precision=p, macro_recall=r, macro_f1=f, inference_ms_per_image=ms))
        f1s[m.name] = precision_recall_fscore_support(y, pred, labels=range(len(SLUGS)), zero_division=0)[2]
        (RESULTS / f"classification_report_{key}.txt").write_text(
            classification_report(y, pred, labels=range(len(SLUGS)), target_names=names, digits=3, zero_division=0))
        cm = confusion_matrix(y, pred, labels=range(len(SLUGS)), normalize="true")
        plt.figure(figsize=(11, 9))
        sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=names, yticklabels=names,
                    cbar=False, annot_kws={"size": 7})
        plt.title(f"Confusion matrix (test, row-normalized) – {m.name}")
        plt.xlabel("Predicted"); plt.ylabel("True"); plt.xticks(rotation=45, ha="right")
        plt.tight_layout(); plt.savefig(RESULTS / f"confusion_{key}.png", dpi=130); plt.close()
        print(rows[-1], flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "comparison.csv", index=False)
    md = df.copy()
    for c in md.columns[1:-1]:
        md[c] = (md[c] * 100).map("{:.1f}%".format)
    md["inference_ms_per_image"] = md["inference_ms_per_image"].map("{:.1f}".format)
    (RESULTS / "comparison.md").write_text(
        f"Test set: {len(test)} images, {len(SLUGS)} classes\n\n" + md.to_markdown(index=False))
    print(md.to_string(index=False))

    # bar chart of main metrics
    long = df.melt(id_vars="model", value_vars=["accuracy", "top3_accuracy", "macro_f1"])
    plt.figure(figsize=(10, 5))
    ax = sns.barplot(long, x="variable", y="value", hue="model", palette="Set2")
    for c in ax.containers:
        ax.bar_label(c, fmt="%.2f", fontsize=8)
    plt.ylim(0, 1.05); plt.ylabel("score"); plt.xlabel(""); plt.title("Model comparison on test set")
    plt.legend(fontsize=8, loc="lower right"); plt.tight_layout()
    plt.savefig(RESULTS / "model_comparison.png", dpi=130); plt.close()

    # per-class F1
    f1df = pd.DataFrame(f1s, index=names)
    f1df.plot.barh(figsize=(9, 8), width=0.8, colormap="Set2")
    plt.xlabel("F1 (test)"); plt.title("Per-class F1"); plt.xlim(0, 1); plt.tight_layout()
    plt.savefig(RESULTS / "per_class_f1.png", dpi=130); plt.close()

    # latency
    plt.figure(figsize=(8, 4))
    sns.barplot(df, y="model", x="inference_ms_per_image", palette="Set2", hue="model", legend=False)
    plt.xlabel("ms / image (CPU, batch=1)"); plt.ylabel(""); plt.title("Inference time")
    plt.tight_layout(); plt.savefig(RESULTS / "inference_time.png", dpi=130); plt.close()

    external_test(models, names)

    # CNN curves
    h = RESULTS / "cnn_history.json"
    if h.exists():
        hist = pd.DataFrame(json.load(open(h)))
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for k, a in zip(["loss", "acc"], ax):
            a.plot(hist.epoch, hist[f"train_{k}"], "o-", label="train")
            a.plot(hist.epoch, hist[f"val_{k}"], "o-", label="val")
            a.axvline(hist[hist.phase == "head"].epoch.max() + 0.5, ls="--", c="gray")
            a.set_title(f"MobileNetV2 {k}"); a.set_xlabel("epoch"); a.legend()
        plt.tight_layout(); plt.savefig(RESULTS / "cnn_training_curves.png", dpi=130); plt.close()


if __name__ == "__main__":
    main()
