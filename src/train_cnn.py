"""Model 2: MobileNetV2 transfer learning (CPU-friendly two-phase schedule).

Phase 1: freeze backbone, train classifier head.
Phase 2: unfreeze the last 5 inverted-residual blocks and fine-tune with a lower LR.
Best checkpoint (val accuracy) -> models/mobilenetv2.pt
"""
import json
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from config import MODELS, RESULTS, SLUGS
from common import ImageDataset, build_mobilenet, load_split, train_tf, eval_tf, set_seed, balanced_weights
import numpy as np

EPOCHS_HEAD, EPOCHS_FT, BS = 5, 12, 32
torch.set_num_threads(4)


def run_epoch(model, loader, crit, opt=None):
    model.train(opt is not None)
    tot = correct = loss_sum = 0
    with torch.set_grad_enabled(opt is not None):
        for x, y in loader:
            out = model(x); loss = crit(out, y)
            if opt:
                opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += loss.item() * len(y); correct += (out.argmax(1) == y).sum().item(); tot += len(y)
    return loss_sum / tot, correct / tot


def main():
    set_seed(); MODELS.mkdir(exist_ok=True); RESULTS.mkdir(exist_ok=True)
    dtr = load_split("train")
    tr = DataLoader(ImageDataset(dtr, train_tf), BS, shuffle=True, num_workers=3)
    va = DataLoader(ImageDataset(load_split("val"), eval_tf), 64, num_workers=3)
    model = build_mobilenet()
    # class weights -> the small "trash" class is not ignored
    cw = torch.tensor(np.bincount(dtr.y).sum() / (len(SLUGS) * np.bincount(dtr.y)), dtype=torch.float32)
    crit = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.1)
    crit_val = nn.CrossEntropyLoss()
    hist, best = [], 0.0

    def fit(epochs, opt, sched, phase):
        nonlocal best
        for ep in range(epochs):
            t = time.time()
            trl, tra = run_epoch(model, tr, crit, opt)
            val, vaa = run_epoch(model, va, crit_val)
            sched.step()
            hist.append(dict(phase=phase, epoch=len(hist) + 1, train_loss=trl, train_acc=tra, val_loss=val, val_acc=vaa))
            tag = ""
            if vaa > best:
                best = vaa; torch.save(model.state_dict(), MODELS / "mobilenetv2.pt"); tag = " *saved*"
            print(f"[{phase}] ep{len(hist):02d} train {trl:.3f}/{tra:.3f}  val {val:.3f}/{vaa:.3f}  {time.time()-t:.0f}s{tag}", flush=True)

    # Phase 1 - head only
    for p in model.features.parameters():
        p.requires_grad = False
    opt = torch.optim.AdamW(model.classifier.parameters(), lr=1e-3, weight_decay=1e-4)
    fit(EPOCHS_HEAD, opt, torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS_HEAD), "head")

    # Phase 2 - fine-tune last blocks
    for p in model.features[-5:].parameters():
        p.requires_grad = True
    opt = torch.optim.AdamW([{"params": model.features[-5:].parameters(), "lr": 2e-4},
                             {"params": model.classifier.parameters(), "lr": 5e-4}], weight_decay=1e-4)
    fit(EPOCHS_FT, opt, torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS_FT), "finetune")

    json.dump(hist, open(RESULTS / "cnn_history.json", "w"), indent=1)
    print("best val acc", best)


if __name__ == "__main__":
    main()
