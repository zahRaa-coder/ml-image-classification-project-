"""Leakage guard: find near-duplicate images that pHash missed (crops, resizes,
colour edits, watermarks) using cosine similarity of ImageNet MobileNetV2 embeddings.

Nothing is deleted. Outputs:
  data/near_duplicates.csv   - every pair with similarity >= SIM (a, b, sim, same_class)
  data/dup_groups.csv        - path -> group id (connected components of those pairs);
                               split.py keeps each group inside ONE split -> no leakage
  results/near_duplicates_preview.jpg - side-by-side preview of the top pairs
"""
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset
from config import ROOT, CLEAN_DIR, SLUGS, DATA, RESULTS
from common import build_mobilenet, eval_tf, embed, load_rgb

SIM = 0.92  # cosine similarity threshold (checked visually on the preview sheet)
torch.set_num_threads(4)


class Files(Dataset):
    def __init__(self, paths): self.p = paths
    def __len__(self): return len(self.p)
    def __getitem__(self, i): return eval_tf(load_rgb(ROOT / self.p[i]))


def main():
    paths = [str(p.relative_to(ROOT)) for s in SLUGS for p in sorted((CLEAN_DIR / s).glob("*.jpg"))]
    labels = [p.split("/")[2] for p in paths]
    m = build_mobilenet().eval()
    with torch.no_grad():
        E = torch.cat([embed(m, x) for x in DataLoader(Files(paths), 64, num_workers=3)])
    E = torch.nn.functional.normalize(E, dim=1)
    S = (E @ E.T).numpy(); np.fill_diagonal(S, 0)
    ii, jj = np.where(np.triu(S) >= SIM)
    pairs = pd.DataFrame({"a": [paths[i] for i in ii], "b": [paths[j] for j in jj],
                          "sim": S[ii, jj].round(4), "same_class": [labels[i] == labels[j] for i, j in zip(ii, jj)]})
    pairs.sort_values("sim", ascending=False).to_csv(DATA / "near_duplicates.csv", index=False)

    # union-find -> groups
    parent = list(range(len(paths)))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for i, j in zip(ii, jj):
        parent[find(i)] = find(j)
    pd.DataFrame({"path": paths, "group": [find(i) for i in range(len(paths))]}).to_csv(DATA / "dup_groups.csv", index=False)

    print(f"{len(paths)} images, {len(pairs)} near-duplicate pairs (sim >= {SIM}), "
          f"{int((~pairs.same_class).sum()) if len(pairs) else 0} cross-class")
    # preview sheet of the top pairs
    top = pairs.sort_values("sim", ascending=False).head(40)
    if len(top):
        W = 160; sheet = Image.new("RGB", (W * 4 + 30, (W + 22) * ((len(top) + 1) // 2)), "white")
        d = ImageDraw.Draw(sheet)
        for k, r in enumerate(top.itertuples()):
            x0, y0 = (k % 2) * (2 * W + 30), (k // 2) * (W + 22)
            for n, p in enumerate([r.a, r.b]):
                im = load_rgb(ROOT / p); im.thumbnail((W, W)); sheet.paste(im, (x0 + n * W, y0))
            d.text((x0, y0 + W + 4), f"{r.sim:.3f} {'same' if r.same_class else 'CROSS'}", fill="black")
        RESULTS.mkdir(exist_ok=True); sheet.save(RESULTS / "near_duplicates_preview.jpg", quality=85)


if __name__ == "__main__":
    main()
