"""Step 2c: verify the labels of the images collected online.

Each image got its label from the Commons category / search query that returned it.
Web results are noisy (paintings, people, buildings, the wrong material ...), so every
image is checked with CLIP (openai/clip-vit-base-patch32, zero-shot):

  * class score  = sum of CLIP probabilities of the prompts describing each class
  * other score  = sum of probabilities of "not a waste photo" prompts
  * verdict: ok          - CLIP's best class == assigned label, its score >= MIN_LABEL
                           and the "other" score < MAX_OTHER
             uncertain   - right class but low score
             mismatch    - CLIP prefers another waste class   (suggested label kept in csv)
             irrelevant  - not a photo of a waste item

Images with verdict != ok are written to data/rejected.txt (reversible: nothing is
deleted - remove a line to bring an image back). The full table is data/label_check.csv,
and results/label_check_<class>.jpg shows samples of each verdict for visual review.
CLIP is only a labelling assistant; it is NOT one of the course models.
"""
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from transformers import CLIPModel, CLIPProcessor
from config import ROOT, CLEAN_DIR, SLUGS, DATA, RESULTS, REJECTED_TXT
from common import load_rgb

PROMPTS = {
    "cardboard": ["a photo of a cardboard box", "a photo of corrugated cardboard",
                  "a photo of flattened cardboard waste"],
    "glass": ["a photo of a glass bottle", "a photo of a glass jar", "a photo of empty glass bottles"],
    "metal": ["a photo of an aluminium drink can", "a photo of a metal tin can",
              "a photo of crushed metal cans", "a photo of aluminium foil"],
    "paper": ["a photo of crumpled paper", "a photo of waste paper", "a photo of old newspapers",
              "a photo of shredded paper", "a photo of a paper bag"],
    "plastic": ["a photo of a plastic bottle", "a photo of a plastic bag", "a photo of a plastic container",
                "a photo of plastic waste"],
    "trash": ["a photo of a chip bag", "a photo of a candy wrapper", "a photo of a styrofoam cup",
              "a photo of cigarette butts", "a photo of mixed litter on the ground", "a photo of a used tissue"],
}
OTHER = ["a painting", "a drawing or illustration", "a diagram or chart", "a map", "a portrait of a person",
         "a group of people", "a building or street", "a landscape", "a page of text", "a logo or icon",
         "a museum exhibit", "a truck or vehicle", "an animal", "a crowd of people",
         "a recycling bin or waste container", "a plate of food", "a shop or supermarket shelf",
         "a machine or factory"]
MIN_LABEL = 0.40   # minimum CLIP score of the assigned class
MAX_OTHER = 0.35   # maximum score of the "not a waste item" prompts


def _feat(x):
    """transformers >= 5 returns a model output object instead of a tensor."""
    return x if isinstance(x, torch.Tensor) else x.pooler_output


def main():
    torch.set_num_threads(4)
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    texts = [p for s in SLUGS for p in PROMPTS[s]] + OTHER
    owner = [s for s in SLUGS for _ in PROMPTS[s]] + ["other"] * len(OTHER)
    with torch.no_grad():
        T = _feat(model.get_text_features(**proc(text=texts, return_tensors="pt", padding=True)))
        T = T / T.norm(dim=-1, keepdim=True)
    rows = []
    for s in SLUGS:
        files = sorted((CLEAN_DIR / s).glob("*.jpg"))
        for i in range(0, len(files), 32):
            batch = files[i:i + 32]
            with torch.no_grad():
                I = _feat(model.get_image_features(**proc(images=[load_rgb(f) for f in batch], return_tensors="pt")))
                I = I / I.norm(dim=-1, keepdim=True)
                P = (100 * I @ T.T).softmax(-1).numpy()
            for f, p in zip(batch, P):
                score = {c: float(p[[k for k, o in enumerate(owner) if o == c]].sum()) for c in SLUGS + ["other"]}
                best = max(SLUGS, key=score.get)
                if score["other"] >= MAX_OTHER:
                    verdict = "irrelevant"
                elif best != s:
                    verdict = "mismatch"
                else:
                    verdict = "ok" if score[s] >= MIN_LABEL else "uncertain"
                rows.append({"path": str(f.relative_to(ROOT)), "label": s, "verdict": verdict,
                             "clip_best_class": best, "label_score": round(score[s], 3),
                             "other_score": round(score["other"], 3)})
        print(s, pd.Series([r["verdict"] for r in rows if r["label"] == s]).value_counts().to_dict(), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(DATA / "label_check.csv", index=False)

    # write exclusions (keep manual review lines that are already there)
    old = set(REJECTED_TXT.read_text().split()) if REJECTED_TXT.exists() else set()
    REJECTED_TXT.write_text("\n".join(sorted(old | set(df[df.verdict != "ok"].path))) + "\n")

    # sample sheets for visual review
    RESULTS.mkdir(exist_ok=True)
    for s in SLUGS:
        d = df[df.label == s]
        C, sheet = 150, Image.new("RGB", (150 * 8, 150 * 6), "white")
        draw = ImageDraw.Draw(sheet); k = 0
        for v, n in [("ok", 32), ("uncertain", 8), ("mismatch", 4), ("irrelevant", 4)]:
            for r in d[d.verdict == v].sample(min(n, (d.verdict == v).sum()), random_state=0).itertuples():
                im = load_rgb(ROOT / r.path); im.thumbnail((C - 4, C - 16))
                x, y = (k % 8) * C, (k // 8) * C
                sheet.paste(im, (x + 2, y + 14))
                draw.text((x + 3, y + 1), f"{v[:4]} {r.clip_best_class[:6]}", fill="green" if v == "ok" else "red")
                k += 1
        sheet.save(RESULTS / f"label_check_{s}.jpg", quality=80)
    print(df.groupby(["label", "verdict"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
