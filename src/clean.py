"""Step 2b: build data/clean/ from data/raw/ (raw files are never modified or deleted).

Filters, in order (every decision is logged to data/clean_report.csv):
  1. source/licence policy  - No-Derivatives (ND) licences are excluded; by default only
     files with a recorded licence in
     data/manifest.csv (Wikimedia Commons / Openverse). Bing images (unknown licence)
     are skipped unless USE_UNLICENSED = True in config.py
  2. rejected on the review page (data/rejected.txt)
  3. broken / unreadable file
  4. too small (min side < MIN_SIDE) or very blurry (variance of Laplacian < BLUR_MIN)
  5. exact / near-identical duplicate (perceptual hash, Hamming <= 6) across ALL classes
     and sources -> the same photo can never end up in two splits or two classes
Output: data/clean/<class>/<raw file stem>.jpg (RGB JPEG, longest side <= 512).
"""
import csv
import shutil
import cv2
import imagehash
import numpy as np
from PIL import Image, ImageOps
from config import (RAW_DIR, CLEAN_DIR, SLUGS, MIN_SIDE, REJECTED_TXT, MANIFEST_CSV, DATA,
                    USE_UNLICENSED)

HASH_DIST = 6
MAX_SIDE = 512
BLUR_MIN = 8.0   # TrashNet: plain backgrounds have low Laplacian variance; 40 rejected sharp images


def main():
    if CLEAN_DIR.exists():
        shutil.rmtree(CLEAN_DIR)            # derived folder only; data/raw is untouched
    rejected = set(REJECTED_TXT.read_text().split()) if REJECTED_TXT.exists() else set()
    manifest = {}
    if MANIFEST_CSV.exists():
        with open(MANIFEST_CSV, encoding="utf-8") as fh:
            manifest = {r["file"]: r for r in csv.DictReader(fh)}
    seen, report = [], []
    print(f"{'class':18s} {'kept':>5s} {'nolic':>6s} {'review':>6s} {'broken':>6s} {'small':>6s} {'blur':>5s} {'dup':>5s}")
    for slug in SLUGS:
        out = CLEAN_DIR / slug
        out.mkdir(parents=True, exist_ok=True)
        c = dict(kept=0, nolic=0, review=0, broken=0, small=0, blur=0, dup=0)
        for p in sorted((RAW_DIR / slug).glob("*")):
            key = f"data/raw/{slug}/{p.name}"
            dst = out / f"{p.stem}.jpg"
            src = manifest.get(key, {}).get("source", "bing_icrawler")
            lic = manifest.get(key, {}).get("license", "unknown")
            reason = None
            if lic in ("", "unknown") and not USE_UNLICENSED:
                reason = "nolic"
            elif "ND" in lic.replace("-", " ").split():         # No-Derivatives: resizing for training = modification
                reason = "nolic"
            elif f"data/clean/{slug}/{dst.name}" in rejected:
                reason = "review"
            else:
                try:
                    with Image.open(p) as im:
                        im.verify()
                    im = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
                except Exception:
                    reason = "broken"
                if reason is None and min(im.size) < MIN_SIDE:
                    reason = "small"
                if reason is None:
                    g = cv2.cvtColor(np.asarray(im.resize((256, 256))), cv2.COLOR_RGB2GRAY)
                    if cv2.Laplacian(g, cv2.CV_64F).var() < BLUR_MIN:
                        reason = "blur"
                if reason is None:
                    h = imagehash.phash(im)
                    dup_of = next((sp for sh, sp in seen if h - sh <= HASH_DIST), None)
                    if dup_of:
                        reason = f"dup"
                    else:
                        seen.append((h, str(dst)))
            if reason:
                c[reason] += 1
            else:
                im.thumbnail((MAX_SIDE, MAX_SIDE)); im.save(dst, quality=92); c["kept"] += 1
            report.append({"raw": key, "class": slug, "source": src, "license": lic,
                           "decision": reason or "kept",
                           "clean": f"data/clean/{slug}/{dst.name}" if not reason else ""})
        print(f"{slug:18s} " + " ".join(f"{v:5d} " for v in c.values()))
    with open(DATA / "clean_report.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, list(report[0])); w.writeheader(); w.writerows(report)
    print("TOTAL kept:", sum(r["decision"] == "kept" for r in report))


if __name__ == "__main__":
    main()
