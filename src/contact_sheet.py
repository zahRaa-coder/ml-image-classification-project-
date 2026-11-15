"""Numbered contact sheets for visual review of a class (nothing is deleted).

    python contact_sheet.py tavuk_sis      -> results/review_sheets/tavuk_sis_01.jpg ...
The numbers map to data/review_sheets/<class>_index.csv (number -> clean file path).
"""
import sys
import pandas as pd
from PIL import Image, ImageDraw
from config import CLEAN_DIR, RESULTS, DATA, ROOT
from common import load_rgb

COLS, ROWS, CELL = 6, 5, 200


def main(slug):
    files = sorted((CLEAN_DIR / slug).glob("*.jpg"))
    out = RESULTS / "review_sheets"; out.mkdir(parents=True, exist_ok=True)
    (DATA / "review_sheets").mkdir(exist_ok=True)
    pd.DataFrame({"n": range(1, len(files) + 1), "path": [str(f.relative_to(ROOT)) for f in files]}) \
        .to_csv(DATA / "review_sheets" / f"{slug}_index.csv", index=False)
    per = COLS * ROWS
    for s in range(0, len(files), per):
        sheet = Image.new("RGB", (COLS * CELL, ROWS * CELL), "white"); d = ImageDraw.Draw(sheet)
        for k, f in enumerate(files[s:s + per]):
            im = load_rgb(f); im.thumbnail((CELL - 6, CELL - 24))
            x, y = (k % COLS) * CELL, (k // COLS) * CELL
            sheet.paste(im, (x + 3, y + 20))
            d.rectangle([x + 2, y + 2, x + 44, y + 18], fill="black")
            d.text((x + 6, y + 4), f"#{s + k + 1}", fill="yellow")
        sheet.save(out / f"{slug}_{s // per + 1:02d}.jpg", quality=85)
    print(f"{len(files)} images -> {(len(files) + per - 1) // per} sheets in {out}")


if __name__ == "__main__":
    main(sys.argv[1])
