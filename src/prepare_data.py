"""Download TrashNet (MIT licence) as an EXTERNAL TEST SET -> data/trashnet + data/trashnet_manifest.csv.
(Training images are collected online by collect_online.py.)

Source : https://github.com/garythung/trashnet  (Gary Thung & Mindy Yang, Stanford 2016)
Mirror : https://huggingface.co/datasets/garythung/trashnet  (official, by the author)
Every image gets a provenance row (source, URL, licence, author) in data/manifest.csv.
"""
import csv
import urllib.request
import zipfile
from config import DATA, TRASHNET_DIR as RAW_DIR, TRASHNET_MANIFEST as MANIFEST_CSV

URL = "https://huggingface.co/datasets/garythung/trashnet/resolve/main/dataset-resized.zip"
ZIP = DATA / "external" / "trashnet" / "dataset-resized.zip"
FIELDS = ["file", "class", "source", "title", "page_url", "original_url", "download_url",
          "license", "license_url", "author"]


def main():
    if not ZIP.exists():
        ZIP.parent.mkdir(parents=True, exist_ok=True)
        print("downloading", URL); urllib.request.urlretrieve(URL, ZIP)
    rows = []
    with zipfile.ZipFile(ZIP) as z:
        for i in z.infolist():
            parts = i.filename.split("/")
            if "__MACOSX" in i.filename or not i.filename.endswith(".jpg") or len(parts) != 3:
                continue
            cls, name = parts[1], parts[2]
            (RAW_DIR / cls).mkdir(parents=True, exist_ok=True)
            dst = RAW_DIR / cls / f"trashnet_{name}"
            dst.write_bytes(z.read(i))
            rows.append({"file": f"data/trashnet/{cls}/{dst.name}", "class": cls, "source": "trashnet",
                         "title": name, "page_url": "https://github.com/garythung/trashnet",
                         "original_url": "https://huggingface.co/datasets/garythung/trashnet",
                         "download_url": URL, "license": "MIT",
                         "license_url": "https://github.com/garythung/trashnet/blob/master/LICENSE",
                         "author": "Gary Thung, Mindy Yang"})
    with open(MANIFEST_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, FIELDS); w.writeheader(); w.writerows(rows)
    print(f"{len(rows)} images -> {RAW_DIR}")


if __name__ == "__main__":
    main()
