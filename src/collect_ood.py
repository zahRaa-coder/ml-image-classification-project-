"""Collect openly-licensed photos that are NOT waste (people, animals, places, objects ...)
-> data/ood/ + data/ood_manifest.csv.  Used only to calibrate and test the "not sure" check."""
import csv
import hashlib
from config import OOD_DIR, DATA
import collect_online as co

QUERIES = ["portrait photograph", "dog", "cat", "horse", "bird", "landscape mountains", "beach", "city street",
           "car", "bicycle", "chair furniture", "sofa living room", "laptop computer", "smartphone", "flowers garden",
           "tree forest", "plate of food", "fruit market", "football match", "musical instrument", "shoes", "clock"]
PER_QUERY = 15


def main():
    OOD_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for q in QUERIES:
        n = 0
        for it in co.commons_urls(q, limit=60):
            if n >= PER_QUERY or "ND" in it["license"].replace("-", " ").split():
                continue
            name = OOD_DIR / f"ood_{q.split()[0]}_{hashlib.md5(it['thumb'].encode()).hexdigest()[:10]}.jpg"
            if not name.exists():
                r = co.polite_get(it["thumb"])
                if r is None or not r.headers.get("content-type", "").startswith("image"):
                    continue
                name.write_bytes(r.content)
            rows.append({"file": f"data/ood/{name.name}", "class": "not_waste", "query": q,
                         **{k: v for k, v in it.items() if k != "thumb"}, "download_url": it["thumb"]})
            n += 1
        print(q, n, flush=True)
    with open(DATA / "ood_manifest.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, list(rows[0])); w.writeheader(); w.writerows(rows)
    print("total", len(rows))


if __name__ == "__main__":
    main()
