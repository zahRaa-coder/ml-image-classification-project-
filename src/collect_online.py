"""Step 1: collect waste images ONLINE and label them (course requirement:
"images will be collected online and labeled").

Sources (only openly-licensed photos, licence recorded per image):
  * Wikimedia Commons API - curated categories + keyword search
  * Openverse API         - Creative-Commons photos (mostly Flickr)
Label = the class whose category / query returned the image; labels are then
verified (label_check.py + visual review) before training.

Every file gets a provenance row in data/manifest.csv
(file, class, source, page URL, licence, author).

    python collect_online.py            # all classes
    python collect_online.py paper      # one class
    python collect_online.py --commons  &  python collect_online.py --openverse   # parallel
    python -c "import collect_online as c; c.merge_manifests()"
"""
import csv
import hashlib
import html
import re
import sys
import time
import requests
from config import RAW_DIR, MANIFEST_CSV, SLUGS

TARGET_RAW = 520 if "--extra" not in sys.argv else 900   # max raw files per class
IIPROP = {"iiprop": "url|mime|extmetadata", "iiurlwidth": 512,
          "iiextmetadatafilter": "LicenseShortName|LicenseUrl|Artist|UsageTerms"}
HEADERS = {"User-Agent": "EcoScanAI-student-project/1.0 (university ML course; python-requests)"}
COMMONS_API = "https://commons.wikimedia.org/w/api.php"
MANIFEST = {}

# class -> (Commons categories, search queries [Commons + Openverse])
SOURCES = {
    "cardboard": (["Cardboard boxes", "Corrugated fiberboard"],
                  ["cardboard box", "corrugated cardboard", "cardboard waste", "flattened cardboard boxes", "cardboard recycling"]),
    "glass":     (["Glass bottles", "Glass jars"],
                  ["glass bottle", "glass jar", "empty glass bottle", "beer bottle", "wine bottle empty"]),
    "metal":     (["Beverage cans", "Aluminium cans"],
                  ["aluminium can", "soda can", "tin can", "crushed can", "food can empty", "aluminum foil"]),
    "paper":     (["Shredded paper", "Crumpled paper", "Waste paper"],
                  ["crumpled paper", "waste paper", "paper recycling", "newspaper recycling", "scrap paper", "paper bag"]),
    "plastic":   (["Plastic bottles", "PET bottles", "Plastic bags", "Plastic waste"],
                  ["plastic bottle", "plastic bag", "plastic container", "plastic cup", "plastic packaging waste"]),
    "trash":     (["Cigarette butts", "Expanded polystyrene", "Litter"],
                  ["chip bag", "candy wrapper", "styrofoam cup", "food wrapper litter", "cigarette butts", "used tissue"]),
}


def polite_get(url, params=None, tries=6, **kw):
    """GET with throttling + exponential backoff on 429/5xx (Wikimedia rate limits)."""
    wait = 2.0
    for _ in range(tries):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=30, **kw)
        except requests.RequestException:
            time.sleep(wait); wait *= 2; continue
        if r.status_code == 200:
            time.sleep(0.6)
            return r
        if r.status_code in (429, 500, 502, 503, 504):
            ra = float(r.headers.get("Retry-After", wait))
            if ra > 30:        # long ban for this URL -> skip it rather than stall
                return None
            time.sleep(ra); wait *= 2; continue
        return None
    return None


def _clean(x):
    return html.unescape(re.sub(r"<[^>]+>", "", x or "")).strip()


def page_meta(p):
    """Provenance dict for one Commons file page, or None if not a usable bitmap."""
    ii = (p.get("imageinfo") or [{}])[0]
    if ii.get("mime") not in ("image/jpeg", "image/png", "image/webp") or not ii.get("thumburl") \
            or "thumbnail_unscaled" in ii["thumburl"]:
        return None
    em = ii.get("extmetadata", {})
    return {"thumb": ii["thumburl"], "source": "wikimedia_commons", "title": p.get("title", ""),
            "page_url": ii.get("descriptionurl", ""), "original_url": ii.get("url", ""),
            "license": _clean(em.get("LicenseShortName", {}).get("value")),
            "license_url": _clean(em.get("LicenseUrl", {}).get("value")),
            "author": _clean(em.get("Artist", {}).get("value"))[:200]}


def commons_urls(query, limit=150):
    """Provenance dicts (incl. 512px thumbnail URL) of bitmap files matching `query`."""
    urls, offset = [], 0
    while len(urls) < limit:
        r = polite_get(COMMONS_API, params={
            "action": "query", "format": "json", "generator": "search",
            "gsrsearch": f"{query} filetype:bitmap", "gsrnamespace": 6,
            "gsrlimit": 50, "gsroffset": offset, "prop": "imageinfo", "maxlag": 5, **IIPROP})
        if r is None:
            break
        data = r.json()
        for p in data.get("query", {}).get("pages", {}).values():
            m = page_meta(p)
            if m:
                urls.append(m)
        offset = data.get("continue", {}).get("gsroffset")
        if offset is None:
            break
    return urls[:limit]


def commons_category(cat, limit=250, depth=1):
    """File thumbnails in a Commons category (+ one level of sub-categories)."""
    urls, cont = [], {}
    while len(urls) < limit:
        r = polite_get(COMMONS_API, params={
            "action": "query", "format": "json", "generator": "categorymembers",
            "gcmtitle": f"Category:{cat}", "gcmtype": "file|subcat", "gcmlimit": 50,
            "prop": "imageinfo", "maxlag": 5, **IIPROP, **cont})
        if r is None:
            break
        data = r.json()
        for p in data.get("query", {}).get("pages", {}).values():
            if p.get("ns") == 14 and depth > 0:
                urls += commons_category(p["title"].split(":", 1)[1], limit // 2, depth - 1)
                continue
            m = page_meta(p)
            if m:
                urls.append(m)
        if "continue" not in data:
            break
        cont = {k: v for k, v in data["continue"].items()}
    return urls[:limit]


def download(items, out, prefix):
    """Download Commons thumbnails (skip existing) and record provenance for every file."""
    n = 0
    for it in items:
        name = out / f"{prefix}_{hashlib.md5(it['thumb'].encode()).hexdigest()[:12]}.jpg"
        if not name.exists():
            if len(list(out.iterdir())) >= TARGET_RAW:
                continue
            r = polite_get(it["thumb"])
            if r is None or not r.headers.get("content-type", "").startswith("image"):
                continue
            name.write_bytes(r.content); n += 1
        MANIFEST[f"{out.name}/{name.name}"] = {"file": f"data/raw/{out.name}/{name.name}", "class": out.name,
                                              **{k: v for k, v in it.items() if k != "thumb"},
                                              "download_url": it["thumb"]}
    return n


FIELDS = ["file", "class", "source", "title", "page_url", "original_url", "download_url",
          "license", "license_url", "author"]


def merge_manifests():
    """Merge the per-process manifests (data/manifest_*.csv) into data/manifest.csv."""
    from config import MANIFEST_CSV as M
    rows = {}
    for p in [M] + sorted(M.parent.glob("manifest_*.csv")):
        if p.exists():
            with open(p, encoding="utf-8") as fh:
                rows.update({r["file"]: r for r in csv.DictReader(fh)})
    with open(M, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, FIELDS); w.writeheader()
        for k in sorted(rows):
            w.writerow(rows[k])
    print(f"manifest: {len(rows)} rows")


def write_manifest():
    """Merge with the existing manifest; files without a Commons record are marked as Bing."""
    rows = {}
    if MANIFEST_CSV.exists():
        with open(MANIFEST_CSV, encoding="utf-8") as fh:
            rows = {r["file"]: r for r in csv.DictReader(fh)}
    rows.update({r["file"]: r for r in MANIFEST.values()})
    for d in RAW_DIR.iterdir():
        for f in d.iterdir():
            key = f"data/raw/{d.name}/{f.name}"
            if key not in rows and not f.name.startswith(("commons_", "openverse_", "trashnet_")):
                rows[key] = {"file": key, "class": d.name, "source": "bing_icrawler",
                             "license": "unknown", "author": "unknown",
                             "page_url": "unknown (icrawler did not record the URL)"}
    with open(MANIFEST_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, FIELDS); w.writeheader()
        for k in sorted(rows):
            w.writerow({f: rows[k].get(f, "") for f in FIELDS})


OPENVERSE_API = "https://api.openverse.org/v1/images/"


def openverse_items(query, max_pages=12):
    """Creative-Commons images (mostly Flickr) via the Openverse API, with license info.
    Wikimedia results are skipped (already covered by the Commons API)."""
    items = []
    for page in range(1, max_pages + 1):
        r = polite_get(OPENVERSE_API, params={"q": query, "page_size": 20, "page": page})
        if r is None:
            break
        j = r.json()
        for x in j.get("results", []):
            if x.get("source") == "wikimedia" or not x.get("url"):
                continue
            lic = f"CC {x.get('license', '').upper()} {x.get('license_version') or ''}".strip()
            if x.get("license") in ("cc0", "pdm"):
                lic = x["license"].upper()
            items.append({"thumb": x["url"], "source": f"openverse/{x.get('source')}",
                          "title": (x.get("title") or "")[:150], "page_url": x.get("foreign_landing_url", ""),
                          "original_url": x["url"], "license": lic,
                          "license_url": x.get("license_url", ""), "author": (x.get("creator") or "")[:200]})
        if page >= j.get("page_count", 0):
            break
    return items




# second round of queries (classes that ended below 300 verified images)
EXTRA_QUERIES = {
    "cardboard": ["cardboard boxes pile", "empty carton box", "cardboard packaging", "moving boxes", "pizza box"],
    "glass": ["glass bottles recycling", "jam jar empty", "glass waste", "green glass bottle", "broken glass bottle", "glass jars"],
    "metal": ["empty cans", "aluminium cans recycling", "beer can", "canned food tin", "metal scrap cans"],
    "paper": ["old newspapers", "newspaper stack", "magazines pile", "paper waste pile", "torn paper", "notebook paper",
              "stack of paper", "paper sheets", "wrapping paper", "egg carton", "paper cup", "office paper waste"],
    "plastic": ["PET bottle", "plastic bottles pile", "plastic food container", "yogurt cup", "plastic waste beach", "shampoo bottle"],
    "trash": ["candy wrapper", "styrofoam waste", "chips packet", "used napkin", "food packaging litter", "snack wrapper"],
}
MODE = "all"   # all | commons | openverse  (two processes can run in parallel)


def collect(slug):
    out = RAW_DIR / slug
    out.mkdir(parents=True, exist_ok=True)
    cats, queries = SOURCES[slug]
    seen = set()
    def take(items, prefix):
        items = [i for i in items if i["thumb"] not in seen]
        seen.update(i["thumb"] for i in items)
        download(items, out, prefix)
    if "--extra" in sys.argv:
        for q in EXTRA_QUERIES.get(slug, []):
            take(commons_urls(q, limit=120), "commons")
        write_manifest()
        print(f"[extra] {slug:10s} files = {len(list(out.iterdir()))}", flush=True)
        return
    if MODE in ("all", "commons"):
        for c in cats:
            take(commons_category(c, limit=400), "commons")
        for q in queries:
            take(commons_urls(q, limit=100), "commons")
    if MODE in ("all", "openverse"):
        for q in queries:
            take(openverse_items(q, max_pages=6), "openverse")
    write_manifest()
    n = sum(1 for f in out.iterdir() if not f.name.startswith("trashnet_"))
    print(f"[done] {slug:10s} online images = {n}", flush=True)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--commons" in sys.argv:
        MODE = "commons"
    if "--openverse" in sys.argv:
        MODE = "openverse"
    if MODE != "all":   # separate manifest per process, merged by merge_manifest()
        MANIFEST_CSV = MANIFEST_CSV.with_name(f"manifest_{MODE}{'_extra' if '--extra' in sys.argv else ''}{'_' + '_'.join(args) if args else ''}.csv")
    for s in (args or SLUGS):
        collect(s)
