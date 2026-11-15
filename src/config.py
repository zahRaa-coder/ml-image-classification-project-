"""Central configuration for EcoScan AI (waste classification for recycling)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW_DIR = DATA / "raw"          # images collected online by collect_online.py (one folder per class)
TRASHNET_DIR = DATA / "trashnet"  # TrashNet photos - used ONLY as an external test set
TRASHNET_MANIFEST = DATA / "trashnet_manifest.csv"
CLEAN_DIR = DATA / "clean"      # filtered / deduplicated copies
SPLIT_CSV = DATA / "splits.csv" # path,label,group,split
REJECTED_TXT = DATA / "rejected.txt"  # images excluded on the review page
MANIFEST_CSV = DATA / "manifest.csv"  # provenance: file, class, source, url, license, author
BINS_JSON = DATA / "bins.json"        # disposal info shown on the website
OOD_DIR = DATA / "ood"                # out-of-scope images (not waste) for the "unknown" test
MODELS = ROOT / "models"
RESULTS = ROOT / "results"

SEED = 42
IMG_SIZE = 224
MIN_SIDE = 224          # drop images smaller than this (pixels)
USE_UNLICENSED = False  # only images with a recorded licence are used

# Training data: collected online (Wikimedia Commons + Openverse/Flickr, open licences)
# External test : TrashNet (Gary Thung & Mindy Yang, Stanford 2016) - MIT licence
# slug -> display name
CLASSES = {
    "cardboard": "Cardboard",
    "glass": "Glass",
    "metal": "Metal",
    "paper": "Paper",
    "plastic": "Plastic",
    "trash": "General waste",
}
SLUGS = list(CLASSES.keys())
DISPLAY = dict(CLASSES)
