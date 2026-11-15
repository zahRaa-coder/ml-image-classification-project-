# Data

The image files are not stored in this repository (size). Every image is listed with its
source page, licence and author:

- `manifest.csv` – all images collected online (Wikimedia Commons / Openverse)
- `splits.csv` – the 1,861 verified images and their train / val / test split
- `label_check.csv`, `rejected.txt`, `clean_report.csv` – labelling and cleaning decisions
- `ood_manifest.csv` – non-waste photos used for the "not sure" check
- `trashnet_manifest.csv` – TrashNet external test set
- `bins.json` – disposal information shown on the website

Download the images again with `python src/collect_online.py`, `python src/collect_ood.py`
and `python src/prepare_data.py`.
