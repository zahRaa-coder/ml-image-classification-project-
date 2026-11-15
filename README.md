# EcoScan AI ♻️ — Waste classification for recycling

**Course:** BLM22325E – Introduction to Machine Learning, Fall 2026
**Instructor:** Dr. Zeki Kuş

**Team:** Zahraa Esteif · Leen Alkurdi

**Report:** [`docs/EcoScan_AI_Project_Report.docx`](docs/EcoScan_AI_Project_Report.docx)

The user uploads a photo of a single piece of waste. The website predicts its type
(cardboard, glass, metal, paper, plastic, general waste), shows the **top-3 predictions with
confidence**, and explains **which bin it belongs in**, with tips. A **Grad-CAM heatmap** shows
which part of the photo the CNN based its decision on (Explainable AI). Photos that are not waste
(people, animals, places, objects …) are reported as **"not sure"** instead of being forced into a class.

> **Note:** the image files are not stored in this repository (size). All images are listed with
> their source URL, licence and author in `data/` (see [`data/README.md`](data/README.md)) and can be
> downloaded again with the collection scripts. The trained models are included, so the website runs
> without the images.

---

## 1. Data — collected online and labelled

As required by the course, the training images were **collected online and labelled** by us
(`src/collect_online.py`, `src/label_check.py`).

| | |
|---|---|
| Sources | **Wikimedia Commons API** (curated categories such as *Glass bottles*, *Beverage cans*, *Cigarette butts* + keyword search) and **Openverse API** (Creative-Commons photos from Flickr) |
| Queries | English keywords per class, e.g. *cardboard box, glass jar, soda can, crumpled paper, plastic bottle, chip bag* (full list in `SOURCES` / `EXTRA_QUERIES`) |
| Licences | only openly-licensed photos (CC BY, CC BY-SA, CC0, public domain …); **No-Derivatives licences excluded** |
| Provenance | `data/manifest.csv` — file, class, source, page URL, licence, author for **every** image |
| Initial label | the class whose category / query returned the image |

**Labelling / label verification** (`src/label_check.py`): web results are noisy (people, buildings,
old posters, the wrong material …). Every image is checked with the zero-shot model **CLIP**
(ViT-B/32) against descriptions of the 6 classes and of "not a waste item" content; an image is kept
only if CLIP agrees with the label with enough confidence. Rejected images are listed in
`data/rejected.txt` (nothing is deleted) and the decision per image is in `data/label_check.csv`. The review page
(`python src/review.py`) can exclude further images by hand. CLIP is only a labelling assistant –
it is not one of the compared models.

**External test set:** **TrashNet** (Thung & Yang, Stanford 2016, MIT licence,
https://github.com/garythung/trashnet) – studio photos from a completely different source, used
**only for testing** how well the models generalise (`data/trashnet/`, `data/trashnet_manifest.csv`).

**Cleaning** (`src/clean.py`, every decision logged in `data/clean_report.csv`, raw files never modified):
broken files, images < 224 px, extremely blurry images, and exact / near-identical duplicates
(perceptual hash, Hamming ≤ 6, checked across *all* classes) are excluded.

**Leakage protection:**
1. pHash de-duplication across all classes *before* splitting.
2. `src/near_duplicates.py` finds visually near-identical photos that pHash misses
   (cosine similarity of MobileNetV2 features ≥ 0.92) and groups them.
3. `src/split.py` uses **StratifiedGroupKFold** → each group stays inside one split
   (70 / 15 / 15, stratified by class) and the script **fails** if any near-duplicate pair crosses splits.

The class counts after cleaning and label verification are in `results/split_counts.csv`;
all models use **class weights** to compensate for the remaining imbalance.

## 2. Models

| # | Model | Input |
|---|---|---|
| 1 | **Random Forest** and **XGBoost** | handcrafted features: HSV colour histogram (256) + HOG (1764) + LBP (28) |
| 2 | **MobileNetV2** transfer learning (PyTorch) | image 224×224 |
| 3 | **Hybrid**: MobileNetV2 ImageNet embeddings (1280-d) → **XGBoost** | image 224×224 |

* **Preprocessing (MobileNetV2):** exactly the official `MobileNet_V2_Weights.IMAGENET1K_V2`
  transform: RGB → resize 232 → centre-crop 224 → [0,1] → ImageNet mean/std normalisation.
* **Data augmentation:** CNN – random resized crop, flips, rotation, colour jitter, grayscale,
  random erasing. Tree and hybrid models – 2 extra augmented copies of every *training* image
  (never of validation / test images).
* **CNN schedule (CPU friendly):** 5 epochs head only → 12 epochs fine-tuning the last 5 blocks,
  AdamW + cosine LR, label smoothing; best epoch chosen on the validation set.

## 3. Unknown images ("not sure")

`src/ood.py` compares two confidence scores — max softmax probability and cosine similarity to the
nearest class centroid of the fine-tuned CNN embeddings — using 315 openly-licensed non-waste photos (people, animals, landscapes, vehicles, furniture,
electronics, food … collected by `src/collect_ood.py`)
(`data/ood/`, provenance in `data/ood_manifest.csv`), split 50/50 into *ood-val* (choose the score)
and *ood-test* (report only). The threshold is set so that 95 % of real waste images in the
validation set are accepted. Results: `results/ood_report.csv`.

## 4. Results (held-out test set)

See `results/comparison.md` (accuracy, top-3 accuracy, macro precision / recall / F1, inference time),
`results/confusion_*.png`, `results/per_class_f1.png`, `results/model_comparison.png`,
`results/cnn_training_curves.png`, `results/classification_report_*.txt`.

Test set: 279 images, 6 classes

| model                             | accuracy   | top3_accuracy   | macro_precision   | macro_recall   | macro_f1   |   inference_ms_per_image |
|:----------------------------------|:-----------|:----------------|:------------------|:---------------|:-----------|-------------------------:|
| Random Forest (HSV+HOG+LBP)       | 44.8%      | 79.9%           | 45.3%             | 42.6%          | 42.8%      |                    191.8 |
| XGBoost (HSV+HOG+LBP)             | 46.2%      | 81.7%           | 50.5%             | 44.7%          | 45.7%      |                     14.1 |
| MobileNetV2 (fine-tuned CNN)      | 81.7%      | 97.1%           | 80.6%             | 80.2%          | 80.0%      |                     18.2 |
| Hybrid (CNN embeddings + XGBoost) | 78.1%      | 97.5%           | 76.5%             | 75.7%          | 75.7%      |                     39   |

**Best model: fine-tuned MobileNetV2** (used by default on the website). The handcrafted-feature
tree models struggle with real-world web photos (cluttered backgrounds, many viewpoints) – the
deep features of the CNN / hybrid are clearly better.

**External test (TrashNet, a different source never used for training)** – `results/comparison_external.md`:

| model                             | accuracy   | top3_accuracy   | macro_precision   | macro_recall   | macro_f1   |
|:----------------------------------|:-----------|:----------------|:------------------|:---------------|:-----------|
| Random Forest (HSV+HOG+LBP)       | 21.5%      | 62.2%           | 24.5%             | 21.5%          | 15.9%      |
| XGBoost (HSV+HOG+LBP)             | 25.4%      | 64.0%           | 31.0%             | 25.4%          | 22.1%      |
| MobileNetV2 (fine-tuned CNN)      | 58.5%      | 89.6%           | 57.7%             | 58.5%          | 57.6%      |
| Hybrid (CNN embeddings + XGBoost) | 58.6%      | 90.7%           | 57.9%             | 58.6%          | 57.9%      |

Accuracy drops on TrashNet (studio photos on a white background) – a typical *domain shift*; the
CNN still ranks the correct class in its top-3 for ~90 % of the images.

**Unknown-image detection** (CNN centroid score, threshold 0.413): accepts **96.1%** of real waste
test images but rejects only **29.1%** of non-waste photos (AUROC 0.819). Because the
web training images themselves contain varied scenes, separating "waste" from "anything else" is
much harder than with studio photos; this is listed as a limitation.

**Images per split (after cleaning + label verification):**

| label     |   train |   val |   test |
|:----------|--------:|------:|-------:|
| cardboard |     205 |    45 |     44 |
| glass     |     279 |    60 |     60 |
| metal     |     249 |    54 |     54 |
| paper     |     130 |    27 |     28 |
| plastic   |     244 |    51 |     51 |
| trash     |     196 |    42 |     42 |
| total     |    1303 |   279 |    279 |

## 5. How to run

Requirements: Python 3.10+, ~3 GB disk, CPU is enough (no GPU needed).

```bash
python -m venv .venv
```

```bash
# Windows: .venv\Scripts\activate      Linux / macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Collect the images online (~1 hour because of API rate limits; already done, the images are in `data/raw`):

```bash
python src/collect_online.py
```

Run the rest of the pipeline (TrashNet external test download → clean → label check → split → train 3 models → unknown threshold → evaluate),
about 40 minutes on a 4-core CPU:

```bash
python run_pipeline.py
```

Start the website and open http://127.0.0.1:5000 :

```bash
python app/app.py
```

Optional – review images in the browser (click wrong images to exclude them; nothing is deleted),
then re-run from the cleaning step:

```bash
python src/review.py
```

```bash
python run_pipeline.py --from clean
```

## 6. Project structure

```
EcoScan-AI/
├── docs/                project report
├── app/                 Flask website (app.py, templates/index.html, static/style.css, static/app.js)
├── data/
│   ├── raw/             images collected online (per class) – never modified
│   ├── trashnet/        TrashNet photos – external test set only
│   ├── clean/           filtered copies used for training
│   ├── ood/             non-waste images for the "unknown" test
│   ├── manifest.csv     provenance + licence of every collected image
│   ├── label_check.csv  label verification result per image
│   ├── rejected.txt     images excluded from training (reversible)
│   ├── clean_report.csv why each image was kept / excluded
│   ├── splits.csv       train / val / test assignment (+ duplicate group)
│   └── bins.json        disposal information shown on the website
├── models/              trained models
├── results/             metrics, tables and plots
├── src/                 pipeline scripts (collect_online, prepare_data, clean, label_check, near_duplicates, split,
│                        features, train_tree, train_cnn, train_hybrid, ood, evaluate, review)
└── run_pipeline.py
```

## 7. Limitations

* Web images are noisy: after CLIP verification roughly 85 % of the sampled labels were visually
  correct (estimate from the review sheets); remaining errors can be removed on the review page.
* *Paper* has the fewest verified images (~185); class weights compensate.
* The "not sure" gate is weak on web-trained models (rejects ~29 % of non-waste photos at 95 % acceptance of real waste).
* Disposal guidance is general; recycling rules differ between municipalities.

## 8. Credits & licences

* Training images: Wikimedia Commons / Flickr (via Openverse) contributors – licence and author of
  every image in `data/manifest.csv`.
* External test set: TrashNet © 2017 Gary Thung — MIT licence.
* CLIP (label verification): OpenAI, MIT licence.
* Non-waste test photos: Wikimedia Commons, under the licences listed per
  file in `data/ood_manifest.csv` (No-Derivatives licences excluded).
* MobileNetV2 ImageNet weights: torchvision (BSD-3-Clause).
