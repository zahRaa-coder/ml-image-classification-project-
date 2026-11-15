"""EcoScan AI - Flask web app (waste classification for recycling).

Run from the project root:  python app/app.py   ->  http://127.0.0.1:5000
"""
import base64
import io
import json
import sys
import time
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import BINS_JSON, SLUGS, DISPLAY, RESULTS        # noqa: E402
from inference import load_all, overlay_cam, load_ood, confidence_score  # noqa: E402

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB
_bins = json.loads(BINS_JSON.read_text(encoding="utf-8"))
BINS = {k: v for k, v in _bins.items() if not k.startswith("_")}
BIN_NOTE = _bins["_meta"]["note"]
print("Loading models ...")
MODELS = load_all()
OOD_CFG, CENTROIDS = load_ood()
DEFAULT_MODEL = "cnn"
METRICS = {}
if (RESULTS / "comparison.csv").exists():
    import pandas as pd
    METRICS = {r.model: r._asdict() for r in pd.read_csv(RESULTS / "comparison.csv").itertuples(index=False)}


def to_b64(img: Image.Image) -> str:
    buf = io.BytesIO(); img.save(buf, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


@app.route("/")
def index():
    models = [{"key": k, "name": m.name, "acc": METRICS.get(m.name, {}).get("accuracy")}
              for k, m in MODELS.items()]
    return render_template("index.html", models=models, default=DEFAULT_MODEL,
                           class_info=[{"slug": s, "name": DISPLAY[s], "color": BINS[s]["color"]} for s in SLUGS],
                           bin_note=BIN_NOTE)


@app.route("/predict", methods=["POST"])
def predict():
    f = request.files.get("image")
    if not f:
        return jsonify(error="No image uploaded"), 400
    try:
        img = ImageOps.exif_transpose(Image.open(f.stream)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        return jsonify(error="File is not a valid image"), 400
    key = request.form.get("model", DEFAULT_MODEL)
    model = MODELS.get(key, MODELS[DEFAULT_MODEL])

    t = time.perf_counter()
    proba = model.predict_proba(img)
    ms = (time.perf_counter() - t) * 1000
    top = proba.argsort()[::-1][:3]
    preds = [{"slug": SLUGS[i], "name": DISPLAY[SLUGS[i]], "confidence": float(proba[i])} for i in top]
    # "unknown" check: score below the threshold calibrated on the validation set
    # The gate always uses the CNN centroid score: it separates waste / non-waste best
    # (results/ood_report.csv); the selected model is used for the classification itself.
    score, thr, method = confidence_score(MODELS, "cnn", img, proba, OOD_CFG, CENTROIDS)
    unknown = score < thr

    # Grad-CAM always comes from the CNN (the only differentiable model), for the top-1 class
    cnn = MODELS["cnn"]
    cam = cnn.gradcam(img, class_idx=int(top[0]))
    small = img.copy(); small.thumbnail((448, 448))
    return jsonify(model=model.name, inference_ms=round(ms, 1), predictions=preds,
                   info=BINS[SLUGS[top[0]]], all_info={p["slug"]: BINS[p["slug"]] for p in preds},
                   gradcam=to_b64(overlay_cam(img, cam)), original=to_b64(small),
                   unknown=bool(unknown), score=round(score, 3), threshold=round(thr, 3),
                   score_method=method)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
