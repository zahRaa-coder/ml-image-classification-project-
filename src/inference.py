"""Loading the trained models + single-image prediction + Grad-CAM."""
import joblib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from xgboost import XGBClassifier
from config import MODELS, SLUGS
from common import build_mobilenet, eval_tf, embed
from features import extract

torch.set_num_threads(4)


class CNNModel:
    name = "MobileNetV2 (fine-tuned CNN)"
    def __init__(self):
        self.m = build_mobilenet(pretrained=False)
        self.m.load_state_dict(torch.load(MODELS / "mobilenetv2.pt", map_location="cpu"))
        self.m.eval()

    @torch.no_grad()
    def predict_proba(self, img: Image.Image):
        return F.softmax(self.m(eval_tf(img)[None]), 1)[0].numpy()

    def gradcam(self, img: Image.Image, class_idx=None):
        """Grad-CAM on the last conv block (features[-1], 7x7x1280). Returns HxW in [0,1]."""
        x = eval_tf(img)[None]
        f = self.m.features(x)          # last conv block output: 1x1280x7x7
        f.retain_grad()
        logits = self.m.classifier(F.adaptive_avg_pool2d(f, 1).flatten(1))
        idx = int(logits.argmax()) if class_idx is None else class_idx
        self.m.zero_grad(); logits[0, idx].backward()
        w = f.grad.mean((2, 3), keepdim=True)
        cam = F.relu((w * f).sum(1))[0].detach()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.numpy()


class HybridModel:
    name = "Hybrid (CNN embeddings + XGBoost)"
    def __init__(self):
        self.backbone = build_mobilenet(pretrained=True).eval()
        self.xgb = XGBClassifier(); self.xgb.load_model(MODELS / "xgb_cnn_embeddings.json")

    @torch.no_grad()
    def predict_proba(self, img):
        x = eval_tf(img)[None]
        e = (embed(self.backbone, x) + embed(self.backbone, torch.flip(x, [3]))) / 2
        return self.xgb.predict_proba(e.numpy())[0]


class HandcraftedModel:
    def __init__(self, kind):
        if kind == "rf":
            self.name = "Random Forest (HSV+HOG+LBP)"
            self.clf = joblib.load(MODELS / "rf_handcrafted.joblib")
        else:
            self.name = "XGBoost (HSV+HOG+LBP)"
            self.clf = XGBClassifier(); self.clf.load_model(MODELS / "xgb_handcrafted.json")

    def predict_proba(self, img):
        return self.clf.predict_proba(extract(np.asarray(img))[None])[0]


def load_all():
    return {"rf": HandcraftedModel("rf"), "xgb": HandcraftedModel("xgb"),
            "cnn": CNNModel(), "hybrid": HybridModel()}


def overlay_cam(img: Image.Image, cam: np.ndarray, alpha=0.45) -> Image.Image:
    import matplotlib
    img = img.copy(); img.thumbnail((448, 448))
    heat = Image.fromarray(np.uint8(cam * 255)).resize(img.size, Image.BICUBIC)
    rgb = (matplotlib.colormaps["jet"](np.asarray(heat) / 255.0)[..., :3] * 255).astype(np.uint8)
    return Image.blend(img, Image.fromarray(rgb), alpha)


# ---------------------------------------------------------------- unknown images
def load_ood():
    import json
    p = MODELS / "ood.json"
    cfg = json.load(open(p)) if p.exists() else {}
    c = MODELS / "cnn_centroids.npy"
    return cfg, (np.load(c) if c.exists() else None)


@torch.no_grad()
def confidence_score(models, key, img, proba, ood_cfg, centroids):
    """Returns (score, threshold, method) for the chosen model; score < threshold => unknown."""
    cfg = ood_cfg.get(key)
    if not cfg:
        return float(proba.max()), 0.0, "msp"
    if cfg["method"] == "centroid" and centroids is not None:
        e = F.normalize(embed(models["cnn"].m, eval_tf(img)[None]), dim=1)[0].numpy()
        return float((centroids @ e).max()), cfg["threshold"], "centroid"
    return float(proba.max()), cfg["threshold"], "msp"
