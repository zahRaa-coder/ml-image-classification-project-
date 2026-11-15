"""Handcrafted features for the tree-based models: HSV colour histogram + HOG + LBP."""
import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern

FEAT_SIZE = 128

def extract(img_rgb: np.ndarray) -> np.ndarray:
    img = cv2.resize(img_rgb, (FEAT_SIZE, FEAT_SIZE), interpolation=cv2.INTER_AREA)
    # 1) HSV colour histogram (16 H x 4 S x 4 V = 256 bins) - colour is very
    #    informative for food (red lentil soup, green sarma, ...)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h = cv2.calcHist([hsv], [0, 1, 2], None, [16, 4, 4], [0, 180, 0, 256, 0, 256]).flatten()
    h /= h.sum() + 1e-8
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) HOG - shape / edge structure (e.g. skewers, round lahmacun)
    hg = hog(gray, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2),
             feature_vector=True)
    # 3) LBP - texture (uniform, P=8, R=1 -> 10 bins) on two scales
    lbps = []
    for P, R in [(8, 1), (16, 2)]:
        lbp = local_binary_pattern(gray, P, R, method="uniform")
        hist, _ = np.histogram(lbp, bins=P + 2, range=(0, P + 2))
        lbps.append(hist / (hist.sum() + 1e-8))
    return np.concatenate([h, hg, *lbps]).astype(np.float32)
