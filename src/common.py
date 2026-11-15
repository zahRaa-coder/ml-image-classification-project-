"""Shared helpers: dataset, transforms, MobileNetV2 builder, image loading."""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import models, transforms
from config import ROOT, SPLIT_CSV, SLUGS, IMG_SIZE, SEED

# Official preprocessing of torchvision's MobileNet_V2_Weights.IMAGENET1K_V2:
# RGB, resize shorter side to 232 (bilinear), centre-crop 224, scale to [0,1],
# normalise with ImageNet mean/std.  (Keras' preprocess_input [-1,1] does NOT apply here.)
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
RESIZE = 232
LABEL2IDX = {s: i for i, s in enumerate(SLUGS)}

train_tf = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomVerticalFlip(0.2),
    transforms.ColorJitter(0.3, 0.3, 0.2, 0.03),
    transforms.RandomGrayscale(0.05),
    transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.15))])
eval_tf = transforms.Compose([
    transforms.Resize(RESIZE), transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def set_seed(seed=SEED):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def load_rgb(path) -> Image.Image:
    im = Image.open(path)
    return ImageOps.exif_transpose(im).convert("RGB")


def load_split(split=None) -> pd.DataFrame:
    df = pd.read_csv(SPLIT_CSV)
    df["y"] = df.label.map(LABEL2IDX)
    return df if split is None else df[df.split == split].reset_index(drop=True)


class ImageDataset(Dataset):
    def __init__(self, df, tf):
        self.paths, self.ys, self.tf = list(df.path), list(df.y), tf
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, i):
        return self.tf(load_rgb(ROOT / self.paths[i])), self.ys[i]


def build_mobilenet(num_classes=len(SLUGS), pretrained=True):
    w = models.MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None
    m = models.mobilenet_v2(weights=w)
    m.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.last_channel, num_classes))
    return m


def embed(model, x):
    """1280-d global-average-pooled MobileNetV2 features."""
    f = model.features(x)
    return nn.functional.adaptive_avg_pool2d(f, 1).flatten(1)


# Augmentation for the non-CNN models (tree / hybrid): returns a PIL image.
# Applied ONLY to training images, to create extra augmented copies.
pil_aug = transforms.Compose([
    transforms.RandomResizedCrop(320, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(0.2),
    transforms.RandomRotation(20),
    transforms.ColorJitter(0.3, 0.3, 0.2, 0.03)])
AUG_COPIES = 2  # extra augmented copies per training image


def balanced_weights(y):
    """Per-sample weights inversely proportional to class frequency."""
    counts = np.bincount(y)
    return (len(y) / (len(counts) * counts))[y]
