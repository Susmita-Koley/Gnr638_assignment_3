"""
Assignment 3 — Data transforms: light augmentation (RGB LEVIR-CD).

Augmentation: Light — random horizontal and vertical flips only.
No color jitter, no rotation (matches assignment spec).
"""
import random
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image


MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


def to_tensor_normalize(img: Image.Image) -> torch.Tensor:
    t = TF.to_tensor(img)
    t = TF.normalize(t, MEAN, STD)
    return t

def label_to_tensor(label: Image.Image) -> torch.Tensor:
    import numpy as np
    lbl = torch.from_numpy((np.array(label) > 0).astype('int64'))
    return lbl


def train_transform(img1: Image.Image, img2: Image.Image, label: Image.Image):
    """Light augmentation: random flips, preserving VHR resolution on full 1024x1024."""
    # Random horizontal flip
    if random.random() > 0.5:
        img1  = TF.hflip(img1)
        img2  = TF.hflip(img2)
        label = TF.hflip(label)

    # Random vertical flip
    if random.random() > 0.5:
        img1  = TF.vflip(img1)
        img2  = TF.vflip(img2)
        label = TF.vflip(label)

    return (
        to_tensor_normalize(img1),
        to_tensor_normalize(img2),
        label_to_tensor(label),
    )


def val_transform(img1: Image.Image, img2: Image.Image, label: Image.Image):
    """Validation uses full 1024x1024 image."""
    return (
        to_tensor_normalize(img1),
        to_tensor_normalize(img2),
        label_to_tensor(label),
    )


def tta_transforms(img1: Image.Image, img2: Image.Image):
    """Test-Time Augmentation: 4 variants (original + H-flip + V-flip + HV-flip).

    Returns list of (t1, t2) tensor pairs. Average softmax scores across all 4.
    """
    variants = []
    for hflip in [False, True]:
        for vflip in [False, True]:
            a = TF.hflip(img1) if hflip else img1
            b = TF.hflip(img2) if hflip else img2
            if vflip:
                a = TF.vflip(a)
                b = TF.vflip(b)
            variants.append((to_tensor_normalize(a), to_tensor_normalize(b)))
    return variants   # 4 pairs
