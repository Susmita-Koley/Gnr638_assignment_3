"""
Assignment 3 — Dataset: LEVIR-CD (150 sample subset, seed=777).

Uses the SAME LEVIR-CD dataset as Assignments 1 & 2 but with seed=777,
yielding a different reproducible 150-sample subset.
Augmentation: Light (H/V flips only).

Expected folder structure:
    data/
    ├── train/
    │   ├── A/        ← pre-change RGB images
    │   ├── B/        ← post-change RGB images
    │   └── label/    ← binary change masks
    ├── val/
    └── test/
"""
import os
import random as _rnd
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from .transforms import train_transform, val_transform


class LEVIRDataset(Dataset):
    """LEVIR-CD dataset with seed=777 for Assignment 3 subset.

    Parameters
    ----------
    root        : top-level data directory
    split       : 'train', 'val', or 'test'
    augment     : apply light augmentation (flips only)
    max_samples : truncate after seed-shuffle
    seed        : 777 for Assignment 3
    """

    IMG_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}

    def __init__(self, root, split='train', augment=False,
                 max_samples=None, seed=777):
        self.root      = Path(root)
        self.transform = train_transform if augment else val_transform

        split_dir = self.root / split
        a_dir     = split_dir / 'A'
        b_dir     = split_dir / 'B'
        lbl_dir   = split_dir / 'label'

        if not a_dir.exists():
            raise FileNotFoundError(
                f"A/ not found at {a_dir}. Check your LEVIR-CD data path."
            )

        fnames = sorted([
            f for f in os.listdir(a_dir)
            if Path(f).suffix.lower() in self.IMG_EXTENSIONS
        ])

        # Seed-based shuffle → unique subset vs Assignments 1 & 2
        rng = _rnd.Random(seed)
        rng.shuffle(fnames)

        if max_samples is not None:
            fnames = fnames[:max_samples]

        self.samples = [
            {'img1': a_dir / f, 'img2': b_dir / f, 'label': lbl_dir / f}
            for f in fnames
        ]
        self.fnames = fnames

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s     = self.samples[idx]
        img1  = Image.open(s['img1']).convert('RGB')
        img2  = Image.open(s['img2']).convert('RGB')
        label = Image.open(s['label']).convert('L')
        return self.transform(img1, img2, label)


def get_loaders(data_root, batch_size=8, num_workers=2,
                max_train=None, max_val=None, seed=777):
    train_ds = LEVIRDataset(data_root, split='train',
                            augment=True, max_samples=max_train, seed=seed)
    val_ds   = LEVIRDataset(data_root, split='val',
                            augment=False, max_samples=max_val, seed=seed)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader
