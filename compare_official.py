"""
compare_official.py — Run our model AND the official SNUNet-CD on the EXACT SAME
subset of LEVIR-CD, then print a side-by-side metric comparison.

USAGE
-----

Step 1: Clone the official repo into this assignment directory:
    git clone https://github.com/likyoo/Siam-NestedUNet official_snunet

Step 2: Run our model training first (or use a checkpoint):
    python train.py --data_root data --epochs 50

Step 3: Run this comparison script:
    python compare_official.py --data_root data --our_ckpt checkpoints/best_model.pt

HOW IT WORKS
------------
- Uses our dataset loader (with the fixed seed) to select the exact same N images.
- Saves the selected filenames to `checkpoints/eval_filenames.json`.
- Loads our trained model and evaluates on those images.
- Loads the official SNUNet_ECAM model and evaluates on those same images,
  using a fresh (randomly-initialized) or trained model.
- Prints and saves a comparison table.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from models.snunet import SNUNet_ECAM as OurModel, count_parameters
from utils.dataset import LEVIRDataset
from utils.metrics import compute_metrics
from utils.transforms import val_transform


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s | %(levelname)s | %(message)s')


def get_args():
    p = argparse.ArgumentParser(description='Official vs Our SNUNet Comparison')
    p.add_argument('--data_root',    type=str, default='data',
                   help='Root of LEVIR-CD with train/val/test subfolders')
    p.add_argument('--split',        type=str, default='test',
                   choices=['train', 'val', 'test'])
    p.add_argument('--max_samples',  type=int, default=100,
                   help='Evaluate on this many images (same for both models)')
    p.add_argument('--seed',         type=int, default=777,
                   help='Must match the seed used during training')
    p.add_argument('--our_ckpt',     type=str, default='checkpoints/best_model.pt',
                   help='Path to our trained model checkpoint')
    p.add_argument('--official_ckpt',type=str, default=None,
                   help='(Optional) path to official model checkpoint (.pt)')
    p.add_argument('--official_path',type=str, default='official_snunet',
                   help='Path to cloned official Siam-NestedUNet repo')
    p.add_argument('--in_channels',  type=int, default=3)
    p.add_argument('--out_dir',      type=str, default='checkpoints',
                   help='Where to save comparison results')
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helper
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(model, dataset, device) -> dict:
    """Evaluate a model on the given dataset. Returns metrics dict."""
    model.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc='Evaluating', leave=False):
            img1, img2, label = dataset[i]
            img1  = img1.unsqueeze(0).float().to(device)
            img2  = img2.unsqueeze(0).float().to(device)

            outs = model(img1, img2)
            pred = outs[-1].argmax(dim=1).squeeze(0).cpu().numpy()   # (H, W)

            all_preds.append(pred.flatten())
            all_labels.append(label.numpy().flatten())

    return compute_metrics(
        np.concatenate(all_preds),
        np.concatenate(all_labels)
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args   = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Device: {device}')

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Select the evaluation subset ──────────────────────────────────────────
    dataset = LEVIRDataset(
        root=args.data_root,
        split=args.split,
        augment=False,
        max_samples=args.max_samples,
        seed=args.seed,
    )
    logging.info(f'Evaluation subset: {len(dataset)} samples '
                 f'(split={args.split}, seed={args.seed})')

    # Save the exact filenames used — so anyone can reproduce
    filenames_path = out_dir / 'eval_filenames.json'
    with open(filenames_path, 'w') as f:
        json.dump({'split': args.split, 'seed': args.seed,
                   'filenames': dataset.fnames}, f, indent=2)
    logging.info(f'Saved filenames → {filenames_path}')

    results = {}

    # ── Our model ─────────────────────────────────────────────────────────────
    logging.info('\n[OUR IMPLEMENTATION]')
    our_model = OurModel(in_channels=args.in_channels, out_channels=2)
    our_params = count_parameters(our_model)

    if Path(args.our_ckpt).exists():
        ckpt = torch.load(args.our_ckpt, map_location=device)
        our_model.load_state_dict(ckpt['state_dict'])
        logging.info(f'  Loaded checkpoint: {args.our_ckpt} (epoch {ckpt["epoch"]})')
    else:
        logging.warning(f'  No checkpoint found at {args.our_ckpt} — using random weights!')

    our_model = our_model.to(device)
    our_metrics = evaluate_model(our_model, dataset, device)
    results['our_model'] = {'params': our_params, **our_metrics}

    # ── Official model ─────────────────────────────────────────────────────────
    official_path = Path(args.official_path)
    if official_path.exists():
        logging.info(f'\n[OFFICIAL IMPLEMENTATION] ({official_path})')
        try:
            sys.path.insert(0, str(official_path))
            from models.Models import SNUNet_ECAM as OfficialModel
            official_model = OfficialModel(in_ch=args.in_channels, out_ch=2)
            official_params = sum(p.numel() for p in official_model.parameters()
                                  if p.requires_grad)

            if args.official_ckpt and Path(args.official_ckpt).exists():
                ckpt = torch.load(args.official_ckpt, map_location=device)
                # Official repo saves the full model object
                if isinstance(ckpt, dict):
                    if 'state_dict' in ckpt:
                        official_model.load_state_dict(ckpt['state_dict'])
                    else:
                        official_model.load_state_dict(ckpt)
                else:
                    official_model = ckpt
                logging.info(f'  Loaded official checkpoint: {args.official_ckpt}')
            else:
                logging.warning('  No official checkpoint — using random weights for architecture check only.')

            official_model = official_model.to(device)
            off_metrics = evaluate_model(official_model, dataset, device)
            results['official_model'] = {'params': official_params, **off_metrics}

        except ImportError as e:
            logging.error(f'  Import error: {e}')
            logging.error('  Make sure you cloned: https://github.com/likyoo/Siam-NestedUNet')
        finally:
            sys.path.pop(0)
    else:
        logging.info(f'\n[OFFICIAL IMPLEMENTATION] not found at {official_path}')
        logging.info('  Clone: git clone https://github.com/likyoo/Siam-NestedUNet official_snunet')

    # ── Print comparison table ─────────────────────────────────────────────────
    print('\n' + '='*70)
    print(f'  COMPARISON: {args.split} split | {len(dataset)} samples | seed={args.seed}')
    print('='*70)
    header = f"{'Metric':<18} {'Our Model':>15} {'Official':>15}"
    print(header)
    print('-'*70)

    for key in ['params', 'precision', 'recall', 'f1', 'iou', 'oa']:
        our_val = results.get('our_model', {}).get(key, 'N/A')
        off_val = results.get('official_model', {}).get(key, 'N/A')
        our_str = f'{our_val:>15,}' if key == 'params' and isinstance(our_val, int) \
                  else f'{our_val:>15.4f}' if isinstance(our_val, float) else f'{our_val:>15}'
        off_str = f'{off_val:>15,}' if key == 'params' and isinstance(off_val, int) \
                  else f'{off_val:>15.4f}' if isinstance(off_val, float) else f'{off_val:>15}'
        print(f'{key:<18} {our_str} {off_str}')

    print('='*70)

    # Save JSON results
    results_path = out_dir / 'comparison_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logging.info(f'\nResults saved → {results_path}')


if __name__ == '__main__':
    main()
