"""
Assignment 1 — Evaluation Script
Loads the best checkpoint and evaluates on the test split.
Reports: Precision, Recall, F1, IoU, OA
"""

import argparse
import logging
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from models.snunet import SNUNet_ECAM
from utils.dataset import LEVIRDataset
from utils.metrics import compute_metrics
from torch.utils.data import DataLoader


def get_args():
    parser = argparse.ArgumentParser(description='Assignment 1 — Evaluation')
    parser.add_argument('--data_root',   type=str, default='data')
    parser.add_argument('--checkpoint',  type=str, default='checkpoints/best_model.pt')
    parser.add_argument('--batch_size',  type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--split',       type=str, default='test',
                        choices=['train', 'val', 'test'])
    parser.add_argument('--in_channels', type=int, default=3)
    return parser.parse_args()


def main():
    args = get_args()
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s | %(levelname)s | %(message)s')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Evaluating on {device}')

    # ── Model ─────────────────────────────────────────────────────────────────
    model = SNUNet_ECAM(in_channels=args.in_channels, out_channels=2)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['state_dict'])
    model = model.to(device).eval()
    logging.info(f'Loaded checkpoint: epoch {ckpt["epoch"]}')

    # ── Data ──────────────────────────────────────────────────────────────────
    dataset = LEVIRDataset(args.data_root, split=args.split,
                           augment=False, max_samples=args.max_samples)
    loader  = DataLoader(dataset, batch_size=args.batch_size,
                         shuffle=False, num_workers=args.num_workers)
    logging.info(f'Dataset: {len(dataset)} samples ({args.split} split)')

    # ── Inference ─────────────────────────────────────────────────────────────
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for img1, img2, labels in tqdm(loader, desc='Evaluating'):
            img1   = img1.float().to(device)
            img2   = img2.float().to(device)
            labels = labels.long()

            preds = model(img1, img2)
            pred_map = preds[-1].argmax(dim=1).cpu().numpy()  # ensemble output

            all_preds.append(pred_map.flatten())
            all_labels.append(labels.numpy().flatten())

    all_preds  = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # ── Metrics ───────────────────────────────────────────────────────────────
    metrics = compute_metrics(all_preds, all_labels)

    logging.info('─' * 50)
    logging.info(f'  Precision : {metrics["precision"]:.4f}')
    logging.info(f'  Recall    : {metrics["recall"]:.4f}')
    logging.info(f'  F1 Score  : {metrics["f1"]:.4f}')
    logging.info(f'  IoU       : {metrics["iou"]:.4f}')
    logging.info(f'  OA        : {metrics["oa"]:.4f}')
    logging.info('─' * 50)

    results_path = Path(args.checkpoint).parent / f'eval_{args.split}.json'
    with open(results_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logging.info(f'Results saved → {results_path}')


if __name__ == '__main__':
    main()
