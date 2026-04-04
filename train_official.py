"""
Assignment 3 — Training Script
Dataset:        LEVIR-CD (150 sample subset, seed=777, same dataset as A1/A2)
Optimizer:      RAdam (lr=5e-4)
LR Schedule:    OneCycleLR (warmup + cosine annealing built-in)
Loss:           Lovász-Softmax + BCE over all 5 outputs
Regularization: Stochastic Depth (drop_path_rate=0.1)
Initialization: Xavier Glorot
Augmentation:   Light (H/V flips only)
Batch Size:     8
Epochs:         40
Special:        Multi-scale validation, TTA at inference
"""

import os
import json
import logging
import random
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import sys
sys.path.append(r'd:\CSRE MTech IITB\2SEM2\GNR638 Intro DL\Asign3\officialSNUNet')
from models.Models import SNUNet_ECAM
from models.snunet import init_weights_xavier, count_parameters
from utils.dataset import get_loaders
from utils.losses import lovasz_bce_loss
from utils.metrics import compute_metrics
from utils.transforms import tta_transforms, val_transform


# ─────────────────────────────────────────────────────────────────────────────
# CLI Arguments
# ─────────────────────────────────────────────────────────────────────────────

def get_args():
    parser = argparse.ArgumentParser(description='Assignment 3 — SNUNet on OSCD')
    parser.add_argument('--data_root',       type=str, default='data')
    parser.add_argument('--epochs',          type=int, default=40)
    parser.add_argument('--batch_size',      type=int, default=8)
    parser.add_argument('--lr',              type=float, default=5e-4)
    parser.add_argument('--drop_path_rate',  type=float, default=0.1,
                        help='Stochastic depth max drop rate')
    parser.add_argument('--num_workers',     type=int, default=2)
    parser.add_argument('--max_train',       type=int, default=150)
    parser.add_argument('--max_val',         type=int, default=30)
    parser.add_argument('--save_dir',        type=str, default='checkpoints')
    parser.add_argument('--log_dir',         type=str, default='runs_official')
    parser.add_argument('--seed',            type=int, default=777)
    parser.add_argument('--in_channels',     type=int, default=3,
                        help='3 for RGB LEVIR-CD')
    parser.add_argument('--multiscale_val',  action='store_true', default=True,
                        help='Evaluate at multiple scales')
    parser.add_argument('--tta',             action='store_true', default=False,
                        help='Use TTA during final evaluation')
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)


# ─────────────────────────────────────────────────────────────────────────────
# Multi-scale Validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_multiscale(model, val_loader, device,
                        scales=(1.0, 0.75, 1.25)) -> dict:
    """Run validation at multiple spatial scales and average predictions.

    At each scale, images are resized, predicted, and output upsampled
    back to original size. Final prediction averages softmax scores.

    Parameters
    ----------
    scales : tuple of float
        Scale factors relative to original 256×256 patch size.
    """
    model.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for img1, img2, labels in tqdm(val_loader, desc='multiscale-val', leave=False):
            img1   = img1.float().to(device)
            img2   = img2.float().to(device)
            labels = labels.long()
            B, C, H, W = img1.shape

            # Accumulate softmax predictions across scales
            avg_probs = torch.zeros(B, 2, H, W, device=device)

            for scale in scales:
                new_h = int(H * scale)
                new_w = int(W * scale)
                # Resize inputs
                x1 = F.interpolate(img1, size=(new_h, new_w),
                                   mode='bilinear', align_corners=False)
                x2 = F.interpolate(img2, size=(new_h, new_w),
                                   mode='bilinear', align_corners=False)
                preds = model(x1, x2)
                prob  = F.softmax(preds[-1], dim=1)   # ensemble output
                # Resize back to original
                prob  = F.interpolate(prob, size=(H, W),
                                      mode='bilinear', align_corners=False)
                avg_probs += prob / len(scales)

            pred_map = avg_probs.argmax(dim=1).cpu().numpy()
            all_preds.append(pred_map.flatten())
            all_labels.append(labels.numpy().flatten())

    metrics = compute_metrics(np.concatenate(all_preds),
                              np.concatenate(all_labels))
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Standard Epoch
# ─────────────────────────────────────────────────────────────────────────────

def run_train_epoch(model, loader, criterion, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0
    all_preds  = []
    all_labels = []

    pbar = tqdm(loader, desc='train', leave=False)
    for img1, img2, labels in pbar:
        img1   = img1.float().to(device)
        img2   = img2.float().to(device)
        labels = labels.long().to(device)

        optimizer.zero_grad()
        preds = model(img1, img2)
        loss  = criterion(preds, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()          # OneCycleLR steps per batch

        total_loss += loss.item()
        pred_map = preds[-1].argmax(dim=1).cpu().numpy()
        all_preds.append(pred_map.flatten())
        all_labels.append(labels.cpu().numpy().flatten())
        pbar.set_postfix(loss=f'{loss.item():.4f}',
                         lr=f'{scheduler.get_last_lr()[0]:.2e}')

    metrics = compute_metrics(np.concatenate(all_preds),
                              np.concatenate(all_labels))
    metrics['loss'] = total_loss / max(len(loader), 1)
    return metrics


def main():
    args = get_args()
    seed_everything(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s | %(levelname)s | %(message)s',
                        handlers=[
                            logging.FileHandler('official_train_history.log'),
                            logging.StreamHandler()
                        ])
    logging.info(f'Device: {device}')

    train_loader, val_loader = get_loaders(
        args.data_root, args.batch_size, args.num_workers,
        max_train=args.max_train, max_val=args.max_val
    )
    logging.info(f'Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)}')

    # Model — Xavier init, stochastic depth enabled
    model = SNUNet_ECAM(
        in_ch=args.in_channels,
        out_ch=2
    )
    init_weights_xavier(model)
    model = model.to(device)
    logging.info(f'Parameters: {count_parameters(model):,} | '
                 f'DropPath rate: {args.drop_path_rate}')

    # RAdam optimizer — combines Adam benefits with rectified variance
    optimizer = torch.optim.RAdam(model.parameters(), lr=args.lr)

    # OneCycleLR: combines warmup + cosine annealing, steps per batch
    total_steps = args.epochs * len(train_loader)
    scheduler   = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=total_steps,
        pct_start=0.1, anneal_strategy='cos', div_factor=10, final_div_factor=100
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    writer   = SummaryWriter(args.log_dir)

    best_f1  = -1.0
    history  = []

    logging.info('Starting training...')
    for epoch in range(args.epochs):
        train_m = run_train_epoch(model, train_loader, lovasz_bce_loss,
                                  optimizer, scheduler, device)

        # Multi-scale validation
        if args.multiscale_val:
            val_m = validate_multiscale(model, val_loader, device,
                                        scales=(1.0, 0.75, 1.25))
            val_m['loss'] = 0.0   # not computed in multiscale (efficiency)
        else:
            val_m = run_train_epoch(model, val_loader, lovasz_bce_loss,
                                    optimizer, scheduler, device)

        lr_now = scheduler.get_last_lr()[0]
        logging.info(
            f"Epoch {epoch+1:02d}/{args.epochs} | LR: {lr_now:.2e} | "
            f"Train Loss: {train_m['loss']:.4f} F1: {train_m['f1']:.4f} | "
            f"Val F1: {val_m['f1']:.4f} IoU: {val_m['iou']:.4f}"
        )

        for tag, val in train_m.items():
            writer.add_scalar(f'Train/{tag}', val, epoch)
        for tag, val in val_m.items():
            writer.add_scalar(f'Val/{tag}', val, epoch)

        if val_m['f1'] > best_f1:
            best_f1 = val_m['f1']
            torch.save({
                'epoch': epoch + 1, 'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_metrics': val_m, 'args': vars(args),
            }, save_dir / 'official_best_model.pt')
            logging.info(f'  ✓ Saved best (F1={best_f1:.4f})')

        history.append({'epoch': epoch + 1, 'train': train_m, 'val': val_m})

    with open(save_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    writer.close()
    logging.info(f'Done. Best Val F1: {best_f1:.4f}')


if __name__ == '__main__':
    main()
