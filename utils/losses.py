"""
Assignment 3 — Loss Function: Lovász-Softmax + BCE.

Applies (Lovász + BCE) over all 5 deep-supervision outputs.

- Lovász-Softmax: directly optimises the IoU metric via its convex
  Lovász extension. Handles class imbalance well for small changed regions.
- BCE (CrossEntropyLoss): standard classification loss for stability.

Reference: Berman et al., "The Lovász-Softmax Loss: A Tractable Surrogate
           for the Optimization of the Intersection-Over-Union Measure in
           Neural Networks." CVPR 2018.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Lovász extension helpers
# ─────────────────────────────────────────────────────────────────────────────

def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    """Compute the Lovász gradient for a sorted binary label vector.

    The gradient of the Lovász extension of the Jaccard loss.
    gt_sorted : 1-D tensor of ground-truth labels sorted by decreasing errors.
    """
    p    = len(gt_sorted)
    gts  = gt_sorted.sum()
    # Cumulative sum of GT labels for sorted errors
    intersection = gts - gt_sorted.float().cumsum(0)
    union        = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard      = 1.0 - intersection / (union + 1e-10)
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def _lovasz_softmax_flat(probs: torch.Tensor,
                          labels: torch.Tensor,
                          classes: str = 'present') -> torch.Tensor:
    """Lovász-Softmax loss for flattened (N,) inputs.

    probs  : (C, N) — per-class softmax probabilities
    labels : (N,)   — ground-truth class indices
    """
    C = probs.shape[0]
    losses = []
    class_range = list(range(C)) if classes == 'all' else range(C)

    for c in class_range:
        fg = (labels == c).float()
        if classes == 'present' and fg.sum() == 0:
            continue
        class_pred  = probs[c]                              # (N,)
        errors      = (fg - class_pred).abs()              # per-pixel error
        errors_sort, perm = torch.sort(errors, descending=True)
        fg_sorted   = fg[perm]
        losses.append(torch.dot(errors_sort, _lovasz_grad(fg_sorted)))

    if not losses:
        return probs.sum() * 0.0
    return torch.stack(losses).mean()


def lovasz_softmax(pred: torch.Tensor, target: torch.Tensor,
                   classes: str = 'present') -> torch.Tensor:
    """Lovász-Softmax loss for a (B,C,H,W) prediction tensor.

    pred   : (B, 2, H, W) — logits
    target : (B, H, W)    — long class labels {0, 1}
    """
    probs  = F.softmax(pred, dim=1)                         # (B, C, H, W)
    B, C, H, W = probs.shape

    # Flatten to (C, B*H*W)
    probs_flat  = probs.permute(1, 0, 2, 3).contiguous().view(C, -1)
    target_flat = target.contiguous().view(-1)

    return _lovasz_softmax_flat(probs_flat, target_flat, classes)


# ─────────────────────────────────────────────────────────────────────────────
# Combined Lovász + BCE over all deep-supervision heads
# ─────────────────────────────────────────────────────────────────────────────

_ce = nn.CrossEntropyLoss()


def lovasz_bce_loss(predictions: tuple, target: torch.Tensor) -> torch.Tensor:
    """Sum of (Lovász-Softmax + CrossEntropy) over all 5 SNUNet outputs.

    Parameters
    ----------
    predictions : tuple of (B,2,H,W) tensors
    target      : LongTensor (B,H,W)
    """
    loss = torch.tensor(0.0, device=target.device)
    for pred in predictions:
        loss = loss + lovasz_softmax(pred, target) + _ce(pred, target)
    return loss
