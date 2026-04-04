"""Shared metrics for change detection evaluation."""
import numpy as np


def compute_metrics(preds: np.ndarray, labels: np.ndarray) -> dict:
    """Compute precision, recall, F1, IoU (Jaccard), and OA for binary CD.

    Parameters
    ----------
    preds  : np.ndarray, shape (N,) — predicted binary labels
    labels : np.ndarray, shape (N,) — ground-truth binary labels

    Returns
    -------
    dict with keys: precision, recall, f1, iou, oa
    """
    TP = np.sum((preds == 1) & (labels == 1))
    TN = np.sum((preds == 0) & (labels == 0))
    FP = np.sum((preds == 1) & (labels == 0))
    FN = np.sum((preds == 0) & (labels == 1))

    eps = 1e-10
    precision = TP / (TP + FP + eps)
    recall    = TP / (TP + FN + eps)
    f1        = 2 * precision * recall / (precision + recall + eps)
    iou       = TP / (TP + FP + FN + eps)
    oa        = (TP + TN) / (TP + TN + FP + FN + eps)

    return {
        'precision': float(precision),
        'recall':    float(recall),
        'f1':        float(f1),
        'iou':       float(iou),
        'oa':        float(oa),
    }
