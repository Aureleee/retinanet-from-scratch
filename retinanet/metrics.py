"""VOC-style mean average precision at a fixed IoU threshold (all-point interpolation)."""
import numpy as np
import torch

from .box_ops import box_iou


def average_precision(recall, precision):
    r = np.concatenate([[0.0], recall, [1.0]])
    p = np.concatenate([[0.0], precision, [0.0]])
    for i in range(len(p) - 2, -1, -1):
        p[i] = max(p[i], p[i + 1])
    idx = np.where(r[1:] != r[:-1])[0]
    return float(np.sum((r[idx + 1] - r[idx]) * p[idx + 1]))


def mean_ap(preds, gts, num_classes, iou_thr=0.5):
    """preds / gts: lists (one entry per image) of dicts with boxes, labels (+ scores for preds)."""
    aps = []
    for c in range(num_classes):
        dets, n_gt, gt_by_img = [], 0, {}
        for i, (p, g) in enumerate(zip(preds, gts)):
            gb = g["boxes"][g["labels"] == c]
            gt_by_img[i] = (gb, torch.zeros(len(gb), dtype=torch.bool))
            n_gt += len(gb)
            m = p["labels"] == c
            dets += [(float(s), i, b) for s, b in zip(p["scores"][m], p["boxes"][m])]
        if n_gt == 0:
            continue
        dets.sort(key=lambda d: -d[0])
        tp = np.zeros(len(dets)); fp = np.zeros(len(dets))
        for k, (_, i, b) in enumerate(dets):
            gb, used = gt_by_img[i]
            if len(gb) == 0:
                fp[k] = 1
                continue
            iou = box_iou(b[None], gb)[0]
            j = int(iou.argmax())
            if iou[j] >= iou_thr and not used[j]:
                tp[k] = 1; used[j] = True
            else:
                fp[k] = 1
        tp, fp = np.cumsum(tp), np.cumsum(fp)
        aps.append(average_precision(tp / n_gt, tp / np.maximum(tp + fp, 1e-9)))
    return float(np.mean(aps)) if aps else 0.0
