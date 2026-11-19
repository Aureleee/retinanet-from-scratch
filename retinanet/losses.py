"""Anchor matching, focal loss and box regression loss."""
import torch
import torch.nn.functional as F

from .box_ops import box_iou, encode


def match(anchors, gt_boxes, pos_thr=0.5, neg_thr=0.4):
    """Returns for each anchor the index of its gt (>= 0), -1 for background, -2 to ignore."""
    n = anchors.shape[0]
    if gt_boxes.numel() == 0:
        return torch.full((n,), -1, dtype=torch.long, device=anchors.device)
    iou = box_iou(anchors, gt_boxes)                 # (A, G)
    best_iou, best_gt = iou.max(1)
    idx = torch.full((n,), -2, dtype=torch.long, device=anchors.device)
    idx[best_iou < neg_thr] = -1
    pos = best_iou >= pos_thr
    idx[pos] = best_gt[pos]
    # every gt gets at least its best anchor (low-quality matches, as in the paper's code)
    best_anchor_iou, _ = iou.max(0)
    a_ids, g_ids = torch.nonzero((iou == best_anchor_iou[None]) & (best_anchor_iou[None] > 0), as_tuple=True)
    idx[a_ids] = g_ids
    return idx


def sigmoid_focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    """FL(p_t) = -alpha_t (1 - p_t)^gamma log(p_t), summed."""
    p = logits.sigmoid()
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    return (alpha_t * (1 - p_t) ** gamma * ce).sum()


def retinanet_loss(cls_out, box_out, anchors, targets, num_classes, alpha=0.25, gamma=2.0, beta=1 / 9):
    """cls_out: (B, A, K) logits, box_out: (B, A, 4), targets: list of dicts with boxes / labels."""
    cls_losses, box_losses = [], []
    for b, t in enumerate(targets):
        idx = match(anchors, t["boxes"])
        pos, valid = idx >= 0, idx >= -1
        num_pos = max(1, int(pos.sum()))

        cls_t = torch.zeros_like(cls_out[b])
        if pos.any():
            cls_t[pos, t["labels"][idx[pos]]] = 1.0
        cls_losses.append(sigmoid_focal_loss(cls_out[b][valid], cls_t[valid], alpha, gamma) / num_pos)

        if pos.any():
            reg_t = encode(anchors[pos], t["boxes"][idx[pos]])
            box_losses.append(F.smooth_l1_loss(box_out[b][pos], reg_t, beta=beta, reduction="sum") / num_pos)
        else:
            box_losses.append(box_out[b].sum() * 0)
    return torch.stack(cls_losses).mean(), torch.stack(box_losses).mean()
