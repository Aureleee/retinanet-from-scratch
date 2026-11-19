"""Box utilities: IoU, encoding / decoding of regression targets, NMS."""
import torch


def box_area(b):
    return (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)


def box_iou(a, b):
    """a: (N, 4), b: (M, 4) in xyxy. Returns (N, M)."""
    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    inter = (rb - lt).clamp(min=0).prod(-1)
    return inter / (box_area(a)[:, None] + box_area(b)[None, :] - inter + 1e-9)


# standard deviations used to normalise regression targets (as in Detectron)
STD = torch.tensor([0.1, 0.1, 0.2, 0.2])


def encode(anchors, gt):
    """xyxy anchors + matched gt boxes -> (dx, dy, dw, dh) targets."""
    aw, ah = anchors[:, 2] - anchors[:, 0], anchors[:, 3] - anchors[:, 1]
    ax, ay = anchors[:, 0] + 0.5 * aw, anchors[:, 1] + 0.5 * ah
    gw, gh = gt[:, 2] - gt[:, 0], gt[:, 3] - gt[:, 1]
    gx, gy = gt[:, 0] + 0.5 * gw, gt[:, 1] + 0.5 * gh
    t = torch.stack([(gx - ax) / aw, (gy - ay) / ah, torch.log(gw / aw), torch.log(gh / ah)], 1)
    return t / STD.to(t)


def decode(anchors, deltas, max_ratio=4.135):
    d = deltas * STD.to(deltas)
    aw, ah = anchors[:, 2] - anchors[:, 0], anchors[:, 3] - anchors[:, 1]
    ax, ay = anchors[:, 0] + 0.5 * aw, anchors[:, 1] + 0.5 * ah
    cx, cy = ax + d[:, 0] * aw, ay + d[:, 1] * ah
    w = aw * d[:, 2].clamp(max=max_ratio).exp()
    h = ah * d[:, 3].clamp(max=max_ratio).exp()
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)


def nms(boxes, scores, thr):
    """Greedy NMS. Returns kept indices sorted by decreasing score."""
    order = scores.argsort(descending=True)
    keep = []
    while order.numel():
        i = order[0]
        keep.append(i)
        if order.numel() == 1:
            break
        iou = box_iou(boxes[i][None], boxes[order[1:]])[0]
        order = order[1:][iou <= thr]
    return torch.stack(keep) if keep else torch.empty(0, dtype=torch.long, device=boxes.device)


def batched_nms(boxes, scores, labels, thr):
    """Class-aware NMS: offset boxes per class so different classes never overlap."""
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)
    offset = labels.to(boxes) * (boxes.max() + 1)
    return nms(boxes + offset[:, None], scores, thr)
