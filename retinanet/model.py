"""RetinaNet: ResNet + FPN (P3..P7) + class / box subnets, trained with focal loss."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .anchors import Anchors
from .backbone import build_backbone
from .box_ops import batched_nms, decode
from .losses import retinanet_loss


class FPN(nn.Module):
    """Top-down pathway with lateral 1x1 convs, plus P6 / P7 computed from C5."""

    def __init__(self, in_channels, ch=256):
        super().__init__()
        self.lat = nn.ModuleList([nn.Conv2d(c, ch, 1) for c in in_channels])
        self.out = nn.ModuleList([nn.Conv2d(ch, ch, 3, padding=1) for _ in in_channels])
        self.p6 = nn.Conv2d(in_channels[-1], ch, 3, 2, 1)
        self.p7 = nn.Conv2d(ch, ch, 3, 2, 1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.zeros_(m.bias)

    def forward(self, c3, c4, c5):
        l3, l4, l5 = self.lat[0](c3), self.lat[1](c4), self.lat[2](c5)
        m4 = l4 + F.interpolate(l5, size=l4.shape[-2:], mode="nearest")   # residual merge
        m3 = l3 + F.interpolate(m4, size=l3.shape[-2:], mode="nearest")
        p3, p4, p5 = self.out[0](m3), self.out[1](m4), self.out[2](l5)
        p6 = self.p6(c5)
        p7 = self.p7(F.relu(p6))
        return [p3, p4, p5, p6, p7]


class Head(nn.Module):
    """4 x (conv 3x3 + ReLU) then a final conv. Shared across pyramid levels."""

    def __init__(self, out_per_anchor, num_anchors, ch=256, prior=None):
        super().__init__()
        layers = []
        for _ in range(4):
            layers += [nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(inplace=True)]
        self.tower = nn.Sequential(*layers)
        self.pred = nn.Conv2d(ch, num_anchors * out_per_anchor, 3, padding=1)
        self.k = out_per_anchor
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)
        if prior is not None:
            # start with P(object) = prior everywhere so the many easy negatives don't blow up the loss
            nn.init.constant_(self.pred.bias, -math.log((1 - prior) / prior))

    def forward(self, feats):
        outs = []
        for f in feats:
            o = self.pred(self.tower(f))                         # (B, A*k, H, W)
            b, _, h, w = o.shape
            outs.append(o.view(b, -1, self.k, h, w).permute(0, 3, 4, 1, 2).reshape(b, -1, self.k))
        return torch.cat(outs, 1)                                # (B, sum H*W*A, k)


class RetinaNet(nn.Module):
    def __init__(self, num_classes, depth="resnet50", pretrained_backbone=False,
                 score_thr=0.05, nms_thr=0.5, max_det=100, topk_per_level=1000):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = build_backbone(depth, pretrained_backbone)
        self.fpn = FPN(self.backbone.out_channels)
        self.anchors = Anchors()
        A = self.anchors.num_per_loc
        self.cls_head = Head(num_classes, A, prior=0.01)
        self.box_head = Head(4, A)
        self.score_thr, self.nms_thr, self.max_det, self.topk = score_thr, nms_thr, max_det, topk_per_level

    def forward(self, images, targets=None):
        feats = self.fpn(*self.backbone(images))
        cls_out, box_out = self.cls_head(feats), self.box_head(feats)
        shapes = [f.shape[-2:] for f in feats]
        anchors = self.anchors(shapes, images.device)
        if targets is not None:
            l_cls, l_box = retinanet_loss(cls_out, box_out, anchors, targets, self.num_classes)
            return {"cls": l_cls, "box": l_box, "total": l_cls + l_box}
        return self.postprocess(cls_out, box_out, anchors, shapes, images.shape[-2:])

    @torch.no_grad()
    def postprocess(self, cls_out, box_out, anchors, shapes, img_size):
        A = self.anchors.num_per_loc
        sizes = [h * w * A for h, w in shapes]
        results = []
        for b in range(cls_out.shape[0]):
            boxes_all, scores_all, labels_all = [], [], []
            start = 0
            for n in sizes:
                scores = cls_out[b, start:start + n].sigmoid().flatten()
                keep = scores > self.score_thr
                scores, ids = scores[keep], keep.nonzero().squeeze(1)
                if scores.numel() > self.topk:
                    scores, top = scores.topk(self.topk)
                    ids = ids[top]
                a_ids, labels = ids // self.num_classes, ids % self.num_classes
                boxes = decode(anchors[start:start + n][a_ids], box_out[b, start:start + n][a_ids])
                boxes[:, 0::2] = boxes[:, 0::2].clamp(0, img_size[1])
                boxes[:, 1::2] = boxes[:, 1::2].clamp(0, img_size[0])
                boxes_all.append(boxes); scores_all.append(scores); labels_all.append(labels)
                start += n
            boxes, scores, labels = torch.cat(boxes_all), torch.cat(scores_all), torch.cat(labels_all)
            keep = batched_nms(boxes, scores, labels, self.nms_thr)[: self.max_det]
            results.append({"boxes": boxes[keep], "scores": scores[keep], "labels": labels[keep]})
        return results
