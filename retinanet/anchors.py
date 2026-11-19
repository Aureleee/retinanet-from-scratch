"""Anchor generation: 3 scales x 3 aspect ratios per location on P3..P7."""
import math
import torch


class Anchors:
    def __init__(self, strides=(8, 16, 32, 64, 128), sizes=(32, 64, 128, 256, 512),
                 ratios=(0.5, 1.0, 2.0), scales=(1.0, 2 ** (1 / 3), 2 ** (2 / 3))):
        self.strides, self.sizes = strides, sizes
        self.ratios, self.scales = ratios, scales
        self.num_per_loc = len(ratios) * len(scales)
        self._cache = {}

    def base(self, size):
        out = []
        for r in self.ratios:
            for s in self.scales:
                area = (size * s) ** 2
                w = math.sqrt(area / r)
                h = w * r
                out.append([-w / 2, -h / 2, w / 2, h / 2])
        return torch.tensor(out)

    def __call__(self, feat_shapes, device):
        """feat_shapes: list of (H, W) for P3..P7. Returns (sum_l H*W*A, 4) in xyxy.

        Order matches the head output: level, then y, then x, then anchor.
        """
        key = (tuple(feat_shapes), str(device))
        if key in self._cache:
            return self._cache[key]
        all_a = []
        for (h, w), stride, size in zip(feat_shapes, self.strides, self.sizes):
            ys = (torch.arange(h, device=device) + 0.5) * stride
            xs = (torch.arange(w, device=device) + 0.5) * stride
            yy, xx = torch.meshgrid(ys, xs, indexing="ij")
            shifts = torch.stack([xx, yy, xx, yy], -1).reshape(-1, 1, 4)
            all_a.append((shifts + self.base(size).to(device)[None]).reshape(-1, 4))
        anchors = torch.cat(all_a)
        self._cache[key] = anchors
        return anchors
