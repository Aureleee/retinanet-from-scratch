"""Quick check that the whole pipeline learns: train on synthetic shapes (squares vs discs) on CPU.

python sanity_check.py
"""
import random

import torch

from retinanet import RetinaNet
from retinanet.metrics import mean_ap


def make_sample(size=128):
    img = torch.rand(3, size, size) * 0.2
    boxes, labels = [], []
    for _ in range(random.randint(1, 3)):
        s = random.randint(24, 56)
        x0, y0 = random.randint(0, size - s), random.randint(0, size - s)
        col = torch.rand(3, 1, 1) * 0.5 + 0.5
        if random.random() < 0.5:
            img[:, y0:y0 + s, x0:x0 + s] = col
            labels.append(0)
        else:
            yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
            mask = (xx - x0 - s / 2) ** 2 + (yy - y0 - s / 2) ** 2 <= (s / 2) ** 2
            img[:, mask] = col.view(3, 1)
            labels.append(1)
        boxes.append([x0, y0, x0 + s, y0 + s])
    return img, {"boxes": torch.tensor(boxes, dtype=torch.float32), "labels": torch.tensor(labels)}


def main(steps=400, batch=8, seed=0):
    random.seed(seed); torch.manual_seed(seed)
    val = [make_sample() for _ in range(64)]
    model = RetinaNet(num_classes=2, depth="resnet18")
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    for it in range(steps):
        model.train()
        imgs, targets = zip(*[make_sample() for _ in range(batch)])
        losses = model(torch.stack(imgs), list(targets))
        opt.zero_grad(); losses["total"].backward(); opt.step()
        if it % 50 == 0 or it == steps - 1:
            model.eval()
            with torch.no_grad():
                preds = []
                for k in range(0, len(val), 16):
                    preds += model(torch.stack([v[0] for v in val[k:k + 16]]))
            m = mean_ap(preds, [v[1] for v in val], 2)
            print(f"step {it:4d}  cls {losses['cls'].item():.3f}  box {losses['box'].item():.3f}  val mAP@0.5 {m:.3f}")


if __name__ == "__main__":
    main()
