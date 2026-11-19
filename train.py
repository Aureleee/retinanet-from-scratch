"""Train RetinaNet on Pascal VOC.

python train.py --data ./data --epochs 24 --depth resnet50 --pretrained
"""
import argparse
import math
import os
import time

import torch

from retinanet import RetinaNet
from retinanet.data import VOC, VOC_CLASSES, collate
from evaluate import evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--depth", default="resnet50")
    ap.add_argument("--pretrained", action="store_true", help="ImageNet weights for the backbone")
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--min-size", type=int, default=512)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="checkpoints")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    train_ds = VOC(args.data, "2007", "trainval", train=True, min_size=args.min_size, download=args.download)
    val_ds = VOC(args.data, "2007", "test", train=False, min_size=args.min_size, download=args.download)
    loader = torch.utils.data.DataLoader(train_ds, args.batch, shuffle=True, num_workers=args.workers,
                                         collate_fn=collate, drop_last=True)

    model = RetinaNet(len(VOC_CLASSES), args.depth, args.pretrained).to(dev)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=1e-4)
    total = args.epochs * len(loader)
    warmup = 500
    # linear warmup, then cosine decay
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda it: (it + 1) / warmup if it < warmup else 0.5 * (1 + math.cos(math.pi * (it - warmup) / (total - warmup))))
    scaler = torch.amp.GradScaler(enabled=dev == "cuda")
    os.makedirs(args.out, exist_ok=True)

    it = 0
    for ep in range(args.epochs):
        model.train()
        t0 = time.time()
        for imgs, targets in loader:
            imgs = imgs.to(dev)
            targets = [{k: v.to(dev) for k, v in t.items()} for t in targets]
            with torch.autocast(dev, enabled=dev == "cuda"):
                losses = model(imgs, targets)
            opt.zero_grad(set_to_none=True)
            scaler.scale(losses["total"]).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            scaler.step(opt); scaler.update(); sched.step()
            if it % 50 == 0:
                print(f"ep {ep} it {it} cls {losses['cls'].item():.3f} box {losses['box'].item():.3f} lr {sched.get_last_lr()[0]:.4f}")
            it += 1
        torch.save(model.state_dict(), f"{args.out}/retinanet_{args.depth}_ep{ep}.pt")
        m = evaluate(model, val_ds, dev)
        print(f"epoch {ep} done in {time.time() - t0:.0f}s  mAP@0.5 = {m:.4f}")


if __name__ == "__main__":
    main()
