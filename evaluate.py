"""Evaluate a checkpoint on VOC 2007 test (mAP@0.5).

python evaluate.py --data ./data --ckpt checkpoints/retinanet_resnet50_ep23.pt
"""
import argparse

import torch

from retinanet import RetinaNet
from retinanet.data import VOC, VOC_CLASSES, collate
from retinanet.metrics import mean_ap


@torch.no_grad()
def evaluate(model, ds, dev, batch=8):
    model.eval()
    loader = torch.utils.data.DataLoader(ds, batch, collate_fn=collate, num_workers=2)
    preds, gts = [], []
    for imgs, targets in loader:
        out = model(imgs.to(dev))
        preds += [{k: v.cpu() for k, v in o.items()} for o in out]
        gts += targets
    return mean_ap(preds, gts, model.num_classes)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--depth", default="resnet50")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = RetinaNet(len(VOC_CLASSES), args.depth).to(dev)
    model.load_state_dict(torch.load(args.ckpt, map_location=dev))
    print(f"mAP@0.5 = {evaluate(model, VOC(args.data, '2007', 'test', train=False), dev):.4f}")
