"""Pascal VOC loading, resize / flip augmentation and batching."""
import random
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

VOC_CLASSES = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
               "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
               "train", "tvmonitor"]
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class Resize:
    """Shorter side to `min_size`, longer side capped at `max_size`."""

    def __init__(self, min_size=512, max_size=853):
        self.min_size, self.max_size = min_size, max_size

    def __call__(self, img, boxes):
        h, w = img.shape[-2:]
        s = min(self.min_size / min(h, w), self.max_size / max(h, w))
        nh, nw = int(round(h * s)), int(round(w * s))
        img = F.interpolate(img[None], size=(nh, nw), mode="bilinear", align_corners=False)[0]
        return img, boxes * s


class RandomHFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, boxes):
        if random.random() < self.p:
            w = img.shape[-1]
            img = img.flip(-1)
            boxes = boxes.clone()
            boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
        return img, boxes


class VOC(torch.utils.data.Dataset):
    """Wraps torchvision's VOC parser and returns (normalised image, {boxes, labels})."""

    def __init__(self, root, year="2007", split="trainval", train=True, min_size=512, max_size=853, download=False):
        from torchvision.datasets import VOCDetection
        self.ds = VOCDetection(root, year=year, image_set=split, download=download)
        self.resize = Resize(min_size, max_size)
        self.flip = RandomHFlip() if train else None
        self.keep_difficult = not train

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        img, ann = self.ds[i]
        boxes, labels = [], []
        objs = ann["annotation"]["object"]
        objs = objs if isinstance(objs, list) else [objs]
        for o in objs:
            if o.get("difficult", "0") == "1" and not self.keep_difficult:
                continue
            b = o["bndbox"]
            boxes.append([float(b["xmin"]) - 1, float(b["ymin"]) - 1, float(b["xmax"]), float(b["ymax"])])
            labels.append(VOC_CLASSES.index(o["name"]))
        img = torch.from_numpy(np.asarray(img.convert("RGB"), dtype=np.float32) / 255).permute(2, 0, 1)
        boxes = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        img, boxes = self.resize(img, boxes)
        if self.flip:
            img, boxes = self.flip(img, boxes)
        img = (img - MEAN) / STD
        return img, {"boxes": boxes, "labels": torch.tensor(labels, dtype=torch.long)}


def collate(batch, divisor=128):
    """Pad images to a common size divisible by the largest stride."""
    imgs, targets = zip(*batch)
    h = max(i.shape[1] for i in imgs)
    w = max(i.shape[2] for i in imgs)
    h, w = (h + divisor - 1) // divisor * divisor, (w + divisor - 1) // divisor * divisor
    out = torch.zeros(len(imgs), 3, h, w)
    for k, im in enumerate(imgs):
        out[k, :, : im.shape[1], : im.shape[2]] = im
    return out, list(targets)
