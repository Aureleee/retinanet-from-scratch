import torch

from retinanet import RetinaNet
from retinanet.anchors import Anchors
from retinanet.box_ops import box_iou, decode, encode, nms
from retinanet.losses import match, sigmoid_focal_loss


def test_encode_decode_roundtrip():
    a = torch.tensor([[10., 10., 50., 70.], [0., 0., 32., 32.]])
    g = torch.tensor([[12., 8., 60., 66.], [4., 2., 30., 40.]])
    assert torch.allclose(decode(a, encode(a, g)), g, atol=1e-4)


def test_iou_and_nms():
    b = torch.tensor([[0., 0., 10., 10.], [1., 1., 11., 11.], [20., 20., 30., 30.]])
    assert abs(box_iou(b[:1], b[1:2]).item() - 81 / 119) < 1e-5
    keep = nms(b, torch.tensor([0.9, 0.8, 0.7]), 0.5)
    assert keep.tolist() == [0, 2]


def test_anchor_count():
    shapes = [(64, 64), (32, 32), (16, 16), (8, 8), (4, 4)]
    assert Anchors()(shapes, "cpu").shape == (sum(h * w for h, w in shapes) * 9, 4)


def test_match_gives_every_gt_an_anchor():
    anchors = Anchors()([(8, 8), (4, 4), (2, 2), (1, 1), (1, 1)], "cpu")
    gt = torch.tensor([[3., 3., 9., 9.]])            # tiny box, no anchor above 0.5
    assert (match(anchors, gt) == 0).any()


def test_focal_loss_downweights_easy_examples():
    t = torch.zeros(1)
    easy, hard = sigmoid_focal_loss(torch.tensor([-5.]), t), sigmoid_focal_loss(torch.tensor([2.]), t)
    assert easy < hard * 1e-3


def test_forward_train_and_eval():
    m = RetinaNet(num_classes=3, depth="resnet18")
    x = torch.randn(2, 3, 256, 256)
    t = [{"boxes": torch.tensor([[20., 30., 120., 140.]]), "labels": torch.tensor([1])},
         {"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.long)}]
    losses = m(x, t)
    losses["total"].backward()
    assert torch.isfinite(losses["total"])
    m.eval()
    out = m(x)
    assert len(out) == 2 and out[0]["boxes"].shape[1] == 4
