"""ResNet backbone written from scratch.

Layer names mirror torchvision so ImageNet weights can be loaded with a plain
`load_state_dict` if you want a pretrained start. Returns C3, C4, C5.
"""
import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, cin, cout, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        idt = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + idt)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, cin, width, stride=1, downsample=None):
        super().__init__()
        cout = width * self.expansion
        self.conv1 = nn.Conv2d(cin, width, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, cout, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(cout)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        idt = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.relu(out + idt)


CONFIGS = {
    "resnet18": (BasicBlock, [2, 2, 2, 2]),
    "resnet34": (BasicBlock, [3, 4, 6, 3]),
    "resnet50": (Bottleneck, [3, 4, 6, 3]),
    "resnet101": (Bottleneck, [3, 4, 23, 3]),
}


class ResNet(nn.Module):
    def __init__(self, depth="resnet50"):
        super().__init__()
        block, layers = CONFIGS[depth]
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], 2)
        self.layer3 = self._make_layer(block, 256, layers[2], 2)
        self.layer4 = self._make_layer(block, 512, layers[3], 2)
        self.out_channels = [128 * block.expansion, 256 * block.expansion, 512 * block.expansion]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _make_layer(self, block, width, n, stride=1):
        down = None
        if stride != 1 or self.inplanes != width * block.expansion:
            down = nn.Sequential(
                nn.Conv2d(self.inplanes, width * block.expansion, 1, stride, bias=False),
                nn.BatchNorm2d(width * block.expansion),
            )
        blocks = [block(self.inplanes, width, stride, down)]
        self.inplanes = width * block.expansion
        blocks += [block(self.inplanes, width) for _ in range(n - 1)]
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return c3, c4, c5


def build_backbone(depth="resnet50", pretrained=False):
    net = ResNet(depth)
    if pretrained:
        # only the weights come from torchvision, the architecture above is ours
        import torchvision
        tv = getattr(torchvision.models, depth)(weights="DEFAULT")
        missing, _ = net.load_state_dict(tv.state_dict(), strict=False)
        assert not missing, missing
    return net
