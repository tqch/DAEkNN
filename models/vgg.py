import torch
import torch.nn as nn
import torch.nn.functional as F


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        cfg1 = [64, 64, 'M']
        cfg2 = [128, 128, 'M']
        cfg3 = [256, 256, 256, 'M']
        cfg4 = [512, 512, 512, 'M']
        cfg5 = [512, 512, 512, 'M']
        self.f1 = self._make_layers(cfg1, 3)
        self.f2 = self._make_layers(cfg2, 64)
        self.f3 = self._make_layers(cfg3, 128)
        self.f4 = self._make_layers(cfg4, 256)
        self.f5 = self._make_layers(cfg5, 512)
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 10),
        )

    def _make_layers(self, cfg, in_channels):
        layers = []
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)

    def forward(self, x):
        out1 = self.f1(x)
        out2 = self.f2(out1)
        out3 = self.f3(out2)
        out4 = self.f4(out3)
        out5 = self.f5(out4)
        out = out5.view(out5.size(0), -1)
        out = self.classifier(out)
        return out
